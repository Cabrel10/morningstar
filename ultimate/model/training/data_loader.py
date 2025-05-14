import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
from sklearn.preprocessing import RobustScaler
import logging  # Importer le module logging

from config.config import Config  # Importer la classe Config

# Initialiser le logger pour ce module
logger = logging.getLogger(__name__)


def load_data(file_path):
    """Charge les données prétraitées à partir d'un fichier Parquet ou CSV."""
    file_path_str = str(file_path)  # Assurer que c'est une chaîne
    if file_path_str.endswith(".parquet"):
        logger.info(f"Chargement des données depuis Parquet: {file_path_str}")
        return pd.read_parquet(file_path)
    elif file_path_str.endswith(".csv"):
        logger.info(f"Chargement des données depuis CSV: {file_path_str}")
        # Essayer de détecter l'index 'timestamp' si présent
        try:
            # Lire d'abord les colonnes pour vérifier si 'timestamp' existe
            cols = pd.read_csv(file_path, nrows=0).columns
            if "timestamp" in cols:
                return pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"Erreur lors de la lecture CSV avec index timestamp: {e}. Tentative sans index.")
            return pd.read_csv(file_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_path_str}. Attendu .parquet ou .csv")


# La fonction _preprocess_market_regime n'est plus nécessaire car le mapping
# et le cast seront faits dans load_and_split_data directement sur le dictionnaire y_labels.


def _convert_labels_to_tensors(data, label_columns, label_mappings):
    """Convertit les labels en tenseurs TensorFlow avec gestion des mappings."""
    y_tensors = {}

    for name in label_columns:
        if name == "sl_tp":
            if "level_sl" in data.columns and "level_tp" in data.columns:
                sl_tp_df = data[["level_sl", "level_tp"]].copy()
                sl_tp_df["level_sl"] = pd.to_numeric(sl_tp_df["level_sl"], errors="coerce")
                sl_tp_df["level_tp"] = pd.to_numeric(sl_tp_df["level_tp"], errors="coerce")
                # Remplacer NaN/inf par 0.0 pour la conversion en tenseur
                sl_tp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                sl_tp_df.fillna(0.0, inplace=True)
                y_tensors[name] = tf.convert_to_tensor(sl_tp_df.values, dtype=tf.float32)
            else:
                print(f"WARN: Colonnes 'level_sl' ou 'level_tp' non trouvées pour générer le label 'sl_tp'.")
            continue

        if name not in data.columns:
            print(f"WARN: Colonne de label '{name}' non trouvée dans les données.")

            continue  # Ignore ce label s'il manque

        series = data[name]
        values, dtype = _process_label_series(series, name, label_mappings)

        try:
            y_tensors[name] = tf.convert_to_tensor(values, dtype=dtype)
        except Exception as e:
            print(f"Erreur conversion {name} (dtype={dtype}): {values[:10]}")
            raise e

    return y_tensors


# Removed _handle_sl_tp function as scaling and normalization are moved to training_script.py


def _process_label_series(series, name, label_mappings):
    """Traite une série de labels et retourne les valeurs + dtype approprié."""
    dtype = tf.float32
    values = series.values

    # Si un mapping est fourni, l'utiliser
    if label_mappings and name in label_mappings:
        return _apply_label_mapping(series, label_mappings[name])

    # Si la colonne est de type object (textuelle) et aucun mapping n'a été appliqué, factoriser.
    if series.dtype == "object":
        print(f"INFO: Factorisation automatique de la colonne label '{name}' (type object, pas de mapping fourni).")
        # Remplacer les NaN par une chaîne placeholder avant factorisation pour éviter -1, puis les remettre à 0 après.
        # pd.factorize convertit NaN en -1.
        placeholder = "__NAN_PLACEHOLDER__"
        series_filled_for_factorization = series.fillna(placeholder)
        factorized_values, uniques = pd.factorize(series_filled_for_factorization)

        # Si le placeholder a été factorisé, remplacer son code par 0 (ou une autre valeur pour NaN si désiré)
        # Ceci est plus robuste si on veut que NaN devienne 0 après factorisation.
        # Cependant, pd.factorize met -1 pour les NaN, ce qui est souvent géré par les modèles d'embedding.
        # Pour être cohérent avec le fillna(0) des labels numériques, on peut vouloir que NaN devienne 0.
        # Mais pour la factorisation pure, -1 est le standard pour NaN.
        # Le test actuel s'attend à ce que NaN devienne -1 puis soit traité.
        # Laissons pd.factorize gérer NaN comme -1 pour l'instant.
        # values = factorized_values
        # Si on veut que NaN devienne 0 après factorisation (comme pour les numériques):
        # nan_code = -1 # pd.factorize met -1 pour les NaN
        # if placeholder in uniques:
        #     nan_code = list(uniques).index(placeholder)
        # factorized_values[factorized_values == nan_code] = 0 # Remplace le code du NaN par 0
        # Ou plus simple si on accepte -1 pour NaN:
        values = factorized_values
        return values.astype(np.int64), tf.int64

    # Si c'est une colonne catégorielle connue mais déjà numérique (ou devenue numérique après factorisation ci-dessus)
    # Cette condition est maintenant partiellement couverte par la précédente si le label était object.
    # Elle s'applique si le label était déjà numérique mais est sémantiquement catégoriel.
    if name in ["trading_signal", "market_regime"]:  # Et implicitement series.dtype n'est pas 'object' ici
        # Remplacer les NaN par 0 (ou une autre valeur par défaut appropriée)
        # avant de caster en int64 pour éviter les erreurs de conversion de NaN en int.
        values = series.fillna(0).values
        dtype = tf.int64  # S'assurer que le dtype est entier pour TF
    elif pd.api.types.is_numeric_dtype(series.dtype):  # Pour les autres labels numériques
        values = series.fillna(0).values  # Remplacer NaN par 0
        # dtype reste tf.float32 par défaut, sauf si modifié par un mapping

    return values, dtype


def _apply_label_mapping(series, mapping_dict):
    """Applique un mapping de valeurs textuelles à numériques."""
    str_series = series.astype(str).str.strip()
    mapped = str_series.map(lambda x, m=mapping_dict: m.get(x.strip(), float("nan")))

    if mapped.isnull().any():
        unmapped = str_series[mapped.isnull()].unique()
        raise ValueError(f"Valeurs non mappées dans '{series.name}': {list(unmapped)}")

    return pd.to_numeric(mapped, downcast="integer").values, tf.int64


def load_and_split_data(
    file_path,
    label_columns=None,
    label_mappings=None,
    as_tensor=False,
    num_mcp_features=128,
    num_technical_features=38,
    num_llm_features=768,
):
    """
    Charge et sépare les données en features techniques, LLM, MCP, instrument et labels.

    Args:
        file_path (str or Path): Chemin vers le fichier Parquet.
        label_columns (list, optional): Noms des colonnes de labels à extraire.
        label_mappings (dict, optional): Mappings pour convertir les labels textuels en entiers.
        as_tensor (bool, optional): Si True, retourne les données sous forme de tf.Tensor.
        num_mcp_features (int): Nombre attendu de features MCP.
        num_technical_features (int): Nombre attendu de features techniques.
        num_llm_features (int): Nombre attendu de features LLM.

    Returns:
        tuple: Un tuple contenant (X_features, y_labels)
               X_features (dict): Dictionnaire mappant les noms d'entrée ('technical_input',
                                  'llm_input', 'mcp_input', 'instrument_input') à leurs
                                  données (np.ndarray ou tf.Tensor).
               y_labels (dict): Dictionnaire mappant les noms de sortie aux labels
                                (np.ndarray, pd.Series, pd.DataFrame ou tf.Tensor).
    """
    if label_columns is None:
        # Mise à jour des labels par défaut selon les spécifications Agent 4
        label_columns = ["signal", "volatility_quantiles", "market_regime", "sl_tp"]

    cfg = Config()  # Charger la config pour accéder aux mappings
    data = load_data(file_path)

    # Le prétraitement de market_regime est déplacé après l'extraction de y_labels

    # Vérification des colonnes de labels requises (celles qui ne sont pas 'sl_tp')
    required_label_cols = [col for col in label_columns if col != "sl_tp"]
    missing_labels = [col for col in required_label_cols if col not in data.columns]
    if missing_labels:
        raise ValueError(f"Colonnes de labels manquantes: {missing_labels}")
    # Vérifier aussi les colonnes sources pour sl_tp si sl_tp est demandé
    if "sl_tp" in label_columns and ("level_sl" not in data.columns or "level_tp" not in data.columns):
        raise ValueError("Colonnes 'level_sl' et/ou 'level_tp' manquantes pour générer 'sl_tp'.")

    # --- Extraction des Features ---
    all_cols = set(data.columns)

    # 1. Identifier les colonnes LLM
    llm_cols = sorted([col for col in all_cols if col.startswith("bert_")])  # Modifié: llm_ -> bert_

    # 2. Identifier les colonnes MCP
    mcp_cols = sorted([col for col in all_cols if col.startswith("mcp_")])

    # 3. Identifier la colonne Instrument
    instrument_col = "instrument_type"  # Nom conventionnel
    instrument_cols = []
    if instrument_col in all_cols:
        instrument_cols = [instrument_col]  # Liste pour cohérence
        print(f"INFO: Colonne instrument '{instrument_col}' trouvée.")
    else:
        print(f"WARN: Colonne '{instrument_col}' non trouvée. L'input instrument ne sera pas généré.")

    # 4. Identifier les colonnes de labels sources (y compris celles pour sl_tp)
    # Utiliser les label_columns demandés + les sources potentielles pour sl_tp
    label_source_cols = set(required_label_cols)  # required_label_cols défini plus haut
    if "sl_tp" in label_columns:
        label_source_cols.update(["level_sl", "level_tp"])

    # 5. Déduire les colonnes techniques par exclusion
    # Ajouter les colonnes OHLCV de base et autres colonnes non-feature/système à exclure
    base_cols_to_exclude = {
        "open",
        "high",
        "low",
        "close",
        "volume",  # OHLCV
        "trading_signal",
        "volatility",
        "news_snippets",  # Autres non-features
        "split",
        "symbol",
        "timestamp_numeric",
        "timestamp",  # Colonnes système/catégorielles/date
        "Unnamed: 0",  # Index potentiel du CSV
        "level_sl",
        "level_tp",  # Cibles SL/TP à exclure des features
    }

    # Identifier les colonnes HMM pour les extraire séparément
    hmm_cols = sorted([col for col in all_cols if col.startswith("hmm_")])

    excluded_for_tech = (
        set(llm_cols) | set(mcp_cols) | set(instrument_cols) | set(hmm_cols) | label_source_cols | base_cols_to_exclude
    )
    feature_cols = sorted([col for col in all_cols if col not in excluded_for_tech])

    # --- Debug Logging Amélioré ---
    print(f"--- Debug: Data Loader Feature/Label Extraction ---")
    print(f"Total columns in Parquet: {len(all_cols)}")
    print(f"Requested label columns: {label_columns}")
    print(f"Source columns for labels: {sorted(list(label_source_cols))}")
    print(f"Excluded columns for tech features: {sorted(list(excluded_for_tech))}")
    print(
        f"Detected Technical Features ({len(feature_cols)}): {feature_cols[:5]}...{feature_cols[-5:] if len(feature_cols)>5 else ''}"
    )
    print(f"Detected LLM Features ({len(llm_cols)}): {llm_cols[:2]}...{llm_cols[-2:] if len(llm_cols)>2 else ''}")
    print(f"Detected MCP Features ({len(mcp_cols)}): {mcp_cols[:2]}...{mcp_cols[-2:] if len(mcp_cols)>2 else ''}")
    print(f"Detected HMM Features ({len(hmm_cols)}): {hmm_cols}")
    print(f"Detected Instrument Column: {instrument_cols}")
    print(f"--- End Debug ---")

    # --- Validation des Dimensions ---
    if len(feature_cols) != num_technical_features:
        raise ValueError(f"{num_technical_features} features techniques requises (trouvé {len(feature_cols)}).")
    if len(llm_cols) != num_llm_features:
        raise ValueError(f"{num_llm_features} embeddings LLM requis (trouvé {len(llm_cols)}).")
    if len(mcp_cols) != num_mcp_features:
        raise ValueError(f"{num_mcp_features} features MCP requises (trouvé {len(mcp_cols)}).")
    # Pas de validation de dimension pour instrument_type car c'est une seule colonne
    print(
        f"Dimensions validées - Tech: {len(feature_cols)}, LLM: {len(llm_cols)}, MCP: {len(mcp_cols)}, Instrument: {len(instrument_cols)}"
    )

    # --- Préparation des Données X et Y ---
    X_features = {}

    # Ajouter les features seulement si les colonnes correspondantes existent
    if feature_cols:
        # Remplacer NaN par 0 AVANT de convertir en numpy array et float32
        x_technical_data = data[feature_cols].fillna(0).values.astype(np.float32)
        X_features["technical_input"] = x_technical_data

    if llm_cols:
        x_llm_data = data[llm_cols].fillna(0).values.astype(np.float32)
        X_features["cryptobert_input"] = x_llm_data  # Utilise cryptobert_input pour les colonnes bert_*

    if mcp_cols:
        x_mcp_data = data[mcp_cols].fillna(0).values.astype(np.float32)
        X_features["mcp_input"] = x_mcp_data

    if hmm_cols:
        # S'assurer que les colonnes HMM sont numériques et remplir NaN
        hmm_data = data[hmm_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        X_features["hmm_input"] = hmm_data

    if instrument_cols:
        x_instrument_data, _ = pd.factorize(data[instrument_col])
        X_features["instrument_input"] = x_instrument_data.astype(np.int64)

    # Ajouter d'autres inputs (sentiment, market) s'ils existent et sont traités
    # Exemple (à adapter si ces colonnes existent) :
    # sentiment_cols = sorted([col for col in all_cols if col.startswith('sentiment_')])
    # if sentiment_cols:
    #     x_sentiment_data = data[sentiment_cols].values.astype(np.float32)
    #     X_features["sentiment_input"] = x_sentiment_data
    # market_cols = sorted([col for col in all_cols if col.startswith('market_')])
    # if market_cols:
    #     x_market_data = data[market_cols].values.astype(np.float32)
    #     X_features["market_input"] = x_market_data

    # Préparer le dictionnaire Y (labels)
    y_labels = {}
    if as_tensor:
        # Convertir X en Tensors
        for key, value in X_features.items():
            # Déterminer le dtype cible pour le tenseur
            target_dtype = tf.int64 if key == "instrument_input" else tf.float32
            try:
                # S'assurer que la conversion est possible (ex: pas de string vers int)
                if target_dtype == tf.int64 and not np.issubdtype(value.dtype, np.integer):
                    # Gérer le cas où instrument_input n'est pas déjà int (factorize devrait le faire)
                    print(
                        f"WARN: Tentative de conversion de {key} (dtype {value.dtype}) en int64 échouerait. Vérifier la factorisation."
                    )
                    # On pourrait tenter une conversion explicite ici si nécessaire, mais factorize le fait déjà.
                X_features[key] = tf.convert_to_tensor(value, dtype=target_dtype)
            except Exception as e:
                print(
                    f"Erreur conversion X['{key}'] (source dtype: {value.dtype}) en Tensor (target dtype={target_dtype}): {value[:5]}"
                )
                raise e
        # Convertir Y en Tensors
        y_labels = _convert_labels_to_tensors(data, label_columns, label_mappings)
    else:
        # Retourner Y comme Series/DataFrame pandas
        y_labels = {}
        for col in label_columns:
            if col in data.columns:
                y_labels[col] = data[col]
            elif col == "sl_tp" and "level_sl" in data.columns and "level_tp" in data.columns:
                # Gérer sl_tp séparément: retourner les colonnes brutes, gérer NaN/inf
                sl_tp_df = data[["level_sl", "level_tp"]].copy()
                sl_tp_df["level_sl"] = pd.to_numeric(sl_tp_df["level_sl"], errors="coerce")
                sl_tp_df["level_tp"] = pd.to_numeric(sl_tp_df["level_tp"], errors="coerce")
                sl_tp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                nan_count = sl_tp_df.isnull().sum().sum()
                if nan_count > 0:
                    print(f"INFO: Found and filled {nan_count} NaN/inf values in raw SL/TP labels with 0.")
                sl_tp_df.fillna(0, inplace=True)
                y_labels["sl_tp"] = sl_tp_df  # Return as DataFrame
            else:
                print(f"WARN: Colonne label '{col}' demandée mais non trouvée dans les données.")

        # Appliquer le mapping et caster en int32 pour les labels catégoriels
        label_mappings_from_config = cfg.get_config("data.label_mappings", {})
        for label_name, label_series in y_labels.items():
            if label_name in label_mappings_from_config:
                mapping = label_mappings_from_config[label_name]
                print(f"INFO: Application du mapping pour '{label_name}': {mapping}")
                # Assurer que les clés du mapping sont du même type que les valeurs de la série
                try:
                    # Tenter de convertir les clés du mapping en type de la série si nécessaire
                    series_dtype = label_series.dtype
                    if pd.api.types.is_numeric_dtype(series_dtype):
                        safe_mapping = {type(label_series.iloc[0])(k): v for k, v in mapping.items()}
                    else:  # Si la série est object/string, les clés YAML devraient être ok
                        safe_mapping = mapping

                    mapped_series = label_series.map(safe_mapping)

                    # Vérifier les NaN après le mapping
                    nan_after_map = mapped_series.isnull().sum()
                    if nan_after_map > 0:
                        unmapped_values = label_series[mapped_series.isnull()].unique()
                        print(
                            f"WARN: {nan_after_map} valeurs de '{label_name}' n'ont pas pu être mappées: {list(unmapped_values)}. Elles seront remplacées par NaN puis 0."
                        )
                        mapped_series = mapped_series.fillna(0)  # Remplacer NaN par 0 (ou une autre valeur par défaut)

                    y_labels[label_name] = mapped_series.astype("int32")  # Caster en int32
                    print(f"INFO: Label '{label_name}' mappé et casté en int32.")

                except Exception as e:
                    print(f"ERREUR lors du mapping/cast de '{label_name}': {e}")
                    print(f"       Mapping utilisé: {mapping}")
                    print(f"       Premières valeurs de la série: {label_series.head().tolist()}")
                    # Optionnel: lever l'erreur ou continuer avec les données non mappées/castées
                    # raise e
            elif (
                isinstance(label_series, pd.Series) and label_series.dtype == "object" and label_name not in ["sl_tp"]
            ):  # Vérifier que c'est une Series avant d'accéder à dtype
                print(
                    f"INFO: Factorisation automatique du label textuel '{label_name}' car aucun mapping n'a été fourni."
                )
                factorized_values, _ = pd.factorize(label_series)
                y_labels[label_name] = pd.Series(
                    factorized_values, index=label_series.index, name=label_series.name
                ).astype("int32")
            elif label_name == "sl_tp":  # Assurer que sl_tp est un DataFrame avant d'essayer de le caster
                if isinstance(y_labels[label_name], pd.DataFrame):
                    y_labels[label_name] = y_labels[label_name].astype("float32")
                # Si ce n'est pas un DataFrame (par ex. si les colonnes sources manquaient), il ne sera pas dans y_labels ou sera None
                print(f"INFO: Label '{label_name}' casté en float32.")
                print(f"INFO: Label '{label_name}' casté en float32.")
            # Ajouter d'autres casts si nécessaire pour d'autres types de labels

    # Conversion en Tensors si demandé (déplacé après la création complète de X_features)
    # Le bloc dupliqué pour sl_tp a été supprimé ici. Le traitement se fait dans la boucle ci-dessus.

    # Conversion en Tensors si demandé
    if as_tensor:
        for key, value in X_features.items():
            target_dtype = tf.int64 if key == "instrument_input" else tf.float32
            try:
                # S'assurer que les NaN sont remplacés par 0 avant la conversion en tenseur pour les floats
                if target_dtype == tf.float32 and isinstance(value, np.ndarray):
                    value_filled = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    value_filled = value

                if target_dtype == tf.int64 and not np.issubdtype(value_filled.dtype, np.integer):
                    print(
                        f"WARN: Tentative de conversion de {key} (dtype {value_filled.dtype}) en int64 échouerait. Vérifier la factorisation."
                    )
                X_features[key] = tf.convert_to_tensor(value_filled, dtype=target_dtype)
            except Exception as e:
                print(
                    f"Erreur conversion X['{key}'] (source dtype: {value.dtype if isinstance(value, np.ndarray) else type(value)}) en Tensor (target dtype={target_dtype}): {value[:5] if isinstance(value, np.ndarray) else value}"
                )
                raise e
        y_labels = _convert_labels_to_tensors(data, label_columns, label_mappings)

    # Retourner aussi la liste des noms de features techniques utilisées
    # feature_cols a été défini plus haut comme la liste des colonnes techniques déduites.
    # C'est cette liste qui est pertinente pour ExplanationDecoder si les inputs techniques sont passés "tels quels"
    # ou si l'attention se fait sur ces features initiales.
    # Si les features techniques sont transformées par des couches denses avant le ReasoningModule,
    # alors feature_names pour ExplanationDecoder devrait refléter cela (plus complexe).
    # Pour l'instant, on retourne la liste des colonnes techniques brutes.
    # Le script d'entraînement devra sauvegarder cette liste.
    # La fonction preprocess_data dans predict_with_reasoning.py retourne déjà une liste de feature_names.
    # Assurons-nous que la logique est cohérente.
    # feature_cols ici sont les noms des colonnes techniques brutes.
    # La liste des features pour ExplanationDecoder doit correspondre à l'entrée 'features' du ReasoningModule.
    # Si 'features' du ReasoningModule est la concaténation de toutes les features encodées,
    # alors feature_names est plus complexe.
    # Pour l'instant, supposons que feature_names se réfère aux colonnes techniques initiales.
    
    # La variable `feature_cols` contient les noms des colonnes techniques.
    # `llm_cols`, `mcp_cols`, `hmm_cols` contiennent les autres.
    # `ExplanationDecoder` prend `feature_names` qui est la liste des noms des features
    # sur lesquelles l'attention est calculée ou qui sont pertinentes pour l'explication.
    # Si l'attention dans ReasoningModule se fait sur `merged_features` (après encodage individuel),
    # alors `feature_names` devrait correspondre aux dimensions de `merged_features`.
    # C'est un point complexe.
    # Pour une première version, on peut retourner `feature_cols` (noms des features techniques brutes).
    # Le script d'entraînement sauvegardera cette liste.
    # `predict_with_reasoning.preprocess_data` retourne déjà une liste `feature_names` construite
    # en concaténant technical_cols, llm_cols, mcp_cols etc. C'est cette liste qui est utilisée
    # pour initialiser ExplanationDecoder dans `predict_with_reasoning.py`.
    # Donc, `load_and_split_data` devrait retourner une liste similaire pour la cohérence.
    
    # Construire la liste complète des noms de features dans l'ordre où elles sont concaténées
    # pour former l'entrée principale du ReasoningModule (si c'est `merged_features`).
    # L'ordre est: tech, llm, mcp, hmm, instrument, sentiment, cryptobert, market (selon build_reasoning_model)
    # Cependant, `ExplanationDecoder` est initialisé avec une liste plate de noms.
    # Il est plus probable que `feature_names` pour `ExplanationDecoder` se réfère aux features
    # *avant* leur encodage individuel si l'attention est interprétée à ce niveau.
    # Pour l'instant, nous allons retourner la liste des colonnes techniques brutes,
    # car c'est souvent sur celles-ci que l'on veut voir l'importance.
    # Le script d'entraînement (enhanced_reasoning_training.py) devra sauvegarder cette liste.
    
    # La fonction `preprocess_data` de `predict_with_reasoning.py` construit `feature_names` comme:
    # technical_cols + llm_cols + mcp_cols + hmm_cols + sentiment_cols + cryptobert_cols + market_info_cols
    # Faisons de même ici pour la cohérence.
    
    all_input_feature_names = []
    if feature_cols: all_input_feature_names.extend(feature_cols)
    if llm_cols: all_input_feature_names.extend(llm_cols) # ou cryptobert_cols si c'est le nom utilisé
    if mcp_cols: all_input_feature_names.extend(mcp_cols)
    if hmm_cols: all_input_feature_names.extend(hmm_cols)
    # Ajouter instrument, sentiment, cryptobert, market si elles sont traitées comme des listes de features
    # Pour l'instant, on se limite à celles qui sont clairement des listes de features.
    # instrument_input est un ID, pas une liste de features.

    return X_features, y_labels, all_input_feature_names
