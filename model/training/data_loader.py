import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
from sklearn.preprocessing import RobustScaler

from config.config import Config # Importer la classe Config

def load_data(file_path):
    """Charge les données prétraitées à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

# La fonction _preprocess_market_regime n'est plus nécessaire car le mapping
# et le cast seront faits dans load_and_split_data directement sur le dictionnaire y_labels.

def _convert_labels_to_tensors(data, label_columns, label_mappings):
    """Convertit les labels en tenseurs TensorFlow avec gestion des mappings."""
    y_tensors = {}

    for name in label_columns:
        # Pour les autres labels, vérifier qu'ils existent comme colonnes
        # NOTE: 'sl_tp' is handled separately after scaling in the training script
        if name == 'sl_tp':
            continue # Skip sl_tp here, it's processed later

        if name not in data.columns:
            print(f"WARN: Colonne de label '{name}' non trouvée dans les données.")
            
            continue # Ignore ce label s'il manque

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

    # Si c'est une colonne catégorielle connue et textuelle, sans mapping fourni, factoriser
    if name in ['trading_signal', 'market_regime'] and series.dtype == 'object':
        print(f"INFO: Factorisation automatique de la colonne label '{name}'.")
        values, _ = pd.factorize(series)
        return values.astype(np.int64), tf.int64 # Retourner les codes entiers

    # Si c'est une colonne catégorielle connue mais déjà numérique
    if name in ['trading_signal', 'market_regime']:
        dtype = tf.int64 # S'assurer que le dtype est entier pour TF

    return values, dtype

def _apply_label_mapping(series, mapping_dict):
    """Applique un mapping de valeurs textuelles à numériques."""
    str_series = series.astype(str).str.strip()
    mapped = str_series.map(lambda x, m=mapping_dict: m.get(x.strip(), float('nan')))

    if mapped.isnull().any():
        unmapped = str_series[mapped.isnull()].unique()
        raise ValueError(f"Valeurs non mappées dans '{series.name}': {list(unmapped)}")

    return pd.to_numeric(mapped, downcast='integer').values, tf.int64

def load_and_split_data(file_path, label_columns=None, label_mappings=None, as_tensor=False,
                        num_mcp_features=128, num_technical_features=38, num_llm_features=768):
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
        label_columns = ['signal', 'volatility_quantiles', 'market_regime', 'sl_tp']

    cfg = Config() # Charger la config pour accéder aux mappings
    data = load_data(file_path)
    
    # Le prétraitement de market_regime est déplacé après l'extraction de y_labels

    # Vérification des colonnes de labels requises (celles qui ne sont pas 'sl_tp')
    required_label_cols = [col for col in label_columns if col != 'sl_tp']
    missing_labels = [col for col in required_label_cols if col not in data.columns]
    if missing_labels:
        raise ValueError(f"Colonnes de labels manquantes: {missing_labels}")
    # Vérifier aussi les colonnes sources pour sl_tp si sl_tp est demandé
    if 'sl_tp' in label_columns and ('level_sl' not in data.columns or 'level_tp' not in data.columns):
        raise ValueError("Colonnes 'level_sl' et/ou 'level_tp' manquantes pour générer 'sl_tp'.")

    # --- Extraction des Features ---
    all_cols = set(data.columns)

    # 1. Identifier les colonnes LLM
    llm_cols = sorted([col for col in all_cols if col.startswith('bert_')]) # Modifié: llm_ -> bert_

    # 2. Identifier les colonnes MCP
    mcp_cols = sorted([col for col in all_cols if col.startswith('mcp_')])

    # 3. Identifier la colonne Instrument
    instrument_col = 'instrument_type' # Nom conventionnel
    instrument_cols = []
    if instrument_col in all_cols:
        instrument_cols = [instrument_col] # Liste pour cohérence
        print(f"INFO: Colonne instrument '{instrument_col}' trouvée.")
    else:
        print(f"WARN: Colonne '{instrument_col}' non trouvée. L'input instrument ne sera pas généré.")

    # 4. Identifier les colonnes de labels sources (y compris celles pour sl_tp)
    # Utiliser les label_columns demandés + les sources potentielles pour sl_tp
    label_source_cols = set(required_label_cols) # required_label_cols défini plus haut
    if 'sl_tp' in label_columns:
        label_source_cols.update(['level_sl', 'level_tp'])

    # 5. Déduire les colonnes techniques par exclusion
    # Ajouter les colonnes OHLCV de base et autres colonnes non-feature à exclure
    base_cols_to_exclude = {'open', 'high', 'low', 'close', 'volume', 'trading_signal', 'volatility', 'news_snippets'} # Ajout de news_snippets
    
    # Identifier les colonnes HMM pour les extraire séparément
    hmm_cols = sorted([col for col in all_cols if col.startswith('hmm_')])
    
    excluded_for_tech = set(llm_cols) | set(mcp_cols) | set(instrument_cols) | set(hmm_cols) | label_source_cols | base_cols_to_exclude
    feature_cols = sorted([
        col for col in all_cols
        if col not in excluded_for_tech
    ])

    # --- Debug Logging Amélioré ---
    print(f"--- Debug: Data Loader Feature/Label Extraction ---")
    print(f"Total columns in Parquet: {len(all_cols)}")
    print(f"Requested label columns: {label_columns}")
    print(f"Source columns for labels: {sorted(list(label_source_cols))}")
    print(f"Excluded columns for tech features: {sorted(list(excluded_for_tech))}")
    print(f"Detected Technical Features ({len(feature_cols)}): {feature_cols[:5]}...{feature_cols[-5:] if len(feature_cols)>5 else ''}")
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
    print(f"Dimensions validées - Tech: {len(feature_cols)}, LLM: {len(llm_cols)}, MCP: {len(mcp_cols)}, Instrument: {len(instrument_cols)}")

    # --- Préparation des Données X et Y ---
    X_features = {}

    # Ajouter les features seulement si les colonnes correspondantes existent
    if feature_cols:
        x_technical_data = data[feature_cols].values.astype(np.float32)
        X_features["technical_input"] = x_technical_data
    
    if llm_cols:
        x_llm_data = data[llm_cols].values.astype(np.float32)
        X_features["cryptobert_input"] = x_llm_data # Utilise cryptobert_input pour les colonnes bert_*
        
    if mcp_cols:
        x_mcp_data = data[mcp_cols].values.astype(np.float32)
        X_features["mcp_input"] = x_mcp_data
        
    if hmm_cols:
        # S'assurer que les colonnes HMM sont numériques
        hmm_data = data[hmm_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
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
                     print(f"WARN: Tentative de conversion de {key} (dtype {value.dtype}) en int64 échouerait. Vérifier la factorisation.")
                     # On pourrait tenter une conversion explicite ici si nécessaire, mais factorize le fait déjà.
                X_features[key] = tf.convert_to_tensor(value, dtype=target_dtype)
            except Exception as e:
                 print(f"Erreur conversion X['{key}'] (source dtype: {value.dtype}) en Tensor (target dtype={target_dtype}): {value[:5]}")
                 raise e
        # Convertir Y en Tensors
        y_labels = _convert_labels_to_tensors(data, label_columns, label_mappings)
    else:
        # Retourner Y comme Series/DataFrame pandas
        y_labels = {}
        for col in label_columns:
             if col in data.columns:
                 y_labels[col] = data[col]
             elif col == 'sl_tp' and 'level_sl' in data.columns and 'level_tp' in data.columns:
                 # Gérer sl_tp séparément: retourner les colonnes brutes, gérer NaN/inf
                 sl_tp_df = data[['level_sl', 'level_tp']].copy()
                 sl_tp_df['level_sl'] = pd.to_numeric(sl_tp_df['level_sl'], errors='coerce')
                 sl_tp_df['level_tp'] = pd.to_numeric(sl_tp_df['level_tp'], errors='coerce')
                 sl_tp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                 nan_count = sl_tp_df.isnull().sum().sum()
                 if nan_count > 0:
                     print(f"INFO: Found and filled {nan_count} NaN/inf values in raw SL/TP labels with 0.")
                 sl_tp_df.fillna(0, inplace=True)
                 y_labels['sl_tp'] = sl_tp_df # Return as DataFrame
             else:
                 print(f"WARN: Colonne label '{col}' demandée mais non trouvée dans les données.")

        # Appliquer le mapping et caster en int32 pour les labels catégoriels
        label_mappings_from_config = cfg.get_config('data.label_mappings', {})
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
                    else: # Si la série est object/string, les clés YAML devraient être ok
                        safe_mapping = mapping
                    
                    mapped_series = label_series.map(safe_mapping)
                    
                    # Vérifier les NaN après le mapping
                    nan_after_map = mapped_series.isnull().sum()
                    if nan_after_map > 0:
                         unmapped_values = label_series[mapped_series.isnull()].unique()
                         print(f"WARN: {nan_after_map} valeurs de '{label_name}' n'ont pas pu être mappées: {list(unmapped_values)}. Elles seront remplacées par NaN puis 0.")
                         mapped_series = mapped_series.fillna(0) # Remplacer NaN par 0 (ou une autre valeur par défaut)

                    y_labels[label_name] = mapped_series.astype('int32') # Caster en int32
                    print(f"INFO: Label '{label_name}' mappé et casté en int32.")

                except Exception as e:
                    print(f"ERREUR lors du mapping/cast de '{label_name}': {e}")
                    print(f"       Mapping utilisé: {mapping}")
                    print(f"       Premières valeurs de la série: {label_series.head().tolist()}")
                    # Optionnel: lever l'erreur ou continuer avec les données non mappées/castées
                    # raise e 
            elif label_name == 'sl_tp':
                 # Assurer que sl_tp est float32
                 y_labels[label_name] = y_labels[label_name].astype('float32')
                 print(f"INFO: Label '{label_name}' casté en float32.")
            # Ajouter d'autres casts si nécessaire pour d'autres types de labels


    # Conversion en Tensors si demandé (déplacé après la création complète de X_features)
    # Le bloc dupliqué pour sl_tp a été supprimé ici. Le traitement se fait dans la boucle ci-dessus.

    # Conversion en Tensors si demandé
    if as_tensor:
        for key, value in X_features.items():
            target_dtype = tf.int64 if key == "instrument_input" else tf.float32
            try:
                if target_dtype == tf.int64 and not np.issubdtype(value.dtype, np.integer):
                     print(f"WARN: Tentative de conversion de {key} (dtype {value.dtype}) en int64 échouerait. Vérifier la factorisation.")
                X_features[key] = tf.convert_to_tensor(value, dtype=target_dtype)
            except Exception as e:
                 print(f"Erreur conversion X['{key}'] (source dtype: {value.dtype}) en Tensor (target dtype={target_dtype}): {value[:5]}")
                 raise e
        y_labels = _convert_labels_to_tensors(data, label_columns, label_mappings)


    return X_features, y_labels
