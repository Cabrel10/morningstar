import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
from sklearn.preprocessing import RobustScaler

def load_data(file_path):
    """Charge les données prétraitées à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

def _preprocess_market_regime(data):
    """Prétraite la colonne market_regime en appliquant le mapping de config.yaml."""
    if 'market_regime' not in data.columns:
        print("WARN: Colonne 'market_regime' manquante dans le dataset")
        return data

    # Charger la configuration
    config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Récupérer le mapping des régimes de marché
    mapping = config.get('training', {}).get('label_mappings', {}).get('market_regime', {})
    if not mapping:
        print("WARN: Mapping des régimes de marché non trouvable dans config.yaml")
        return data

    print(f"INFO: Applique mapping market_regime: {mapping}")

    # Appliquer le mapping
    original_values = data['market_regime'].unique()
    data['market_regime'] = data['market_regime'].map(lambda x: mapping.get(x, 0))  # Utiliser 0 (sideways) comme valeur par défaut au lieu de -1
    
    # Vérifier s'il y a des valeurs non mappées
    new_values = data['market_regime'].unique()
    unmapped = [val for val in original_values if val not in mapping and val != -1]
    if unmapped:
        print(f"WARN: Valeurs market_regime non mappées: {unmapped}")

    return data

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

    data = load_data(file_path)
    
    # Prétraiter la colonne market_regime pour la convertir en entiers
    if 'market_regime' in data.columns and 'market_regime' in label_columns:
        data = _preprocess_market_regime(data)

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
    llm_cols = sorted([col for col in all_cols if col.startswith('llm_')])

    # 2. Identifier les colonnes MCP
    mcp_cols = sorted([col for col in all_cols if col.startswith('mcp_')])

    # 3. Identifier la colonne Instrument
    instrument_col = 'instrument_type' # Nom conventionnel
    if instrument_col not in all_cols:
        raise ValueError(f"Colonne '{instrument_col}' manquante pour l'input instrument.")
    instrument_cols = [instrument_col] # Liste pour cohérence

    # 4. Identifier les colonnes de labels sources (y compris celles pour sl_tp)
    # Utiliser les label_columns demandés + les sources potentielles pour sl_tp
    label_source_cols = set(required_label_cols) # required_label_cols défini plus haut
    if 'sl_tp' in label_columns:
        label_source_cols.update(['level_sl', 'level_tp'])

    # 5. Déduire les colonnes techniques par exclusion
    # Ajouter les colonnes OHLCV de base et autres colonnes non-feature à exclure
    base_cols_to_exclude = {'open', 'high', 'low', 'close', 'volume', 'trading_signal', 'volatility'}
    
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
    x_technical_data = data[feature_cols].values.astype(np.float32)
    x_llm_data = data[llm_cols].values.astype(np.float32)
    x_mcp_data = data[mcp_cols].values.astype(np.float32)
    # Pour l'instrument, on suppose qu'il contient des IDs entiers ou des catégories
    # qui seront gérés par une couche Embedding dans le modèle.
    # S'assurer qu'il est numérique ou le convertir si nécessaire.
    # Pour l'instrument, convertir en codes entiers (factorize)
    x_instrument_data, _ = pd.factorize(data[instrument_col])
    x_instrument_data = x_instrument_data.astype(np.int64) # Assurer int64 pour TF

    # Préparer le dictionnaire X
    X_features = {
        "technical_input": x_technical_data,
        "llm_input": x_llm_data,
        "mcp_input": x_mcp_data,
        "instrument_input": x_instrument_data
    }

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
        y_labels = {col: data[col] for col in label_columns if col in data.columns and col != 'sl_tp'}
        # Gérer sl_tp séparément: retourner les colonnes brutes, gérer NaN/inf
        if 'sl_tp' in label_columns and 'level_sl' in data.columns and 'level_tp' in data.columns:
            sl_tp_df = data[['level_sl', 'level_tp']].copy()
            # Convertir en numérique et gérer NaN/inf
            sl_tp_df['level_sl'] = pd.to_numeric(sl_tp_df['level_sl'], errors='coerce')
            sl_tp_df['level_tp'] = pd.to_numeric(sl_tp_df['level_tp'], errors='coerce')
            sl_tp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Remplacer NaN par 0 (ou une autre stratégie si nécessaire)
            # Log NaN count before filling
            nan_count = sl_tp_df.isnull().sum().sum()
            if nan_count > 0:
                print(f"INFO: Found and filled {nan_count} NaN/inf values in raw SL/TP labels with 0.")
            sl_tp_df.fillna(0, inplace=True)
            y_labels['sl_tp'] = sl_tp_df # Return as DataFrame

    # Ajouter l'entrée 'hmm_input' au dictionnaire de features
    if as_tensor:
        X_features['hmm_input'] = tf.convert_to_tensor(data[hmm_cols].values, dtype=tf.float32)
    else:
        X_features['hmm_input'] = data[hmm_cols].values

    return X_features, y_labels
