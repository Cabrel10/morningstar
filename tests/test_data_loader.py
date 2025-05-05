import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Ajouter le répertoire racine au PYTHONPATH pour les imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.training.data_loader import load_and_split_data

# --- Configuration des Tests ---
TEST_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
TEST_FILE_PATH = TEST_DATA_DIR / 'btc_test.parquet' # Utiliser le fichier de test existant
EXPECTED_TECH_FEATURES = 38
EXPECTED_LLM_FEATURES = 768
EXPECTED_MCP_FEATURES = 0 # Le fichier de test ne contient pas de features MCP
# Utiliser les labels par défaut du script training_script.py
DEFAULT_LABEL_COLUMNS = ['market_regime', 'sl_tp'] 

# Vérifier si le fichier de test existe
if not TEST_FILE_PATH.exists():
    pytest.skip(f"Fichier de données de test non trouvé: {TEST_FILE_PATH}", allow_module_level=True)

# --- Fixture pour charger les données une seule fois (optionnel, mais peut accélérer) ---
# Pour l'instant, on charge dans chaque test pour simplicité

# --- Tests ---

def test_load_and_split_file_not_found():
    """Vérifie qu'une erreur est levée si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_and_split_data("non_existent_file.parquet")

def test_load_and_split_returns_dicts_numpy():
    """Vérifie que la fonction retourne deux dictionnaires (format NumPy/Pandas)."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=False,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    assert isinstance(X_dict, dict)
    assert isinstance(y_dict, dict)

def test_load_and_split_returns_dicts_tensor():
    """Vérifie que la fonction retourne deux dictionnaires (format Tensor)."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=True,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    assert isinstance(X_dict, dict)
    assert isinstance(y_dict, dict)

def test_load_and_split_x_keys():
    """Vérifie les clés attendues dans le dictionnaire X_features."""
    X_dict, _ = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=False,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    expected_keys = {"technical_input", "llm_input", "mcp_input", "instrument_input"}
    assert set(X_dict.keys()) == expected_keys

def test_load_and_split_y_keys():
    """Vérifie les clés attendues dans le dictionnaire y_labels."""
    # Note: Ce test suppose que les colonnes 'market_regime', 'level_sl', 'level_tp' 
    # existent dans btc_test.parquet pour générer les labels demandés.
    try:
        _, y_dict = load_and_split_data(
            TEST_FILE_PATH,
            label_columns=DEFAULT_LABEL_COLUMNS,
            as_tensor=False,
            num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
        )
        assert set(y_dict.keys()) == set(DEFAULT_LABEL_COLUMNS)
    except ValueError as e:
        pytest.fail(f"Erreur lors du chargement des labels (vérifier si les colonnes existent dans le fichier): {e}")

def test_load_and_split_shapes_numpy():
    """Vérifie les shapes des arrays NumPy retournés."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=False,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    # Obtenir le nombre d'échantillons (supposé identique pour toutes les features/labels)
    n_samples = len(X_dict['technical_input'])
    assert n_samples > 0 # S'assurer que le fichier n'est pas vide
    
    assert X_dict['technical_input'].shape == (n_samples, EXPECTED_TECH_FEATURES)
    assert X_dict['llm_input'].shape == (n_samples, EXPECTED_LLM_FEATURES)
    assert X_dict['mcp_input'].shape == (n_samples, EXPECTED_MCP_FEATURES)
    assert X_dict['instrument_input'].shape == (n_samples,) # C'est un vecteur d'IDs
    
    if 'market_regime' in y_dict:
        assert y_dict['market_regime'].shape == (n_samples,)
    if 'sl_tp' in y_dict:
        # sl_tp est retourné comme DataFrame (n_samples, 2) en mode NumPy
        assert y_dict['sl_tp'].shape == (n_samples, 2) 

def test_load_and_split_dtypes_numpy():
    """Vérifie les dtypes des arrays NumPy retournés."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=False,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    assert X_dict['technical_input'].dtype == np.float32
    assert X_dict['llm_input'].dtype == np.float32
    assert X_dict['mcp_input'].dtype == np.float32
    # instrument_input peut être int ou object selon la source, le modèle gère via Embedding
    assert pd.api.types.is_numeric_dtype(X_dict['instrument_input']) or pd.api.types.is_object_dtype(X_dict['instrument_input'])
    
    if 'market_regime' in y_dict:
        # Peut être numérique (si mappé) ou object (si brut et catégoriel)
         assert pd.api.types.is_numeric_dtype(y_dict['market_regime']) or pd.api.types.is_object_dtype(y_dict['market_regime'])
    if 'sl_tp' in y_dict:
         assert y_dict['sl_tp']['level_sl'].dtype == np.float64 or y_dict['sl_tp']['level_sl'].dtype == np.float32
         assert y_dict['sl_tp']['level_tp'].dtype == np.float64 or y_dict['sl_tp']['level_tp'].dtype == np.float32


def test_load_and_split_shapes_tensor():
    """Vérifie les shapes des Tensors TensorFlow retournés."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=True,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    n_samples = X_dict['technical_input'].shape[0]
    assert n_samples > 0
    
    tf.debugging.assert_shapes([
        (X_dict['technical_input'], (n_samples, EXPECTED_TECH_FEATURES)),
        (X_dict['llm_input'], (n_samples, EXPECTED_LLM_FEATURES)),
        (X_dict['mcp_input'], (n_samples, EXPECTED_MCP_FEATURES)),
        (X_dict['instrument_input'], (n_samples,)), # Shape (n_samples,) pour TF
    ])
    
    if 'market_regime' in y_dict:
         tf.debugging.assert_shapes([(y_dict['market_regime'], (n_samples,))])
    if 'sl_tp' in y_dict:
         tf.debugging.assert_shapes([(y_dict['sl_tp'], (n_samples, 2))])


def test_load_and_split_dtypes_tensor():
    """Vérifie les dtypes des Tensors TensorFlow retournés."""
    X_dict, y_dict = load_and_split_data(
        TEST_FILE_PATH,
        label_columns=DEFAULT_LABEL_COLUMNS,
        as_tensor=True,
        num_mcp_features=EXPECTED_MCP_FEATURES # Spécifier 0 MCP features pour ce test
    )
    assert X_dict['technical_input'].dtype == tf.float32
    assert X_dict['llm_input'].dtype == tf.float32
    assert X_dict['mcp_input'].dtype == tf.float32
    assert X_dict['instrument_input'].dtype == tf.int64 # Doit être int pour Embedding layer
    
    if 'market_regime' in y_dict:
        assert y_dict['market_regime'].dtype == tf.int64 # Catégoriel
    if 'sl_tp' in y_dict:
        assert y_dict['sl_tp'].dtype == tf.float32 # Régression

# TODO: Ajouter des tests pour vérifier la gestion des erreurs (colonnes manquantes, dims incorrectes)
# Ceci nécessiterait de créer des fichiers Parquet de test spécifiques.
