import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Assurez-vous que le chemin d'importation est correct par rapport à la racine du projet
# Si les tests sont lancés depuis la racine, l'importation suivante devrait fonctionner.
# Sinon, ajustez le sys.path ou la structure.
from utils.feature_engineering import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_stochastics,
    integrate_llm_context,  # Importer pour mocker
    apply_feature_pipeline,
)


# Fixture pour charger les données mock une seule fois pour tous les tests du module
@pytest.fixture(scope="module")
def mock_data():
    """Charge les données depuis le fichier mock CSV (maintenant 100 lignes)."""
    df = pd.read_csv("tests/fixtures/mock_data.csv", parse_dates=["timestamp"], index_col="timestamp")
    # La condition skip n'est plus nécessaire avec 100 lignes
    # min_required_rows = 30 # Seuil raisonnable pour la plupart des indicateurs
    # if len(df) < min_required_rows:
    #      pytest.skip(f"Mock data too short ({len(df)} rows, need {min_required_rows}) for window calculations, skipping feature tests.")
    return df


# --- Tests pour les fonctions individuelles ---


def test_compute_sma(mock_data):
    """Teste le calcul de la SMA."""
    result = compute_sma(mock_data, period=5)
    assert isinstance(result, pd.Series)
    assert result.dtype == float  # Vérifier le type
    assert not result.isnull().all(), "SMA result should not be all NaN"
    # Les premières 'period-1' valeurs seront NaN
    period = 5
    assert result.iloc[: period - 1].isnull().all()
    assert result.iloc[period - 1 :].notnull().all(), "SMA should have values after the initial window"


def test_compute_ema(mock_data):
    """Teste le calcul de l'EMA."""
    result = compute_ema(mock_data, period=5)
    assert isinstance(result, pd.Series)
    assert result.dtype == float
    assert not result.isnull().all()
    period = 5
    assert result.iloc[: period - 1].isnull().all()  # EMA a aussi besoin de données initiales
    assert result.iloc[period - 1 :].notnull().all()


def test_compute_rsi(mock_data):
    """Teste le calcul du RSI."""
    result = compute_rsi(mock_data, period=14)
    assert isinstance(result, pd.Series)
    assert result.dtype == float
    assert not result.isnull().all()
    period = 14  # Default RSI period
    # Le RSI nécessite 'period' points pour la première valeur
    assert result.iloc[:period].isnull().all()  # Les 'period' premières valeurs sont NaN
    assert result.iloc[period:].notnull().all()


def test_compute_macd(mock_data):
    """Teste le calcul du MACD."""
    result_df = compute_macd(mock_data)
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["MACD", "MACDs", "MACDh"]
    assert all(result_df.dtypes == float)  # Vérifier les types
    assert not result_df.isnull().all().all()
    # MACD dépend des EMA, vérifier après la plus longue période EMA (slow=26)
    longest_ema_period = 26  # Default slow EMA
    # La ligne de signal a besoin de 'signal' (9) points de MACD calculés
    first_valid_idx = longest_ema_period + 9 - 2  # Index où tout devrait être non-NaN
    assert result_df.iloc[first_valid_idx:].notnull().all().all()


def test_compute_bollinger_bands(mock_data):
    """Teste le calcul des Bandes de Bollinger."""
    result_df = compute_bollinger_bands(mock_data, period=20)
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["BBU", "BBM", "BBL"]
    assert all(result_df.dtypes == float)
    assert not result_df.isnull().all().all()
    period = 20  # Default BBands period
    # Les bandes nécessitent 'period-1' points pour la SMA initiale
    assert result_df.iloc[period - 1 :].notnull().all().all()


def test_compute_atr(mock_data):
    """Teste le calcul de l'ATR."""
    result = compute_atr(mock_data, period=14)
    assert isinstance(result, pd.Series)
    assert result.dtype == float
    assert not result.isnull().all()
    period = 14  # Default ATR period
    # ATR nécessite 'period' points pour la première valeur
    assert result.iloc[period:].notnull().all()  # Vérifier à partir de l'indice 'period'


def test_compute_stochastics(mock_data):
    """Teste le calcul des Stochastiques."""
    result_df = compute_stochastics(mock_data)
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["STOCHk", "STOCHd"]
    assert all(result_df.dtypes == float)
    assert not result_df.isnull().all().all()
    # Stochastics dépendent des périodes k, d, smooth_k
    k, d, smooth_k = 14, 3, 3  # Defaults
    first_valid_idx = k + smooth_k + d - 3  # Index approx où %D devient valide
    assert result_df.iloc[first_valid_idx:].notnull().all().all()


# --- Test pour la fonction placeholder LLM (vérifier qu'elle ajoute les colonnes) ---


def test_integrate_llm_context(mock_data):
    """Teste la fonction placeholder d'intégration LLM."""
    df_copy = mock_data.copy()
    result_df = integrate_llm_context(df_copy)
    assert isinstance(result_df, pd.DataFrame)
    assert "llm_context_summary" in result_df.columns
    assert "llm_embedding" in result_df.columns
    assert isinstance(result_df["llm_context_summary"].iloc[0], str)  # Vérifier type
    assert isinstance(result_df["llm_embedding"].iloc[0], str)  # Vérifier type
    assert result_df["llm_context_summary"].iloc[0] == "Placeholder LLM Summary"
    assert result_df["llm_embedding"].iloc[0] == "[0.1, -0.2, 0.3]"


# --- Test pour le pipeline complet ---


# Mocker la fonction LLM pour éviter les appels réels ou les placeholders non désirés
@patch("utils.feature_engineering.integrate_llm_context")
def test_apply_feature_pipeline(mock_integrate_llm, mock_data):
    """Teste le pipeline complet de feature engineering."""

    # Configurer le mock pour retourner un DataFrame avec les colonnes attendues
    def mock_llm_func(df):
        df["llm_context_summary"] = "Mocked LLM Summary"
        df["llm_embedding"] = "[Mocked Embedding]"
        return df

    mock_integrate_llm.side_effect = mock_llm_func

    # 1. Tester sans LLM
    df_no_llm = apply_feature_pipeline(mock_data.copy(), include_llm=False)
    assert isinstance(df_no_llm, pd.DataFrame)
    # Vérifier que les colonnes LLM ne sont PAS présentes
    assert "llm_context_summary" not in df_no_llm.columns
    assert "llm_embedding" not in df_no_llm.columns
    # Vérifier que les colonnes techniques sont présentes et ont le bon type (float)
    expected_tech_cols = [
        "SMA_short",
        "SMA_long",
        "EMA_short",
        "EMA_long",
        "RSI",
        "MACD",
        "MACDs",
        "MACDh",
        "BBU",
        "BBM",
        "BBL",
        "ATR",
        "STOCHk",
        "STOCHd",
    ]
    for col in expected_tech_cols:
        assert col in df_no_llm.columns
        assert df_no_llm[col].dtype == float
    # Vérifier qu'il n'y a pas de NaN après dropna()
    assert not df_no_llm.isnull().any().any(), "DataFrame ne devrait pas avoir de NaN après apply_feature_pipeline"

    # 2. Tester avec LLM (mocké)
    df_with_llm = apply_feature_pipeline(mock_data.copy(), include_llm=True)
    assert isinstance(df_with_llm, pd.DataFrame)
    # Vérifier que le mock a été appelé
    mock_integrate_llm.assert_called_once()
    # Vérifier que les colonnes LLM (mockées) sont présentes
    assert "llm_context_summary" in df_with_llm.columns
    assert "llm_embedding" in df_with_llm.columns
    # Accéder au premier élément valide après dropna(), si le DataFrame n'est pas vide
    assert not df_with_llm.empty, "DataFrame ne devrait pas être vide après pipeline avec LLM mock"
    first_valid_index = df_with_llm.index[0]
    assert df_with_llm.loc[first_valid_index, "llm_context_summary"] == "Mocked LLM Summary"
    # Vérifier que les colonnes techniques sont aussi présentes et ont le bon type
    for col in expected_tech_cols:
        assert col in df_with_llm.columns
        assert df_with_llm[col].dtype == float
    # Vérifier qu'il n'y a pas de NaN après dropna()
    assert (
        not df_with_llm.isnull().any().any()
    ), "DataFrame ne devrait pas avoir de NaN après apply_feature_pipeline avec LLM mock"


def test_apply_feature_pipeline_missing_columns(mock_data):
    """Teste que le pipeline lève une erreur si des colonnes manquent."""
    df_incomplete = mock_data.drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        apply_feature_pipeline(df_incomplete)
