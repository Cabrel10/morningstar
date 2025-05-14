import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import de la fonction à tester
from data.pipelines.data_pipeline import run_pipeline
from utils.feature_engineering import apply_feature_pipeline  # Pour simuler les erreurs

# Configuration pour les tests
PIPELINE_MODULE_PATH = "data.pipelines.data_pipeline"
NUM_TECHNICAL_FEATURES = 38  # Nombre attendu de features techniques


@pytest.fixture
def base_mock_df():
    """Fixture pour un DataFrame de base mock."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:15:00"]),
            "open": [1000, 1005],
            "high": [1010, 1015],
            "low": [990, 1000],
            "close": [1005, 1010],
            "volume": [100, 110],
        }
    ).set_index("timestamp")


@pytest.fixture
def mock_apply_feature_pipeline():
    """Fixture pour mocker apply_feature_pipeline."""
    with patch(f"{PIPELINE_MODULE_PATH}.apply_feature_pipeline") as mock:
        yield mock


def test_run_pipeline_feature_validation_success(mock_apply_feature_pipeline, base_mock_df, tmp_path):
    """
    Teste que run_pipeline réussit si apply_feature_pipeline retourne un DataFrame valide.
    """

    # Configurer le mock pour retourner un DataFrame avec 38 colonnes
    def valid_feature_side_effect(df):
        # Ajouter des colonnes pour atteindre 38
        while len(df.columns) < NUM_TECHNICAL_FEATURES:
            df[f"extra_feature_{len(df.columns)}"] = 0  # Ajouter des colonnes pour atteindre 38
        return df

    mock_apply_feature_pipeline.side_effect = valid_feature_side_effect

    # Définir les mocks pour les autres étapes (pour que le pipeline s'exécute)
    with patch(f"{PIPELINE_MODULE_PATH}.load_raw_data", return_value=base_mock_df), patch(
        f"{PIPELINE_MODULE_PATH}.clean_data", side_effect=lambda df: df
    ), patch("utils.llm_integration.LLMIntegration", autospec=True) as MockLLM, patch(
        f"{PIPELINE_MODULE_PATH}.pd.DataFrame.to_parquet"
    ) as mock_to_parquet:

        # Configurer le mock LLM pour retourner des embeddings
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.get_embeddings.return_value = np.zeros(768)

        # Définir les chemins d'entrée/sortie
        input_path = "dummy_input.csv"
        output_path = tmp_path / "output.parquet"

        # Exécuter le pipeline
        success = run_pipeline(input_path, str(output_path), limit=10)
        assert success, "Le pipeline devrait réussir avec des features valides"


def test_run_pipeline_feature_validation_failure(mock_apply_feature_pipeline, base_mock_df, tmp_path):
    """
    Teste que run_pipeline gère correctement une exception si apply_feature_pipeline
    ne retourne pas un DataFrame avec 38 colonnes.
    """
    # Configurer le mock pour lever une ValueError (simulant une erreur de validation)
    mock_apply_feature_pipeline.side_effect = ValueError("Nombre incorrect de features techniques")

    # Définir les mocks pour les autres étapes (pour que le pipeline s'exécute jusqu'à l'étape de features)
    with patch(f"{PIPELINE_MODULE_PATH}.load_raw_data", return_value=base_mock_df), patch(
        f"{PIPELINE_MODULE_PATH}.clean_data", side_effect=lambda df: df
    ), patch(f"{PIPELINE_MODULE_PATH}.build_labels", side_effect=lambda df: df), patch(
        "utils.llm_integration.LLMIntegration"
    ), patch(
        f"{PIPELINE_MODULE_PATH}.pd.DataFrame.to_parquet"
    ):

        # Définir les chemins d'entrée/sortie
        input_path = "dummy_input.csv"
        output_path = tmp_path / "output.parquet"

        # Exécuter le pipeline
        success = run_pipeline(input_path, str(output_path), limit=10)
        assert not success, "Le pipeline devrait échouer à cause de la validation des features"
