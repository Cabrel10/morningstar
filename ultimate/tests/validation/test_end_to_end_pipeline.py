import unittest
import numpy as np
import time
from unittest.mock import patch, MagicMock
from workflows.trading_workflow import TradingWorkflow


class TestEndToEndPipeline(unittest.TestCase):
    """Test complet du pipeline de trading, du modèle à l'exécution."""

    def setUp(self):
        """Configuration des mocks pour le test end-to-end."""
        self.config = {
            "model": {
                "weights_path": "model/morningstar_v1.h5",
                "num_technical_features": 38,
                "llm_embedding_dim": 768,
            },
            "api": {"base_url": "http://test-api", "timeout": 5},
        }

        # Mock des prédictions du modèle
        self.sample_predictions = {
            "signal": np.array([[0.01, 0.02, 0.1, 0.67, 0.2]]),  # SELL
            "volatility_quantiles": np.array([[0.05, 0.5, 0.95]]),
            "volatility_regime": np.array([[0, 0, 1]]),  # High volatility
            "market_regime": np.array([[0, 1, 0, 0]]),  # Bearish
            "sl_tp": np.array([[0.98, 1.02]]),  # SL 2%, TP 2%
        }

    @patch("workflows.trading_workflow.APIManager")
    @patch("workflows.trading_workflow.MorningstarModel")
    def test_full_pipeline_execution(self, mock_model, mock_api):
        """Teste un cycle complet de récupération prédiction et exécution."""
        # Configuration des mocks
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = self.sample_predictions
        mock_model.return_value = mock_model_instance

        # Données de marché mock
        market_data = {
            "technical": np.random.rand(38),  # format attendu
            "sentiment_embeddings": np.random.rand(768),  # format attendu
        }

        mock_api_instance = MagicMock()
        mock_api_instance.get_market_data.return_value = market_data
        mock_api.return_value = mock_api_instance

        # Exécution du workflow
        workflow = TradingWorkflow(self.config)

        # Appel direct à execute_strategy pour tester le pipeline
        result = workflow.execute_strategy(market_data)

        # Vérifications
        self.assertEqual(result["status"], "success")
        mock_model_instance.predict.assert_called_once()

    @patch("workflows.trading_workflow.APIManager")
    @patch("workflows.trading_workflow.MorningstarModel")
    def test_latency_measurement(self, mock_model, mock_api):
        """Teste que la latence totale est acceptable."""
        # Configuration des mocks avec temporisation
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = lambda *args: (time.sleep(0.1), self.sample_predictions)[1]
        mock_model.return_value = mock_model_instance

        mock_api_instance = MagicMock()
        mock_api_instance.get_market_data.side_effect = lambda: (
            time.sleep(0.05),
            {"technical": np.random.rand(1, 38), "sentiment_embeddings": np.random.rand(1, 768)},
        )[1]
        mock_api.return_value = mock_api_instance

        # Mesure de performance
        workflow = TradingWorkflow(self.config)
        start_time = time.time()
        workflow.execute_strategy({})
        latency = time.time() - start_time

        # Vérification (latence < 200ms)
        self.assertLess(latency, 0.2)

    @patch("workflows.trading_workflow.APIManager")
    @patch("workflows.trading_workflow.MorningstarModel")
    def test_error_recovery(self, mock_model, mock_api):
        """Teste la reprise après erreur."""
        mock_model_instance = MagicMock()
        mock_model_instance.predict.side_effect = [
            Exception("First error"),
            self.sample_predictions,  # Second call succeeds
        ]
        mock_model.return_value = mock_model_instance

        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        workflow = TradingWorkflow(self.config)

        # Données de marché valides
        market_data = {"technical": np.random.rand(38), "sentiment_embeddings": np.random.rand(768)}

        # Premier appel échoue
        result1 = workflow.execute_strategy(market_data)
        self.assertEqual(result1["status"], "error")

        # Deuxième appel réussit
        result2 = workflow.execute_strategy(market_data)
        self.assertEqual(result2["status"], "success")


if __name__ == "__main__":
    import time

    unittest.main()
