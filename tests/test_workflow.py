import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from workflows.trading_workflow import TradingWorkflow

class TestTradingWorkflow(unittest.TestCase):
    """Tests d'intégration pour le workflow de trading."""

    def setUp(self):
        """Initialisation des tests avec une configuration mock."""
        self.config = {
            'model': {
                'weights_path': 'tests/fixtures/mock_model.h5',
                'num_technical_features': 38,
                'llm_embedding_dim': 768,
                'num_mcp_features': 128,
                'instrument_vocab_size': 10,
                'instrument_embedding_dim': 8,
                'num_trading_classes': 5,
                'num_market_regime_classes': 4,
                'num_volatility_quantiles': 3,
                'num_sl_tp_outputs': 2,
                'options_output_dim': 5,
                'futures_output_dim': 3
            },
            'api': {
                'base_url': 'http://mock-api',
                'timeout': 10
            }
        }

        # Mock du modèle et de l'API
        self.mock_model = MagicMock()
        self.mock_api = MagicMock()

        # Configuration des valeurs de retour mock (adaptées au nouveau modèle)
        self.mock_predictions = {
            'signal': np.array([[0.8, 0.1, 0.05, 0.03, 0.02]]),
            'volatility_quantiles': np.array([[0.1, 0.5, 0.9]]),
            'market_regime': np.array([[0.25, 0.25, 0.25, 0.25]]), # 4 classes
            'sl_tp': np.array([[0.95, 1.05]]),
            'conditional_output': np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]) # Exemple
        }

    @patch('workflows.trading_workflow.build_enhanced_hybrid_model') # Mocker la fonction, pas la classe
    @patch('workflows.trading_workflow.APIManager')
    def test_workflow_initialization(self, mock_api, mock_build_model):
        """Teste l'initialisation correcte du workflow."""
        mock_build_model.return_value = self.mock_model # Configurer la valeur de retour du mock
        mock_api.return_value = self.mock_api

        workflow = TradingWorkflow(self.config)

        # Vérifications
        mock_build_model.assert_called_once() # Vérifier que la fonction a été appelée
        mock_api.assert_called_once()
        # self.mock_model.load_weights.assert_called_with('tests/fixtures/mock_model.h5') # Pas de load_weights ici

    @patch('workflows.trading_workflow.build_enhanced_hybrid_model')
    @patch('workflows.trading_workflow.APIManager')
    def test_execute_strategy(self, mock_api, mock_build_model):
        """Teste l'exécution complète de la stratégie."""
        # Configuration des mocks
        mock_build_model.return_value = self.mock_model
        mock_api.return_value = self.mock_api
        self.mock_model.predict.return_value = self.mock_predictions

        # Données de marché mock (adaptées au nouveau modèle)
        market_data = {
            'technical_input': np.random.rand(38),
            'sentiment_embeddings': np.random.rand(768),
            'mcp_input': np.random.rand(128),
            'instrument_input': np.array([0]) # ID instrument (spot par exemple)
        }
        self.mock_api.get_market_data.return_value = market_data

        # Exécution
        workflow = TradingWorkflow(self.config)
        result = workflow.execute_strategy(market_data)

        # Vérifications
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['decisions']['signal'], 'STRONG_BUY')
        self.mock_model.predict.assert_called_once()

    @patch('workflows.trading_workflow.build_enhanced_hybrid_model')
    @patch('workflows.trading_workflow.APIManager')
    def test_error_handling(self, mock_api, mock_build_model):
        """Teste la gestion des erreurs dans le workflow."""
        # Configuration des mocks pour simuler une erreur
        mock_build_model.return_value = self.mock_model
        mock_api.return_value = self.mock_api
        self.mock_model.predict.side_effect = Exception("Mock error")

        # Exécution
        workflow = TradingWorkflow(self.config)
        market_data = {
            'technical_input': np.random.rand(38),
            'sentiment_embeddings': np.random.rand(768),
            'mcp_input': np.random.rand(128),
            'instrument_input': np.array([0])
        }
        result = workflow.execute_strategy(market_data)

        # Vérifications
        self.assertEqual(result['status'], 'error')
        self.assertIn('Mock error', result['error'])

if __name__ == '__main__':
    unittest.main()
