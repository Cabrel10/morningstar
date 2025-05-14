#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module d'inférence du modèle monolithique.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import tempfile
import json

# Ajouter le chemin du modèle monolithique au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../"))
from ultimate.model.monolith_model import MonolithModel
from ultimate.inference.inference_monolith import (
    prepare_inference_data,
    run_inference,
    interpret_predictions
)


class TestInferenceMonolith:
    """
    Tests pour valider le module d'inférence du modèle monolithique.
    """
    
    @pytest.fixture
    def sample_model(self):
        """Fixture pour créer un modèle de test."""
        return MonolithModel()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données de test."""
        # Créer un DataFrame avec des colonnes techniques, embeddings et MCP
        n_samples = 10
        data = {
            # Colonnes techniques
            "open": np.random.uniform(100, 200, n_samples),
            "high": np.random.uniform(150, 250, n_samples),
            "low": np.random.uniform(90, 180, n_samples),
            "close": np.random.uniform(100, 200, n_samples),
            "volume": np.random.uniform(1000, 10000, n_samples),
            
            # Autres colonnes techniques (indicateurs)
            "rsi": np.random.uniform(0, 100, n_samples),
            "macd": np.random.uniform(-10, 10, n_samples),
            "ema": np.random.uniform(90, 210, n_samples),
            
            # Colonne d'embeddings
            "news_embedding": [np.random.normal(0, 1, 768) for _ in range(n_samples)],
            
            # Colonnes MCP
            "market_volatility": np.random.uniform(0, 1, n_samples),
            "trend_strength": np.random.uniform(0, 1, n_samples),
            "correlation_btc": np.random.uniform(-1, 1, n_samples),
            
            # Colonne d'instrument
            "symbol": ["BTC"] * 5 + ["ETH"] * 5
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_metadata(self):
        """Fixture pour créer des métadonnées de test."""
        return {
            "tech_cols": ["open", "high", "low", "close", "volume", "rsi", "macd", "ema"],
            "llm_cols": ["news_embedding"],
            "mcp_cols": ["market_volatility", "trend_strength", "correlation_btc"],
            "instrument_map": {"BTC": 0, "ETH": 1}
        }
    
    def test_prepare_inference_data(self, sample_data, sample_metadata):
        """Teste la préparation des données pour l'inférence."""
        # Préparer les données
        inputs = prepare_inference_data(sample_data, sample_metadata)
        
        # Vérifier que toutes les entrées requises sont présentes
        assert "technical_input" in inputs
        assert "embeddings_input" in inputs
        assert "mcp_input" in inputs
        assert "instrument_input" in inputs
        
        # Vérifier les shapes des entrées
        assert inputs["technical_input"].shape == (10, 8)  # 10 échantillons, 8 features techniques
        assert inputs["embeddings_input"].shape == (10, 768)  # 10 échantillons, dim 768
        assert inputs["mcp_input"].shape == (10, 3)  # 10 échantillons, 3 features MCP
        assert inputs["instrument_input"].shape == (10, 1)  # 10 échantillons, 1 ID d'instrument
        
        # Vérifier que les instruments sont correctement convertis
        expected_instruments = np.array([0] * 5 + [1] * 5).reshape(-1, 1)
        assert np.array_equal(inputs["instrument_input"], expected_instruments)
    
    def test_prepare_inference_data_missing_columns(self, sample_data, sample_metadata):
        """Teste la préparation des données avec des colonnes manquantes."""
        # Créer une copie des données avec des colonnes manquantes
        data_missing = sample_data.drop(columns=["rsi", "macd", "news_embedding"])
        
        # La préparation devrait fonctionner quand même, en générant des avertissements
        inputs = prepare_inference_data(data_missing, sample_metadata)
        
        # Les entrées devraient toujours être présentes, mais avec des dimensions différentes
        assert "technical_input" in inputs
        assert "embeddings_input" in inputs
        assert inputs["technical_input"].shape == (10, 5)  # 10 échantillons, 5 features techniques (3 manquantes)
        assert inputs["embeddings_input"].shape == (10, 768)  # Les embeddings devraient être des zéros
    
    def test_prepare_inference_data_with_nan_values(self, sample_data, sample_metadata):
        """Teste la préparation des données avec des valeurs NaN."""
        # Injecter des NaN dans différentes colonnes
        data_with_nan = sample_data.copy()
        data_with_nan.loc[0, "open"] = np.nan
        data_with_nan.loc[1, "rsi"] = np.nan
        data_with_nan.loc[2, "market_volatility"] = np.nan
        
        # Préparer les données
        inputs = prepare_inference_data(data_with_nan, sample_metadata)
        
        # Vérifier que les entrées ne contiennent pas de NaN
        for key, value in inputs.items():
            assert not np.isnan(value).any(), f"L'entrée {key} contient des NaN"
    
    def test_prepare_inference_data_with_extreme_values(self, sample_data, sample_metadata):
        """Teste la préparation des données avec des valeurs extrêmes."""
        # Créer une copie des données avec des valeurs extrêmes
        data_extreme = sample_data.copy()
        data_extreme.loc[0, "open"] = 1e9  # Valeur très grande
        data_extreme.loc[1, "volume"] = 1e12  # Valeur énorme
        data_extreme.loc[2, "rsi"] = -1000  # Valeur techniquement invalide pour RSI
        
        # Préparer les données
        inputs = prepare_inference_data(data_extreme, sample_metadata)
        
        # La préparation devrait fonctionner malgré les valeurs extrêmes
        assert "technical_input" in inputs
        assert np.isfinite(inputs["technical_input"]).all(), "L'entrée technique contient des valeurs non finies"
    
    def test_run_inference(self, sample_model):
        """Teste l'exécution de l'inférence."""
        # Préparer des données de test
        batch_size = 5
        inputs = {
            "technical_input": np.random.normal(0, 1, (batch_size, 38)),
            "embeddings_input": np.random.normal(0, 1, (batch_size, 768)),
            "mcp_input": np.random.normal(0, 1, (batch_size, 128)),
            "instrument_input": np.random.randint(0, 5, (batch_size, 1))
        }
        
        # Exécuter l'inférence
        predictions = run_inference(sample_model, inputs)
        
        # Vérifier les sorties
        assert "signal_output" in predictions
        assert "sl_tp_output" in predictions
        assert predictions["signal_output"].shape == (batch_size, 3)
        assert predictions["sl_tp_output"].shape == (batch_size, 2)
    
    def test_run_inference_with_edge_cases(self, sample_model):
        """Teste l'inférence avec des cas limites."""
        # Cas limite 1: Batch de taille 1
        inputs_single = {
            "technical_input": np.random.normal(0, 1, (1, 38)),
            "embeddings_input": np.random.normal(0, 1, (1, 768)),
            "mcp_input": np.random.normal(0, 1, (1, 128)),
            "instrument_input": np.random.randint(0, 5, (1, 1))
        }
        predictions_single = run_inference(sample_model, inputs_single)
        assert predictions_single["signal_output"].shape == (1, 3)
        
        # Cas limite 2: Valeurs extrêmes
        inputs_extreme = {
            "technical_input": np.ones((3, 38)) * 1e9,
            "embeddings_input": np.ones((3, 768)) * 1e9,
            "mcp_input": np.ones((3, 128)) * 1e9,
            "instrument_input": np.zeros((3, 1), dtype=np.int32)
        }
        predictions_extreme = run_inference(sample_model, inputs_extreme)
        assert np.isfinite(predictions_extreme["signal_output"]).all()
        assert np.isfinite(predictions_extreme["sl_tp_output"]).all()
        
        # Cas limite 3: Entrées avec des zéros
        inputs_zeros = {
            "technical_input": np.zeros((3, 38)),
            "embeddings_input": np.zeros((3, 768)),
            "mcp_input": np.zeros((3, 128)),
            "instrument_input": np.zeros((3, 1), dtype=np.int32)
        }
        predictions_zeros = run_inference(sample_model, inputs_zeros)
        assert np.isfinite(predictions_zeros["signal_output"]).all()
        assert np.isfinite(predictions_zeros["sl_tp_output"]).all()
    
    def test_interpret_predictions(self):
        """Teste l'interprétation des prédictions."""
        batch_size = 5
        
        # Créer des prédictions de test
        predictions = {
            "signal_output": np.array([
                [0.1, 0.2, 0.7],  # Achat
                [0.7, 0.2, 0.1],  # Vente
                [0.2, 0.6, 0.2],  # Neutre
                [0.3, 0.3, 0.4],  # Achat (faible)
                [0.4, 0.4, 0.2]   # Mixte
            ]),
            "sl_tp_output": np.array([
                [0.95, 1.05],
                [0.92, 1.10],
                [0.94, 1.08],
                [0.93, 1.12],
                [0.91, 1.15]
            ])
        }
        
        # Interpréter les prédictions
        interpreted = interpret_predictions(predictions)
        
        # Vérifier les interprétations
        assert "signal_proba" in interpreted
        assert "signal_score" in interpreted
        assert "stop_loss" in interpreted
        assert "take_profit" in interpreted
        
        # Vérifier les scores de signal (achat - vente)
        expected_scores = np.array([0.6, -0.6, 0.0, 0.1, -0.2])  # Différence entre proba d'achat et de vente
        assert np.allclose(interpreted["signal_score"], expected_scores)
        
        # Vérifier que les SL/TP sont correctement séparés
        assert np.array_equal(interpreted["stop_loss"], predictions["sl_tp_output"][:, 0])
        assert np.array_equal(interpreted["take_profit"], predictions["sl_tp_output"][:, 1])
    
    def test_interpret_predictions_with_classes(self):
        """Teste l'interprétation des prédictions avec classes discrètes."""
        # Créer des prédictions de test
        predictions = {
            "signal_output": np.array([
                [0.1, 0.2, 0.7],  # Achat
                [0.7, 0.2, 0.1],  # Vente
                [0.2, 0.6, 0.2],  # Neutre
            ]),
            "sl_tp_output": np.array([
                [0.95, 1.05],
                [0.92, 1.10],
                [0.94, 1.08],
            ])
        }
        
        # Interpréter les prédictions avec conversion en classes
        interpreted = interpret_predictions(predictions, return_classes=True)
        
        # Vérifier que la conversion en classes a fonctionné
        assert "signal" in interpreted
        assert np.array_equal(interpreted["signal"], [2, 0, 1])  # Classes: 0=Vente, 1=Neutre, 2=Achat
    
    def test_interpret_predictions_with_threshold(self):
        """Teste l'interprétation des prédictions avec seuils personnalisés."""
        # Créer des prédictions de test proches de la frontière
        predictions = {
            "signal_output": np.array([
                [0.3, 0.4, 0.3],  # Proche de neutre
                [0.45, 0.1, 0.45], # Égalité entre achat/vente
                [0.1, 0.45, 0.45], # Égalité entre neutre/achat
            ]),
            "sl_tp_output": np.array([
                [0.95, 1.05],
                [0.92, 1.10],
                [0.94, 1.08],
            ])
        }
        
        # Interpréter avec un seuil plus élevé
        interpreted = interpret_predictions(predictions, signal_threshold=0.4, return_classes=True)
        
        # Vérifier que le seuil est pris en compte
        assert "signal" in interpreted
        # Les prédictions proches du seuil devraient être affectées


if __name__ == "__main__":
    pytest.main() 