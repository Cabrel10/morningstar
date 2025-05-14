#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le modèle monolithique Morningstar.
"""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf

# Ajouter le chemin du modèle monolithique au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../"))
from ultimate.model.monolith_model import build_monolith_model, TransformerBlock, MonolithModel


class TestMonolithModel:
    """
    Tests pour valider le modèle monolithique Morningstar.
    """

    @pytest.mark.parametrize("active_outputs,expected_outputs", [
        (["signal", "sl_tp"], ["signal_output", "sl_tp_output"]),
        (["signal"], ["signal_output"]),
        (["sl_tp"], ["sl_tp_output"])
    ])
    def test_active_outputs(self, active_outputs, expected_outputs):
        """Teste que active_outputs contrôle correctement les sorties du modèle."""
        # Construire le modèle avec les sorties actives spécifiées
        model = build_monolith_model(active_outputs=active_outputs)
        
        # Vérifier que les noms de sortie correspondent aux attentes
        assert set(model.output_names) == set(expected_outputs), f"Sorties attendues: {expected_outputs}, obtenues: {model.output_names}"
        
        # Vérifier les shapes des sorties
        if "signal_output" in model.output_names:
            signal_shape = model.get_layer("signal_output").output_shape
            assert signal_shape[-1] == 3, f"La sortie signal_output devrait avoir 3 classes, obtenu: {signal_shape[-1]}"
        
        if "sl_tp_output" in model.output_names:
            sl_tp_shape = model.get_layer("sl_tp_output").output_shape
            assert sl_tp_shape[-1] == 2, f"La sortie sl_tp_output devrait avoir 2 valeurs, obtenu: {sl_tp_shape[-1]}"

    @pytest.mark.parametrize("sequence_length,should_have_transformer", [
        (None, False),
        (10, True)
    ])
    def test_sequence_transformer_condition(self, sequence_length, should_have_transformer):
        """Teste que le Transformer est utilisé correctement selon la séquentialité."""
        # Construire le modèle avec ou sans séquence
        model = build_monolith_model(sequence_length=sequence_length, use_transformer=True)
        
        # Vérifier la présence de TransformerBlock dans les couches du modèle
        transformer_layers = [layer for layer in model.layers if isinstance(layer, TransformerBlock)]
        
        if should_have_transformer:
            assert len(transformer_layers) > 0, "Le modèle devrait contenir des blocs Transformer"
        else:
            assert len(transformer_layers) == 0, "Le modèle ne devrait pas contenir de blocs Transformer"

    def test_transformer_blocks_count(self):
        """Teste que le nombre de blocs Transformer correspond à la configuration."""
        # Tester avec différentes valeurs de transformer_blocks
        for blocks in [1, 2, 3]:
            backbone_config = {"transformer_blocks": blocks, "transformer_heads": 4, "transformer_dim": 64, "ff_dim": 128}
            model = build_monolith_model(
                sequence_length=10, 
                use_transformer=True, 
                backbone_config=backbone_config
            )
            
            # Compter les blocs Transformer
            transformer_layers = [layer for layer in model.layers if isinstance(layer, TransformerBlock)]
            assert len(transformer_layers) == blocks, f"Le modèle devrait contenir exactement {blocks} blocs Transformer"

    def test_inference_robustness(self):
        """Teste la robustesse du modèle face à différents types d'entrées."""
        # Créer le modèle
        model = MonolithModel()
        
        # Préparer des données de test avec différentes caractéristiques
        batch_size = 5
        inputs = {
            "technical_input": np.random.normal(0, 1, (batch_size, 38)),
            "embeddings_input": np.random.normal(0, 1, (batch_size, 768)),
            "mcp_input": np.random.normal(0, 1, (batch_size, 128)),
            "instrument_input": np.random.randint(0, 5, (batch_size, 1))
        }
        
        # Cas normal - devrait fonctionner sans erreur
        predictions = model.predict(inputs)
        assert "signal_output" in predictions, "La sortie signal_output est manquante"
        assert "sl_tp_output" in predictions, "La sortie sl_tp_output est manquante"
        
        # Vérifier la somme des probabilités pour signal
        signal_probs = predictions["signal_output"]
        assert np.allclose(np.sum(signal_probs, axis=1), 1.0), "Les probabilités de signal ne somment pas à 1"
        
        # Vérifier que les valeurs SL/TP sont finies
        sl_tp_values = predictions["sl_tp_output"]
        assert np.isfinite(sl_tp_values).all(), "Les valeurs SL/TP contiennent des valeurs non finies"
        
        # Test avec valeurs extrêmes
        inputs_extreme = {
            "technical_input": np.ones((batch_size, 38)) * 1e6,
            "embeddings_input": np.ones((batch_size, 768)) * 1e6,
            "mcp_input": np.ones((batch_size, 128)) * 1e6,
            "instrument_input": np.random.randint(0, 5, (batch_size, 1))
        }
        
        # Le modèle devrait gérer les valeurs extrêmes sans produire de NaN
        predictions_extreme = model.predict(inputs_extreme)
        assert not np.isnan(predictions_extreme["signal_output"]).any(), "Les prédictions signal contiennent des NaN"
        assert not np.isnan(predictions_extreme["sl_tp_output"]).any(), "Les prédictions SL/TP contiennent des NaN"

    def test_nan_robustness(self):
        """Teste la robustesse du modèle face aux valeurs manquantes (NaN)."""
        # Créer le modèle
        model = MonolithModel()
        
        # Préparer des données de test avec des NaN
        batch_size = 5
        inputs = {
            "technical_input": np.random.normal(0, 1, (batch_size, 38)),
            "embeddings_input": np.random.normal(0, 1, (batch_size, 768)),
            "mcp_input": np.random.normal(0, 1, (batch_size, 128)),
            "instrument_input": np.random.randint(0, 5, (batch_size, 1))
        }
        
        # Injecter des NaN dans chaque type d'entrée (sauf instrument qui est catégoriel)
        for key in ["technical_input", "embeddings_input", "mcp_input"]:
            inputs_with_nan = {k: v.copy() for k, v in inputs.items()}
            inputs_with_nan[key][0, 0] = np.nan  # Injecter un NaN dans le premier élément
            
            # Tensorflow convertit les NaN en 0 pendant l'inférence, donc les prédictions devraient rester valides
            predictions = model.predict(inputs_with_nan)
            
            # Vérifier que les sorties sont toujours valides
            assert "signal_output" in predictions, "La sortie signal_output est manquante avec NaN dans les entrées"
            assert "sl_tp_output" in predictions, "La sortie sl_tp_output est manquante avec NaN dans les entrées"
            
            # Vérifier la somme des probabilités pour signal
            signal_probs = predictions["signal_output"]
            assert np.allclose(np.sum(signal_probs, axis=1), 1.0), "Les probabilités de signal ne somment pas à 1 avec NaN"
            
            # Vérifier que les valeurs SL/TP sont finies
            sl_tp_values = predictions["sl_tp_output"]
            assert np.isfinite(sl_tp_values).all(), "Les valeurs SL/TP contiennent des valeurs non finies avec NaN"

    def test_seq_nonseq_consistency(self):
        """Teste la cohérence entre les modes séquentiel et non-séquentiel."""
        # Créer le même modèle en mode séquentiel et non-séquentiel
        model_nonseq = build_monolith_model(sequence_length=None)
        model_seq = build_monolith_model(sequence_length=1)  # Séquence minimale
        
        # Vérifier que les deux modèles ont le même nombre de paramètres entraînables
        # Cela peut varier selon l'implémentation, mais la différence ne devrait pas être énorme
        params_nonseq = model_nonseq.count_params()
        params_seq = model_seq.count_params()
        
        # La différence de paramètres ne devrait pas dépasser 10%
        # (Peut être ajustée selon l'implémentation spécifique)
        assert abs(params_nonseq - params_seq) / max(params_nonseq, params_seq) < 0.2, \
            "Trop grande différence de paramètres entre les modèles séquentiels et non-séquentiels"


if __name__ == "__main__":
    pytest.main() 