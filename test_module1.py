#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour le Module 1 (CNN+LSTM Feature Extractor)
Ce script vérifie le bon fonctionnement du modèle hybride avec CNN+LSTM.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajouter le répertoire principal au PYTHONPATH
BASE_DIR = Path('/home/morningstar/Desktop/crypto_robot/Morningstar')
sys.path.append(str(BASE_DIR))

# Importer le modèle
from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model, DEFAULT_INSTRUMENT_VOCAB_SIZE

def test_cnn_lstm_model():
    """
    Test du modèle hybride avec CNN+LSTM pour l'extraction de caractéristiques.
    """
    logger.info("Test du modèle hybride avec CNN+LSTM")
    
    # Paramètres du test
    batch_size = 10
    time_steps = 10
    tech_features = 6  # 6 features techniques par pas de temps (OHLCV + HMM)
    tech_input_shape = (time_steps * tech_features,)
    llm_embedding_dim = 768
    mcp_input_dim = 128
    hmm_input_dim = 4
    
    # Créer des données de test
    technical_data = np.random.normal(0, 1, (batch_size, *tech_input_shape))
    llm_data = np.random.normal(0, 1, (batch_size, llm_embedding_dim))
    mcp_data = np.random.normal(0, 1, (batch_size, mcp_input_dim))
    hmm_data = np.random.normal(0, 1, (batch_size, hmm_input_dim))
    instrument_data = np.random.randint(0, DEFAULT_INSTRUMENT_VOCAB_SIZE, (batch_size, 1))
    
    logger.info(f"Dimensions des données de test:")
    logger.info(f"- Technical: {technical_data.shape}")
    logger.info(f"- LLM: {llm_data.shape}")
    logger.info(f"- MCP: {mcp_data.shape}")
    logger.info(f"- HMM: {hmm_data.shape}")
    logger.info(f"- Instrument: {instrument_data.shape}")
    
    # Construire le modèle
    logger.info("Construction du modèle avec CNN+LSTM")
    model = build_enhanced_hybrid_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=llm_embedding_dim,
        mcp_input_dim=mcp_input_dim,
        hmm_input_dim=hmm_input_dim,
        use_cnn_lstm=True,
        time_steps=time_steps,
        tech_features=tech_features,
        cnn_filters=[32, 64],
        cnn_kernel_sizes=[3, 3],
        cnn_pool_sizes=[2, 2],
        lstm_units=[64],
        bidirectional_lstm=True,
        active_outputs=['signal', 'market_regime']
    )
    
    # Afficher un résumé du modèle
    model.summary()
    
    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss={
            'signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy'
        },
        metrics={
            'signal': ['accuracy'],
            'market_regime': ['accuracy']
        }
    )
    
    # Prédire avec le modèle
    logger.info("Test de prédiction avec le modèle")
    predictions = model.predict({
        'technical_input': technical_data,
        'llm_input': llm_data,
        'mcp_input': mcp_data,
        'hmm_input': hmm_data,
        'instrument_input': instrument_data
    })
    
    # Vérifier les dimensions des sorties
    logger.info("Dimensions des sorties:")
    for name, pred in predictions.items():
        logger.info(f"- {name}: {pred.shape}")
    
    logger.info("Test réussi !")
    return model

# Test du modèle sans CNN+LSTM pour comparaison
def test_standard_model():
    """
    Test du modèle standard sans CNN+LSTM.
    """
    logger.info("Test du modèle standard sans CNN+LSTM")
    
    # Paramètres du test
    batch_size = 10
    tech_input_shape = (60,)  # 60 features techniques
    llm_embedding_dim = 768
    mcp_input_dim = 128
    hmm_input_dim = 4
    
    # Créer des données de test
    technical_data = np.random.normal(0, 1, (batch_size, *tech_input_shape))
    llm_data = np.random.normal(0, 1, (batch_size, llm_embedding_dim))
    mcp_data = np.random.normal(0, 1, (batch_size, mcp_input_dim))
    hmm_data = np.random.normal(0, 1, (batch_size, hmm_input_dim))
    instrument_data = np.random.randint(0, DEFAULT_INSTRUMENT_VOCAB_SIZE, (batch_size, 1))
    
    # Construire le modèle
    logger.info("Construction du modèle standard")
    model = build_enhanced_hybrid_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=llm_embedding_dim,
        mcp_input_dim=mcp_input_dim,
        hmm_input_dim=hmm_input_dim,
        use_cnn_lstm=False,
        active_outputs=['signal', 'market_regime']
    )
    
    # Afficher un résumé du modèle
    model.summary()
    
    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss={
            'signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy'
        },
        metrics={
            'signal': ['accuracy'],
            'market_regime': ['accuracy']
        }
    )
    
    # Prédire avec le modèle
    logger.info("Test de prédiction avec le modèle standard")
    predictions = model.predict({
        'technical_input': technical_data,
        'llm_input': llm_data,
        'mcp_input': mcp_data,
        'hmm_input': hmm_data,
        'instrument_input': instrument_data
    })
    
    # Vérifier les dimensions des sorties
    logger.info("Dimensions des sorties:")
    for name, pred in predictions.items():
        logger.info(f"- {name}: {pred.shape}")
    
    logger.info("Test réussi !")
    return model

if __name__ == "__main__":
    logger.info("=== Test du Module 1: CNN+LSTM Feature Extractor ===")
    
    # Tester le modèle avec CNN+LSTM
    cnn_lstm_model = test_cnn_lstm_model()
    
    # Tester le modèle standard pour comparaison
    standard_model = test_standard_model()
    
    logger.info("Tous les tests ont réussi !")
