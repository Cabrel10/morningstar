#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Exemple de code pour ajuster le modèle en fonction des dimensions du dataset."""

from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model

# Construire le modèle avec les dimensions ajustées
model = build_enhanced_hybrid_model(
    tech_input_shape=(21,),  # Ajustement pour les features techniques
    llm_embedding_dim=10,  # Ajustement pour les embeddings LLM
    # Autres paramètres inchangés
    instrument_vocab_size=10,
    instrument_embedding_dim=8,
    num_trading_classes=5,
    num_market_regime_classes=4,
    num_volatility_quantiles=3,
    num_sl_tp_outputs=2
)

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss={
        'signal': 'categorical_crossentropy',
        'market_regime': 'categorical_crossentropy',
        'volatility_quantiles': 'mse',
        'sl_tp': 'mse'
    },
    metrics={
        'signal': ['accuracy'],
        'market_regime': ['accuracy'],
        'volatility_quantiles': ['mae'],
        'sl_tp': ['mae']
    }
)

# Afficher le résumé du modèle
model.summary()