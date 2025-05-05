#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour créer un modèle compatible avec l'architecture Morningstar.
Ce script génère un modèle avec l'architecture hybride améliorée définie dans le projet
et le sauvegarde dans le répertoire approprié.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import tensorflow as tf
from datetime import datetime

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Importer l'architecture du modèle
from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model

def create_compatible_model(output_dir, model_name=None):
    """
    Crée un modèle compatible avec l'architecture Morningstar et le sauvegarde.
    
    Args:
        output_dir (str): Répertoire où sauvegarder le modèle
        model_name (str, optional): Nom du modèle. Si None, un nom sera généré.
    
    Returns:
        str: Chemin vers le modèle sauvegardé
    """
    print(f"Création d'un modèle compatible avec l'architecture Morningstar...")
    
    # Définir les dimensions d'entrée selon l'architecture du projet
    tech_input_shape = (21,)  # Indicateurs techniques + OHLCV
    llm_embedding_dim = 10    # Embeddings CryptoBERT
    mcp_input_dim = 2         # Market Context Processor
    hmm_input_dim = 1         # Régimes HMM
    
    # Définir les dimensions de sortie
    num_trading_classes = 5      # Signal de trading (5 classes)
    num_market_regime_classes = 4 # Régime de marché (4 classes)
    num_volatility_quantiles = 3  # Quantiles de volatilité (3 valeurs)
    num_sl_tp_outputs = 2         # Stop Loss et Take Profit (2 valeurs)
    
    # Construire le modèle
    model = build_enhanced_hybrid_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=llm_embedding_dim,
        mcp_input_dim=mcp_input_dim,
        hmm_input_dim=hmm_input_dim,
        num_trading_classes=num_trading_classes,
        num_market_regime_classes=num_market_regime_classes,
        num_volatility_quantiles=num_volatility_quantiles,
        num_sl_tp_outputs=num_sl_tp_outputs,
        use_llm=True
    )
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'trading_signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy',
            'volatility_quantiles': 'mse',
            'sl_tp': 'mse'
        },
        metrics={
            'trading_signal': ['accuracy'],
            'market_regime': ['accuracy'],
            'volatility_quantiles': ['mae'],
            'sl_tp': ['mae']
        }
    )
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Générer un nom de modèle s'il n'est pas fourni
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"morningstar_model_{timestamp}"
    
    # Créer un répertoire pour le modèle
    model_dir = output_path / model_name
    model_dir.mkdir(exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = model_dir / "model.h5"
    model.save(str(model_path))
    print(f"Modèle sauvegardé à: {model_path}")
    
    # Sauvegarder les métadonnées du modèle
    metadata = {
        "name": model_name,
        "architecture": "enhanced_hybrid_model",
        "created_at": datetime.now().isoformat(),
        "input_shapes": {
            "technical_input": tech_input_shape[0],
            "llm_input": llm_embedding_dim,
            "mcp_input": mcp_input_dim,
            "hmm_input": hmm_input_dim,
            "instrument_input": 1
        },
        "output_shapes": {
            "trading_signal": num_trading_classes,
            "market_regime": num_market_regime_classes,
            "volatility_quantiles": num_volatility_quantiles,
            "sl_tp": num_sl_tp_outputs
        }
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Métadonnées du modèle sauvegardées à: {metadata_path}")
    
    return str(model_path)

def main():
    parser = argparse.ArgumentParser(description="Créer un modèle compatible avec l'architecture Morningstar")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "model" / "trained"),
                        help="Répertoire où sauvegarder le modèle")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Nom du modèle (optionnel)")
    
    args = parser.parse_args()
    
    model_path = create_compatible_model(args.output_dir, args.model_name)
    print(f"Modèle créé avec succès: {model_path}")

if __name__ == "__main__":
    main()
