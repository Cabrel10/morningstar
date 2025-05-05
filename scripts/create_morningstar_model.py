#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour créer un modèle Morningstar avec capacité de raisonnement.
Ce script génère un modèle compatible avec l'architecture du projet
et le sauvegarde dans le répertoire approprié.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import tensorflow as tf
from datetime import datetime
import logging

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('create_morningstar_model')

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Importer les architectures de modèle
from model.architecture.morningstar_model import MorningstarModel
from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model

def create_morningstar_model(output_dir, model_name=None, use_reasoning=True):
    """
    Crée un modèle Morningstar avec ou sans capacité de raisonnement et le sauvegarde.
    
    Args:
        output_dir (str): Répertoire où sauvegarder le modèle
        model_name (str, optional): Nom du modèle. Si None, un nom sera généré.
        use_reasoning (bool): Si True, utilise le modèle avec capacité de raisonnement
    
    Returns:
        str: Chemin vers le modèle sauvegardé
    """
    logger.info(f"Création d'un modèle Morningstar {'avec' if use_reasoning else 'sans'} capacité de raisonnement...")
    
    # Générer un nom de modèle s'il n'est pas fourni
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"morningstar_{'reasoning_' if use_reasoning else ''}model_{timestamp}"
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Créer un répertoire pour le modèle
    model_dir = output_path / model_name
    model_dir.mkdir(exist_ok=True)
    
    try:
        if use_reasoning:
            # Définir les dimensions d'entrée selon l'architecture du projet
            tech_input_shape = (21,)  # Indicateurs techniques + OHLCV
            llm_embedding_dim = 10    # Embeddings CryptoBERT
            mcp_input_dim = 2         # Market Context Processor
            hmm_input_dim = 1         # Régimes HMM
            sentiment_input_dim = 5   # Sentiment analysis
            cryptobert_input_dim = 768 # CryptoBERT embeddings
            market_input_dim = 10     # Market info
            
            # Définir les dimensions de sortie
            num_market_regime_classes = 4 # Régime de marché (4 classes)
            num_sl_tp_outputs = 2         # Stop Loss et Take Profit (2 valeurs)
            
            # Construire le modèle avec capacité de raisonnement
            model = build_reasoning_model(
                tech_input_shape=tech_input_shape,
                llm_embedding_dim=llm_embedding_dim,
                mcp_input_dim=mcp_input_dim,
                hmm_input_dim=hmm_input_dim,
                sentiment_input_dim=sentiment_input_dim,
                cryptobert_input_dim=cryptobert_input_dim,
                market_input_dim=market_input_dim,
                num_market_regime_classes=num_market_regime_classes,
                num_sl_tp_outputs=num_sl_tp_outputs,
                use_chain_of_thought=True,
                num_reasoning_steps=3,
                num_attention_heads=4,
                feature_names=[f"feature_{i}" for i in range(21)]  # Noms des features pour le décodage des explications
            )
            
            # Compiler le modèle
            model = compile_reasoning_model(model)
            
            logger.info(f"Modèle Morningstar avec capacité de raisonnement créé avec succès.")
        else:
            # Créer le modèle Morningstar standard
            model_wrapper = MorningstarModel(
                model_config={
                    'num_technical_features': 21,
                    'llm_embedding_dim': 10,
                    'mcp_input_dim': 2,
                    'hmm_input_dim': 1,
                    'instrument_vocab_size': 10,
                    'instrument_embedding_dim': 8,
                    'num_signal_classes': 5,
                    'num_market_regimes': 4,
                    'num_volatility_quantiles': 3,
                    'num_sl_tp_outputs': 2
                },
                use_llm=True,
                llm_fallback_strategy='technical_projection'
            )
            
            # Récupérer le modèle Keras sous-jacent
            model = model_wrapper.model
            logger.info(f"Modèle Morningstar standard créé avec succès.")
        
        # Sauvegarder le modèle
        model_path = model_dir / "model.h5"
        model.save(str(model_path))
        logger.info(f"Modèle sauvegardé à: {model_path}")
        
        # Sauvegarder les métadonnées du modèle
        metadata = {
            "name": model_name,
            "architecture": "reasoning_model" if use_reasoning else "morningstar_model",
            "created_at": datetime.now().isoformat(),
            "has_reasoning": use_reasoning,
            "input_shapes": {
                "technical_input": 21,
                "llm_input": 10,
                "mcp_input": 2,
                "hmm_input": 1,
                "instrument_input": 1
            },
            "output_shapes": {
                "market_regime": 4,
                "sl_tp": 2
            }
        }
        
        if use_reasoning:
            metadata["reasoning_capabilities"] = {
                "chain_of_thought": True,
                "attention_mechanism": True,
                "reasoning_steps": 3
            }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées du modèle sauvegardées à: {metadata_path}")
        
        return str(model_path)
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle: {str(e)}")
        raise
    
def main():
    parser = argparse.ArgumentParser(description="Créer un modèle Morningstar avec ou sans capacité de raisonnement")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "model" / "trained"),
                        help="Répertoire où sauvegarder le modèle")
    parser.add_argument("--model-name", type=str, default="morningstar",
                        help="Nom du modèle")
    parser.add_argument("--use-reasoning", action="store_true", default=True,
                        help="Utiliser le modèle avec capacité de raisonnement")
    
    args = parser.parse_args()
    
    try:
        model_path = create_morningstar_model(args.output_dir, args.model_name, args.use_reasoning)
        logger.info(f"Modèle créé avec succès: {model_path}")
    except Exception as e:
        logger.error(f"Échec de la création du modèle: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
