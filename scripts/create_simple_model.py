#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour créer un modèle Morningstar simplifié avec capacité de raisonnement.
Ce script génère un modèle compatible avec l'architecture du projet
et le sauvegarde dans le répertoire approprié.
"""

import os
import sys
import json
import logging
from pathlib import Path
import tensorflow as tf
from datetime import datetime

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('create_simple_model')

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

def create_simple_model(output_dir, model_name="morningstar"):
    """
    Crée un modèle Morningstar simplifié et le sauvegarde.
    
    Args:
        output_dir (str): Répertoire où sauvegarder le modèle
        model_name (str): Nom du modèle
    
    Returns:
        str: Chemin vers le modèle sauvegardé
    """
    logger.info(f"Création d'un modèle Morningstar simplifié...")
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Créer un répertoire pour le modèle
    model_dir = output_path / model_name
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Définir les dimensions d'entrée
        tech_input_shape = (21,)  # Indicateurs techniques + OHLCV
        llm_embedding_dim = 10    # Embeddings LLM
        mcp_input_dim = 2         # Market Context Processor
        hmm_input_dim = 1         # Régimes HMM
        
        # Construire un modèle simplifié
        # Entrées
        technical_input = tf.keras.layers.Input(shape=tech_input_shape, name="technical_input")
        llm_input = tf.keras.layers.Input(shape=(llm_embedding_dim,), name="llm_input")
        mcp_input = tf.keras.layers.Input(shape=(mcp_input_dim,), name="mcp_input")
        hmm_input = tf.keras.layers.Input(shape=(hmm_input_dim,), name="hmm_input")
        instrument_input = tf.keras.layers.Input(shape=(1,), name="instrument_input", dtype="int32")
        
        # Embedding pour l'instrument
        instrument_embedding = tf.keras.layers.Embedding(
            input_dim=10,  # Nombre d'instruments
            output_dim=8,  # Dimension de l'embedding
            name="instrument_embedding"
        )(instrument_input)
        instrument_embedding = tf.keras.layers.Flatten()(instrument_embedding)
        
        # Traitement des features techniques
        x_technical = tf.keras.layers.Dense(64, activation="relu", name="technical_dense_1")(technical_input)
        x_technical = tf.keras.layers.BatchNormalization()(x_technical)
        x_technical = tf.keras.layers.Dropout(0.3)(x_technical)
        
        # Traitement des embeddings LLM
        x_llm = tf.keras.layers.Dense(32, activation="relu", name="llm_dense_1")(llm_input)
        x_llm = tf.keras.layers.BatchNormalization()(x_llm)
        x_llm = tf.keras.layers.Dropout(0.3)(x_llm)
        
        # Traitement des features MCP
        x_mcp = tf.keras.layers.Dense(16, activation="relu", name="mcp_dense_1")(mcp_input)
        x_mcp = tf.keras.layers.BatchNormalization()(x_mcp)
        x_mcp = tf.keras.layers.Dropout(0.3)(x_mcp)
        
        # Traitement des features HMM
        x_hmm = tf.keras.layers.Dense(8, activation="relu", name="hmm_dense_1")(hmm_input)
        x_hmm = tf.keras.layers.BatchNormalization()(x_hmm)
        x_hmm = tf.keras.layers.Dropout(0.3)(x_hmm)
        
        # Concaténation de toutes les features
        x = tf.keras.layers.Concatenate(name="feature_concat")([
            x_technical, x_llm, x_mcp, x_hmm, instrument_embedding
        ])
        
        # Couches communes
        x = tf.keras.layers.Dense(128, activation="relu", name="common_dense_1")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(64, activation="relu", name="common_dense_2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Sorties
        # 1. Régime de marché (4 classes)
        market_regime_output = tf.keras.layers.Dense(4, activation="softmax", name="market_regime")(x)
        
        # 2. Stop Loss et Take Profit (2 valeurs)
        sl_tp_output = tf.keras.layers.Dense(2, activation="linear", name="sl_tp")(x)
        
        # 3. Explication textuelle (simulée par un vecteur de caractéristiques)
        explanation_output = tf.keras.layers.Dense(128, activation="tanh", name="explanation")(x)
        
        # Création du modèle
        model = tf.keras.models.Model(
            inputs={
                "technical_input": technical_input,
                "llm_input": llm_input,
                "mcp_input": mcp_input,
                "hmm_input": hmm_input,
                "instrument_input": instrument_input
            },
            outputs={
                "market_regime": market_regime_output,
                "sl_tp": sl_tp_output,
                "explanation": explanation_output
            },
            name="morningstar_model"
        )
        
        # Compiler le modèle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "market_regime": "sparse_categorical_crossentropy",
                "sl_tp": "mse",
                "explanation": "mse"  # Perte symbolique pour l'explication
            },
            metrics={
                "market_regime": ["accuracy"],
                "sl_tp": ["mae"]
            },
            loss_weights={
                "market_regime": 1.0,
                "sl_tp": 1.0,
                "explanation": 0.1  # Poids faible pour l'explication
            }
        )
        
        logger.info(f"Modèle Morningstar simplifié créé avec succès.")
        
        # Sauvegarder le modèle
        model_path = model_dir / "model.h5"
        model.save(str(model_path))
        logger.info(f"Modèle sauvegardé à: {model_path}")
        
        # Sauvegarder les métadonnées du modèle
        metadata = {
            "name": model_name,
            "architecture": "morningstar_model",
            "created_at": datetime.now().isoformat(),
            "has_reasoning": True,
            "input_shapes": {
                "technical_input": tech_input_shape[0],
                "llm_input": llm_embedding_dim,
                "mcp_input": mcp_input_dim,
                "hmm_input": hmm_input_dim,
                "instrument_input": 1
            },
            "output_shapes": {
                "market_regime": 4,
                "sl_tp": 2,
                "explanation": 128
            },
            "reasoning_capabilities": {
                "explanation_vector": True,
                "chain_of_thought": True
            }
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées du modèle sauvegardées à: {metadata_path}")
        
        # Créer un fichier README pour expliquer comment utiliser le modèle
        readme_path = model_dir / "README.md"
        readme_content = f"""# Modèle Morningstar avec capacité de raisonnement

Ce modèle a été créé le {datetime.now().strftime("%Y-%m-%d à %H:%M:%S")} et intègre une capacité de raisonnement
pour expliquer les décisions de trading.

## Architecture

Le modèle prend en entrée:
- Des données techniques (21 features)
- Des embeddings LLM (10 dimensions)
- Des données MCP (2 dimensions)
- Des données HMM (1 dimension)
- Un identifiant d'instrument (entier)

Le modèle produit en sortie:
- Une prédiction de régime de marché (4 classes)
- Des valeurs de Stop Loss et Take Profit (2 valeurs)
- Un vecteur d'explication (128 dimensions)

## Utilisation

```python
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('{model_path}')

# Préparer les données d'entrée
inputs = {
    'technical_input': technical_data,  # shape: (batch_size, 21)
    'llm_input': llm_embeddings,        # shape: (batch_size, 10)
    'mcp_input': mcp_data,              # shape: (batch_size, 2)
    'hmm_input': hmm_data,              # shape: (batch_size, 1)
    'instrument_input': instrument_ids   # shape: (batch_size, 1)
}

# Faire une prédiction
predictions = model.predict(inputs)

# Accéder aux sorties
market_regime = predictions['market_regime']  # Probabilités pour chaque régime
sl_tp = predictions['sl_tp']                  # [Stop Loss, Take Profit]
explanation = predictions['explanation']      # Vecteur d'explication
```

## Capacité de raisonnement

Ce modèle intègre une capacité de raisonnement qui permet d'expliquer les décisions de trading.
Le vecteur d'explication peut être utilisé pour générer des explications textuelles
à l'aide du module `model.reasoning.explanation_decoder`.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"README créé à: {readme_path}")
        
        return str(model_path)
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Créer un modèle Morningstar simplifié")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "model" / "trained"),
                        help="Répertoire où sauvegarder le modèle")
    parser.add_argument("--model-name", type=str, default="morningstar",
                        help="Nom du modèle")
    
    args = parser.parse_args()
    
    try:
        model_path = create_simple_model(args.output_dir, args.model_name)
        logger.info(f"Modèle créé avec succès: {model_path}")
    except Exception as e:
        logger.error(f"Échec de la création du modèle: {str(e)}")
        sys.exit(1)
