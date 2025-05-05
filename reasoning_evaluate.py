#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'évaluation du modèle avec capacité de raisonnement.
Ce script charge un modèle entraîné et génère des prédictions avec des explications.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.reasoning.reasoning_module import ReasoningModule, ExplanationDecoder
from model.training.reasoning_training import load_data

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Évaluation du modèle avec capacité de raisonnement")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv",
        help="Chemin vers le dataset normalisé"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model/reasoning_model/best_model.h5",
        help="Chemin vers le modèle entraîné"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reasoning",
        help="Répertoire de sortie pour les résultats"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Nombre d'exemples à afficher avec des explications"
    )
    
    return parser.parse_args()

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
    
    Returns:
        Dictionnaire de métriques
    """
    # Prédictions
    predictions = model.predict(X_test)
    
    # Extraire les prédictions
    if isinstance(predictions, dict):
        market_regime_pred = predictions['market_regime']
        sl_tp_pred = predictions['sl_tp']
        reasoning_output = predictions.get('reasoning_output', None)
    else:
        market_regime_pred = predictions[0]
        sl_tp_pred = predictions[1]
        reasoning_output = predictions[2] if len(predictions) > 2 else None
    
    # Convertir les prédictions de régime de marché en classes
    market_regime_classes = np.argmax(market_regime_pred, axis=1)
    true_market_regime = y_test['market_regime']
    
    # Calculer les métriques pour le régime de marché
    market_regime_accuracy = accuracy_score(true_market_regime, market_regime_classes)
    
    # Calculer les métriques pour SL/TP
    sl_tp_rmse = np.sqrt(mean_squared_error(y_test['sl_tp'], sl_tp_pred))
    sl_tp_mae = mean_absolute_error(y_test['sl_tp'], sl_tp_pred)
    
    # Retourner les métriques
    metrics = {
        'market_regime_accuracy': market_regime_accuracy,
        'sl_tp_rmse': sl_tp_rmse,
        'sl_tp_mae': sl_tp_mae
    }
    
    return metrics, market_regime_classes, sl_tp_pred, reasoning_output

def generate_explanations(model, X_test, feature_names, num_examples=5):
    """
    Génère des explications pour les prédictions du modèle.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        feature_names: Noms des features
        num_examples: Nombre d'exemples à afficher
    
    Returns:
        DataFrame avec les prédictions et les explications
    """
    # Limiter le nombre d'exemples
    num_examples = min(num_examples, len(X_test['technical_input']))
    
    # Créer un sous-ensemble pour les exemples
    X_examples = {}
    for key, value in X_test.items():
        X_examples[key] = value[:num_examples]
    
    # Faire des prédictions
    predictions = model.predict(X_examples)
    
    # Extraire les prédictions et les sorties de raisonnement
    if isinstance(predictions, dict):
        market_regime_pred = predictions['market_regime']
        sl_tp_pred = predictions['sl_tp']
        reasoning_output = predictions.get('reasoning_output', None)
        attention_scores = predictions.get('attention_scores', None)
    else:
        market_regime_pred = predictions[0]
        sl_tp_pred = predictions[1]
        reasoning_output = predictions[2] if len(predictions) > 2 else None
        attention_scores = predictions[3] if len(predictions) > 3 else None
    
    # Convertir les prédictions de régime de marché en classes
    market_regime_classes = np.argmax(market_regime_pred, axis=1)
    
    # Créer un décodeur d'explications
    market_regime_names = ['sideways', 'bullish', 'bearish']
    if max(market_regime_classes) < len(market_regime_names):
        regime_names = market_regime_names
    else:
        regime_names = [f'regime_{i}' for i in range(max(market_regime_classes) + 1)]
    
    decoder = ExplanationDecoder(feature_names=feature_names, market_regime_names=regime_names)
    
    # Générer des explications pour chaque exemple
    explanations = []
    for i in range(num_examples):
        example_dict = {
            'example_id': i,
            'market_regime_pred': market_regime_classes[i],
            'market_regime_confidence': np.max(market_regime_pred[i]),
            'sl_pred': sl_tp_pred[i][0],
            'tp_pred': sl_tp_pred[i][1],
        }
        
        # Ajouter des explications si disponibles
        if reasoning_output is not None and attention_scores is not None:
            # Extraire les scores d'attention pour cet exemple
            example_attention = attention_scores[i]
            
            # Générer des explications
            market_regime_explanation = decoder.decode_market_regime_explanation(
                market_regime_classes[i],
                reasoning_output[i],
                example_attention,
                top_k=5
            )
            
            sl_tp_explanation = decoder.decode_sl_tp_explanation(
                sl_tp_pred[i][0],
                sl_tp_pred[i][1],
                reasoning_output[i],
                reasoning_output[i],
                example_attention,
                top_k=5
            )
            
            example_dict['market_regime_explanation'] = market_regime_explanation
            example_dict['sl_explanation'] = sl_tp_explanation[0]
            example_dict['tp_explanation'] = sl_tp_explanation[1]
        
        explanations.append(example_dict)
    
    return pd.DataFrame(explanations)

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Charger les données
    logger.info(f"Chargement des données depuis {args.data_path}")
    X_train, X_val, y_train, y_val, X_test, y_test, feature_names = load_data(
        args.data_path, validation_split=0.1, random_state=42
    )
    
    # Charger le modèle
    logger.info(f"Chargement du modèle depuis {args.model_path}")
    
    # Au lieu de charger directement le modèle, nous allons le reconstruire et charger les poids
    try:
        # Essayer d'abord de charger le modèle directement
        custom_objects = {
            'ReasoningModule': ReasoningModule
        }
        model = load_model(args.model_path, custom_objects=custom_objects)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.warning(f"Erreur lors du chargement du modèle: {e}")
        logger.info("Reconstruction du modèle et chargement des poids...")
        
        # Reconstruire le modèle
        from model.architecture.reasoning_model import build_reasoning_model
        
        # Déterminer les dimensions d'entrée à partir des données de test
        tech_input_shape = X_test['technical_input'].shape[1:]
        llm_input_shape = X_test['llm_input'].shape[1:]
        mcp_input_shape = X_test['mcp_input'].shape[1:]
        hmm_input_shape = X_test['hmm_input'].shape[1:]
        instrument_vocab_size = np.max(X_test['instrument_input']) + 1
        
        # Construire le modèle avec les mêmes paramètres que lors de l'entraînement
        model = build_reasoning_model(
            tech_input_shape=tech_input_shape,
            llm_embedding_dim=llm_input_shape[0] if len(llm_input_shape) > 0 else 1,
            mcp_input_dim=mcp_input_shape[0] if len(mcp_input_shape) > 0 else 1,
            hmm_input_dim=hmm_input_shape[0] if len(hmm_input_shape) > 0 else 4,
            instrument_vocab_size=int(instrument_vocab_size),
            instrument_embedding_dim=8,
            num_market_regime_classes=2,  # Adapté à notre cas
            num_sl_tp_outputs=2,
            l2_reg=0.01,
            dropout_rate=0.3,
            use_batch_norm=True,
            num_reasoning_steps=2,
            reasoning_units=128,
            num_attention_heads=4,
            attention_key_dim=64,
            feature_names=feature_names
        )
        
        # Compiler le modèle
        model = compile_reasoning_model(model, learning_rate=0.001)
        
        # Charger les poids
        model.load_weights(args.model_path)
        logger.info("Modèle reconstruit et poids chargés avec succès")
    
    # Évaluer le modèle
    logger.info("Évaluation du modèle sur l'ensemble de test")
    metrics, market_regime_classes, sl_tp_pred, reasoning_output = evaluate_model(model, X_test, y_test)
    
    # Afficher les métriques
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value}")
    
    # Générer des explications
    logger.info("Génération des explications pour les exemples de test")
    explanations_df = generate_explanations(model, X_test, feature_names, args.num_examples)
    
    # Sauvegarder les explications
    explanations_path = os.path.join(args.output_dir, 'explanations.csv')
    explanations_df.to_csv(explanations_path, index=False)
    logger.info(f"Explications sauvegardées dans {explanations_path}")
    
    # Afficher quelques exemples d'explications
    logger.info("\nExemples d'explications:")
    for _, row in explanations_df.iterrows():
        logger.info(f"\nExemple {row['example_id']}:")
        logger.info(f"Prédiction de régime de marché: {row['market_regime_pred']} (confiance: {row['market_regime_confidence']:.2f})")
        logger.info(f"Prédiction de SL/TP: SL={row['sl_pred']:.4f}, TP={row['tp_pred']:.4f}")
        
        if 'market_regime_explanation' in row:
            logger.info(f"Explication du régime de marché: {row['market_regime_explanation']}")
            logger.info(f"Explication du SL: {row['sl_explanation']}")
            logger.info(f"Explication du TP: {row['tp_explanation']}")

if __name__ == "__main__":
    main()
