#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic reasoning training pipeline (legacy)
"""

import os, sys
# Ajout pour que 'from model...' fonctionne, en priorité
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import tensorflow as tf
from config.config import Config
from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.training.data_loader import load_and_split_data

def main():
    parser = argparse.ArgumentParser("Reasoning Training")
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-dir', default='outputs/reasoning')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config()

    # Chargement et split
    # Assurez-vous que les dates de split sont bien gérées par load_and_split_data
    # ou ajustez l'appel si nécessaire.
    # Récupérer les dimensions des features depuis la config
    num_technical = cfg.get_config('model.num_technical', 0)
    num_mcp = cfg.get_config('model.num_mcp', 0)
    num_cryptobert = cfg.get_config('model.num_cryptobert', 0) # Utilise la nouvelle clé
    num_hmm = cfg.get_config('model.num_hmm', 0)
    num_sentiment = cfg.get_config('model.num_sentiment', 0)
    num_market = cfg.get_config('model.num_market', 0)
    instrument_vocab_size = cfg.get_config('model.instrument_vocab_size', 0)

    # Passer les dimensions correctes à load_and_split_data
    # Note: load_and_split_data utilise num_llm_features pour les colonnes bert_*,
    # nous devons donc passer num_cryptobert à cet argument.
    X, y = load_and_split_data(
        file_path=args.data_path,
        label_columns=cfg.get_config('data.label_columns'),
        as_tensor=False,
        num_technical_features=num_technical,
        num_llm_features=num_cryptobert, # Passe num_cryptobert à num_llm_features
        num_mcp_features=num_mcp
        # Les autres dimensions (hmm, sentiment, market) ne sont pas directement utilisées
        # par load_and_split_data pour la validation, mais la logique d'extraction les gère.
    )

    # Construction & compilation
    # Comme pour enhanced_reasoning_training.py, assurez-vous que les paramètres du modèle
    # sont correctement passés, soit via la config, soit en les dérivant de X.
    model_params = cfg.get_config('model.reasoning_architecture')
    # Passer toutes les dimensions nécessaires à build_reasoning_model
    model_params['tech_input_shape'] = (num_technical,) if num_technical > 0 else None
    model_params['mcp_input_dim'] = num_mcp
    model_params['cryptobert_input_dim'] = num_cryptobert # Utilise la nouvelle clé
    model_params['hmm_input_dim'] = num_hmm
    model_params['sentiment_input_dim'] = num_sentiment
    model_params['market_input_dim'] = num_market
    model_params['instrument_vocab_size'] = instrument_vocab_size
    # Supprimer llm_embedding_dim si build_reasoning_model ne l'utilise plus
    if 'llm_embedding_dim' in model_params: del model_params['llm_embedding_dim'] 
    
    # Récupérer les sorties actives pour la construction et la compilation
    active_outputs = cfg.get_config('model.active_outputs', ['market_regime', 'sl_tp', 'reasoning'])
    model_params['active_outputs'] = active_outputs
    
    # Récupérer le nombre de classes pour market_regime à partir du mapping
    market_regime_mapping = cfg.get_config('data.label_mappings.market_regime', {})
    if market_regime_mapping:
        # Le nombre de classes est le nombre d'indices uniques + 1 (car les indices commencent à 0)
        num_market_regime_classes = max(market_regime_mapping.values()) + 1
    else:
        # Utiliser la valeur par défaut du modèle si aucun mapping n'est trouvé
        # (ou définir une valeur par défaut plus sûre ici)
        num_market_regime_classes = 2 # Valeur par défaut de build_reasoning_model
        print("WARN: Mapping market_regime non trouvé dans la config, utilise num_classes=2 par défaut.")
        
    model_params['num_market_regime_classes'] = num_market_regime_classes

    model = build_reasoning_model(**model_params)
    
    # Récupérer le learning rate pour la compilation
    learning_rate_config = cfg.get_config('training.learning_rate', 0.001)
    try:
        # Assurer explicitement que c'est un float
        learning_rate = float(learning_rate_config) 
    except (ValueError, TypeError):
        print(f"WARN: Impossible de convertir learning_rate '{learning_rate_config}' en float. Utilisation de 0.001 par défaut.")
        learning_rate = 0.001
        
    compile_reasoning_model(model, learning_rate=learning_rate, active_outputs=active_outputs)

    # Entraînement
    ckpt_path = os.path.join(args.output_dir, 'best_model.h5')
    if not ckpt_path.endswith(('.h5', '.keras', '.weights.h5')): # TF >= 2.11
        ckpt_path += '.keras'
        
    final_model_path = os.path.join(args.output_dir, 'final_model.h5')
    if not final_model_path.endswith(('.h5', '.keras', '.weights.h5')): # TF >= 2.11
        final_model_path += '.keras'

    # Préparer les données pour Keras (dictionnaire d'inputs et de labels)
    # X est déjà un dictionnaire d'inputs numpy
    # y est un dictionnaire de labels Series pandas
    
    # Convertir les labels Series en numpy array pour Keras
    y_train = {}
    for label_name, label_series in y.items():
        if label_name in model.output_names: # S'assurer que le label correspond à une sortie du modèle
             y_train[label_name] = label_series.values
        else:
             print(f"WARN: Label '{label_name}' trouvé dans les données mais pas dans les sorties du modèle ({model.output_names}). Il sera ignoré.")

    # Vérifier si y_train contient au moins un label attendu par le modèle
    if not any(key in y_train for key in model.output_names):
         raise ValueError(f"Aucun des labels chargés ({list(y.keys())}) ne correspond aux sorties attendues du modèle ({model.output_names}). Vérifiez 'data.label_columns' dans config.yaml.")

    history = model.fit(
        X, y_train, # Utiliser y_train (numpy arrays)
        epochs=cfg.get_config('training.epochs', 50),
        batch_size=cfg.get_config('training.batch_size', 64),
        validation_split=cfg.get_config('data.validation_split', 0.2),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path,
                save_best_only=True
            )
        ]
    )

    model.save(final_model_path)

if __name__ == '__main__':
    main()
