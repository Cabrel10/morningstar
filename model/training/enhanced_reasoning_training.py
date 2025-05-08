#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training pipeline for the enhanced reasoning model
"""

import os, sys
# ⚙️ Ajout pour que 'from model.architecture...' fonctionne, en priorité
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import tensorflow as tf
import json # Ajout de l'import json
from config.config import Config
from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.training.data_loader import load_and_split_data
from model.reasoning.reasoning_module import ExplanationDecoder # Correction de l'import

def main():
    parser = argparse.ArgumentParser("Enhanced Reasoning Training")
    parser.add_argument('--data-path', required=True, help='Path to enriched data file')
    parser.add_argument('--output-dir', default='outputs/enhanced', help='Where to save models and logs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config()

    # Chargement des données
    # Récupérer les dimensions des features depuis la config
    num_technical = cfg.get_config('model.num_technical', 0)
    num_mcp = cfg.get_config('model.num_mcp', 0)
    num_cryptobert = cfg.get_config('model.num_cryptobert', 0)
    num_hmm = cfg.get_config('model.num_hmm', 0)
    num_sentiment = cfg.get_config('model.num_sentiment', 0)
    num_market = cfg.get_config('model.num_market', 0)
    instrument_vocab_size = cfg.get_config('model.instrument_vocab_size', 0)

    X, y = load_and_split_data(
        file_path=args.data_path,
        label_columns=cfg.get_config('data.label_columns'), # Utilise data.label_columns
        # split_ratios=cfg.get_config('data.split_ratios'), # Supprimé
        as_tensor=False,
        num_technical_features=num_technical,
        num_llm_features=num_cryptobert, # Passe num_cryptobert à num_llm_features
        num_mcp_features=num_mcp
    )

    # Construction et compilation du modèle
    # Assurez-vous que cfg.get_config('model.reasoning_architecture') retourne les bons paramètres
    # pour build_reasoning_model, notamment les dimensions d'entrée.
    # Il faudra peut-être inspecter X pour obtenir ces dimensions si elles ne sont pas dans la config.
    
    # Exemple de récupération des dimensions si nécessaire (à adapter)
    # tech_input_shape = X['technical_input'].shape[1:] if isinstance(X, dict) and 'technical_input' in X else (X.shape[1],)
    # sentiment_input_dim = X['sentiment_input'].shape[1] if isinstance(X, dict) and 'sentiment_input' in X and X['sentiment_input'].ndim > 1 else 1
    # cryptobert_input_dim = X['cryptobert_input'].shape[1] if isinstance(X, dict) and 'cryptobert_input' in X and X['cryptobert_input'].ndim > 1 else 1
    # market_input_dim = X['market_input'].shape[1] if isinstance(X, dict) and 'market_input' in X and X['market_input'].ndim > 1 else 1
    # hmm_input_dim = X['hmm_input'].shape[1] if isinstance(X, dict) and 'hmm_input' in X and X['hmm_input'].ndim > 1 else 1
    # instrument_vocab_size = int(np.max(X['instrument_input']) + 1) if isinstance(X, dict) and 'instrument_input' in X else 10 

    model_params = cfg.get_config('model.reasoning_architecture')
    # Ici, il faudrait s'assurer que model_params contient toutes les clés attendues par build_reasoning_model
    # ou les ajouter dynamiquement à partir de X si nécessaire.
    # Pour l'instant, on suppose que la config est complète.
    # Passer toutes les dimensions nécessaires à build_reasoning_model
    model_params['tech_input_shape'] = (num_technical,) if num_technical > 0 else None
    model_params['mcp_input_dim'] = num_mcp
    model_params['cryptobert_input_dim'] = num_cryptobert
    model_params['hmm_input_dim'] = num_hmm
    model_params['sentiment_input_dim'] = num_sentiment
    model_params['market_input_dim'] = num_market
    model_params['instrument_vocab_size'] = instrument_vocab_size
    if 'llm_embedding_dim' in model_params: del model_params['llm_embedding_dim']

    # Récupérer les sorties actives pour la construction et la compilation
    active_outputs = cfg.get_config('model.active_outputs', ['market_regime', 'sl_tp', 'reasoning'])
    model_params['active_outputs'] = active_outputs

    # Récupérer le nombre de classes pour market_regime à partir du mapping
    market_regime_mapping = cfg.get_config('data.label_mappings.market_regime', {})
    if market_regime_mapping:
        num_market_regime_classes = max(market_regime_mapping.values()) + 1
    else:
        num_market_regime_classes = 2
        print("WARN: Mapping market_regime non trouvé dans la config, utilise num_classes=2 par défaut.")
    model_params['num_market_regime_classes'] = num_market_regime_classes

    model = build_reasoning_model(**model_params)

    # Récupérer le learning rate pour la compilation
    learning_rate_config = cfg.get_config('training.learning_rate', 0.001)
    try:
        learning_rate = float(learning_rate_config)
    except (ValueError, TypeError):
        print(f"WARN: Impossible de convertir learning_rate '{learning_rate_config}' en float. Utilisation de 0.001 par défaut.")
        learning_rate = 0.001
    compile_reasoning_model(model, learning_rate=learning_rate, active_outputs=active_outputs)

    # Callbacks
    ckpt_path = os.path.join(args.output_dir, 'best_model.h5')
    if not ckpt_path.endswith(('.h5', '.keras', '.weights.h5')): # TF >= 2.11
        ckpt_path += '.keras'

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True
    )
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, 'tb'))

    # Entraînement
    # Préparer les données y pour Keras
    y_train = {}
    for label_name, label_series in y.items():
        if label_name in model.output_names:
             y_train[label_name] = label_series.values
        else:
             print(f"WARN: Label '{label_name}' trouvé dans les données mais pas dans les sorties du modèle ({model.output_names}). Il sera ignoré.")

    if not any(key in y_train for key in model.output_names):
         raise ValueError(f"Aucun des labels chargés ({list(y.keys())}) ne correspond aux sorties attendues du modèle ({model.output_names}). Vérifiez 'data.label_columns' dans config.yaml.")

    history = model.fit(
        X, y_train, # Utiliser y_train (numpy arrays)
        validation_split=cfg.get_config('training.validation_split', 0.2), # Corrigé: training.validation_split
        epochs=cfg.get_config('training.epochs', 100),
        batch_size=cfg.get_config('training.batch_size', 64),
        callbacks=[ckpt, es, tb]
    )

    # Génération d'exemples d'explication
    # Assurez-vous que ExplanationDecoder est compatible avec le modèle et les données X
    # Récupérer les noms de features si possible (nécessaire pour ExplanationDecoder)
    # Note: data_loader ne retourne pas les noms de features pour l'instant.
    # Il faudrait modifier data_loader ou passer une liste statique si connue.
    # feature_names_list = cfg.get_config('model.feature_names', None) 
    # if feature_names_list is None:
    #     # Essayer de déduire de X (si X contient les noms ou si on peut les reconstruire)
    #     # Pour l'instant, on passe une liste vide ou None
    #     print("WARN: Noms de features non fournis à ExplanationDecoder.")
    #     feature_names_list = [] 
        
    decoder = ExplanationDecoder(feature_names=[]) # Passer une liste vide pour l'instant
    
    # Préparer les données pour l'explication (prendre les 5 premiers exemples de X)
    # Si X est un dictionnaire, il faut prendre les 5 premiers de chaque entrée.
    # Si X est un tenseur/array unique, c'est plus simple.
    # L'exemple original utilise X['technical_input'][:5]
    # Cela suppose que X est un dictionnaire et que 'technical_input' est la clé pertinente pour ExplanationDecoder.
    # Adapter si la structure de X est différente ou si ExplanationDecoder attend autre chose.
    
    data_for_explanation = None
    if isinstance(X, dict) and 'technical_input' in X:
        data_for_explanation = X['technical_input'][:5]
    elif not isinstance(X, dict): # Supposons que X est un array/tenseur
        data_for_explanation = X[:5]
    else:
        print("Avertissement: Impossible de déterminer les données pour l'explication. 'technical_input' non trouvé dans X ou X n'est pas un dictionnaire.")
        # Fournir des données factices ou gérer l'erreur
        # Pour l'instant, on passe None, ce qui causera probablement une erreur dans generate_chain_of_thought_explanation
        # si data_for_explanation est requis.
        
    # La génération d'explication nécessite des prédictions et potentiellement des données brutes.
    # generate_chain_of_thought_explanation attend (model, data), mais devrait plutôt attendre
    # les prédictions et les données brutes correspondantes.
    # Pour l'instant, nous commentons cette partie car elle nécessite une refonte.
    # if data_for_explanation is not None:
    #     # predictions = model.predict(data_for_explanation) # Obtenir les prédictions
    #     # examples = decoder.generate_chain_of_thought_explanation(market_data={}, predictions=predictions) # Adapter l'appel
    #     # with open(os.path.join(args.output_dir, 'explanations.json'), 'w') as f:
    #     #     json.dump(examples, f, indent=2)
    #     print("INFO: Génération d'explications commentée pour l'instant.")
    # else: # Commenté car le if correspondant est commenté
    #     print("Impossible de générer les explications car les données d'entrée n'ont pas pu être préparées.")

if __name__ == '__main__':
    main()
