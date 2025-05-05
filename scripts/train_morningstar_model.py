#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement du modèle Morningstar.

Ce script permet d'entraîner le modèle Morningstar avec une architecture hybride
qui combine plusieurs sources de données et produit plusieurs sorties.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, 
    Concatenate, Attention, MultiHeadAttention
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, AdamW
from tensorflow.keras.regularizers import l2
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au chemin pour importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))

from app_modules.utils import (
    BASE_DIR, DATA_DIR, MODEL_DIR, REPORTS_DIR,
    load_dataset
)

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement du modèle Morningstar")
    
    # Arguments obligatoires
    parser.add_argument("--dataset", required=True, help="Chemin vers le dataset d'entraînement")
    parser.add_argument("--model-name", required=True, help="Nom du modèle à sauvegarder")
    
    # Paramètres d'architecture
    parser.add_argument("--tech-input-dim", type=int, default=21, help="Dimension des entrées techniques")
    parser.add_argument("--llm-embedding-dim", type=int, default=10, help="Dimension des embeddings LLM")
    parser.add_argument("--mcp-input-dim", type=int, default=2, help="Dimension des entrées MCP")
    parser.add_argument("--hmm-input-dim", type=int, default=1, help="Dimension des entrées HMM")
    parser.add_argument("--num-trading-classes", type=int, default=5, help="Nombre de classes de trading")
    parser.add_argument("--num-market-regime-classes", type=int, default=4, help="Nombre de classes de régime de marché")
    parser.add_argument("--num-volatility-quantiles", type=int, default=3, help="Nombre de quantiles de volatilité")
    parser.add_argument("--num-sl-tp-outputs", type=int, default=2, help="Nombre de sorties SL/TP")
    
    # Paramètres d'entraînement
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille du batch")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Taux d'apprentissage")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "rmsprop", "sgd", "adamw"], help="Optimiseur")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Proportion des données pour la validation")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Patience pour early stopping")
    
    # Paramètres avancés
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Taux de dropout")
    parser.add_argument("--l2-reg", type=float, default=0.001, help="Coefficient de régularisation L2")
    parser.add_argument("--use-batch-norm", action="store_true", help="Utiliser Batch Normalization")
    parser.add_argument("--num-reasoning-steps", type=int, default=3, help="Nombre d'étapes de raisonnement")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Nombre de têtes d'attention")
    parser.add_argument("--use-chain-of-thought", action="store_true", help="Activer Chain-of-Thought")
    
    # Options de sauvegarde
    parser.add_argument("--save-best-only", action="store_true", help="Sauvegarder uniquement le meilleur modèle")
    parser.add_argument("--save-weights-only", action="store_true", help="Sauvegarder uniquement les poids")
    parser.add_argument("--save-metadata", action="store_true", help="Sauvegarder les métadonnées")
    
    return parser.parse_args()

def create_morningstar_model(args):
    """
    Crée le modèle Morningstar avec l'architecture spécifiée.
    
    Args:
        args: Arguments de la ligne de commande
        
    Returns:
        model: Modèle Morningstar compilé
    """
    # Définir les entrées
    tech_input = Input(shape=(args.tech_input_dim,), name="technical_input")
    llm_input = Input(shape=(args.llm_embedding_dim,), name="llm_input")
    mcp_input = Input(shape=(args.mcp_input_dim,), name="mcp_input")
    hmm_input = Input(shape=(args.hmm_input_dim,), name="hmm_input")
    instrument_input = Input(shape=(1,), name="instrument_input")
    
    # Traitement des entrées techniques
    x_tech = Dense(64, activation="relu", kernel_regularizer=l2(args.l2_reg))(tech_input)
    if args.use_batch_norm:
        x_tech = BatchNormalization()(x_tech)
    x_tech = Dropout(args.dropout_rate)(x_tech)
    x_tech = Dense(32, activation="relu", kernel_regularizer=l2(args.l2_reg))(x_tech)
    
    # Traitement des embeddings LLM
    x_llm = Dense(32, activation="relu", kernel_regularizer=l2(args.l2_reg))(llm_input)
    if args.use_batch_norm:
        x_llm = BatchNormalization()(x_llm)
    x_llm = Dropout(args.dropout_rate)(x_llm)
    x_llm = Dense(16, activation="relu", kernel_regularizer=l2(args.l2_reg))(x_llm)
    
    # Traitement des entrées MCP
    x_mcp = Dense(16, activation="relu", kernel_regularizer=l2(args.l2_reg))(mcp_input)
    if args.use_batch_norm:
        x_mcp = BatchNormalization()(x_mcp)
    x_mcp = Dropout(args.dropout_rate)(x_mcp)
    x_mcp = Dense(8, activation="relu", kernel_regularizer=l2(args.l2_reg))(x_mcp)
    
    # Traitement des entrées HMM
    x_hmm = Dense(8, activation="relu", kernel_regularizer=l2(args.l2_reg))(hmm_input)
    if args.use_batch_norm:
        x_hmm = BatchNormalization()(x_hmm)
    x_hmm = Dropout(args.dropout_rate)(x_hmm)
    x_hmm = Dense(4, activation="relu", kernel_regularizer=l2(args.l2_reg))(x_hmm)
    
    # Traitement du type d'instrument
    x_instrument = Dense(4, activation="relu", kernel_regularizer=l2(args.l2_reg))(instrument_input)
    if args.use_batch_norm:
        x_instrument = BatchNormalization()(x_instrument)
    x_instrument = Dropout(args.dropout_rate)(x_instrument)
    
    # Fusion des entrées
    merged = Concatenate()([x_tech, x_llm, x_mcp, x_hmm, x_instrument])
    
    # Couches communes
    x = Dense(128, activation="relu", kernel_regularizer=l2(args.l2_reg))(merged)
    if args.use_batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(args.dropout_rate)(x)
    
    x = Dense(64, activation="relu", kernel_regularizer=l2(args.l2_reg))(x)
    if args.use_batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(args.dropout_rate)(x)
    
    # Mécanisme d'attention si demandé
    if args.use_chain_of_thought:
        # Utiliser MultiHeadAttention pour le mécanisme d'attention
        attention_output = MultiHeadAttention(
            num_heads=args.num_attention_heads,
            key_dim=16
        )(x, x)
        x = Concatenate()([x, attention_output])
        
        # Étapes de raisonnement Chain-of-Thought
        reasoning_outputs = []
        for i in range(args.num_reasoning_steps):
            reasoning_step = Dense(64, activation="relu", name=f"reasoning_step_{i+1}")(x)
            if args.use_batch_norm:
                reasoning_step = BatchNormalization()(reasoning_step)
            reasoning_step = Dropout(args.dropout_rate)(reasoning_step)
            reasoning_outputs.append(reasoning_step)
        
        # Combiner les étapes de raisonnement
        if reasoning_outputs:
            reasoning_concat = Concatenate()(reasoning_outputs)
            x = Concatenate()([x, reasoning_concat])
    
    # Couche finale commune
    x = Dense(32, activation="relu", kernel_regularizer=l2(args.l2_reg))(x)
    
    # Sorties multiples
    trading_output = Dense(args.num_trading_classes, activation="softmax", name="trading_signal")(x)
    regime_output = Dense(args.num_market_regime_classes, activation="softmax", name="market_regime")(x)
    volatility_output = Dense(args.num_volatility_quantiles, activation="sigmoid", name="volatility_quantiles")(x)
    sl_tp_output = Dense(args.num_sl_tp_outputs, activation="linear", name="sl_tp_values")(x)
    
    # Créer le modèle
    model = Model(
        inputs=[tech_input, llm_input, mcp_input, hmm_input, instrument_input],
        outputs=[trading_output, regime_output, volatility_output, sl_tp_output]
    )
    
    # Compiler le modèle avec l'optimiseur spécifié
    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == "rmsprop":
        optimizer = RMSprop(learning_rate=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = SGD(learning_rate=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = AdamW(learning_rate=args.learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss={
            "trading_signal": "categorical_crossentropy",
            "market_regime": "categorical_crossentropy",
            "volatility_quantiles": "binary_crossentropy",
            "sl_tp_values": "mse"
        },
        metrics={
            "trading_signal": ["accuracy"],
            "market_regime": ["accuracy"],
            "volatility_quantiles": ["accuracy"],
            "sl_tp_values": ["mae"]
        }
    )
    
    return model

def prepare_data(dataset_path, args):
    """
    Prépare les données pour l'entraînement.
    
    Args:
        dataset_path: Chemin vers le dataset
        args: Arguments de la ligne de commande
        
    Returns:
        X: Dictionnaire des entrées
        y: Dictionnaire des sorties
        feature_scaler: Scaler pour les features
        target_scaler: Scaler pour les cibles
    """
    # Charger le dataset
    df = load_dataset(dataset_path)
    
    # Vérifier si le dataset contient les colonnes nécessaires
    required_columns = {
        "technical": args.tech_input_dim,
        "llm_embedding": args.llm_embedding_dim,
        "mcp": args.mcp_input_dim,
        "hmm_regime": args.hmm_input_dim,
        "instrument_type": 1,
        "trading_signal": args.num_trading_classes,
        "market_regime": args.num_market_regime_classes,
        "volatility": args.num_volatility_quantiles,
        "sl_tp": args.num_sl_tp_outputs
    }
    
    # Créer un dictionnaire pour stocker les entrées
    X = {}
    y = {}
    
    # Extraire les colonnes techniques
    tech_cols = [col for col in df.columns if col.startswith("tech_")]
    if len(tech_cols) < args.tech_input_dim:
        print(f"Attention: Le nombre de colonnes techniques ({len(tech_cols)}) est inférieur à la dimension spécifiée ({args.tech_input_dim})")
        # Compléter avec des zéros si nécessaire
        for i in range(len(tech_cols), args.tech_input_dim):
            df[f"tech_{i}"] = 0
        tech_cols = [f"tech_{i}" for i in range(args.tech_input_dim)]
    elif len(tech_cols) > args.tech_input_dim:
        print(f"Attention: Le nombre de colonnes techniques ({len(tech_cols)}) est supérieur à la dimension spécifiée ({args.tech_input_dim})")
        tech_cols = tech_cols[:args.tech_input_dim]
    
    X["technical_input"] = df[tech_cols].values
    
    # Extraire les colonnes LLM
    llm_cols = [col for col in df.columns if col.startswith("llm_")]
    if len(llm_cols) < args.llm_embedding_dim:
        print(f"Attention: Le nombre de colonnes LLM ({len(llm_cols)}) est inférieur à la dimension spécifiée ({args.llm_embedding_dim})")
        # Compléter avec des zéros si nécessaire
        for i in range(len(llm_cols), args.llm_embedding_dim):
            df[f"llm_{i}"] = 0
        llm_cols = [f"llm_{i}" for i in range(args.llm_embedding_dim)]
    elif len(llm_cols) > args.llm_embedding_dim:
        print(f"Attention: Le nombre de colonnes LLM ({len(llm_cols)}) est supérieur à la dimension spécifiée ({args.llm_embedding_dim})")
        llm_cols = llm_cols[:args.llm_embedding_dim]
    
    X["llm_input"] = df[llm_cols].values
    
    # Extraire les colonnes MCP
    mcp_cols = [col for col in df.columns if col.startswith("mcp_")]
    if len(mcp_cols) < args.mcp_input_dim:
        print(f"Attention: Le nombre de colonnes MCP ({len(mcp_cols)}) est inférieur à la dimension spécifiée ({args.mcp_input_dim})")
        # Compléter avec des zéros si nécessaire
        for i in range(len(mcp_cols), args.mcp_input_dim):
            df[f"mcp_{i}"] = 0
        mcp_cols = [f"mcp_{i}" for i in range(args.mcp_input_dim)]
    elif len(mcp_cols) > args.mcp_input_dim:
        print(f"Attention: Le nombre de colonnes MCP ({len(mcp_cols)}) est supérieur à la dimension spécifiée ({args.mcp_input_dim})")
        mcp_cols = mcp_cols[:args.mcp_input_dim]
    
    X["mcp_input"] = df[mcp_cols].values
    
    # Extraire les colonnes HMM
    hmm_cols = [col for col in df.columns if col.startswith("hmm_")]
    if len(hmm_cols) < args.hmm_input_dim:
        print(f"Attention: Le nombre de colonnes HMM ({len(hmm_cols)}) est inférieur à la dimension spécifiée ({args.hmm_input_dim})")
        # Compléter avec des zéros si nécessaire
        for i in range(len(hmm_cols), args.hmm_input_dim):
            df[f"hmm_{i}"] = 0
        hmm_cols = [f"hmm_{i}" for i in range(args.hmm_input_dim)]
    elif len(hmm_cols) > args.hmm_input_dim:
        print(f"Attention: Le nombre de colonnes HMM ({len(hmm_cols)}) est supérieur à la dimension spécifiée ({args.hmm_input_dim})")
        hmm_cols = hmm_cols[:args.hmm_input_dim]
    
    X["hmm_input"] = df[hmm_cols].values
    
    # Extraire le type d'instrument
    if "instrument_type" in df.columns:
        X["instrument_input"] = df[["instrument_type"]].values
    else:
        print("Attention: La colonne 'instrument_type' n'existe pas, utilisation de valeurs par défaut (0)")
        X["instrument_input"] = np.zeros((len(df), 1))
    
    # Extraire les sorties
    # Signal de trading (one-hot encoding)
    if "trading_signal" in df.columns:
        trading_signal = pd.get_dummies(df["trading_signal"], prefix="signal")
        if trading_signal.shape[1] != args.num_trading_classes:
            print(f"Attention: Le nombre de classes de trading ({trading_signal.shape[1]}) ne correspond pas à la valeur spécifiée ({args.num_trading_classes})")
            # Ajuster le nombre de classes si nécessaire
            if trading_signal.shape[1] < args.num_trading_classes:
                for i in range(trading_signal.shape[1], args.num_trading_classes):
                    trading_signal[f"signal_{i}"] = 0
            else:
                trading_signal = trading_signal.iloc[:, :args.num_trading_classes]
        y["trading_signal"] = trading_signal.values
    else:
        print("Attention: La colonne 'trading_signal' n'existe pas, utilisation de valeurs aléatoires")
        y["trading_signal"] = np.random.rand(len(df), args.num_trading_classes)
        y["trading_signal"] = y["trading_signal"] / y["trading_signal"].sum(axis=1, keepdims=True)
    
    # Régime de marché (one-hot encoding)
    if "market_regime" in df.columns:
        market_regime = pd.get_dummies(df["market_regime"], prefix="regime")
        if market_regime.shape[1] != args.num_market_regime_classes:
            print(f"Attention: Le nombre de classes de régime ({market_regime.shape[1]}) ne correspond pas à la valeur spécifiée ({args.num_market_regime_classes})")
            # Ajuster le nombre de classes si nécessaire
            if market_regime.shape[1] < args.num_market_regime_classes:
                for i in range(market_regime.shape[1], args.num_market_regime_classes):
                    market_regime[f"regime_{i}"] = 0
            else:
                market_regime = market_regime.iloc[:, :args.num_market_regime_classes]
        y["market_regime"] = market_regime.values
    else:
        print("Attention: La colonne 'market_regime' n'existe pas, utilisation de valeurs aléatoires")
        y["market_regime"] = np.random.rand(len(df), args.num_market_regime_classes)
        y["market_regime"] = y["market_regime"] / y["market_regime"].sum(axis=1, keepdims=True)
    
    # Quantiles de volatilité
    volatility_cols = [col for col in df.columns if col.startswith("volatility_")]
    if len(volatility_cols) == args.num_volatility_quantiles:
        y["volatility_quantiles"] = df[volatility_cols].values
    else:
        print(f"Attention: Le nombre de colonnes de volatilité ({len(volatility_cols)}) ne correspond pas à la valeur spécifiée ({args.num_volatility_quantiles})")
        # Créer des colonnes de volatilité aléatoires
        y["volatility_quantiles"] = np.random.rand(len(df), args.num_volatility_quantiles)
    
    # Stop Loss et Take Profit
    sl_tp_cols = [col for col in df.columns if col.startswith("sl_") or col.startswith("tp_")]
    if len(sl_tp_cols) == args.num_sl_tp_outputs:
        y["sl_tp_values"] = df[sl_tp_cols].values
    else:
        print(f"Attention: Le nombre de colonnes SL/TP ({len(sl_tp_cols)}) ne correspond pas à la valeur spécifiée ({args.num_sl_tp_outputs})")
        # Créer des colonnes SL/TP aléatoires
        y["sl_tp_values"] = np.random.rand(len(df), args.num_sl_tp_outputs) * 0.1  # Valeurs entre 0 et 0.1
    
    # Créer des scalers pour les features et les cibles
    feature_scaler = {
        "technical_input": {"mean": X["technical_input"].mean(axis=0), "std": X["technical_input"].std(axis=0) + 1e-8},
        "llm_input": {"mean": X["llm_input"].mean(axis=0), "std": X["llm_input"].std(axis=0) + 1e-8},
        "mcp_input": {"mean": X["mcp_input"].mean(axis=0), "std": X["mcp_input"].std(axis=0) + 1e-8},
        "hmm_input": {"mean": X["hmm_input"].mean(axis=0), "std": X["hmm_input"].std(axis=0) + 1e-8},
        "instrument_input": {"mean": X["instrument_input"].mean(axis=0), "std": X["instrument_input"].std(axis=0) + 1e-8}
    }
    
    target_scaler = {
        "sl_tp_values": {"mean": y["sl_tp_values"].mean(axis=0), "std": y["sl_tp_values"].std(axis=0) + 1e-8}
    }
    
    # Normaliser les entrées
    for key in X:
        if key != "instrument_input":  # Ne pas normaliser le type d'instrument
            X[key] = (X[key] - feature_scaler[key]["mean"]) / feature_scaler[key]["std"]
    
    # Normaliser les sorties SL/TP
    y["sl_tp_values"] = (y["sl_tp_values"] - target_scaler["sl_tp_values"]["mean"]) / target_scaler["sl_tp_values"]["std"]
    
    return X, y, feature_scaler, target_scaler

def main():
    """Fonction principale."""
    args = parse_args()
    
    # Créer les répertoires de sortie s'ils n'existent pas
    model_dir = MODEL_DIR / "trained" / "morningstar"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = MODEL_DIR / "logs" / args.model_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Préparer les données
    print(f"Préparation des données à partir du dataset {args.dataset}...")
    X, y, feature_scaler, target_scaler = prepare_data(args.dataset, args)
    
    # Créer le modèle
    print("Création du modèle Morningstar...")
    model = create_morningstar_model(args)
    
    # Afficher le résumé du modèle
    model.summary()
    
    # Définir les callbacks
    callbacks = []
    
    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True
        )
    )
    
    # Model checkpoint
    if args.save_best_only:
        callbacks.append(
            ModelCheckpoint(
                filepath=str(model_dir / f"{args.model_name}_best.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=args.save_weights_only
            )
        )
    
    # Reduce LR on plateau
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.early_stopping_patience // 2,
            min_lr=1e-6
        )
    )
    
    # TensorBoard
    callbacks.append(
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1
        )
    )
    
    # Entraîner le modèle
    print(f"Entraînement du modèle Morningstar pour {args.epochs} époques...")
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modèle final
    model_path = model_dir / f"{args.model_name}.h5"
    if args.save_weights_only:
        model.save_weights(str(model_path))
        print(f"Poids du modèle sauvegardés dans {model_path}")
    else:
        model.save(str(model_path))
        print(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarder les métadonnées si demandé
    if args.save_metadata:
        metadata = {
            "model_name": args.model_name,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": {
                "tech_input_dim": args.tech_input_dim,
                "llm_embedding_dim": args.llm_embedding_dim,
                "mcp_input_dim": args.mcp_input_dim,
                "hmm_input_dim": args.hmm_input_dim,
                "num_trading_classes": args.num_trading_classes,
                "num_market_regime_classes": args.num_market_regime_classes,
                "num_volatility_quantiles": args.num_volatility_quantiles,
                "num_sl_tp_outputs": args.num_sl_tp_outputs,
                "use_chain_of_thought": args.use_chain_of_thought,
                "num_reasoning_steps": args.num_reasoning_steps,
                "num_attention_heads": args.num_attention_heads
            },
            "training": {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "validation_split": args.validation_split,
                "early_stopping_patience": args.early_stopping_patience,
                "dropout_rate": args.dropout_rate,
                "l2_reg": args.l2_reg,
                "use_batch_norm": args.use_batch_norm
            },
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "history": {
                key: [float(val) for val in history.history[key]]
                for key in history.history.keys()
            }
        }
        
        metadata_path = model_dir / f"{args.model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Métadonnées sauvegardées dans {metadata_path}")
    
    print("Entraînement terminé avec succès!")
    
    # Afficher les métriques finales
    val_loss = history.history["val_loss"][-1]
    print(f"Perte finale sur la validation: {val_loss:.4f}")
    
    for output in ["trading_signal", "market_regime", "volatility_quantiles", "sl_tp_values"]:
        if f"val_{output}_accuracy" in history.history:
            val_acc = history.history[f"val_{output}_accuracy"][-1]
            print(f"Précision finale de {output} sur la validation: {val_acc:.4f}")
        elif f"val_{output}_mae" in history.history:
            val_mae = history.history[f"val_{output}_mae"][-1]
            print(f"MAE finale de {output} sur la validation: {val_mae:.4f}")

if __name__ == "__main__":
    main()
