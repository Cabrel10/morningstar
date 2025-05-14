#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement pour le modèle monolithique Morningstar.

Ce script permet de:
1. Charger les données prétraitées
2. Préparer les features et les labels
3. Créer et configurer le modèle monolithique
4. Entraîner le modèle
5. Évaluer les performances
6. Sauvegarder le modèle et les métriques
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any, Union

# Ajouter le chemin du projet au PYTHONPATH
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Ajustez selon votre structure
sys.path.append(str(PROJECT_ROOT))

# Import du modèle monolithique
from monolith_model import MonolithModel

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_monolith")


def load_data(data_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Parquet.
    
    Args:
        data_path: Chemin vers le fichier de données
        
    Returns:
        DataFrame des données chargées
    """
    logger.info(f"Chargement des données depuis {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    logger.info(f"Données chargées: {df.shape[0]} échantillons, {df.shape[1]} colonnes")
    return df


def prepare_features(
    df: pd.DataFrame,
    tech_feature_cols: List[str],
    embedding_cols: List[str],
    mcp_cols: List[str],
    instrument_col: str,
    target_cols: List[str],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    sequence_length: Optional[int] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Prépare les features et les labels pour l'entraînement, la validation et le test.
    
    Args:
        df: DataFrame avec les données
        tech_feature_cols: Liste des colonnes de features techniques
        embedding_cols: Liste des colonnes d'embeddings
        mcp_cols: Liste des colonnes MCP
        instrument_col: Nom de la colonne d'instrument
        target_cols: Liste des colonnes cibles
        test_size: Proportion de données pour le test
        validation_size: Proportion de données pour la validation
        sequence_length: Longueur de séquence pour les données (optionnel)
        
    Returns:
        Dictionnaire contenant les jeux d'entraînement, validation et test
    """
    logger.info("Préparation des features et labels")
    
    # Encoder la colonne d'instrument en entiers
    instruments = df[instrument_col].unique()
    instrument_map = {inst: i for i, inst in enumerate(instruments)}
    df['instrument_encoded'] = df[instrument_col].map(instrument_map)
    
    # Extraire les features
    X_tech = df[tech_feature_cols].values
    X_emb = df[embedding_cols].values if embedding_cols else np.zeros((len(df), 1))
    X_mcp = df[mcp_cols].values if mcp_cols else np.zeros((len(df), 1))
    X_inst = df['instrument_encoded'].values.reshape(-1, 1)
    
    # Extraire les targets
    if len(target_cols) >= 3:
        # Cas: signal (3 classes) + sl/tp (2 valeurs)
        signal_cols = target_cols[:3]  # {sell, neutral, buy}
        sltp_cols = target_cols[3:5] if len(target_cols) >= 5 else []
        
        y_signal = df[signal_cols].values
        y_sltp = df[sltp_cols].values if sltp_cols else None
    else:
        # Cas: signal binaire
        y_signal = df[target_cols].values
        y_sltp = None
    
    # Séparation train/val/test (temporal split)
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * validation_size)
    
    # Indices pour la séparation (en préservant l'ordre temporel)
    test_idx = slice(-n_test, None)
    val_idx = slice(-n_test-n_val, -n_test)
    train_idx = slice(None, -n_test-n_val)
    
    # Fonction pour créer des séquences si nécessaire
    def create_sequences(data, seq_length):
        if seq_length is None:
            return data
        
        n = len(data)
        seq_data = []
        for i in range(n - seq_length + 1):
            seq_data.append(data[i:i+seq_length])
        return np.array(seq_data)
    
    # Préparer les datasets
    if sequence_length is not None:
        # Cas séquentiel
        X_tech_seq = create_sequences(X_tech, sequence_length)
        X_emb_seq = X_emb[sequence_length-1:]
        X_mcp_seq = X_mcp[sequence_length-1:]
        X_inst_seq = X_inst[sequence_length-1:]
        y_signal_seq = y_signal[sequence_length-1:]
        y_sltp_seq = y_sltp[sequence_length-1:] if y_sltp is not None else None
        
        # Ajuster les indices
        n_seq = len(X_tech_seq)
        n_test_seq = int(n_seq * test_size)
        n_val_seq = int(n_seq * validation_size)
        
        test_idx_seq = slice(-n_test_seq, None)
        val_idx_seq = slice(-n_test_seq-n_val_seq, -n_test_seq)
        train_idx_seq = slice(None, -n_test_seq-n_val_seq)
        
        # Appliquer les indices
        train_data = {
            "technical_input": X_tech_seq[train_idx_seq],
            "embeddings_input": X_emb_seq[train_idx_seq],
            "mcp_input": X_mcp_seq[train_idx_seq],
            "instrument_input": X_inst_seq[train_idx_seq]
        }
        
        val_data = {
            "technical_input": X_tech_seq[val_idx_seq],
            "embeddings_input": X_emb_seq[val_idx_seq],
            "mcp_input": X_mcp_seq[val_idx_seq],
            "instrument_input": X_inst_seq[val_idx_seq]
        }
        
        test_data = {
            "technical_input": X_tech_seq[test_idx_seq],
            "embeddings_input": X_emb_seq[test_idx_seq],
            "mcp_input": X_mcp_seq[test_idx_seq],
            "instrument_input": X_inst_seq[test_idx_seq]
        }
        
        train_targets = {"signal_output": y_signal_seq[train_idx_seq]}
        val_targets = {"signal_output": y_signal_seq[val_idx_seq]}
        test_targets = {"signal_output": y_signal_seq[test_idx_seq]}
        
        if y_sltp_seq is not None:
            train_targets["sl_tp_output"] = y_sltp_seq[train_idx_seq]
            val_targets["sl_tp_output"] = y_sltp_seq[val_idx_seq]
            test_targets["sl_tp_output"] = y_sltp_seq[test_idx_seq]
    else:
        # Cas non-séquentiel
        train_data = {
            "technical_input": X_tech[train_idx],
            "embeddings_input": X_emb[train_idx],
            "mcp_input": X_mcp[train_idx],
            "instrument_input": X_inst[train_idx]
        }
        
        val_data = {
            "technical_input": X_tech[val_idx],
            "embeddings_input": X_emb[val_idx],
            "mcp_input": X_mcp[val_idx],
            "instrument_input": X_inst[val_idx]
        }
        
        test_data = {
            "technical_input": X_tech[test_idx],
            "embeddings_input": X_emb[test_idx],
            "mcp_input": X_mcp[test_idx],
            "instrument_input": X_inst[test_idx]
        }
        
        train_targets = {"signal_output": y_signal[train_idx]}
        val_targets = {"signal_output": y_signal[val_idx]}
        test_targets = {"signal_output": y_signal[test_idx]}
        
        if y_sltp is not None:
            train_targets["sl_tp_output"] = y_sltp[train_idx]
            val_targets["sl_tp_output"] = y_sltp[val_idx]
            test_targets["sl_tp_output"] = y_sltp[test_idx]
    
    logger.info(f"Préparation terminée: {len(train_data['technical_input'])} échantillons d'entraînement, " 
                f"{len(val_data['technical_input'])} échantillons de validation, "
                f"{len(test_data['technical_input'])} échantillons de test")
    
    return {
        "train": {"inputs": train_data, "targets": train_targets},
        "val": {"inputs": val_data, "targets": val_targets},
        "test": {"inputs": test_data, "targets": test_targets},
        "metadata": {
            "instrument_map": instrument_map,
            "sequence_length": sequence_length,
            "tech_feature_cols": tech_feature_cols,
            "embedding_cols": embedding_cols,
            "mcp_cols": mcp_cols,
            "target_cols": target_cols
        }
    }


def train_monolith_model(
    data_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    model_config: Optional[Dict[str, Any]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    output_dir: str = "./model_output"
) -> Tuple[MonolithModel, Dict[str, Any]]:
    """
    Entraîne le modèle monolithique.
    
    Args:
        data_dict: Dictionnaire des données d'entraînement, validation et test
        model_config: Configuration du modèle (optionnel)
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille du batch
        patience: Patience pour early stopping
        output_dir: Répertoire de sortie pour les résultats
        
    Returns:
        Modèle entraîné et historique d'entraînement
    """
    logger.info("Configuration et entraînement du modèle monolithique")
    
    # Extraire les données
    train_data = data_dict["train"]["inputs"]
    train_targets = data_dict["train"]["targets"]
    val_data = data_dict["val"]["inputs"]
    val_targets = data_dict["val"]["targets"]
    metadata = data_dict["metadata"]
    
    # Inférer les dimensions d'entrée
    if model_config is None:
        model_config = {}
    
    # Configurer les dimensions d'entrée à partir des données
    tech_input_shape = train_data["technical_input"].shape[1:]
    if len(tech_input_shape) == 2:  # Cas séquentiel
        model_config["sequence_length"] = tech_input_shape[0]
        model_config["tech_input_shape"] = (tech_input_shape[1],)
    else:
        model_config["tech_input_shape"] = tech_input_shape
    
    model_config["embeddings_input_shape"] = train_data["embeddings_input"].shape[1]
    model_config["mcp_input_shape"] = train_data["mcp_input"].shape[1]
    model_config["instrument_vocab_size"] = len(metadata["instrument_map"])
    
    # Configurer les sorties
    active_outputs = []
    if "signal_output" in train_targets:
        active_outputs.append("signal")
        model_config["head_config"] = model_config.get("head_config", {})
        model_config["head_config"]["signal"] = model_config["head_config"].get("signal", {})
        model_config["head_config"]["signal"]["classes"] = train_targets["signal_output"].shape[1]
    
    if "sl_tp_output" in train_targets:
        active_outputs.extend(["sl", "tp"])
        model_config["head_config"] = model_config.get("head_config", {})
        model_config["head_config"]["sl_tp"] = model_config["head_config"].get("sl_tp", {})
        model_config["head_config"]["sl_tp"]["outputs"] = train_targets["sl_tp_output"].shape[1]
    
    model_config["active_outputs"] = active_outputs
    
    # Créer le modèle
    model = MonolithModel(config=model_config)
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv')
        )
    ]
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Enregistrer la configuration du modèle
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Enregistrer les métadonnées
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        # Convertir les listes numpy en listes Python pour la sérialisation
        serializable_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                serializable_metadata[k] = {str(key): int(val) for key, val in v.items()}
            elif isinstance(v, (list, np.ndarray)):
                serializable_metadata[k] = list(v)
            else:
                serializable_metadata[k] = v
        json.dump(serializable_metadata, f, indent=2)
    
    # Entraîner le modèle
    logger.info(f"Début de l'entraînement sur {len(train_data['technical_input'])} échantillons")
    history = model.model.fit(
        train_data,
        train_targets,
        validation_data=(val_data, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modèle
    model_path = os.path.join(output_dir, 'monolith_model.keras')
    model.save(model_path)
    logger.info(f"Modèle sauvegardé à {model_path}")
    
    # Évaluer sur le jeu de test
    test_data = data_dict["test"]["inputs"]
    test_targets = data_dict["test"]["targets"]
    
    test_results = model.model.evaluate(test_data, test_targets, verbose=1)
    test_metrics = {}
    
    # Associer les métriques aux noms de sorties
    metrics_per_output = {}
    for i, metric_name in enumerate(model.model.metrics_names):
        if '_' in metric_name:
            output_name, metric_type = metric_name.split('_', 1)
            if output_name not in metrics_per_output:
                metrics_per_output[output_name] = {}
            metrics_per_output[output_name][metric_type] = test_results[i]
        else:
            test_metrics[metric_name] = test_results[i]
    
    test_metrics["per_output"] = metrics_per_output
    
    # Sauvegarder les métriques de test
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Métriques de test: {test_metrics}")
    
    return model, {
        "history": history.history,
        "test_metrics": test_metrics,
        "model_config": model_config
    }


def main():
    """Fonction principale d'entraînement."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîne le modèle monolithique Morningstar")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers le fichier de données (parquet ou csv)")
    parser.add_argument("--output-dir", type=str, default="./model_output", help="Répertoire de sortie")
    parser.add_argument("--config", type=str, help="Chemin vers le fichier de configuration JSON (optionnel)")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument("--patience", type=int, default=10, help="Patience pour early stopping")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion de données pour le test")
    parser.add_argument("--val-size", type=float, default=0.1, help="Proportion de données pour la validation")
    parser.add_argument("--seq-length", type=int, help="Longueur de séquence (optionnel)")
    
    args = parser.parse_args()
    
    # Charger la configuration si fournie
    model_config = None
    if args.config:
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    # Charger les données
    df = load_data(args.data)
    
    # Pour cet exemple, nous allons supposer un schéma de colonnes typique
    # Dans une utilisation réelle, cela devrait être configuré via un fichier de config
    
    # Identifier les colonnes par type
    all_cols = df.columns.tolist()
    target_cols = [col for col in all_cols if col.startswith('target_')]
    tech_cols = [col for col in all_cols if col.startswith('tech_') or col in ['open', 'high', 'low', 'close', 'volume']]
    embedding_cols = [col for col in all_cols if col.startswith('embedding_')]
    mcp_cols = [col for col in all_cols if col.startswith('mcp_')]
    
    # Utiliser 'instrument' comme colonne d'instrument si présente, sinon créer une colonne fictive
    if 'instrument' in all_cols:
        instrument_col = 'instrument'
    else:
        # Ajouter une colonne fictive si nécessaire
        df['instrument'] = 'default_instrument'
        instrument_col = 'instrument'
    
    # Préparer les données
    data_dict = prepare_features(
        df=df,
        tech_feature_cols=tech_cols,
        embedding_cols=embedding_cols,
        mcp_cols=mcp_cols,
        instrument_col=instrument_col,
        target_cols=target_cols,
        test_size=args.test_size,
        validation_size=args.val_size,
        sequence_length=args.seq_length
    )
    
    # Entraîner le modèle
    trained_model, results = train_monolith_model(
        data_dict=data_dict,
        model_config=model_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    logger.info("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main() 