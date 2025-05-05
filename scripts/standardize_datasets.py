#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour standardiser les datasets de plusieurs actifs et les préparer pour l'entraînement.
Utilise le module data_standardizer pour gérer la standardisation des données.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parents[1]))

# Importer nos modules personnalisés
from data_processors.data_standardizer import standardize_datasets
from config.config_manager import ConfigManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Standardisation des datasets pour Morningstar")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Répertoire contenant les datasets prétraités"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/standardized",
        help="Répertoire de sortie pour les données standardisées"
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Liste d'actifs à standardiser, séparés par des virgules (si non spécifié, utilise tous les actifs de config.yaml)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion de données pour le test"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion de données pour la validation"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour la reproductibilité"
    )
    parser.add_argument(
        "--stratify-col",
        type=str,
        default="market_regime",
        help="Colonne pour la stratification"
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Chemin vers un fichier JSON avec la configuration des features"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="standardized_multi_crypto_dataset",
        help="Nom du dataset standardisé"
    )
    return parser.parse_args()

def main():
    """Fonction principale."""
    # Parser les arguments
    args = parse_args()
    
    # Charger la configuration
    config = ConfigManager()
    
    # Créer les répertoires d'entrée et de sortie
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Déterminer les actifs à standardiser
    if args.assets:
        assets = args.assets.split(',')
    else:
        assets = config.get_training_assets()
    
    logger.info(f"Standardisation des datasets pour {len(assets)} actifs: {assets}")
    
    # Construire les chemins des fichiers d'entrée
    input_paths = []
    for asset in assets:
        # Si l'actif est déjà un chemin complet, l'utiliser directement
        if asset.endswith('.parquet'):
            asset_path = input_dir / asset
        else:
            # Sinon, construire le chemin avec le format standard
            asset_path = input_dir / f"{asset}_processed.parquet"
        
        # Vérifier si le fichier existe
        if asset_path.exists():
            input_paths.append(asset_path)
        else:
            logger.warning(f"Fichier non trouvé pour l'actif {asset}: {asset_path}")
    
    if not input_paths:
        logger.error("Aucun fichier d'entrée trouvé. Arrêt du script.")
        sys.exit(1)
    
    # Charger la configuration des features si spécifiée
    feature_config = {}
    if args.feature_config:
        feature_config_path = Path(args.feature_config)
        if feature_config_path.exists():
            with open(feature_config_path, 'r') as f:
                feature_config = json.load(f)
            logger.info(f"Configuration des features chargée depuis {feature_config_path}")
        else:
            logger.warning(f"Fichier de configuration des features non trouvé: {feature_config_path}")
    
    # Déterminer les colonnes de features et de labels
    # Si la configuration des features est spécifiée, l'utiliser
    if feature_config:
        feature_columns = feature_config.get("feature_columns", [])
        label_columns = feature_config.get("label_columns", [])
        technical_cols = feature_config.get("technical_cols", None)
        llm_cols = feature_config.get("llm_cols", None)
        mcp_cols = feature_config.get("mcp_cols", None)
        hmm_cols = feature_config.get("hmm_cols", None)
    else:
        # Sinon, détecter automatiquement à partir du premier fichier
        logger.info(f"Détection automatique des colonnes à partir de {input_paths[0]}")
        sample_df = pd.read_parquet(input_paths[0])
        
        # Colonnes de labels connues
        label_columns = ['signal', 'market_regime']
        sl_tp_cols = ['level_sl', 'level_tp']
        
        # Colonnes d'asset_id
        asset_id_col = 'asset_id'
        
        # Toutes les autres colonnes sont des features
        feature_columns = [col for col in sample_df.columns 
                          if col not in label_columns + sl_tp_cols + [asset_id_col]]
        
        # Détecter les colonnes techniques, LLM, MCP et HMM
        technical_cols = [col for col in feature_columns 
                         if col.startswith(('rsi_', 'macd_', 'bbands_', 'ema_', 'sma_', 'atr_', 'adx_')) or 
                         col in ['open', 'high', 'low', 'close', 'volume', 'returns']]
        
        llm_cols = [col for col in feature_columns if col.startswith('cryptobert_dim_')]
        mcp_cols = [col for col in feature_columns 
                   if col.startswith(('market_', 'global_', 'sentiment_'))]
        hmm_cols = [col for col in feature_columns if col.startswith('hmm_')]
        
        logger.info(f"Colonnes détectées:")
        logger.info(f"  - Features: {len(feature_columns)} colonnes")
        logger.info(f"  - Labels: {label_columns}")
        logger.info(f"  - Techniques: {len(technical_cols)} colonnes")
        logger.info(f"  - LLM: {len(llm_cols)} colonnes")
        logger.info(f"  - MCP: {len(mcp_cols)} colonnes")
        logger.info(f"  - HMM: {len(hmm_cols)} colonnes")
    
    # Standardiser les datasets
    success = standardize_datasets(
        input_paths=input_paths,
        output_dir=output_dir,
        feature_columns=feature_columns,
        label_columns=label_columns,
        technical_cols=technical_cols,
        llm_cols=llm_cols,
        mcp_cols=mcp_cols,
        hmm_cols=hmm_cols,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        stratify_col=args.stratify_col
    )
    
    if success:
        logger.info(f"Standardisation terminée avec succès! Données sauvegardées dans {output_dir}")
    else:
        logger.error("Erreur lors de la standardisation des datasets.")
        sys.exit(1)

if __name__ == "__main__":
    main()
