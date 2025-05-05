#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour vérifier la compatibilité du modèle avec le dataset enrichi.
Vérifie les dimensions d'entrée et de sortie et génère un rapport de compatibilité.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Importer nos modules personnalisés
from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model

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
    parser = argparse.ArgumentParser(description="Vérification de la compatibilité du modèle avec le dataset enrichi")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/enriched/enriched_dataset.parquet",
        help="Chemin vers le dataset enrichi"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Répertoire de sortie pour le rapport de compatibilité"
    )
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    Charge le dataset enrichi.
    
    Args:
        dataset_path: Chemin vers le dataset enrichi
    
    Returns:
        DataFrame avec le dataset enrichi
    """
    logger.info(f"Chargement du dataset depuis {dataset_path}")
    try:
        df = pd.read_parquet(dataset_path)
        logger.info(f"Dataset chargé avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        sys.exit(1)

def analyze_dataset_structure(df):
    """
    Analyse la structure du dataset pour déterminer les dimensions d'entrée.
    
    Args:
        df: DataFrame avec le dataset enrichi
    
    Returns:
        Dictionnaire avec les dimensions d'entrée
    """
    logger.info("Analyse de la structure du dataset")
    
    # Identifier les colonnes techniques (indicateurs techniques)
    technical_columns = [
        col for col in df.columns if col.startswith(('rsi_', 'macd_', 'bbands_', 'ema_', 'sma_', 'atr_', 'adx_')) or 
        col in ['open', 'high', 'low', 'close', 'volume', 'returns']
    ]
    
    # Identifier les colonnes CryptoBERT
    cryptobert_columns = [col for col in df.columns if col.startswith('cryptobert_dim_')]
    
    # Identifier les colonnes de sentiment
    sentiment_columns = [col for col in df.columns if col == 'sentiment_score']
    
    # Identifier les colonnes HMM
    hmm_columns = [col for col in df.columns if col == 'hmm_regime']
    
    # Identifier les colonnes de métriques de marché
    market_columns = [col for col in df.columns if col.startswith('market_')]
    
    # Identifier les colonnes globales
    global_columns = [col for col in df.columns if col.startswith('global_')]
    
    # Combiner les colonnes de métriques de marché et globales pour former MCP
    mcp_columns = market_columns + global_columns
    
    # Calculer les dimensions
    tech_dim = len(technical_columns)
    llm_dim = len(cryptobert_columns)  # CryptoBERT est utilisé comme entrée LLM
    mcp_dim = len(mcp_columns) + len(sentiment_columns)  # Inclure le sentiment dans MCP
    hmm_dim = len(hmm_columns)
    
    # Afficher les résultats
    logger.info(f"Nombre de colonnes techniques: {tech_dim}")
    logger.info(f"Nombre de colonnes CryptoBERT (LLM): {llm_dim}")
    logger.info(f"Nombre de colonnes MCP (métriques de marché + sentiment): {mcp_dim}")
    logger.info(f"Nombre de colonnes HMM: {hmm_dim}")
    
    return {
        'tech_dim': tech_dim,
        'llm_dim': llm_dim,
        'mcp_dim': mcp_dim,
        'hmm_dim': hmm_dim,
        'technical_columns': technical_columns,
        'cryptobert_columns': cryptobert_columns,
        'sentiment_columns': sentiment_columns,
        'hmm_columns': hmm_columns,
        'market_columns': market_columns,
        'global_columns': global_columns,
        'mcp_columns': mcp_columns
    }

def check_model_compatibility(dataset_dims):
    """
    Vérifie la compatibilité du modèle avec les dimensions du dataset.
    
    Args:
        dataset_dims: Dictionnaire avec les dimensions du dataset
    
    Returns:
        Dictionnaire avec les résultats de la vérification
    """
    logger.info("Vérification de la compatibilité du modèle")
    
    # Dimensions attendues par le modèle
    model_tech_dim = 38  # Valeur par défaut dans le modèle
    model_llm_dim = 768  # Valeur par défaut dans le modèle
    model_mcp_dim = 128  # Valeur par défaut dans le modèle
    model_hmm_dim = 4    # Valeur par défaut dans le modèle
    
    # Vérifier la compatibilité
    tech_compatible = dataset_dims['tech_dim'] == model_tech_dim
    llm_compatible = dataset_dims['llm_dim'] == model_llm_dim
    mcp_compatible = dataset_dims['mcp_dim'] <= model_mcp_dim  # MCP peut être plus petit
    hmm_compatible = dataset_dims['hmm_dim'] <= model_hmm_dim  # HMM peut être plus petit
    
    # Ajustements nécessaires
    tech_adjustment = None if tech_compatible else f"Ajuster tech_input_shape=({dataset_dims['tech_dim']},)"
    llm_adjustment = None if llm_compatible else f"Ajuster llm_embedding_dim={dataset_dims['llm_dim']}"
    mcp_adjustment = None if mcp_compatible else f"Ajuster mcp_input_dim={dataset_dims['mcp_dim']}"
    hmm_adjustment = None if hmm_compatible else f"Ajuster hmm_input_dim={dataset_dims['hmm_dim']}"
    
    # Résultats
    compatibility_results = {
        'tech_compatible': tech_compatible,
        'llm_compatible': llm_compatible,
        'mcp_compatible': mcp_compatible,
        'hmm_compatible': hmm_compatible,
        'overall_compatible': tech_compatible and llm_compatible and mcp_compatible and hmm_compatible,
        'tech_adjustment': tech_adjustment,
        'llm_adjustment': llm_adjustment,
        'mcp_adjustment': mcp_adjustment,
        'hmm_adjustment': hmm_adjustment
    }
    
    # Afficher les résultats
    logger.info(f"Compatibilité technique: {tech_compatible}")
    logger.info(f"Compatibilité LLM: {llm_compatible}")
    logger.info(f"Compatibilité MCP: {mcp_compatible}")
    logger.info(f"Compatibilité HMM: {hmm_compatible}")
    logger.info(f"Compatibilité globale: {compatibility_results['overall_compatible']}")
    
    if not compatibility_results['overall_compatible']:
        logger.warning("Le modèle n'est pas compatible avec le dataset. Ajustements nécessaires:")
        for adjustment in [tech_adjustment, llm_adjustment, mcp_adjustment, hmm_adjustment]:
            if adjustment:
                logger.warning(f"  - {adjustment}")
    else:
        logger.info("Le modèle est compatible avec le dataset. Aucun ajustement nécessaire.")
    
    return compatibility_results

def generate_compatibility_report(dataset_dims, compatibility_results, output_dir):
    """
    Génère un rapport de compatibilité.
    
    Args:
        dataset_dims: Dictionnaire avec les dimensions du dataset
        compatibility_results: Dictionnaire avec les résultats de la vérification
        output_dir: Répertoire de sortie pour le rapport
    """
    logger.info("Génération du rapport de compatibilité")
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Créer le rapport
    report = {
        'dataset_dimensions': {
            'tech_dim': dataset_dims['tech_dim'],
            'llm_dim': dataset_dims['llm_dim'],
            'mcp_dim': dataset_dims['mcp_dim'],
            'hmm_dim': dataset_dims['hmm_dim']
        },
        'model_expected_dimensions': {
            'tech_dim': 38,
            'llm_dim': 768,
            'mcp_dim': 128,
            'hmm_dim': 4
        },
        'compatibility': {
            'tech_compatible': compatibility_results['tech_compatible'],
            'llm_compatible': compatibility_results['llm_compatible'],
            'mcp_compatible': compatibility_results['mcp_compatible'],
            'hmm_compatible': compatibility_results['hmm_compatible'],
            'overall_compatible': compatibility_results['overall_compatible']
        },
        'adjustments_needed': {
            'tech_adjustment': compatibility_results['tech_adjustment'],
            'llm_adjustment': compatibility_results['llm_adjustment'],
            'mcp_adjustment': compatibility_results['mcp_adjustment'],
            'hmm_adjustment': compatibility_results['hmm_adjustment']
        },
        'column_groups': {
            'technical_columns': dataset_dims['technical_columns'],
            'cryptobert_columns': dataset_dims['cryptobert_columns'],
            'sentiment_columns': dataset_dims['sentiment_columns'],
            'hmm_columns': dataset_dims['hmm_columns'],
            'market_columns': dataset_dims['market_columns'],
            'global_columns': dataset_dims['global_columns'],
            'mcp_columns': dataset_dims['mcp_columns']
        }
    }
    
    # Sauvegarder le rapport
    report_file = output_path / "model_compatibility_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Rapport de compatibilité sauvegardé dans {report_file}")
    
    # Générer un exemple de code pour ajuster le modèle si nécessaire
    if not compatibility_results['overall_compatible']:
        # Créer un fichier d'exemple pour ajuster le modèle
        example_file = output_path / "model_adjustment_example.py"
        
        # Contenu du fichier d'exemple
        example_content = [
            "#!/usr/bin/env python",
            "# -*- coding: utf-8 -*-",
            "",
            "\"\"\"Exemple de code pour ajuster le modèle en fonction des dimensions du dataset.\"\"\"",
            "",
            "from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model",
            "",
            "# Construire le modèle avec les dimensions ajustées",
            "model = build_enhanced_hybrid_model("
        ]
        
        # Ajouter les ajustements nécessaires
        if compatibility_results['tech_adjustment']:
            example_content.append(f"    tech_input_shape=({dataset_dims['tech_dim']},),  # Ajustement pour les features techniques")
        if compatibility_results['llm_adjustment']:
            example_content.append(f"    llm_embedding_dim={dataset_dims['llm_dim']},  # Ajustement pour les embeddings LLM")
        if compatibility_results['mcp_adjustment']:
            example_content.append(f"    mcp_input_dim={dataset_dims['mcp_dim']},  # Ajustement pour les features MCP")
        if compatibility_results['hmm_adjustment']:
            example_content.append(f"    hmm_input_dim={dataset_dims['hmm_dim']},  # Ajustement pour les features HMM")
        
        # Ajouter les autres paramètres inchangés
        example_content.extend([
            "    # Autres paramètres inchangés",
            "    instrument_vocab_size=10,",
            "    instrument_embedding_dim=8,",
            "    num_trading_classes=5,",
            "    num_market_regime_classes=4,",
            "    num_volatility_quantiles=3,",
            "    num_sl_tp_outputs=2",
            ")",
            "",
            "# Compiler le modèle",
            "model.compile(",
            "    optimizer='adam',",
            "    loss={",
            "        'signal': 'categorical_crossentropy',",
            "        'market_regime': 'categorical_crossentropy',",
            "        'volatility_quantiles': 'mse',",
            "        'sl_tp': 'mse'",
            "    },",
            "    metrics={",
            "        'signal': ['accuracy'],",
            "        'market_regime': ['accuracy'],",
            "        'volatility_quantiles': ['mae'],",
            "        'sl_tp': ['mae']",
            "    }",
            ")",
            "",
            "# Afficher le résumé du modèle",
            "model.summary()"
        ])
        
        # Écrire le contenu dans le fichier
        with open(example_file, 'w') as f:
            f.write('\n'.join(example_content))
        
        logger.info(f"Exemple de code pour ajuster le modèle sauvegardé dans {example_file}")

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Charger le dataset
    df = load_dataset(args.dataset_path)
    
    # Analyser la structure du dataset
    dataset_dims = analyze_dataset_structure(df)
    
    # Vérifier la compatibilité du modèle
    compatibility_results = check_model_compatibility(dataset_dims)
    
    # Générer le rapport de compatibilité
    generate_compatibility_report(dataset_dims, compatibility_results, args.output_dir)
    
    # Afficher le résultat final
    if compatibility_results['overall_compatible']:
        logger.info("Le modèle est compatible avec le dataset. Aucun ajustement nécessaire.")
    else:
        logger.warning("Le modèle n'est pas compatible avec le dataset. Veuillez consulter le rapport pour les ajustements nécessaires.")

if __name__ == "__main__":
    main()
