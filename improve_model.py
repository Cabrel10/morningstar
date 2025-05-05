#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour orchestrer l'amu00e9lioration du modu00e8le Morningstar.
Ce script exu00e9cute toutes les u00e9tapes du processus d'amu00e9lioration :
1. Pru00e9paration des donnu00e9es avec normalisation robuste et u00e9quilibrage des classes
2. Optimisation des hyperparamu00e8tres avec un algorithme gu00e9nu00e9tique
3. Entrau00eenement du modu00e8le avec les hyperparamu00e8tres optimaux
4. u00c9valuation du modu00e8le et simulation de trading
"""

import os
import logging
import argparse
import subprocess
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """
    Exu00e9cute une commande shell et affiche les logs.
    
    Args:
        command: Commande u00e0 exu00e9cuter
        description: Description de la commande
    
    Returns:
        Code de retour de la commande
    """
    logger.info(f"Exu00e9cution de: {description}")
    logger.info(f"Commande: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Afficher les logs en temps ru00e9el
    for line in process.stdout:
        logger.info(line.strip())
    
    # Attendre la fin de l'exu00e9cution
    process.wait()
    
    # Afficher les erreurs s'il y en a
    if process.returncode != 0:
        for line in process.stderr:
            logger.error(line.strip())
        logger.error(f"Erreur lors de l'exu00e9cution de: {description}")
    else:
        logger.info(f"Exu00e9cution ru00e9ussie: {description}")
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Amu00e9liore le modu00e8le Morningstar avec des techniques avancu00e9es.')
    parser.add_argument('--input-data', type=str, default='/home/morningstar/Desktop/crypto_robot/Morningstar/data/processed/multi_crypto_dataset_prepared.parquet', help='Chemin vers le dataset d\'entru00e9e')
    parser.add_argument('--output-dir', type=str, default='/home/morningstar/Desktop/crypto_robot/Morningstar/model/improved', help='Ru00e9pertoire de sortie pour les ru00e9sultats')
    parser.add_argument('--skip-data-prep', action='store_true', help='Sauter l\'u00e9tape de pru00e9paration des donnu00e9es')
    parser.add_argument('--skip-hyperopt', action='store_true', help='Sauter l\'u00e9tape d\'optimisation des hyperparamu00e8tres')
    parser.add_argument('--skip-training', action='store_true', help='Sauter l\'u00e9tape d\'entrau00eenement')
    parser.add_argument('--skip-evaluation', action='store_true', help='Sauter l\'u00e9tape d\'u00e9valuation')
    parser.add_argument('--population-size', type=int, default=20, help='Taille de la population pour l\'algorithme gu00e9nu00e9tique')
    parser.add_argument('--generations', type=int, default=10, help='Nombre de gu00e9nu00e9rations pour l\'algorithme gu00e9nu00e9tique')
    
    args = parser.parse_args()
    
    # Cru00e9er les ru00e9pertoires nu00e9cessaires
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'evaluation'), exist_ok=True)
    
    # 1. Pru00e9paration des donnu00e9es
    normalized_data_path = os.path.join(Path(args.input_data).parent, 'normalized', Path(args.input_data).stem + '_normalized.csv')
    if not args.skip_data_prep:
        logger.info("u00c9tape 1: Pru00e9paration des donnu00e9es avec normalisation robuste et u00e9quilibrage des classes")
        command = f"python3 /home/morningstar/Desktop/crypto_robot/Morningstar/scripts/prepare_improved_dataset.py --input {args.input_data} --output {normalized_data_path}"
        if run_command(command, "Pru00e9paration des donnu00e9es") != 0:
            logger.error("Erreur lors de la pru00e9paration des donnu00e9es. Arru00eat du processus.")
            return
    else:
        logger.info("u00c9tape 1: Pru00e9paration des donnu00e9es sautu00e9e")
    
    # 2. Optimisation des hyperparamu00e8tres
    model_path = os.path.join(args.output_dir, 'morningstar_improved.h5')
    if not args.skip_hyperopt and not args.skip_training:
        logger.info("u00c9tape 2: Optimisation des hyperparamu00e8tres avec un algorithme gu00e9nu00e9tique")
        command = f"python3 /home/morningstar/Desktop/crypto_robot/Morningstar/model/training/genetic_optimizer.py --data-path {normalized_data_path} --output-dir {args.output_dir} --population-size {args.population_size} --generations {args.generations}"
        if run_command(command, "Optimisation des hyperparamu00e8tres") != 0:
            logger.error("Erreur lors de l'optimisation des hyperparamu00e8tres. Arru00eat du processus.")
            return
    else:
        logger.info("u00c9tape 2: Optimisation des hyperparamu00e8tres sautu00e9e")
    
    # 3. u00c9valuation du modu00e8le
    if not args.skip_evaluation and os.path.exists(model_path):
        logger.info("u00c9tape 3: u00c9valuation du modu00e8le et simulation de trading")
        scalers_path = os.path.join(Path(normalized_data_path).parent, 'scalers', 'scalers.npz')
        command = f"python3 /home/morningstar/Desktop/crypto_robot/Morningstar/improved_evaluate.py --data-path {normalized_data_path} --model-path {model_path} --scalers-path {scalers_path} --output-dir {os.path.join(args.output_dir, 'evaluation')}"
        if run_command(command, "u00c9valuation du modu00e8le") != 0:
            logger.error("Erreur lors de l'u00e9valuation du modu00e8le.")
    else:
        logger.info("u00c9tape 3: u00c9valuation du modu00e8le sautu00e9e")
    
    logger.info("Processus d'amu00e9lioration du modu00e8le terminu00e9")

if __name__ == "__main__":
    main()
