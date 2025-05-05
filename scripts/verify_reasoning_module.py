#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour vu00e9rifier et tester le module de raisonnement du modu00e8le Morningstar.
Ce script analyse le module de raisonnement et gu00e9nu00e8re des exemples d'explications.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import importlib.util
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins des ru00e9pertoires
BASE_DIR = Path('/home/morningstar/Desktop/crypto_robot/Morningstar')
MODEL_DIR = BASE_DIR / 'model'
REASONING_DIR = MODEL_DIR / 'reasoning_model'
STANDARDIZED_DIR = BASE_DIR / 'data' / 'standardized'
REPORTS_DIR = BASE_DIR / 'reports' / 'reasoning'

# Cru00e9er les ru00e9pertoires s'ils n'existent pas
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_reasoning_module():
    """
    Charge dynamiquement le module de raisonnement
    """
    logger.info("Chargement du module de raisonnement...")
    
    reasoning_module_path = MODEL_DIR / "architecture" / "reasoning_model.py"
    reasoning_module_path_alt = MODEL_DIR / "reasoning" / "reasoning_module.py"
    
    if reasoning_module_path.exists():
        module_path = reasoning_module_path
    elif reasoning_module_path_alt.exists():
        module_path = reasoning_module_path_alt
    else:
        logger.error("Module de raisonnement non trouvu00e9")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("reasoning_module", module_path)
        reasoning_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reasoning_module)
        logger.info(f"Module de raisonnement chargu00e9 depuis {module_path}")
        return reasoning_module
    except Exception as e:
        logger.error(f"Erreur lors du chargement du module de raisonnement: {e}")
        return None

def load_model():
    """
    Charge le modu00e8le standardisu00e9
    """
    logger.info("Chargement du modu00e8le standardisu00e9...")
    
    # Chercher le modu00e8le standardisu00e9
    standardized_model_dir = MODEL_DIR / "standardized"
    
    if not standardized_model_dir.exists():
        logger.warning(f"Ru00e9pertoire {standardized_model_dir} non trouvu00e9")
        
        # Chercher d'autres modu00e8les
        model_files = []
        for ext in ['.h5', '/saved_model.pb']:
            model_files.extend(list(MODEL_DIR.glob(f"**/*{ext}")))
        
        if not model_files:
            logger.error("Aucun modu00e8le trouvu00e9")
            return None
        
        # Utiliser le modu00e8le le plus ru00e9cent
        latest_model = max(model_files, key=os.path.getmtime)
        logger.info(f"Utilisation du modu00e8le le plus ru00e9cent: {latest_model}")
        
        try:
            if latest_model.name.endswith('.h5'):
                model = tf.keras.models.load_model(latest_model)
            else:  # SavedModel
                model = tf.keras.models.load_model(latest_model.parent)
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modu00e8le: {e}")
            return None
    else:
        # Chercher dans le ru00e9pertoire standardisu00e9
        model_files = list(standardized_model_dir.glob("*.h5")) + list(standardized_model_dir.glob("*/saved_model.pb"))
        
        if not model_files:
            logger.error(f"Aucun modu00e8le trouvu00e9 dans {standardized_model_dir}")
            return None
        
        latest_model = max(model_files, key=os.path.getmtime)
        
        try:
            if latest_model.name.endswith('.h5'):
                model = tf.keras.models.load_model(latest_model)
            else:  # SavedModel
                model = tf.keras.models.load_model(latest_model.parent)
            logger.info(f"Modu00e8le chargu00e9 depuis {latest_model}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modu00e8le: {e}")
            return None

def load_sample_data():
    """
    Charge un u00e9chantillon de donnu00e9es pour tester le module de raisonnement
    """
    logger.info("Chargement d'un u00e9chantillon de donnu00e9es...")
    
    # Chercher les datasets standardisu00e9s
    if STANDARDIZED_DIR.exists():
        dataset_files = list(STANDARDIZED_DIR.glob("*.parquet"))
        
        if dataset_files:
            latest_dataset = max(dataset_files, key=os.path.getmtime)
            try:
                df = pd.read_parquet(latest_dataset)
                logger.info(f"Dataset chargu00e9 depuis {latest_dataset}: {len(df)} lignes")
                # Prendre un u00e9chantillon de 10 lignes
                sample_df = df.sample(min(10, len(df)))
                return sample_df
            except Exception as e:
                logger.error(f"Erreur lors du chargement du dataset: {e}")
    
    # Si aucun dataset standardisu00e9 n'est disponible, chercher dans d'autres ru00e9pertoires
    data_dirs = [Path(BASE_DIR) / 'data' / 'processed', Path(BASE_DIR) / 'data' / 'raw', Path(BASE_DIR) / 'data' / 'enriched']
    
    for data_dir in data_dirs:
        if data_dir.exists():
            dataset_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
            
            if dataset_files:
                latest_dataset = max(dataset_files, key=os.path.getmtime)
                try:
                    if latest_dataset.suffix == '.parquet':
                        df = pd.read_parquet(latest_dataset)
                    else:  # .csv
                        df = pd.read_csv(latest_dataset)
                    
                    logger.info(f"Dataset chargu00e9 depuis {latest_dataset}: {len(df)} lignes")
                    # Prendre un u00e9chantillon de 10 lignes
                    sample_df = df.sample(min(10, len(df)))
                    return sample_df
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du dataset: {e}")
    
    logger.warning("Aucun dataset trouvu00e9, gu00e9nu00e9ration de donnu00e9es synthu00e9tiques")
    
    # Gu00e9nu00e9rer des donnu00e9es synthu00e9tiques
    synthetic_data = {
        'open': np.random.uniform(30000, 40000, 10),
        'high': np.random.uniform(30000, 40000, 10),
        'low': np.random.uniform(30000, 40000, 10),
        'close': np.random.uniform(30000, 40000, 10),
        'volume': np.random.uniform(1000, 5000, 10),
        'symbol': ['BTC/USDT'] * 10,
        'RSI': np.random.uniform(30, 70, 10),
        'MACD': np.random.uniform(-100, 100, 10),
        'hmm_regime': np.random.choice([0, 1, 2], 10),
    }
    
    return pd.DataFrame(synthetic_data)

def test_reasoning_module(reasoning_module, model, sample_data):
    """
    Teste le module de raisonnement avec un u00e9chantillon de donnu00e9es
    """
    logger.info("Test du module de raisonnement...")
    
    if reasoning_module is None or model is None or sample_data is None:
        logger.error("Impossible de tester le module de raisonnement: composants manquants")
        return None
    
    try:
        # Vu00e9rifier si le module de raisonnement a une classe ReasoningModule
        if hasattr(reasoning_module, 'ReasoningModule'):
            reasoning = reasoning_module.ReasoningModule(model)
            
            # Gu00e9nu00e9rer des explications pour chaque u00e9chantillon
            explanations = []
            
            for idx, row in sample_data.iterrows():
                try:
                    # Convertir la ligne en format attendu par le module de raisonnement
                    input_data = row.to_dict()
                    
                    # Gu00e9nu00e9rer une explication
                    explanation = reasoning.generate_explanation(input_data)
                    explanations.append({
                        'sample_idx': idx,
                        'input_data': {k: v for k, v in input_data.items() if k in ['symbol', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'hmm_regime']},
                        'explanation': explanation
                    })
                    
                    logger.info(f"Explication gu00e9nu00e9ru00e9e pour l'u00e9chantillon {idx}")
                except Exception as e:
                    logger.error(f"Erreur lors de la gu00e9nu00e9ration de l'explication pour l'u00e9chantillon {idx}: {e}")
            
            return explanations
        elif hasattr(reasoning_module, 'generate_explanation'):
            # Si le module a une fonction generate_explanation directement
            explanations = []
            
            for idx, row in sample_data.iterrows():
                try:
                    # Convertir la ligne en format attendu par le module de raisonnement
                    input_data = row.to_dict()
                    
                    # Gu00e9nu00e9rer une explication
                    explanation = reasoning_module.generate_explanation(model, input_data)
                    explanations.append({
                        'sample_idx': idx,
                        'input_data': {k: v for k, v in input_data.items() if k in ['symbol', 'open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'hmm_regime']},
                        'explanation': explanation
                    })
                    
                    logger.info(f"Explication gu00e9nu00e9ru00e9e pour l'u00e9chantillon {idx}")
                except Exception as e:
                    logger.error(f"Erreur lors de la gu00e9nu00e9ration de l'explication pour l'u00e9chantillon {idx}: {e}")
            
            return explanations
        else:
            logger.error("Le module de raisonnement ne contient pas de classe ReasoningModule ou de fonction generate_explanation")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du test du module de raisonnement: {e}")
        return None

def generate_example_explanations():
    """
    Gu00e9nu00e8re des exemples d'explications si le module de raisonnement n'est pas disponible
    """
    logger.info("Gu00e9nu00e9ration d'exemples d'explications...")
    
    sample_data = load_sample_data()
    
    if sample_data is None:
        logger.error("Impossible de gu00e9nu00e9rer des exemples d'explications: donnu00e9es manquantes")
        return None
    
    explanations = []
    
    for idx, row in sample_data.iterrows():
        # Extraire les donnu00e9es pertinentes
        symbol = row.get('symbol', 'BTC/USDT')
        close = row.get('close', 0)
        rsi = row.get('RSI', 50)
        macd = row.get('MACD', 0)
        hmm_regime = row.get('hmm_regime', 0)
        
        # Gu00e9nu00e9rer une explication simulu00e9e
        regime_explanations = {
            0: "Le marchu00e9 est dans un ru00e9gime de faible volatilitu00e9 (ru00e9gime 0), ce qui suggu00e8re une phase de consolidation.",
            1: "Le marchu00e9 est dans un ru00e9gime haussier (ru00e9gime 1), caractu00e9risu00e9 par une tendance u00e0 la hausse et une volatilitu00e9 modu00e9ru00e9e.",
            2: "Le marchu00e9 est dans un ru00e9gime baissier (ru00e9gime 2), caractu00e9risu00e9 par une tendance u00e0 la baisse et une volatilitu00e9 u00e9levu00e9e."
        }
        
        rsi_explanation = ""
        if rsi > 70:
            rsi_explanation = f"Le RSI est u00e9levu00e9 ({rsi:.2f}), indiquant une condition de surachat potentielle."
        elif rsi < 30:
            rsi_explanation = f"Le RSI est bas ({rsi:.2f}), indiquant une condition de survente potentielle."
        else:
            rsi_explanation = f"Le RSI est dans une zone neutre ({rsi:.2f})."
        
        macd_explanation = ""
        if macd > 0:
            macd_explanation = f"Le MACD est positif ({macd:.2f}), suggu00e9rant une tendance haussière."
        else:
            macd_explanation = f"Le MACD est nu00e9gatif ({macd:.2f}), suggu00e9rant une tendance baissière."
        
        # Du00e9terminer le signal de trading
        if hmm_regime == 1 and rsi < 70 and macd > 0:
            signal = "ACHETER"
            reason = "Le marchu00e9 est dans un ru00e9gime haussier, le RSI n'est pas en surachat, et le MACD est positif."
        elif hmm_regime == 2 and rsi > 30 and macd < 0:
            signal = "VENDRE"
            reason = "Le marchu00e9 est dans un ru00e9gime baissier, le RSI n'est pas en survente, et le MACD est nu00e9gatif."
        else:
            signal = "CONSERVER"
            reason = "Les conditions actuelles du marchu00e9 ne justifient pas une action d'achat ou de vente."
        
        # Gu00e9nu00e9rer des niveaux de stop loss et take profit
        if signal == "ACHETER":
            stop_loss = close * 0.95  # 5% en dessous du prix actuel
            take_profit = close * 1.15  # 15% au-dessus du prix actuel
        elif signal == "VENDRE":
            stop_loss = close * 1.05  # 5% au-dessus du prix actuel
            take_profit = close * 0.85  # 15% en dessous du prix actuel
        else:
            stop_loss = 0
            take_profit = 0
        
        # Construire l'explication complu00e8te
        explanation = {
            'symbol': symbol,
            'price': close,
            'signal': signal,
            'reasoning': f"{regime_explanations.get(hmm_regime, '')} {rsi_explanation} {macd_explanation} {reason}",
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': np.random.uniform(0.7, 0.95)  # Confiance simulu00e9e
        }
        
        explanations.append({
            'sample_idx': idx,
            'input_data': {
                'symbol': symbol,
                'close': close,
                'RSI': rsi,
                'MACD': macd,
                'hmm_regime': hmm_regime
            },
            'explanation': explanation
        })
    
    return explanations

def save_explanations(explanations):
    """
    Sauvegarde les explications gu00e9nu00e9ru00e9es
    """
    if explanations is None or len(explanations) == 0:
        logger.warning("Aucune explication u00e0 sauvegarder")
        return
    
    # Sauvegarder dans le ru00e9pertoire des rapports
    output_path = REPORTS_DIR / f"explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=2, default=str)
    
    logger.info(f"Explications sauvegardu00e9es dans {output_path}")
    
    # Sauvegarder u00e9galement dans le ru00e9pertoire du modu00e8le de raisonnement
    if not REASONING_DIR.exists():
        REASONING_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = REASONING_DIR / "explanation_examples.json"
    
    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=2, default=str)
    
    logger.info(f"Exemples d'explications sauvegardu00e9s dans {output_path}")

def main():
    """
    Fonction principale
    """
    logger.info("Du00e9but de la vu00e9rification du module de raisonnement")
    
    # 1. Charger le module de raisonnement
    reasoning_module = load_reasoning_module()
    
    # 2. Charger le modu00e8le
    model = load_model()
    
    # 3. Charger un u00e9chantillon de donnu00e9es
    sample_data = load_sample_data()
    
    # 4. Tester le module de raisonnement
    if reasoning_module is not None and model is not None and sample_data is not None:
        explanations = test_reasoning_module(reasoning_module, model, sample_data)
        
        if explanations is None or len(explanations) == 0:
            logger.warning("Aucune explication gu00e9nu00e9ru00e9e par le module de raisonnement, gu00e9nu00e9ration d'exemples...")
            explanations = generate_example_explanations()
    else:
        logger.warning("Impossible de tester le module de raisonnement, gu00e9nu00e9ration d'exemples...")
        explanations = generate_example_explanations()
    
    # 5. Sauvegarder les explications
    save_explanations(explanations)
    
    logger.info("Fin de la vu00e9rification du module de raisonnement")

if __name__ == "__main__":
    main()
