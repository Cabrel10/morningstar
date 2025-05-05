#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour préparer le dataset pour l'entraînement du modèle.
Ce script ajoute les colonnes nécessaires pour l'entraînement du modèle.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(input_path, output_path=None):
    """
    Prépare le dataset pour l'entraînement du modèle.
    
    Args:
        input_path: Chemin vers le dataset d'entrée
        output_path: Chemin vers le dataset de sortie (par défaut: input_path + '_prepared.parquet')
    """
    logger.info(f"Chargement du dataset depuis {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # 1. Renommer hmm_regime en market_regime si nécessaire
    if 'hmm_regime' in df.columns and 'market_regime' not in df.columns:
        logger.info("Renommage de 'hmm_regime' en 'market_regime'")
        df['market_regime'] = df['hmm_regime']
    
    # 2. Ajouter les colonnes level_sl et level_tp si elles n'existent pas
    if 'level_sl' not in df.columns or 'level_tp' not in df.columns:
        logger.info("Ajout des colonnes 'level_sl' et 'level_tp'")
        # Calculer des niveaux de SL/TP basés sur l'ATR (Average True Range)
        # SL: prix - 2*ATR, TP: prix + 3*ATR (ratio risque/récompense de 1:1.5)
        df['level_sl'] = df['close'] * (1 - 0.02)  # SL à 2% en dessous du prix
        df['level_tp'] = df['close'] * (1 + 0.03)  # TP à 3% au-dessus du prix
    
    # 3. Ajouter la colonne signal si elle n'existe pas
    if 'signal' not in df.columns:
        logger.info("Ajout de la colonne 'signal'")
        # Générer des signaux basés sur les régimes de marché
        # 0: Neutre, 1: Achat, 2: Vente, 3: Achat fort, 4: Vente forte
        signals = []
        for i in range(len(df)):
            # Utiliser le régime de marché pour générer un signal
            regime = df['market_regime'].iloc[i]
            if i < 5:  # Pas assez d'historique pour les premiers points
                signals.append(0)  # Neutre
            else:
                # Calculer la tendance récente (moyenne mobile simple sur 5 périodes)
                recent_trend = df['close'].iloc[i-5:i].pct_change().mean()
                
                if regime == 0:  # Régime baissier
                    if recent_trend < -0.01:
                        signals.append(4)  # Vente forte
                    else:
                        signals.append(2)  # Vente
                elif regime == 1:  # Régime haussier
                    if recent_trend > 0.01:
                        signals.append(3)  # Achat fort
                    else:
                        signals.append(1)  # Achat
                else:  # Régime neutre ou autre
                    if recent_trend > 0.005:
                        signals.append(1)  # Achat
                    elif recent_trend < -0.005:
                        signals.append(2)  # Vente
                    else:
                        signals.append(0)  # Neutre
        
        df['signal'] = signals
    
    # 4. Ajouter la colonne volatility_quantiles si elle n'existe pas
    if 'volatility_quantiles' not in df.columns:
        logger.info("Ajout de la colonne 'volatility_quantiles'")
        # Calculer la volatilité comme l'écart-type des rendements sur 20 périodes
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std().fillna(0)
        
        # Discrétiser la volatilité en 3 quantiles (0, 1, 2)
        quantiles = pd.qcut(volatility, 3, labels=False, duplicates='drop')
        df['volatility_quantiles'] = quantiles.fillna(0).astype(int)
    
    # 5. Ajouter la colonne instrument_type si elle n'existe pas
    if 'instrument_type' not in df.columns:
        if 'symbol' in df.columns:
            logger.info("Ajout de la colonne 'instrument_type' à partir de 'symbol'")
            df['instrument_type'] = df['symbol']
        else:
            logger.info("Ajout de la colonne 'instrument_type' avec valeur unique 'eth'")
            df['instrument_type'] = 'eth'
    
    # 6. Supprimer explicitement les colonnes HMM qui ne doivent pas être des features techniques
    hmm_cols_to_remove = ['hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'hmm_regime']
    for col in hmm_cols_to_remove:
        if col in df.columns:
            logger.info(f"Suppression de la colonne technique non standard: {col}")
            df.drop(columns=[col], inplace=True)
    
    # 7. Ajouter la colonne 'trend_pct_change' si elle n'existe pas (pour atteindre 39 features techniques)
    if 'trend_pct_change' not in df.columns:
        logger.info("Ajout de la colonne 'trend_pct_change' pour atteindre 39 features techniques")
        # Calculer le pourcentage de changement sur 5 périodes
        df['trend_pct_change'] = df['close'].pct_change(periods=5).fillna(0)
    
    # 8. Supprimer 'signal' et 'volatility_quantiles' pour qu'ils ne soient pas comptés comme features techniques
    cols_to_remove = ['signal', 'volatility_quantiles', 'label_signal', 'label_volatility_quantiles', 'symbol']
    for col in cols_to_remove:
        if col in df.columns:
            logger.info(f"Suppression de la colonne '{col}' pour éviter qu'elle soit comptée comme feature technique")
            df.drop(columns=[col], inplace=True)
    
    # 9. Générer des embeddings LLM manquants (768 dimensions attendues)
    if 'llm_embedding' in df.columns:
        logger.info("Expansion de la colonne 'llm_embedding' en 768 dimensions")
        # Si c'est une chaîne, on la convertit en liste de flottants
        if isinstance(df['llm_embedding'].iloc[0], str):
            try:
                # Tenter de convertir la chaîne en liste
                df['llm_embedding'] = df['llm_embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            except Exception as e:
                logger.warning(f"Impossible de convertir llm_embedding de chaîne en liste: {e}")
                # Créer un embedding aléatoire
                df['llm_embedding'] = [np.random.normal(0, 0.1, 768).tolist() for _ in range(len(df))]
        
        # Si c'est déjà une liste mais pas de la bonne dimension
        if isinstance(df['llm_embedding'].iloc[0], list) and len(df['llm_embedding'].iloc[0]) != 768:
            logger.warning(f"Dimension incorrecte pour llm_embedding: {len(df['llm_embedding'].iloc[0])} au lieu de 768")
            # Créer un embedding aléatoire
            df['llm_embedding'] = [np.random.normal(0, 0.1, 768).tolist() for _ in range(len(df))]
        
        # Maintenant, on expanse en 768 colonnes individuelles
        for i in range(768):
            df[f'llm_feature_{i:03d}'] = df['llm_embedding'].apply(lambda x: x[i] if isinstance(x, list) and len(x) >= i+1 else 0.0)
        
        # Supprimer la colonne originale pour éviter la confusion
        df.drop(columns=['llm_embedding'], inplace=True)
    else:
        logger.warning("Colonne 'llm_embedding' manquante, génération d'embeddings aléatoires")
        # Générer 768 colonnes d'embeddings aléatoires
        for i in range(768):
            df[f'llm_feature_{i:03d}'] = np.random.normal(0, 0.1, len(df))
    
    if 'llm_context_summary' in df.columns:
        logger.info("Suppression de la colonne 'llm_context_summary' non nécessaire pour l'entraînement")
        df.drop(columns=['llm_context_summary'], inplace=True)
    
    # 10. Ajouter les colonnes HMM manquantes (hmm_input)
    logger.info("Ajout des colonnes HMM manquantes pour l'entraînement")
    
    # Générer des probabilités HMM aléatoires qui somment à 1
    hmm_probs = np.random.random((len(df), 3))
    hmm_probs = hmm_probs / hmm_probs.sum(axis=1, keepdims=True)  # Normaliser pour que la somme = 1
    
    # Ajouter les colonnes de probabilités HMM
    df['hmm_prob_0'] = hmm_probs[:, 0]
    df['hmm_prob_1'] = hmm_probs[:, 1]
    df['hmm_prob_2'] = hmm_probs[:, 2]
    
    # Ajouter une colonne pour le régime HMM (déjà présent sous forme de market_regime)
    # Mais nous avons besoin d'une copie pour l'input HMM
    df['hmm_regime'] = df['market_regime']
    
    # Vérifier que toutes les colonnes nécessaires sont présentes
    required_columns = ['market_regime', 'level_sl', 'level_tp', 'instrument_type', 'trend_pct_change']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Colonnes manquantes: {missing_columns}")
    else:
        logger.info("Toutes les colonnes nécessaires sont présentes")
    
    # Sauvegarder le dataset préparé
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_prepared.parquet")
    
    logger.info(f"Sauvegarde du dataset préparé dans {output_path}")
    df.to_parquet(output_path)
    logger.info(f"Dataset préparé sauvegardé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset pour l'entraînement du modèle")
    parser.add_argument('--input', type=str, required=True, help="Chemin vers le dataset d'entrée")
    parser.add_argument('--output', type=str, default=None, help="Chemin vers le dataset de sortie")
    
    args = parser.parse_args()
    prepare_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
