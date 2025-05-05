#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'intégration du raisonnement Chain-of-Thought pour le trading en direct.
Permet d'expliquer les décisions de trading en temps réel.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import joblib
import time
from datetime import datetime, timedelta
import json
import traceback

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Imports du projet
from model.reasoning.reasoning_module import ReasoningModule
from utils.data_preparation import CryptoBERTEmbedder

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveReasoningIntegration:
    """
    Intégration du raisonnement Chain-of-Thought pour le trading en direct.
    """
    
    def __init__(self, model_dir: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialise l'intégration du raisonnement.
        
        Args:
            model_dir: Répertoire du modèle
            config: Configuration optionnelle
        """
        self.model_dir = Path(model_dir)
        self.config = config or {}
        
        # Charger ou initialiser le module de raisonnement
        self.reasoning_module = self._load_reasoning_module()
        
        # Historique des explications
        self.explanation_history = []
        self.max_history_size = self.config.get('max_explanation_history', 100)
        
        logger.info(f"Intégration du raisonnement initialisée pour le modèle: {model_dir}")
    
    def _load_reasoning_module(self) -> ReasoningModule:
        """
        Charge le module de raisonnement depuis le répertoire du modèle.
        
        Returns:
            Module de raisonnement
        """
        reasoning_module_path = self.model_dir / 'metadata' / 'reasoning_module.pkl'
        
        if reasoning_module_path.exists():
            try:
                logger.info(f"Chargement du module de raisonnement depuis {reasoning_module_path}")
                return joblib.load(reasoning_module_path)
            except Exception as e:
                logger.error(f"Erreur lors du chargement du module de raisonnement: {e}")
                logger.info("Initialisation d'un nouveau module de raisonnement")
                return ReasoningModule()
        else:
            logger.info(f"Module de raisonnement non trouvé: {reasoning_module_path}")
            logger.info("Initialisation d'un nouveau module de raisonnement")
            return ReasoningModule()
    
    def generate_explanation(self, market_data: pd.DataFrame, predictions: Dict[str, np.ndarray], 
                             symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Génère une explication Chain-of-Thought pour une prédiction.
        
        Args:
            market_data: Données de marché (OHLCV)
            predictions: Prédictions du modèle
            symbol: Symbole de la paire
            timeframe: Timeframe des données
            
        Returns:
            Dictionnaire contenant l'explication
        """
        try:
            start_time = time.time()
            
            # Générer l'explication
            explanation = self.reasoning_module.generate_chain_of_thought_explanation(
                market_data=market_data,
                predictions=predictions,
                attention_scores=predictions.get('attention_scores')
            )
            
            # Créer le résultat
            result = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'explanation': explanation,
                'signal': self._get_signal_from_predictions(predictions),
                'confidence': self._get_confidence_from_predictions(predictions),
                'processing_time': time.time() - start_time
            }
            
            # Ajouter à l'historique
            self.explanation_history.append(result)
            
            # Limiter la taille de l'historique
            if len(self.explanation_history) > self.max_history_size:
                self.explanation_history = self.explanation_history[-self.max_history_size:]
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'explication: {e}")
            traceback.print_exc()
            
            # Retourner une explication par défaut en cas d'erreur
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'explanation': f"Impossible de générer une explication: {str(e)}",
                'signal': self._get_signal_from_predictions(predictions),
                'confidence': self._get_confidence_from_predictions(predictions),
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _get_signal_from_predictions(self, predictions: Dict[str, np.ndarray]) -> str:
        """
        Extrait le signal de trading des prédictions.
        
        Args:
            predictions: Prédictions du modèle
            
        Returns:
            Signal de trading (Buy, Sell, Hold)
        """
        signal_class = None
        
        if 'signal_class' in predictions:
            signal_class = predictions['signal_class'][-1] if isinstance(predictions['signal_class'], np.ndarray) else predictions['signal_class']
        elif 'signal' in predictions:
            signal_class = np.argmax(predictions['signal'][-1]) if isinstance(predictions['signal'], np.ndarray) else np.argmax(predictions['signal'])
        
        if signal_class is not None:
            if signal_class == 1:
                return "Buy"
            elif signal_class == 2:
                return "Sell"
            else:
                return "Hold"
        
        return "Unknown"
    
    def _get_confidence_from_predictions(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Extrait la confiance du signal de trading des prédictions.
        
        Args:
            predictions: Prédictions du modèle
            
        Returns:
            Confiance du signal (0.0 à 1.0)
        """
        if 'signal_confidence' in predictions:
            return float(predictions['signal_confidence'][-1]) if isinstance(predictions['signal_confidence'], np.ndarray) else float(predictions['signal_confidence'])
        elif 'signal' in predictions:
            signal_probs = predictions['signal'][-1] if isinstance(predictions['signal'], np.ndarray) else predictions['signal']
            return float(np.max(signal_probs))
        
        return 0.0
    
    def save_explanations(self, output_dir: Optional[Path] = None) -> Path:
        """
        Sauvegarde l'historique des explications dans un fichier JSON.
        
        Args:
            output_dir: Répertoire de sortie (par défaut: logs/explanations)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if output_dir is None:
            output_dir = Path('logs/explanations')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer le nom du fichier avec la date
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"explanations_{timestamp}.json"
        
        # Sauvegarder les explications
        with open(output_file, 'w') as f:
            json.dump(self.explanation_history, f, indent=2)
        
        logger.info(f"Explications sauvegardées dans {output_file}")
        
        return output_file
    
    def format_explanation_for_display(self, explanation: Dict[str, Any]) -> str:
        """
        Formate une explication pour l'affichage.
        
        Args:
            explanation: Dictionnaire d'explication
            
        Returns:
            Explication formatée pour l'affichage
        """
        if not explanation or 'explanation' not in explanation:
            return "Aucune explication disponible."
        
        # Extraire les composants de l'explication
        cot_explanation = explanation['explanation']
        signal = explanation.get('signal', 'Unknown')
        confidence = explanation.get('confidence', 0.0)
        timestamp = explanation.get('timestamp', datetime.now().isoformat())
        symbol = explanation.get('symbol', 'Unknown')
        
        # Formater l'horodatage
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        # Construire l'explication formatée
        formatted = f"# Explication du signal de trading pour {symbol}\n\n"
        formatted += f"**Date:** {formatted_time}\n"
        formatted += f"**Signal:** {signal}\n"
        formatted += f"**Confiance:** {confidence:.2%}\n\n"
        
        # Ajouter les sections de l'explication
        if isinstance(cot_explanation, dict):
            if 'market_context' in cot_explanation:
                formatted += f"## Contexte du marché\n{cot_explanation['market_context']}\n\n"
            
            if 'technical_analysis' in cot_explanation:
                formatted += f"## Analyse technique\n{cot_explanation['technical_analysis']}\n\n"
            
            if 'sentiment_analysis' in cot_explanation:
                formatted += f"## Analyse du sentiment\n{cot_explanation['sentiment_analysis']}\n\n"
            
            if 'risk_assessment' in cot_explanation:
                formatted += f"## Évaluation du risque\n{cot_explanation['risk_assessment']}\n\n"
            
            if 'final_reasoning' in cot_explanation:
                formatted += f"## Raisonnement final\n{cot_explanation['final_reasoning']}\n\n"
        else:
            formatted += f"## Explication\n{cot_explanation}\n\n"
        
        return formatted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intégration du raisonnement Chain-of-Thought pour le trading en direct")
    parser.add_argument("--model-dir", required=True, help="Répertoire du modèle")
    parser.add_argument("--data-file", required=True, help="Fichier de données (CSV ou Parquet)")
    parser.add_argument("--output-dir", help="Répertoire de sortie pour les explications")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole de la paire")
    parser.add_argument("--timeframe", default="1h", help="Timeframe des données")
    
    args = parser.parse_args()
    
    # Charger les données
    data_path = Path(args.data_file)
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    else:
        data = pd.read_parquet(data_path)
    
    # Initialiser l'intégration du raisonnement
    reasoning = LiveReasoningIntegration(Path(args.model_dir))
    
    # Simuler des prédictions
    predictions = {
        'signal': np.array([[0.1, 0.8, 0.1]]),  # Exemple: forte probabilité d'achat
        'signal_class': np.array([1]),  # 1 = Buy
        'signal_confidence': np.array([0.8])
    }
    
    # Générer une explication
    explanation = reasoning.generate_explanation(data, predictions, args.symbol, args.timeframe)
    
    # Afficher l'explication formatée
    print(reasoning.format_explanation_for_display(explanation))
    
    # Sauvegarder les explications
    if args.output_dir:
        output_file = reasoning.save_explanations(Path(args.output_dir))
        print(f"Explications sauvegardées dans {output_file}")
