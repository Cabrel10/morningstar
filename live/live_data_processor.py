#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour prétraiter les données en direct pour le modèle Morningstar.
Applique les mêmes transformations que pour l'entraînement aux données live.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import joblib
import time
from datetime import datetime, timedelta
import json
import traceback

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Imports du projet
from utils.data_preparation import CryptoBERTEmbedder
from utils.technical_indicators import add_technical_indicators
from utils.hmm_regime import HMMRegimeDetector
from scripts.normalize_datasets import normalize_dataset, add_synthetic_signals
from live.live_data_handler import LiveDataHandler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDataProcessor:
    """
    Processeur de données live pour le modèle Morningstar.
    Applique les mêmes transformations que pour l'entraînement aux données live.
    """
    
    def __init__(self, model_dir: Path, config: Dict = None, cache_dir: Optional[Path] = None):
        """
        Initialise le processeur de données live.
        
        Args:
            model_dir: Répertoire du modèle
            config: Configuration optionnelle
            cache_dir: Répertoire de cache pour les embeddings
        """
        self.model_dir = Path(model_dir)
        self.config = config or {}
        
        # Initialiser le gestionnaire de données live
        self.data_handler = LiveDataHandler(
            model_dir=self.model_dir,
            scaler_path=self.model_dir / 'metadata' / 'feature_scaler.pkl'
        )
        
        # Charger la configuration des features
        self.feature_config_path = self.model_dir / 'metadata' / 'feature_config.json'
        if self.feature_config_path.exists():
            with open(self.feature_config_path, 'r') as f:
                self.feature_config = json.load(f)
        else:
            logger.warning(f"Configuration des features non trouvée: {self.feature_config_path}")
            self.feature_config = {
                "technical_indicators": True,
                "sentiment_analysis": True,
                "market_regime": True,
                "bert_embeddings": True
            }
        
        # Initialiser les composants de traitement
        self._init_components(cache_dir)
        
        # Historique des données pour les calculs qui nécessitent un historique
        self.history_buffer = {}
        self.min_history_size = 100  # Nombre minimum de points de données pour les calculs techniques
        
        logger.info(f"Processeur de données live initialisé avec le modèle: {model_dir}")
    
    def _init_components(self, cache_dir: Optional[Path] = None):
        """
        Initialise les composants de traitement des données.
        
        Args:
            cache_dir: Répertoire de cache pour les embeddings
        """
        # Initialiser l'embedder CryptoBERT si nécessaire
        if self.feature_config.get("bert_embeddings", True):
            cache_dir = cache_dir or self.model_dir / 'cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.bert_embedder = CryptoBERTEmbedder(cache_dir=cache_dir)
            logger.info(f"CryptoBERT embedder initialisé avec cache: {cache_dir}")
        else:
            self.bert_embedder = None
        
        # Initialiser le détecteur de régime HMM si nécessaire
        if self.feature_config.get("market_regime", True):
            hmm_model_path = self.model_dir / 'metadata' / 'hmm_model.pkl'
            if hmm_model_path.exists():
                self.hmm_detector = joblib.load(hmm_model_path)
                logger.info(f"Détecteur de régime HMM chargé depuis: {hmm_model_path}")
            else:
                self.hmm_detector = HMMRegimeDetector(n_components=4)
                logger.warning(f"Modèle HMM non trouvé, initialisation d'un nouveau modèle")
        else:
            self.hmm_detector = None
    
    def update_history(self, symbol: str, new_data: pd.DataFrame):
        """
        Met à jour l'historique des données pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            new_data: Nouvelles données à ajouter à l'historique
        """
        if symbol not in self.history_buffer:
            self.history_buffer[symbol] = new_data.copy()
        else:
            # Fusionner les nouvelles données avec l'historique existant
            combined = pd.concat([self.history_buffer[symbol], new_data])
            # Supprimer les doublons en gardant la dernière occurrence
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            # Trier par timestamp
            combined = combined.sort_values('timestamp')
            # Limiter la taille de l'historique si nécessaire
            max_history = self.config.get('max_history_size', 1000)
            if len(combined) > max_history:
                combined = combined.iloc[-max_history:]
            
            self.history_buffer[symbol] = combined
        
        logger.debug(f"Historique mis à jour pour {symbol}, taille: {len(self.history_buffer[symbol])}")
    
    def add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les indicateurs techniques aux données.
        
        Args:
            data: DataFrame des données
            
        Returns:
            DataFrame avec les indicateurs techniques ajoutés
        """
        if not self.feature_config.get("technical_indicators", True):
            return data
        
        try:
            # Vérifier que les colonnes requises sont présentes
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"Colonnes manquantes pour les indicateurs techniques: {missing_cols}")
                # Ajouter des colonnes manquantes avec des valeurs par défaut
                for col in missing_cols:
                    if col == 'volume':
                        data[col] = 0
                    else:
                        # Utiliser close pour les autres colonnes de prix manquantes
                        data[col] = data['close'] if 'close' in data.columns else 0
            
            # Ajouter les indicateurs techniques
            result = add_technical_indicators(data)
            
            # Vérifier les valeurs NaN et les remplacer
            for col in result.columns:
                if result[col].isna().any():
                    # Remplacer les NaN par la dernière valeur valide ou 0
                    result[col] = result[col].fillna(method='ffill').fillna(0)
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des indicateurs techniques: {e}")
            traceback.print_exc()
            return data
    
    def add_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute la détection de régime de marché aux données.
        
        Args:
            data: DataFrame des données
            
        Returns:
            DataFrame avec la colonne de régime de marché ajoutée
        """
        if not self.feature_config.get("market_regime", True) or self.hmm_detector is None:
            return data
        
        try:
            # Vérifier qu'il y a suffisamment de données pour la détection de régime
            if len(data) < self.min_history_size:
                logger.warning(f"Données insuffisantes pour la détection de régime: {len(data)} < {self.min_history_size}")
                # Ajouter une colonne de régime par défaut
                data['hmm_regime'] = 1  # Régime neutre par défaut
                return data
            
            # Préparer les données pour le HMM
            features = data[['close']].values
            
            # Détecter le régime
            regimes = self.hmm_detector.predict(features)
            
            # Ajouter la colonne de régime
            data['hmm_regime'] = regimes
            
            return data
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection de régime: {e}")
            traceback.print_exc()
            # Ajouter une colonne de régime par défaut en cas d'erreur
            data['hmm_regime'] = 1  # Régime neutre par défaut
            return data
    
    def add_sentiment_features(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Ajoute les features de sentiment aux données.
        
        Args:
            data: DataFrame des données
            sentiment_data: DataFrame optionnel contenant les données de sentiment
            
        Returns:
            DataFrame avec les features de sentiment ajoutées
        """
        if not self.feature_config.get("sentiment_analysis", True):
            return data
        
        try:
            # Si aucune donnée de sentiment n'est fournie, utiliser des valeurs neutres
            if sentiment_data is None or sentiment_data.empty:
                logger.warning("Aucune donnée de sentiment fournie, utilisation de valeurs neutres")
                data['sentiment_score'] = 0.0
                data['sentiment_magnitude'] = 0.0
                data['sentiment_label'] = 'neutral'
                
                # Ajouter des embeddings nuls si nécessaire
                if self.feature_config.get("bert_embeddings", True) and self.bert_embedder is not None:
                    embedding_dim = 768  # Dimension par défaut pour CryptoBERT
                    for i in range(embedding_dim):
                        data[f'cryptobert_{i}'] = 0.0
                
                return data
            
            # Fusionner les données de sentiment avec les données principales
            # Supposons que sentiment_data a une colonne 'timestamp' compatible
            merged = pd.merge_asof(
                data.sort_values('timestamp'),
                sentiment_data.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            
            # Ajouter les embeddings BERT si nécessaire
            if self.feature_config.get("bert_embeddings", True) and self.bert_embedder is not None:
                if 'text' in merged.columns:
                    # Générer des embeddings pour chaque texte
                    embeddings = []
                    for text in merged['text']:
                        if pd.isna(text) or text == '':
                            # Utiliser un embedding nul pour les textes vides
                            embedding = np.zeros(768)
                        else:
                            embedding = self.bert_embedder.embed_text(text)
                        embeddings.append(embedding)
                    
                    # Ajouter les embeddings comme colonnes
                    embeddings_array = np.vstack(embeddings)
                    for i in range(embeddings_array.shape[1]):
                        merged[f'cryptobert_{i}'] = embeddings_array[:, i]
            
            return merged
        
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des features de sentiment: {e}")
            traceback.print_exc()
            return data
    
    def process_live_data(self, market_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None, symbol: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Traite les données live pour le modèle.
        
        Args:
            market_data: DataFrame des données de marché
            sentiment_data: DataFrame optionnel des données de sentiment
            symbol: Symbole de la paire (pour l'historique)
            
        Returns:
            Dictionnaire avec les entrées pour le modèle
        """
        try:
            start_time = time.time()
            
            # Vérifier que les données ont une colonne timestamp
            if 'timestamp' not in market_data.columns:
                if 'time' in market_data.columns:
                    market_data = market_data.rename(columns={'time': 'timestamp'})
                else:
                    # Créer une colonne timestamp
                    market_data['timestamp'] = pd.date_range(
                        end=datetime.now(), 
                        periods=len(market_data), 
                        freq='1min'
                    )
            
            # Convertir timestamp en datetime si ce n'est pas déjà le cas
            if not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], unit='ms')
            
            # Mettre à jour l'historique si un symbole est fourni
            if symbol:
                self.update_history(symbol, market_data)
                # Utiliser l'historique complet pour les calculs qui nécessitent plus de données
                if symbol in self.history_buffer and len(self.history_buffer[symbol]) >= self.min_history_size:
                    data_to_process = self.history_buffer[symbol].copy()
                else:
                    data_to_process = market_data.copy()
            else:
                data_to_process = market_data.copy()
            
            # Ajouter les indicateurs techniques
            data_with_indicators = self.add_technical_features(data_to_process)
            
            # Ajouter la détection de régime
            data_with_regime = self.add_market_regime(data_with_indicators)
            
            # Ajouter les features de sentiment
            data_with_sentiment = self.add_sentiment_features(data_with_regime, sentiment_data)
            
            # Ajouter les signaux synthétiques si configuré
            if self.feature_config.get("synthetic_signals", False):
                data_with_signals = add_synthetic_signals(data_with_sentiment)
            else:
                data_with_signals = data_with_sentiment
            
            # Normaliser les données
            normalized_data = normalize_dataset(data_with_signals)
            
            # Si nous avons utilisé l'historique complet, extraire seulement les dernières lignes
            # correspondant aux données d'entrée originales
            if symbol and len(normalized_data) > len(market_data):
                normalized_data = normalized_data.iloc[-len(market_data):]
            
            # Gérer les données manquantes
            processed_data = self.data_handler.handle_missing_data(normalized_data)
            
            # Normaliser les données avec le scaler du modèle
            normalized_data = self.data_handler.normalize_live_data(processed_data)
            
            # Préparer les entrées du modèle
            model_inputs = self.data_handler.prepare_model_inputs(normalized_data)
            
            processing_time = time.time() - start_time
            logger.info(f"Données live traitées en {processing_time:.2f}s, taille: {len(normalized_data)}")
            
            return model_inputs
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données live: {e}")
            traceback.print_exc()
            raise
    
    def post_process_predictions(self, predictions: Dict[str, np.ndarray], live_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Post-traite les prédictions du modèle.
        
        Args:
            predictions: Dictionnaire des prédictions du modèle
            live_data: DataFrame des données en direct
            
        Returns:
            Dictionnaire avec les prédictions post-traitées
        """
        return self.data_handler.post_process_predictions(predictions, live_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Processeur de données live pour Morningstar")
    parser.add_argument("--model-dir", required=True, help="Répertoire du modèle")
    parser.add_argument("--input-file", required=True, help="Fichier d'entrée (CSV ou Parquet)")
    parser.add_argument("--sentiment-file", help="Fichier de sentiment (CSV ou Parquet)")
    parser.add_argument("--output-file", help="Fichier de sortie (Parquet)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole de la paire")
    
    args = parser.parse_args()
    
    # Charger les données
    input_path = Path(args.input_file)
    if input_path.suffix == '.csv':
        market_data = pd.read_csv(input_path)
    else:
        market_data = pd.read_parquet(input_path)
    
    # Charger les données de sentiment si fournies
    sentiment_data = None
    if args.sentiment_file:
        sentiment_path = Path(args.sentiment_file)
        if sentiment_path.suffix == '.csv':
            sentiment_data = pd.read_csv(sentiment_path)
        else:
            sentiment_data = pd.read_parquet(sentiment_path)
    
    # Initialiser le processeur de données
    processor = LiveDataProcessor(Path(args.model_dir))
    
    # Traiter les données
    model_inputs = processor.process_live_data(market_data, sentiment_data, args.symbol)
    
    # Afficher les informations sur les entrées du modèle
    for key, value in model_inputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
    # Sauvegarder les entrées préparées si un fichier de sortie est spécifié
    if args.output_file:
        output_path = Path(args.output_file)
        
        # Convertir les entrées en DataFrame
        output_data = pd.DataFrame()
        for key, value in model_inputs.items():
            if len(value.shape) == 2:
                for i in range(value.shape[1]):
                    output_data[f"{key}_{i}"] = value[:, i]
            else:
                output_data[key] = value
        
        # Sauvegarder
        output_data.to_parquet(output_path)
        print(f"Entrées du modèle sauvegardées dans {output_path}")
