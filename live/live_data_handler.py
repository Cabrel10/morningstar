#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour gérer les données en direct pour le modèle Morningstar.
Gère les données manquantes et assure la compatibilité avec le modèle.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import joblib

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDataHandler:
    """Gestionnaire de données en direct pour le modèle Morningstar."""
    
    def __init__(self, model_dir: Path, feature_columns: List[str] = None, scaler_path: Optional[Path] = None):
        """
        Initialise le gestionnaire de données en direct.
        
        Args:
            model_dir: Répertoire du modèle
            feature_columns: Liste des colonnes de features attendues
            scaler_path: Chemin vers le scaler pour la normalisation
        """
        self.model_dir = Path(model_dir)
        self.last_known_values = {}
        
        # Charger la liste des colonnes de features si non fournie
        if feature_columns is None:
            feature_cols_path = self.model_dir / 'metadata' / 'feature_columns.json'
            if feature_cols_path.exists():
                import json
                with open(feature_cols_path, 'r') as f:
                    self.feature_columns = json.load(f)
            else:
                logger.warning(f"Fichier de colonnes de features non trouvé: {feature_cols_path}")
                self.feature_columns = []
        else:
            self.feature_columns = feature_columns
        
        # Charger le scaler si le chemin est fourni
        if scaler_path is not None and scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            scaler_path = self.model_dir / 'metadata' / 'feature_scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler non trouvé: {scaler_path}")
                self.scaler = None
    
    def handle_missing_data(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les données manquantes dans le flux de données en direct.
        
        Args:
            live_data: DataFrame des données en direct
            
        Returns:
            DataFrame avec les données manquantes gérées
        """
        # Vérifier les colonnes manquantes
        missing_columns = [col for col in self.feature_columns if col not in live_data.columns]
        
        # Ajouter les colonnes manquantes
        for col in missing_columns:
            if col in self.last_known_values:
                # Utiliser la dernière valeur connue
                live_data[col] = self.last_known_values[col]
            else:
                # Utiliser une valeur par défaut ou une imputation
                if col.startswith(('rsi_', 'macd_')):
                    live_data[col] = 0  # Valeur neutre pour les oscillateurs
                elif col.startswith('hmm_'):
                    live_data[col] = 1  # Régime neutre par défaut
                elif col.startswith('cryptobert_'):
                    live_data[col] = 0  # Vecteur nul pour les embeddings
                else:
                    # Essayer d'utiliser une colonne similaire
                    col_prefix = col.split('_')[0]
                    similar_cols = [c for c in live_data.columns if c.startswith(col_prefix)]
                    if similar_cols:
                        live_data[col] = live_data[similar_cols[0]]
                    else:
                        live_data[col] = 0
        
        # Gérer les valeurs NaN
        for col in live_data.columns:
            if live_data[col].isna().any():
                if col in self.last_known_values:
                    # Utiliser la dernière valeur connue pour les NaN
                    live_data[col] = live_data[col].fillna(self.last_known_values[col])
                else:
                    # Utiliser la moyenne ou la médiane pour les NaN
                    live_data[col] = live_data[col].fillna(live_data[col].median() if len(live_data) > 1 else 0)
        
        # Mettre à jour les dernières valeurs connues
        for col in self.feature_columns:
            if col in live_data.columns and not live_data[col].isna().all():
                self.last_known_values[col] = live_data[col].iloc[-1]
        
        return live_data
    
    def normalize_live_data(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les données en direct en utilisant le scaler d'entraînement.
        
        Args:
            live_data: DataFrame des données en direct
            
        Returns:
            DataFrame avec les données normalisées
        """
        if self.scaler is None:
            logger.warning("Scaler non disponible. Normalisation impossible.")
            return live_data
        
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = live_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
        
        # Exclure les colonnes de label et d'identifiant
        exclude_cols = ['signal', 'market_regime', 'asset_id', 'symbol', 'timestamp', 'date']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Vérifier si toutes les colonnes nécessaires sont présentes
        missing_cols = [col for col in feature_cols if col not in live_data.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes pour la normalisation: {missing_cols}")
            # Ajouter les colonnes manquantes avec des valeurs par défaut
            for col in missing_cols:
                live_data[col] = 0
        
        # Appliquer le scaling aux caractéristiques
        live_data_scaled = live_data.copy()
        
        try:
            live_data_scaled[feature_cols] = self.scaler.transform(live_data[feature_cols])
        except Exception as e:
            logger.error(f"Erreur lors de la normalisation des données: {str(e)}")
            # Fallback: normalisation simple
            for col in feature_cols:
                if col in live_data.columns:
                    mean = live_data[col].mean()
                    std = live_data[col].std()
                    if std > 0:
                        live_data_scaled[col] = (live_data[col] - mean) / std
                    else:
                        live_data_scaled[col] = 0
        
        return live_data_scaled
    
    def prepare_model_inputs(self, live_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prépare les entrées du modèle à partir des données en direct.
        
        Args:
            live_data: DataFrame des données en direct
            
        Returns:
            Dictionnaire avec les entrées pour le modèle
        """
        # Gérer les données manquantes
        live_data = self.handle_missing_data(live_data)
        
        # Normaliser les données
        live_data_scaled = self.normalize_live_data(live_data)
        
        # Préparer les entrées du modèle
        # Utiliser les mêmes heuristiques que dans standardize_datasets
        technical_cols = [c for c in live_data_scaled.columns if c.startswith(('rsi_', 'macd_', 'bbands_', 'ema_', 'sma_', 'atr_', 'adx_')) or 
                        c in ['open', 'high', 'low', 'close', 'volume', 'returns']]
        
        llm_cols = [c for c in live_data_scaled.columns if c.startswith('cryptobert_dim_')]
        
        mcp_cols = [c for c in live_data_scaled.columns if c.startswith(('market_', 'global_', 'sentiment_'))]
        
        hmm_cols = [c for c in live_data_scaled.columns if c.startswith('hmm_')]
        
        # Factorize pour l'instrument
        if 'asset_id' in live_data_scaled.columns:
            asset_ids, _ = pd.factorize(live_data_scaled['asset_id'])
        else:
            asset_ids = np.zeros(len(live_data_scaled), dtype=np.int64)
        
        # Créer le dictionnaire d'entrée
        model_input = {
            "technical_input": live_data_scaled[technical_cols].values.astype(np.float32),
            "instrument_input": asset_ids.astype(np.int64)
        }
        
        # Ajouter les entrées optionnelles si disponibles
        if llm_cols:
            model_input["llm_input"] = live_data_scaled[llm_cols].values.astype(np.float32)
        else:
            # Fournir un vecteur nul si les embeddings LLM ne sont pas disponibles
            model_input["llm_input"] = np.zeros((len(live_data_scaled), 768), dtype=np.float32)
        
        if mcp_cols:
            model_input["mcp_input"] = live_data_scaled[mcp_cols].values.astype(np.float32)
        else:
            # Fournir un vecteur nul si les features MCP ne sont pas disponibles
            model_input["mcp_input"] = np.zeros((len(live_data_scaled), 128), dtype=np.float32)
        
        if hmm_cols:
            model_input["hmm_input"] = live_data_scaled[hmm_cols].values.astype(np.float32)
        else:
            # Fournir un vecteur nul si les features HMM ne sont pas disponibles
            model_input["hmm_input"] = np.zeros((len(live_data_scaled), 4), dtype=np.float32)
        
        return model_input
    
    def post_process_predictions(self, predictions: Dict[str, np.ndarray], live_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Post-traite les prédictions du modèle.
        
        Args:
            predictions: Dictionnaire des prédictions du modèle
            live_data: DataFrame des données en direct
            
        Returns:
            Dictionnaire avec les prédictions post-traitées
        """
        processed_predictions = {}
        
        # Traiter les prédictions de signal
        if 'signal' in predictions:
            processed_predictions['signal'] = predictions['signal']
            processed_predictions['signal_class'] = np.argmax(predictions['signal'], axis=1)
            processed_predictions['signal_confidence'] = np.max(predictions['signal'], axis=1)
        
        # Traiter les prédictions de régime de marché
        if 'market_regime' in predictions:
            processed_predictions['market_regime'] = predictions['market_regime']
            processed_predictions['market_regime_class'] = np.argmax(predictions['market_regime'], axis=1)
            processed_predictions['market_regime_confidence'] = np.max(predictions['market_regime'], axis=1)
        
        # Traiter les prédictions SL/TP
        if 'sl_tp' in predictions:
            processed_predictions['sl_tp'] = predictions['sl_tp']
            
            # Calculer les niveaux de prix pour SL/TP
            if 'close' in live_data.columns:
                close_prices = live_data['close'].values
                
                # SL/TP en pourcentage
                sl_pct = predictions['sl_tp'][:, 0]
                tp_pct = predictions['sl_tp'][:, 1]
                
                # Calculer les niveaux de prix
                processed_predictions['sl_price'] = close_prices * (1 + sl_pct)
                processed_predictions['tp_price'] = close_prices * (1 + tp_pct)
        
        # Traiter les explications si disponibles
        if 'final_reasoning' in predictions:
            processed_predictions['reasoning'] = predictions['final_reasoning']
        
        if 'attention_scores' in predictions:
            processed_predictions['attention_scores'] = predictions['attention_scores']
        
        return processed_predictions

if __name__ == "__main__":
    # Exemple d'utilisation
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestionnaire de données en direct pour Morningstar")
    parser.add_argument("--model-dir", required=True, help="Répertoire du modèle")
    parser.add_argument("--input-file", required=True, help="Fichier d'entrée (CSV ou Parquet)")
    parser.add_argument("--output-file", help="Fichier de sortie (Parquet)")
    
    args = parser.parse_args()
    
    # Charger les données
    input_path = Path(args.input_file)
    if input_path.suffix == '.csv':
        data = pd.read_csv(input_path)
    else:
        data = pd.read_parquet(input_path)
    
    # Initialiser le gestionnaire de données
    handler = LiveDataHandler(Path(args.model_dir))
    
    # Préparer les données pour le modèle
    model_inputs = handler.prepare_model_inputs(data)
    
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
