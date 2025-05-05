import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Classe pour la détection des régimes de marché à l'aide d'un modèle Hidden Markov.
    """
    def __init__(self, n_components: int = 4, covariance_type: str = "diag", 
                 n_iter: int = 100, random_state: int = 42, feature_set: str = "standard"):
        """
        Initialise le détecteur de régimes de marché.
        
        Args:
            n_components: Nombre d'états cachés (régimes) à détecter
            covariance_type: Type de matrice de covariance pour le HMM
            n_iter: Nombre maximum d'itérations pour l'entraînement
            random_state: Graine pour la reproductibilité
            feature_set: Ensemble de caractéristiques à utiliser ("standard", "extended" ou "minimal")
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.feature_set = feature_set
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.regime_descriptions = {
            0: "Marché stable/neutre",
            1: "Tendance haussière",
            2: "Tendance baissière",
            3: "Forte volatilité"
        }
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les caractéristiques pour le modèle HMM.
        
        Args:
            data: DataFrame contenant les données historiques avec colonnes:
                - close: Prix de clôture
                - high: Prix haut (optionnel)
                - low: Prix bas (optionnel)
                - volume: Volume (optionnel)
            
        Returns:
            np.ndarray: Tableau 2D des caractéristiques normalisées
            
        Raises:
            ValueError: Si la colonne 'close' est manquante
        """
        if 'close' not in data.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'close'")
            
        # Calcul des rendements logarithmiques (plus stables que pct_change)
        returns = np.log(data['close']).diff().fillna(0)
        
        # Calcul de la volatilité mobile (20 périodes)
        volatility = returns.rolling(window=20).std().fillna(0)
        
        # Calcul du ratio de Sharpe (simplifié sans taux sans risque)
        sharpe_ratio = (returns.rolling(window=20).mean() / volatility).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Initialisation du dictionnaire de features
        features_dict = {
            'returns': returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        # Ajout de caractéristiques supplémentaires selon le feature_set
        if self.feature_set in ["standard", "extended"]:
            # True Range comme mesure supplémentaire de volatilité (si high/low disponibles)
            if 'high' in data.columns and 'low' in data.columns:
                tr = pd.DataFrame({
                    'high_low': data['high'] - data['low'],
                    'high_close': (data['high'] - data['close'].shift(1)).abs(),
                    'low_close': (data['low'] - data['close'].shift(1)).abs()
                }).max(axis=1)
                features_dict['true_range'] = tr.rolling(20).mean().fillna(0)
                
            # Volume normalisé (si disponible)
            if 'volume' in data.columns:
                features_dict['volume_change'] = data['volume'].pct_change().rolling(20).mean().fillna(0)
        
        # Caractéristiques étendues pour une meilleure détection des régimes
        if self.feature_set == "extended":
            # Momentum (variation sur 14 périodes)
            features_dict['momentum'] = (data['close'] / data['close'].shift(14) - 1).fillna(0)
            
            # RSI simplifié (14 périodes)
            delta = data['close'].diff().fillna(0)
            gain = delta.where(delta > 0, 0).rolling(window=14).mean().fillna(0)
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean().fillna(0)
            rs = gain / loss.replace(0, np.finfo(float).eps)  # Éviter division par zéro
            features_dict['rsi'] = 100 - (100 / (1 + rs))
            
            # Bandes de Bollinger (écart par rapport à la moyenne mobile)
            sma_20 = data['close'].rolling(window=20).mean().fillna(0)
            std_20 = data['close'].rolling(window=20).std().fillna(0)
            features_dict['bollinger_position'] = (data['close'] - sma_20) / (2 * std_20)
            
            # Variation des prix sur différentes périodes
            for period in [5, 10, 30]:
                features_dict[f'return_{period}d'] = (data['close'] / data['close'].shift(period) - 1).fillna(0)
        
        # Combinaison des caractéristiques
        features = pd.DataFrame(features_dict).fillna(0)
        
        # Enregistrer les noms des features pour référence
        self.feature_names = list(features.columns)
        logger.info(f"Features préparées pour HMM: {', '.join(self.feature_names)}")
        
        return features.values  # Retourne directement le array numpy
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Entraîne le modèle HMM sur les données fournies.
        
        Args:
            data: DataFrame contenant les données historiques
        """
        features = self._prepare_features(data)
        
        # Scaling des features pour de meilleurs résultats
        scaled_features = self.scaler.fit_transform(features)
        
        try:
            # Initialisation et entraînement du modèle GaussianHMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=True
            )
            
            # Entraînement avec gestion des erreurs
            self.model.fit(scaled_features)
            
            logger.info(f"Modèle HMM entraîné avec succès (score: {self.model.score(scaled_features):.2f})")
            logger.info(f"Nombre d'itérations: {self.model.monitor_.iter}")
            
            # Analyse des régimes détectés
            self._analyze_regimes(data, scaled_features)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle HMM: {e}")
            raise
    
    def _analyze_regimes(self, data: pd.DataFrame, scaled_features: np.ndarray) -> None:
        """
        Analyse les régimes détectés pour les caractériser.
        
        Args:
            data: DataFrame original des données
            scaled_features: Features scalées utilisées pour l'entraînement
        """
        if self.model is None:
            return
            
        # Prédire les régimes
        regimes = self.model.predict(scaled_features)
        
        # Analyser les caractéristiques de chaque régime
        regime_stats = {}
        for i in range(self.n_components):
            # Indices des points de données dans ce régime
            regime_indices = np.where(regimes == i)[0]
            if len(regime_indices) == 0:
                continue
                
            # Calculer les statistiques pour ce régime
            regime_returns = np.log(data['close']).diff().fillna(0).iloc[regime_indices].mean()
            regime_volatility = np.log(data['close']).diff().fillna(0).iloc[regime_indices].std()
            
            # Stocker les statistiques
            regime_stats[i] = {
                'count': len(regime_indices),
                'percentage': len(regime_indices) / len(regimes) * 100,
                'avg_return': regime_returns,
                'volatility': regime_volatility,
                'description': self._get_regime_description(regime_returns, regime_volatility)
            }
            
            # Mettre à jour la description du régime
            self.regime_descriptions[i] = regime_stats[i]['description']
        
        # Afficher les statistiques
        for regime, stats in regime_stats.items():
            logger.info(f"Régime {regime}: {stats['count']} points ({stats['percentage']:.1f}%), "  
                      f"Rendement moyen: {stats['avg_return']:.4f}, Volatilité: {stats['volatility']:.4f}, "  
                      f"Description: {stats['description']}")
    
    def _get_regime_description(self, avg_return: float, volatility: float) -> str:
        """
        Génère une description du régime basée sur ses caractéristiques.
        
        Args:
            avg_return: Rendement moyen du régime
            volatility: Volatilité du régime
            
        Returns:
            str: Description du régime
        """
        if volatility > 0.02:  # Seuil de forte volatilité
            if avg_return > 0.001:
                return "Marché haussier volatil"
            elif avg_return < -0.001:
                return "Marché baissier volatil"
            else:
                return "Marché très volatil (range-bound)"
        else:  # Volatilité normale ou faible
            if avg_return > 0.0005:
                return "Tendance haussière stable"
            elif avg_return < -0.0005:
                return "Tendance baissière stable"
            else:
                return "Marché stable/neutre"
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prédit les régimes de marché pour les données fournies.
        
        Args:
            data: DataFrame contenant les données historiques
            
        Returns:
            np.ndarray: Tableau 1D des régimes prédits
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de prédire les régimes")
            
        features = self._prepare_features(data)
        scaled_features = self.scaler.transform(features)
        return self.model.predict(scaled_features)
        
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Obtient les probabilités des régimes pour les données fournies.
        
        Args:
            data: DataFrame contenant les données historiques
            
        Returns:
            np.ndarray: Tableau 2D des probabilités de régime
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de prédire les régimes")
            
        features = self._prepare_features(data)
        scaled_features = self.scaler.transform(features)
        return self.model.predict_proba(scaled_features)
        
    def get_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtient toutes les caractéristiques de régime pour les données fournies.
        
        Args:
            data: DataFrame contenant les données historiques
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant:
                - 'regimes': les régimes prédits
                - 'probabilities': les probabilités de régime
                - 'descriptions': descriptions des régimes
        """
        regimes = self.predict(data)
        probabilities = self.predict_proba(data)
        
        # Créer un mapping des régimes vers leurs descriptions
        regime_descriptions = {i: self.regime_descriptions.get(i, f"Régime {i}") for i in range(self.n_components)}
        
        return {
            'regimes': regimes,
            'probabilities': probabilities,
            'descriptions': regime_descriptions
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modèle HMM et le scaler dans un fichier.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if self.model is None:
            logger.error("Impossible de sauvegarder un modèle non entraîné")
            return False
            
        try:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'feature_set': self.feature_set,
                'feature_names': self.feature_names,
                'regime_descriptions': self.regime_descriptions
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Modèle HMM sauvegardé avec succès dans {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle HMM: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MarketRegimeDetector':
        """
        Charge un modèle HMM et un scaler depuis un fichier.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            MarketRegimeDetector: Instance avec le modèle chargé
            
        Raises:
            ValueError: Si le fichier n'existe pas ou est invalide
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            # Créer une nouvelle instance
            detector = cls(
                n_components=model_data.get('n_components', 4),
                covariance_type=model_data.get('covariance_type', 'diag'),
                feature_set=model_data.get('feature_set', 'standard')
            )
            
            # Charger le modèle et le scaler
            detector.model = model_data.get('model')
            detector.scaler = model_data.get('scaler')
            detector.feature_names = model_data.get('feature_names', [])
            detector.regime_descriptions = model_data.get('regime_descriptions', {})
            
            logger.info(f"Modèle HMM chargé avec succès depuis {filepath}")
            return detector
        except FileNotFoundError:
            logger.error(f"Fichier de modèle non trouvé: {filepath}")
            raise ValueError(f"Fichier de modèle non trouvé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle HMM: {e}")
            raise ValueError(f"Erreur lors du chargement du modèle: {e}")
