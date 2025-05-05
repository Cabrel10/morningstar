# utils/live_preprocessing.py

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Importer les dépendances nécessaires
from utils.feature_engineering import apply_feature_pipeline
from utils.llm_integration import LLMIntegration
from utils.mcp_integration import \
    MCPIntegration  # Importer la nouvelle classe MCP

logger = logging.getLogger(__name__)

# Constantes
MIN_BUFFER_SIZE = 100  # Taille minimale du buffer pour calculer tous les indicateurs (à ajuster précisément)
LLM_EMBEDDING_DIM = 768
MCP_INPUT_DIM = 128

# Liste explicite des 38 features techniques attendues par le modèle
# (Basée sur l'implémentation dans utils.feature_engineering.apply_feature_pipeline)
TECHNICAL_FEATURE_COLUMNS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",  # Base (5)
    "SMA_short",
    "SMA_long",
    "EMA_short",
    "EMA_long",  # Moyennes (4)
    "RSI",  # Momentum (1)
    "MACD",
    "MACDs",
    "MACDh",  # MACD (3)
    "BBU",
    "BBM",
    "BBL",  # Bollinger (3)
    "ATR",  # Volatilité (1)
    "STOCHk",
    "STOCHd",  # Stochastique (2)
    "ADX",
    "CCI",
    "Momentum",
    "ROC",
    "Williams_%R",
    "TRIX",
    "Ultimate_Osc",
    "DPO",  # Momentum Add. (8)
    "OBV",
    "VWMA",
    "CMF",
    "MFI",  # Volume (4)
    "Parabolic_SAR",  # Avancé (1)
    "Ichimoku_Tenkan",
    "Ichimoku_Kijun",
    "Ichimoku_SenkouA",
    "Ichimoku_SenkouB",
    "Ichimoku_Chikou",  # Ichimoku (5)
    "KAMA",  # Adaptative (1)
]
# Vérification rapide du nombre
if len(TECHNICAL_FEATURE_COLUMNS) != 38:
    raise ImportError("La liste TECHNICAL_FEATURE_COLUMNS ne contient pas 38 éléments.")
NUM_TECH_FEATURES = 38


class LiveDataPreprocessor:
    """
    Gère les données de marché en temps réel (OHLCV), calcule les indicateurs techniques
    et prépare les inputs pour le modèle de prédiction live.
    """

    def __init__(self, window_size: int = MIN_BUFFER_SIZE, symbol: str = "BTC/USDT"):
        """
        Initialise le preprocessor.

        Args:
            window_size: La taille minimale requise du buffer pour calculer les indicateurs.
            symbol: Le symbole de la paire tradée (pour logging/contexte).
        """
        self.symbol = symbol
        # Le buffer stockera des dicts OHLCV {'timestamp': ms, 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}
        self.raw_data_buffer = deque(maxlen=window_size * 2)  # Garder une marge
        self.indicator_window_size = window_size
        # DataFrame pour stocker les données avec les indicateurs calculés
        self.data_with_indicators = pd.DataFrame()
        self.last_processed_timestamp = 0

        # Initialiser l'instance LLMIntegration pour charger depuis le cache
        self.llm_integration = LLMIntegration()
        # Initialiser l'instance MCPIntegration
        # TODO: Rendre l'exchange_id configurable si nécessaire
        self.mcp_integration = MCPIntegration(exchange_id="binance")
        # Extraire le symbole de base (ex: BTC de BTC/USDT) pour la recherche dans le cache LLM/MCP
        self.base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
        # Garder le symbole complet pour les appels CCXT (ex: Order Book)
        self.full_symbol = symbol

        logger.info(f"LiveDataPreprocessor initialisé pour {self.symbol} avec window_size={self.indicator_window_size}")

    def add_data(self, data: Dict) -> bool:
        """
        Ajoute une nouvelle donnée OHLCV au buffer brut.

        Args:
            data: Dictionnaire contenant les données OHLCV d'une bougie.
                  Doit contenir 'timestamp', 'open', 'high', 'low', 'close', 'volume'.

        Returns:
            True si la donnée est nouvelle et ajoutée, False sinon.
        """
        timestamp = data.get("timestamp")
        if timestamp is None or timestamp <= self.last_processed_timestamp:
            return False

        required_keys = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(key in data for key in required_keys):
            logger.warning(f"Donnée OHLCV reçue incomplète pour timestamp {timestamp}, clés manquantes: {data.keys()}")
            return False

        self.raw_data_buffer.append(data)
        self.last_processed_timestamp = timestamp
        return True

    def _update_indicators(self) -> bool:
        """
        Met à jour le DataFrame avec les indicateurs techniques en utilisant
        la fonction apply_feature_pipeline.

        Returns:
            True si les indicateurs ont été mis à jour et valides, False sinon.
        """
        if len(self.raw_data_buffer) < self.indicator_window_size:
            # Pas assez de données pour calculer les indicateurs avec la fenêtre requise
            return False

        # Convertir le deque en DataFrame pour le calcul
        df_raw = pd.DataFrame(list(self.raw_data_buffer))
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
        df_raw = df_raw.set_index("timestamp")
        df_raw = df_raw.sort_index()  # Assurer l'ordre chronologique

        # Vérifier si les colonnes OHLCV nécessaires sont présentes
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df_raw.columns for col in required_cols):
            logger.error("Le DataFrame brut ne contient pas toutes les colonnes OHLCV nécessaires après conversion.")
            return False

        try:
            # Appeler la fonction de feature engineering
            # Passer include_llm=False car géré séparément en live
            df_processed = apply_feature_pipeline(df_raw.copy(), include_llm=False)

            # La fonction apply_feature_pipeline effectue déjà des validations (NaN, variance, nombre de colonnes)
            # Si elle réussit sans lever d'erreur, on peut supposer que les données sont valides.
            self.data_with_indicators = df_processed
            logger.debug(f"Indicateurs mis à jour via apply_feature_pipeline. Shape: {self.data_with_indicators.shape}")
            return not self.data_with_indicators.empty

        except ValueError as ve:
            # Capturer les erreurs de validation levées par apply_feature_pipeline
            logger.error(f"Erreur de validation lors du calcul des features live: {ve}")
            self.data_with_indicators = pd.DataFrame()  # Vider en cas d'erreur
            return False
        except Exception as e:
            logger.exception(f"Erreur inattendue lors de l'appel à apply_feature_pipeline: {e}")
            self.data_with_indicators = pd.DataFrame()  # Vider en cas d'erreur
            return False

    def _get_llm_mcp_inputs(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Récupère les inputs LLM (embeddings) depuis le cache local et les inputs MCP
        (actuellement placeholder) pour le dernier point de données.
        """
        llm_input = None

        # 1. Récupérer l'embedding LLM depuis le cache
        if not self.data_with_indicators.empty:
            last_timestamp = self.data_with_indicators.index[-1]
            # S'assurer que le timestamp est bien un objet datetime pour extraire la date
            if isinstance(last_timestamp, pd.Timestamp):
                date_str = last_timestamp.strftime("%Y-%m-%d")
                logger.debug(f"Recherche de l'embedding news pour {self.base_symbol} à la date {date_str}")
                llm_input = self.llm_integration.get_cached_embedding(
                    symbol=self.base_symbol, date_str=date_str, embedding_type="news"
                )
            else:
                logger.error(f"Le dernier index du DataFrame n'est pas un timestamp valide: {last_timestamp}")

        # 2. Gérer le cas où l'embedding n'est pas trouvé (fallback)
        if llm_input is None:
            logger.warning(
                f"Embedding LLM non trouvé dans le cache pour {self.base_symbol} à la date {date_str}. Utilisation d'un vecteur de zéros."
            )
            llm_input = np.zeros(LLM_EMBEDDING_DIM, dtype=np.float32)
        elif llm_input.shape[0] != LLM_EMBEDDING_DIM:
            logger.error(
                f"Dimension de l'embedding LLM chargé ({llm_input.shape[0]}) incorrecte. Attendu: {LLM_EMBEDDING_DIM}. Utilisation d'un vecteur de zéros."
            )
            llm_input = np.zeros(LLM_EMBEDDING_DIM, dtype=np.float32)

        # 3. Récupérer l'input MCP en utilisant MCPIntegration
        try:
            # Utiliser le symbole complet pour les features qui en dépendent (ex: Order Book)
            mcp_input = self.mcp_integration.get_mcp_features(symbol=self.full_symbol)
            if mcp_input.shape[0] != MCP_INPUT_DIM:
                logger.error(
                    f"Dimension de l'input MCP retourné ({mcp_input.shape[0]}) incorrecte. Attendu: {MCP_INPUT_DIM}. Utilisation d'un vecteur de zéros."
                )
                mcp_input = np.zeros(MCP_INPUT_DIM, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des features MCP pour {self.full_symbol}: {e}")
            mcp_input = np.zeros(MCP_INPUT_DIM, dtype=np.float32)  # Fallback en cas d'erreur

        return llm_input, mcp_input

    def get_model_input(self) -> Optional[Tuple[Dict[str, np.ndarray], float]]:
        """
        Prépare et retourne les inputs requis par le modèle et la dernière valeur ATR
        pour la dernière donnée disponible après mise à jour des indicateurs.

        Returns:
            Tuple contenant:
                - Dictionnaire d'inputs modèle ('technical_input', 'llm_input', 'mcp_input').
                - Dernière valeur de l'ATR (float).
            Ou None si les données sont insuffisantes ou l'ATR n'est pas valide.
        """
        # 1. Mettre à jour les indicateurs
        if not self._update_indicators():
            logger.warning("Impossible de générer l'input modèle: indicateurs non à jour ou buffer insuffisant.")
            return None

        # 2. Récupérer la dernière ligne de données avec les indicateurs calculés
        if self.data_with_indicators.empty:
            logger.warning("DataFrame d'indicateurs est vide après mise à jour.")
            return None
        last_data_point = self.data_with_indicators.iloc[-1]

        # 3. Extraire les 38 features techniques définies
        try:
            # Sélectionner les colonnes dans l'ordre défini par TECHNICAL_FEATURE_COLUMNS
            technical_features = last_data_point[TECHNICAL_FEATURE_COLUMNS].values.astype(np.float32)
            if technical_features.shape[0] != NUM_TECH_FEATURES:
                logger.error(
                    f"Nombre incorrect de features techniques extraites ({technical_features.shape[0]} vs {NUM_TECH_FEATURES}). Colonnes disponibles: {last_data_point.index.tolist()}"
                )
                return None
        except KeyError as e:
            logger.error(
                f"Colonne technique manquante dans les données traitées: {e}. Colonnes disponibles: {last_data_point.index.tolist()}"
            )
            return None
        except Exception as e:
            logger.exception(f"Erreur lors de l'extraction des features techniques: {e}")
            return None

        # 4. Extraire la dernière valeur ATR
        try:
            last_atr_value = float(last_data_point["ATR"])
            if pd.isna(last_atr_value) or last_atr_value <= 0:
                logger.warning(
                    f"Valeur ATR invalide ({last_atr_value}) pour le timestamp {last_data_point.name}. Impossible de générer l'input."
                )
                return None
        except KeyError:
            logger.error(
                f"Colonne 'ATR' manquante dans les données traitées. Colonnes: {last_data_point.index.tolist()}"
            )
            return None
        except Exception as e:
            logger.exception(f"Erreur lors de l'extraction de l'ATR: {e}")
            return None

        # 5. Récupérer les inputs LLM et MCP (placeholders actuels)
        llm_input, mcp_input = self._get_llm_mcp_inputs()

        # 6. Construire le dictionnaire final
        model_inputs = {
            "technical_input": technical_features,  # Shape: (NUM_TECH_FEATURES,)
            "llm_input": llm_input,  # Shape: (LLM_EMBEDDING_DIM,)
        }
        if mcp_input is not None:
            # S'assurer que l'input MCP est optionnel si le modèle le gère
            model_inputs["mcp_input"] = mcp_input  # Shape: (MCP_INPUT_DIM,)

        logger.info(f"Input modèle et ATR ({last_atr_value:.4f}) générés pour le timestamp: {last_data_point.name}")
        return model_inputs, last_atr_value
