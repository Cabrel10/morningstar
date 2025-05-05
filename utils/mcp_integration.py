import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

# Configuration
CACHE_DIR = Path("data/mcp_cache")
CACHE_EXPIRY_SECONDS = 300  # 5 minutes pour les données live
ORDER_BOOK_DEPTH = 10  # Nombre de niveaux de profondeur à récupérer
FEATURE_BLOCK_SIZE = 32  # Taille de chaque bloc de features (Order Book, On-chain, etc.)
TOTAL_MCP_DIMENSIONS = 128  # 4 blocs * 32 features

logger = logging.getLogger(__name__)


class MCPIntegration:
    """
    Récupère, calcule et met en cache les features MCP (Market Context Protocol)
    provenant de diverses sources (Order Book, On-chain, Social, Macro).
    """

    def __init__(self, exchange_id: str = "binance"):
        """
        Initialise l'intégration MCP.

        Args:
            exchange_id: L'ID de l'exchange CCXT à utiliser (ex: 'binance', 'kraken').
        """
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.exchange = getattr(ccxt, exchange_id)()
            if not self.exchange.has["fetchOrderBook"]:
                logger.warning(f"L'exchange {exchange_id} ne supporte pas fetchOrderBook via CCXT.")
                self.exchange = None
            else:
                logger.info(f"CCXT initialisé pour l'exchange: {exchange_id}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de CCXT pour {exchange_id}: {e}")
            self.exchange = None

        # TODO: Initialiser les clients API pour On-chain (Glassnode?), Social (Twitter?)
        self.onchain_client = None  # Placeholder
        self.social_client = None  # Placeholder

    def _get_cache_path(self, feature_type: str, symbol: str) -> Path:
        """Retourne le chemin du fichier cache pour un type de feature et un symbole."""
        return self.cache_dir / f"{feature_type}_{symbol.replace('/', '_')}.json"

    def _read_cache(self, feature_type: str, symbol: str) -> Optional[Dict]:
        """Lit les données depuis le cache si elles sont valides."""
        cache_path = self._get_cache_path(feature_type, symbol)
        if cache_path.exists():
            try:
                mtime = cache_path.stat().st_mtime
                if (datetime.now().timestamp() - mtime) < CACHE_EXPIRY_SECONDS:
                    with open(cache_path, "r") as f:
                        data = json.load(f)
                    logger.debug(f"Cache HIT pour {feature_type} - {symbol}")
                    return data
                else:
                    logger.debug(f"Cache EXPIRED pour {feature_type} - {symbol}")
            except Exception as e:
                logger.error(f"Erreur de lecture du cache {cache_path}: {e}")
        logger.debug(f"Cache MISS pour {feature_type} - {symbol}")
        return None

    def _write_cache(self, feature_type: str, symbol: str, data: Dict):
        """Écrit les données dans le cache."""
        cache_path = self._get_cache_path(feature_type, symbol)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cache WRITTEN pour {feature_type} - {symbol}")
        except Exception as e:
            logger.error(f"Erreur d'écriture du cache {cache_path}: {e}")

    # --- Méthodes pour chaque bloc de features ---

    def _get_order_book_features(self, symbol: str) -> np.ndarray:
        """
        Récupère les données du carnet d'ordres et calcule les 32 features associées.
        """
        features = np.zeros(FEATURE_BLOCK_SIZE, dtype=np.float32)
        cached_data = self._read_cache("order_book", symbol)

        if cached_data:
            # TODO: Calculer les features depuis cached_data
            # Exemple simple: utiliser les prix/volumes stockés
            try:
                # Simuler l'extraction de quelques valeurs pour l'exemple
                features[0] = cached_data.get("best_bid", 0)
                features[1] = cached_data.get("best_ask", 0)
                features[2] = cached_data.get("spread", 0)
                # ... remplir les 32 features
                logger.info(f"Features Order Book chargées depuis le cache pour {symbol}")
                return features  # Retourner les features du cache
            except Exception as e:
                logger.error(f"Erreur lors du traitement des données Order Book du cache: {e}")
                # Continuer pour essayer de récupérer les données live

        if not self.exchange:
            logger.warning("CCXT non disponible, impossible de récupérer les données Order Book live.")
            return features  # Retourne des zéros

        try:
            logger.debug(f"Récupération live Order Book pour {symbol}...")
            order_book = self.exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)

            # Extraire les données pertinentes
            best_bid = order_book["bids"][0][0] if order_book["bids"] else 0
            best_ask = order_book["asks"][0][0] if order_book["asks"] else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0

            # TODO: Calculer les 32 features réelles (imbalance, profondeur cumulée, etc.)
            # Placeholder: juste quelques valeurs
            features[0] = best_bid
            features[1] = best_ask
            features[2] = spread
            # ... autres features ...

            # Mettre en cache les données brutes ou calculées
            cache_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                # Ajouter d'autres données brutes si nécessaire pour le cache
                "bids": order_book["bids"],
                "asks": order_book["asks"],
            }
            self._write_cache("order_book", symbol, cache_data)
            logger.info(f"Features Order Book calculées live pour {symbol}")

        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau CCXT (Order Book) pour {symbol}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur d'exchange CCXT (Order Book) pour {symbol}: {e}")
        except Exception as e:
            logger.exception(f"Erreur inattendue (Order Book) pour {symbol}: {e}")

        return features

    def _get_onchain_features(self, symbol: str) -> np.ndarray:
        """
        [Placeholder] Récupère les données on-chain et calcule les 32 features associées.
        Nécessite une API externe (ex: Glassnode) et une clé API.
        """
        features = np.zeros(FEATURE_BLOCK_SIZE, dtype=np.float32)
        logger.warning("Fonctionnalité On-chain non implémentée (nécessite API externe). Retourne des zéros.")
        # TODO: Implémenter la logique d'appel API et de calcul des features on-chain
        # Exemple:
        # if self.onchain_client:
        #     data = self.onchain_client.get_metrics(symbol, ...)
        #     features[0] = data.get('tx_count_24h', 0)
        #     ...
        return features

    def _get_social_features(self, symbol: str) -> np.ndarray:
        """
        [Placeholder] Récupère les données de sentiment social et calcule les 32 features associées.
        Nécessite une API externe (ex: Twitter API) et une clé API.
        """
        features = np.zeros(FEATURE_BLOCK_SIZE, dtype=np.float32)
        logger.warning("Fonctionnalité Social Sentiment non implémentée (nécessite API externe). Retourne des zéros.")
        # TODO: Implémenter la logique d'appel API et de calcul des features sociales
        # Exemple:
        # if self.social_client:
        #     data = self.social_client.get_sentiment(symbol, ...)
        #     features[0] = data.get('sentiment_score', 0.5) # Neutre par défaut
        #     ...
        return features

    def _get_macro_exchange_features(self, symbol: str) -> np.ndarray:
        """
        [Placeholder] Récupère les données macro/exchange et calcule les 32 features associées.
        Peut utiliser CCXT pour funding rates, open interest, et d'autres API pour Fear&Greed, etc.
        """
        features = np.zeros(FEATURE_BLOCK_SIZE, dtype=np.float32)
        logger.warning("Fonctionnalité Macro/Exchange partiellement implémentée. Retourne des zéros pour l'instant.")

        # Exemple avec CCXT (si disponible et supporté)
        if self.exchange:
            try:
                # Funding Rate (pour les marchés futures/perp)
                if self.exchange.has["fetchFundingRate"]:
                    # Le symbole doit être celui du marché future (ex: BTC/USDT:USDT)
                    perp_symbol = f"{symbol}:USDT" if ":" not in symbol else symbol
                    try:
                        funding_rate_data = self.exchange.fetch_funding_rate(perp_symbol)
                        features[0] = funding_rate_data.get("fundingRate", 0)
                        logger.debug(f"Funding rate pour {perp_symbol}: {features[0]}")
                    except ccxt.BadSymbol:
                        logger.warning(f"Symbole {perp_symbol} non trouvé pour fetchFundingRate.")
                    except Exception as e:
                        logger.error(f"Erreur fetchFundingRate pour {perp_symbol}: {e}")

                # Open Interest (pour les marchés futures/perp)
                if self.exchange.has["fetchOpenInterest"]:
                    perp_symbol = f"{symbol}:USDT" if ":" not in symbol else symbol
                    try:
                        open_interest_data = self.exchange.fetch_open_interest(perp_symbol)
                        features[1] = open_interest_data.get("openInterestAmount", 0)  # Ou 'openInterestValue'
                        logger.debug(f"Open interest pour {perp_symbol}: {features[1]}")
                    except ccxt.BadSymbol:
                        logger.warning(f"Symbole {perp_symbol} non trouvé pour fetchOpenInterest.")
                    except Exception as e:
                        logger.error(f"Erreur fetchOpenInterest pour {perp_symbol}: {e}")

            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données Macro/Exchange via CCXT: {e}")

        # TODO: Ajouter appels API pour Fear & Greed Index, Volatilité Implicite, etc.
        # features[2] = get_fear_greed_index() ...

        return features

    # --- Méthode principale ---

    def get_mcp_features(self, symbol: str, timestamp: Optional[datetime] = None) -> np.ndarray:
        """
        Récupère et assemble les 128 features MCP pour un symbole donné.
        Utilise le cache pour éviter les appels API redondants.

        Args:
            symbol: Le symbole de marché (ex: 'BTC/USDT').
            timestamp: Le timestamp pour lequel récupérer les features (peut être utilisé par certaines sources). Non utilisé actuellement.

        Returns:
            Un array numpy de shape (128,) contenant les features MCP, ou un vecteur de zéros en cas d'erreur.
        """
        logger.info(f"Récupération des {TOTAL_MCP_DIMENSIONS} features MCP pour {symbol}...")

        # Extraire le symbole de base si nécessaire (pour certaines API on-chain/social)
        base_symbol = symbol.split("/")[0] if "/" in symbol else symbol

        # Récupérer chaque bloc de features
        order_book_f = self._get_order_book_features(symbol)
        onchain_f = self._get_onchain_features(base_symbol)  # Utilise base_symbol
        social_f = self._get_social_features(base_symbol)  # Utilise base_symbol
        macro_exch_f = self._get_macro_exchange_features(symbol)

        # Concaténer les blocs
        # S'assurer que chaque bloc a la bonne taille (FEATURE_BLOCK_SIZE)
        # Si une fonction retourne un array de mauvaise taille (ex: erreur), elle retourne des zéros de la bonne taille.
        all_features = np.concatenate([order_book_f, onchain_f, social_f, macro_exch_f])

        if all_features.shape[0] != TOTAL_MCP_DIMENSIONS:
            logger.error(
                f"Erreur: Le nombre total de features MCP assemblées ({all_features.shape[0]}) ne correspond pas à {TOTAL_MCP_DIMENSIONS}. Retourne des zéros."
            )
            return np.zeros(TOTAL_MCP_DIMENSIONS, dtype=np.float32)

        # Remplacer les NaN potentiels par 0 (mesure de sécurité)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Features MCP récupérées avec succès pour {symbol}.")
        return all_features.astype(np.float32)


# Exemple d'utilisation (pour test)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    mcp_integrator = MCPIntegration(exchange_id="binance")

    # Tester pour BTC/USDT
    btc_features = mcp_integrator.get_mcp_features("BTC/USDT")
    print("\n--- Features MCP pour BTC/USDT ---")
    print(f"Shape: {btc_features.shape}")
    # Afficher les premières features de chaque bloc pour vérification
    print("Order Book (premières 5):", btc_features[0:5])
    print("On-chain (premières 5):", btc_features[32:37])
    print("Social (premières 5):", btc_features[64:69])
    print("Macro/Exch (premières 5):", btc_features[96:101])

    # Tester pour un autre symbole
    # eth_features = mcp_integrator.get_mcp_features('ETH/USDT')
    # print("\n--- Features MCP pour ETH/USDT ---")
    # print(f"Shape: {eth_features.shape}")
    # print(eth_features[0:5]) # Premières 5
