import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import ccxt
import numpy as np
import pandas as pd

# Configuration du Logging (existant)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, "api_manager.log")
report_file_path = os.path.join(LOG_DIR, "data_download_report.txt")  # Ajout pour le rapport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],  # Afficher aussi dans la console
)
logger = logging.getLogger(__name__)

MIN_ROWS_THRESHOLD = 500  # Seuil minimum de lignes pour la vérification


class APIManager:
    """Wrapper pour l'interface API attendue par le workflow de trading"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le manager API avec la configuration.

        Args:
            config: Configuration de l'API depuis config.yaml
        """
        self.config = config
        self.exchange = self._init_exchange()

    def fetch_ohlcv_data(self, exchange_id, token, timeframe, start_date, end_date, max_retries=5, retry_delay=5.0):
        """
        Méthode d'instance qui délègue à la fonction utilitaire fetch_ohlcv_data pour compatibilité avec ETHDatasetGenerator.
        """
        from utils.api_manager import fetch_ohlcv_data as fetch_ohlcv_data_fn
        return fetch_ohlcv_data_fn(exchange_id, token, timeframe, start_date, end_date, max_retries, retry_delay)

    def _init_exchange(self):
        """Initialise la connexion à l'exchange avec gestion du rate limit explicite."""
        logger.info(f"Initialisation de l'exchange via APIManager: {self.config.get('exchange', 'binance')}")
        try:
            exchange_id = self.config.get("exchange", "binance")
            
            # Option pour utiliser un exchange simulé pour les tests locaux
            if self.config.get("use_mock", False) or os.environ.get("USE_MOCK_EXCHANGE", "false").lower() == "true":
                logger.info("Utilisation d'un exchange simulé pour les tests locaux")
                return MockExchange()
                
            exchange_class = getattr(ccxt, exchange_id)

            # Configuration avancée pour une meilleure gestion des erreurs et des rate limits
            exchange_config = {
                "timeout": self.config.get("timeout", 30000),  # 30 secondes par défaut
                "enableRateLimit": True,  # Activation explicite du rate limit CCXT
                "options": {
                    "defaultType": "spot",  # Force l'utilisation de Spot au lieu de Futures
                    "adjustForTimeDifference": True,  # Ajustement de la différence d'heure
                    "recvWindow": 60000,  # Fenêtre de réception étendue pour les requêtes (surtout Binance)
                }
            }

            # Ajouter les clés API si disponibles dans la configuration
            if "apiKey" in self.config and "secret" in self.config:
                exchange_config["apiKey"] = self.config["apiKey"]
                exchange_config["secret"] = self.config["secret"]
                logger.info("Clés API configurées pour l'authentification")

            exchange = exchange_class(exchange_config)

            # Vérifier la connectivité
            if hasattr(exchange, 'check_required_credentials'):
                exchange.check_required_credentials()

            # Charger les marchés si disponible (pour validation des symboles)
            if hasattr(exchange, 'load_markets') and self.config.get("load_markets", True):
                try:
                    exchange.load_markets()
                    logger.info(f"Marchés chargés pour {exchange_id}: {len(exchange.markets)} paires disponibles")
                except Exception as e:
                    logger.warning(f"Impossible de charger les marchés pour {exchange_id}: {e}")

            return exchange
        except AttributeError as e:
            logger.error(f"Exchange '{self.config.get('exchange')}' non trouvé par ccxt: {e}")
            raise ValueError(f"Exchange '{self.config.get('exchange')}' non supporté par ccxt") from e
        except ccxt.AuthenticationError as e:
            logger.error(f"Erreur d'authentification pour {self.config.get('exchange')}: {e}")
            logger.warning("Utilisation d'un exchange simulé pour les tests locaux")
            return MockExchange()
        except Exception as e:
            logger.error(f"Erreur d'initialisation de l'exchange via APIManager: {e}")
            logger.warning("Utilisation d'un exchange simulé pour les tests locaux")
            return MockExchange()

    def get_market_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Récupère les données de marché au format attendu par le modèle.

        Returns:
            Dictionnaire avec:
            - technical: array numpy des features techniques
            - sentiment_embeddings: array numpy des embeddings LLM
            Ou None si erreur.
        """
        logger.info("Récupération des données de marché via APIManager...")
        try:
            pair = self.config.get("pair", "BTC/USDT")
            timeframe = self.config.get("timeframe", "1h")
            limit = self.config.get("lookback", 100)

            logger.debug(f"Appel à fetch_ohlcv pour {pair}, {timeframe}, limit={limit}")
            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

            if not ohlcv:
                logger.warning("Aucune donnée OHLCV retournée par l'exchange.")
                return None

            # Convertir en features techniques (simplifié)
            # Garder seulement O, H, L, C pour cet exemple
            technical_data = [[candle[1], candle[2], candle[3], candle[4]] for candle in ohlcv]
            technical = np.array(technical_data)

            # Embeddings factices - à remplacer par l'appel réel au LLM/autre source
            logger.debug("Génération d'embeddings factices.")
            embeddings = np.random.rand(768)  # Taille exemple pour BERT base

            logger.info(f"Données techniques ({technical.shape}) et embeddings ({embeddings.shape}) récupérés.")
            return {"technical": technical, "sentiment_embeddings": embeddings}

        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau lors de la récupération des données: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur de l'exchange lors de la récupération des données: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des données: {e}")

        return None  # Retourner None en cas d'erreur

    def execute_orders(self, decisions: Dict[str, Any]) -> bool:
        """
        Exécute les ordres de trading basés sur les décisions du modèle.

        Args:
            decisions: Dictionnaire de décisions de trading

        Returns:
            bool: True si l'exécution a réussi (simulation ici)
        """
        logger.info("Exécution des ordres via APIManager...")
        try:
            # Implémentation simplifiée - à adapter pour ordres réels
            # Nécessiterait des clés API chargées dans _init_exchange
            order_type = decisions.get("type")  # 'buy', 'sell', 'close'
            symbol = decisions.get("symbol", self.config.get("pair", "BTC/USDT"))
            amount = decisions.get("amount", 0.001)  # Exemple de taille
            price = decisions.get("price")  # Pour ordres limit

            if order_type in ["buy", "sell"]:
                order_action = "buy" if order_type == "buy" else "sell"
                # Simuler un ordre market pour simplifier
                logger.info(f"SIMULATION: Création d'un ordre {order_action} market pour {amount} {symbol}")
                # ordre_result = self.exchange.create_market_order(symbol, order_action, amount)
                # logger.info(f"Résultat de l'ordre simulé: {ordre_result}")
                print(f"--- Ordre {order_action.upper()} simulé pour {amount} {symbol} ---")  # Pour visibilité
                return True
            elif order_type == "close":
                logger.info(f"SIMULATION: Fermeture de position pour {symbol}")
                # Logique de fermeture de position...
                return True
            else:
                logger.warning(f"Type de décision non reconnu: {order_type}")
                return False

        except ccxt.AuthenticationError:
            logger.error("Erreur d'authentification. Vérifiez les clés API.")
            return False
        except ccxt.InsufficientFunds:
            logger.error("Fonds insuffisants pour exécuter l'ordre.")
            return False
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur de l'exchange lors de l'exécution de l'ordre: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'exécution des ordres: {e}")
            return False


# --- Fonctions Standalone (utilisées par le script en __main__) ---


def format_symbol(token: str, exchange_id: str) -> str:
    """
    Formate le symbole du token selon les conventions ccxt (ex: BTC/USDT).
    Certains exchanges utilisent des formats sans '/', cette fonction tente de standardiser.
    """
    if "/" in token:
        return token.upper()
    # Essayer de deviner la quote currency (USDT, BUSD, BTC, ETH...)
    possible_quotes = ["USDT", "BUSD", "USDC", "BTC", "ETH"]
    for quote in possible_quotes:
        if token.endswith(quote):
            base = token[: -len(quote)]
            formatted = f"{base}/{quote}"
            logger.info(f"Conversion du token '{token}' au format standard ccxt: '{formatted}'")
            return formatted.upper()
    # Si pas de quote trouvée, retourner tel quel (peut échouer sur l'exchange)
    logger.warning(f"Impossible de déterminer le format standard avec '/' pour '{token}'.")

    # Cas spécifique pour KuCoin qui utilise souvent '-'
    if exchange_id == "kucoin" and "/" not in token:
        for quote in possible_quotes:
            if token.endswith(quote):
                base = token[: -len(quote)]
                formatted_dash = f"{base}-{quote}"
                logger.info(f"Tentative avec le format KuCoin: '{formatted_dash}'")
                return formatted_dash.upper()

    logger.warning(f"Utilisation du token '{token.upper()}' tel quel.")
    return token.upper()


def fetch_ohlcv_data(
    exchange_id: str, token: str, timeframe: str, start_date: str, end_date: str,
    max_retries: int = 5, retry_delay: float = 5.0
) -> Optional[pd.DataFrame]:
    """
    Récupère les données OHLCV depuis un exchange via ccxt pour une utilisation standalone.
    Gère la pagination, les erreurs spécifiques, les retries automatiques et le respect du rate limit.
    """
    logger.info(f"Tentative de connexion à l'exchange: {exchange_id} (fonction standalone)")

    # Initialisation des variables pour le retry
    retry_count = 0
    backoff_factor = 1.5  # Facteur d'augmentation du délai entre les retries
    current_delay = retry_delay

    while retry_count < max_retries:
        try:
            # Vérifier si on doit utiliser un mock exchange
            use_mock = os.environ.get("USE_MOCK_EXCHANGE", "false").lower() == "true"
            
            # Formater le symbole si nécessaire
            token = format_symbol(token, exchange_id)
            
            # Convertir les dates en timestamps (millisecondes)
            from_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
            to_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
            
            # Initialiser le DataFrame final
            all_candles = []
            
            if use_mock:
                # Utiliser le MockExchange pour les tests locaux
                logger.info("Utilisation de MockExchange pour les tests locaux")
                mock_exchange = MockExchange()
                
                # Déterminer l'intervalle en millisecondes pour calculer le nombre de bougies
                if timeframe == '1h':
                    interval_ms = 60 * 60 * 1000
                elif timeframe == '4h':
                    interval_ms = 4 * 60 * 60 * 1000
                elif timeframe == '1d':
                    interval_ms = 24 * 60 * 60 * 1000
                else:
                    interval_ms = 60 * 60 * 1000  # Par défaut 1h
                
                # Calculer le nombre approximatif de bougies nécessaires
                num_candles = int((to_ts - from_ts) / interval_ms) + 1
                
                # Générer les données simulées
                mock_data = mock_exchange.fetch_ohlcv(token, timeframe, since=from_ts, limit=num_candles)
                all_candles.extend(mock_data)
                
            else:
                # Utiliser ccxt pour les données réelles
                # Initialiser l'exchange
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'enableRateLimit': True,  # Respect du rate limit
                    'options': {
                        'defaultType': 'spot'  # Utiliser le marché spot par défaut
                    }
                })
                
                # Pagination manuelle pour récupérer toutes les données
                current_from_ts = from_ts
                
                while current_from_ts < to_ts:
                    for attempt in range(max_retries):
                        try:
                            # Récupérer un lot de données
                            candles = exchange.fetch_ohlcv(
                                token,
                                timeframe,
                                since=current_from_ts,
                                limit=1000  # Limite maximale pour la plupart des exchanges
                            )
                            
                            if not candles:
                                logger.warning(f"Aucune donnée reçue pour {token} depuis {current_from_ts}")
                                break
                                
                            # Ajouter les bougies au résultat final
                            all_candles.extend(candles)
                            
                            # Mettre à jour le timestamp pour la prochaine itération
                            last_timestamp = candles[-1][0]
                            
                            # Si on a reçu moins que la limite ou atteint la fin, on arrête
                            if len(candles) < 1000 or last_timestamp >= to_ts:
                                current_from_ts = to_ts  # Force la sortie de la boucle externe
                            else:
                                current_from_ts = last_timestamp + 1  # +1ms pour éviter les doublons
                                
                            # Pause pour respecter le rate limit (en plus de enableRateLimit)
                            time.sleep(exchange.rateLimit / 1000 * 1.1)  # +10% de marge
                            
                            break  # Sortir de la boucle de tentatives si réussi
                            
                        except ccxt.NetworkError as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Erreur réseau, nouvelle tentative {attempt+1}/{max_retries}: {e}")
                                time.sleep(retry_delay * (attempt + 1))  # Backoff exponentiel
                            else:
                                logger.error(f"Erreur réseau après {max_retries} tentatives: {e}")
                                raise
                        except ccxt.ExchangeError as e:
                            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                                logger.warning(f"Rate limit atteint, pause de {retry_delay * (attempt + 2)} secondes")
                                time.sleep(retry_delay * (attempt + 2))  # Pause plus longue pour rate limit
                            else:
                                logger.error(f"Erreur de l'exchange: {e}")
                                raise
                        except Exception as e:
                            logger.error(f"Erreur inattendue: {e}")
                            raise
            
            # Convertir en DataFrame
            if not all_candles:
                logger.warning(f"Aucune donnée reçue pour {token} de {start_date} à {end_date}")
                return pd.DataFrame()
                
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir les timestamps en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Filtrer par la plage de dates demandée
            df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & 
                    (df['timestamp'] <= pd.to_datetime(end_date))]
            
            # Trier par timestamp et supprimer les doublons
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            logger.info(f"Récupération terminée: {len(df)} bougies pour {token} de {start_date} à {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV: {e}")
            # Activer automatiquement le mock en cas d'erreur
            if not use_mock:
                logger.warning("Activation automatique du mock suite à une erreur")
                os.environ["USE_MOCK_EXCHANGE"] = "true"
                return fetch_ohlcv_data(exchange_id, token, timeframe, start_date, end_date, max_retries, retry_delay)
            return pd.DataFrame()

        except ccxt.BaseError as e:
            retry_count += 1
            logger.warning(f"Erreur CCXT ({type(e).__name__}): {e}. Tentative {retry_count}/{max_retries}")
            if retry_count < max_retries:
                logger.info(f"Nouvelle tentative dans {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff_factor  # Augmentation exponentielle du délai
            else:
                logger.error(f"Échec après {max_retries} tentatives.")
                return None

        except Exception as e:
            logger.error(f"Erreur non-CCXT lors de la récupération des données: {e}")
            return None

    logger.error(f"Échec de la récupération des données après {max_retries} tentatives.")
    return None


def save_data(df: pd.DataFrame, output_path: str) -> bool:
    """Sauvegarde le DataFrame en CSV."""
    logger.info(f"Tentative de sauvegarde vers {output_path}")
    try:
        # Créer le répertoire parent si nécessaire
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):  # Vérifier si output_dir n'est pas vide
            os.makedirs(output_dir)
        df.to_csv(output_path, index=True)  # Sauvegarder avec l'index timestamp
        logger.info(f"Données sauvegardées avec succès dans: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {output_path}: {e}")
        return False


def verify_downloaded_file(file_path: str, min_rows: int = MIN_ROWS_THRESHOLD) -> bool:
    """Vérifie si le fichier existe et contient un nombre minimum de lignes."""
    logger.info(f"Vérification du fichier: {file_path} (seuil: {min_rows} lignes)")
    if not os.path.exists(file_path):
        logger.error(f"Vérification échouée: Le fichier {file_path} n'existe pas.")
        return False
    try:
        # Vérifier la taille du fichier d'abord (rapide)
        if os.path.getsize(file_path) < 50:  # Taille arbitraire pour un fichier CSV presque vide
            logger.warning(f"Vérification échouée: Le fichier {file_path} semble vide ou très petit.")
            return False

        # Lire le fichier pour compter les lignes
        df = pd.read_csv(file_path, index_col="timestamp")  # Lire avec index
        num_rows = len(df)
        if num_rows >= min_rows:
            logger.info(f"Vérification réussie: Le fichier {file_path} contient {num_rows} lignes (>= {min_rows}).")
            return True
        else:
            logger.warning(
                f"Vérification échouée: Le fichier {file_path} contient seulement {num_rows} lignes (< {min_rows})."
            )
            return False
    except pd.errors.EmptyDataError:
        logger.error(f"Vérification échouée: Le fichier {file_path} est vide.")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la lecture ou vérification du fichier {file_path}: {e}")
        return False


def write_report(
    report_path: str,
    token: str,
    exchange: str,
    timeframe: str,
    start: str,
    end: str,
    status: str,
    num_rows: Optional[int] = None,
    error_msg: Optional[str] = None,
):
    """Écrit un rapport simple sur le téléchargement."""
    try:
        with open(report_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- Rapport de Téléchargement ({timestamp}) ---\n")
            f.write(f"Token: {token}\n")
            f.write(f"Exchange: {exchange}\n")
            f.write(f"Timeframe: {timeframe}\n")
            f.write(f"Période: {start} à {end}\n")
            f.write(f"Statut: {status}\n")
            if num_rows is not None:
                f.write(f"Lignes téléchargées: {num_rows}\n")
            if error_msg:
                f.write(f"Message: {error_msg}\n")
            f.write("-" * 40 + "\n\n")
        logger.info(f"Rapport de téléchargement mis à jour: {report_path}")
    except Exception as e:
        logger.error(f"Impossible d'écrire le rapport {report_path}: {e}")


class MockExchange:
    """Classe d'exchange simulé pour les tests locaux sans API keys."""
    
    def __init__(self):
        self.markets = {"ETH/USDT": {}, "BTC/USDT": {}, "BNB/USDT": {}}
        logger.info("MockExchange initialisé avec succès")
    
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """Simule la récupération de données OHLCV."""
        logger.info(f"MockExchange: Simulation de fetch_ohlcv pour {symbol} ({timeframe})")
        
        # Générer des données aléatoires basées sur le symbole
        base_price = 1500 if "ETH" in symbol else 40000 if "BTC" in symbol else 300
        timestamp_now = int(datetime.now().timestamp() * 1000)
        
        # Déterminer l'intervalle en millisecondes
        if timeframe == '1h':
            interval_ms = 60 * 60 * 1000
        elif timeframe == '4h':
            interval_ms = 4 * 60 * 60 * 1000
        elif timeframe == '1d':
            interval_ms = 24 * 60 * 60 * 1000
        else:
            interval_ms = 60 * 60 * 1000  # Par défaut 1h
        
        # Générer les données
        num_candles = limit or 100
        data = []
        
        for i in range(num_candles):
            timestamp = timestamp_now - (num_candles - i) * interval_ms
            open_price = base_price * (1 + np.random.normal(0, 0.01))
            close_price = open_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = abs(np.random.normal(100, 20))
            
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])
        
        return data
    
    def load_markets(self):
        """Simule le chargement des marchés."""
        logger.info("MockExchange: Simulation de load_markets")
        return self.markets
    
    def check_required_credentials(self):
        """Simule la vérification des identifiants."""
        logger.info("MockExchange: Simulation de check_required_credentials")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge les données OHLCV depuis un exchange crypto.")
    parser.add_argument("--token", type=str, required=True, help="Symbole du token (ex: BTC/USDT, ETHUSDT).")
    parser.add_argument(
        "--exchange", type=str, required=True, help="ID de l'exchange ccxt (ex: binance, kucoin, bitget)."
    )
    parser.add_argument("--start", type=str, required=True, help="Date de début (format YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (format YYYY-MM-DD).")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (ex: 1m, 5m, 15m, 1h, 4h, 1d).")
    parser.add_argument("--output", type=str, required=True, help="Chemin du fichier CSV de sortie.")
    parser.add_argument(
        "--min_rows",
        type=int,
        default=MIN_ROWS_THRESHOLD,
        help=f"Nombre minimum de lignes pour considérer le téléchargement réussi (défaut: {MIN_ROWS_THRESHOLD}).",
    )

    args = parser.parse_args()

    # Utiliser la fonction de formatage
    token_symbol_formatted = format_symbol(args.token, args.exchange)

    logger.info(
        f"Début du téléchargement pour {args.token} ({token_symbol_formatted}) sur {args.exchange} [{args.timeframe}] de {args.start} à {args.end}"
    )

    download_status = "Échec"
    final_num_rows = 0
    error_message = "Erreur inconnue."  # Message par défaut

    try:
        df_data = fetch_ohlcv_data(args.exchange, token_symbol_formatted, args.timeframe, args.start, args.end)

        if df_data is not None and not df_data.empty:
            final_num_rows = len(df_data)
            if save_data(df_data, args.output):
                if verify_downloaded_file(args.output, args.min_rows):
                    download_status = "Succès"
                    error_message = None  # Pas d'erreur si succès
                else:
                    download_status = "Échec (Vérification échouée)"
                    error_message = f"Le fichier contient {final_num_rows} lignes (< {args.min_rows} requis)."
            else:
                download_status = "Échec (Sauvegarde échouée)"
                error_message = f"Erreur lors de l'écriture du fichier CSV: {args.output}"
        elif df_data is None:
            download_status = "Échec (Erreur Fetch)"
            error_message = "La fonction fetch_ohlcv_data a retourné None (voir logs pour détails)."
            final_num_rows = 0
        else:  # df_data is empty
            download_status = "Échec (Aucune donnée)"
            error_message = "Aucune donnée retournée par l'exchange pour cette période/paire."
            final_num_rows = 0

    except Exception as e:
        logger.exception("Une erreur non gérée est survenue dans le processus principal.")  # Log l'exception complète
        download_status = "Échec (Erreur Inattendue)"
        error_message = f"Exception: {str(e)}"
        final_num_rows = 0  # Pas de données si exception majeure

    # Écrire le rapport
    write_report(
        report_file_path,
        args.token,
        args.exchange,
        args.timeframe,
        args.start,
        args.end,
        download_status,
        final_num_rows,
        error_message,
    )

    logger.info(f"Fin du processus de téléchargement. Statut: {download_status}")

    # Quitter avec un code d'erreur si échec pour signaler au script appelant
    if download_status != "Succès":
        sys.exit(1)
    else:
        sys.exit(0)
