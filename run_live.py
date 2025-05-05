# run_live.py

import asyncio
import logging
import argparse
import os

# Assurez-vous que le PYTHONPATH inclut la racine du projet ou ajustez les imports
from live.executor import LiveExecutor
from live.monitoring import MetricsLogger
from utils.helpers import load_config  # Assumer que load_config existe

# Configuration simple du logging pour ce script
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lance le robot de trading crypto Morningstar en mode live.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",  # Chemin par défaut
        help="Chemin vers le fichier de configuration YAML principal.",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,  # Par défaut, utilise celui de la config
        help="ID de l'exchange à utiliser (ex: binance, kucoin), surcharge la config.",
    )
    args = parser.parse_args()

    logger.info("Démarrage du robot de trading Morningstar en mode live...")
    logger.info(f"Utilisation du fichier de configuration: {args.config}")
    if args.exchange:
        logger.info(f"Surcharge de l'exchange par ligne de commande: {args.exchange}")

    executor = None  # Pour le bloc finally
    metrics_logger = None  # Pour info

    try:
        # Charger la configuration
        if not os.path.exists(args.config):
            logger.error(f"Le fichier de configuration '{args.config}' n'a pas été trouvé.")
            exit(1)  # Arrêter si la config n'existe pas
        config = load_config(args.config)

        # Instancier le logger de métriques (qui démarre le serveur Prometheus)
        # Il a besoin de la config pour le port
        metrics_logger = MetricsLogger(config=config)
        logger.info("MetricsLogger initialisé.")

        # Instancier l'exécuteur
        # Passer l'override de l'exchange s'il est fourni
        executor = LiveExecutor(config_path=args.config, exchange_id_override=args.exchange)

        # Lancer la boucle principale asyncio
        asyncio.run(executor.run())

    except ValueError as e:
        # Erreurs potentielles de config ou clés API manquantes levées par LiveExecutor.__init__
        logger.error(f"Erreur d'initialisation: {e}")
    except KeyboardInterrupt:
        logger.info("Arrêt manuel détecté (KeyboardInterrupt).")
    except Exception as e:
        logger.exception(f"Une erreur non gérée s'est produite au niveau supérieur: {e}")
    finally:
        # Tentative de fermeture propre de l'executor si il a été créé
        if executor:
            logger.info("Tentative de fermeture propre de l'exécuteur...")
            # La fermeture doit être asynchrone
            asyncio.run(executor.close())
        logger.info("Script run_live.py terminé.")
