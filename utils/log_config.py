import logging
import os
from typing import Optional


def setup_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO):
    """
    Configure le système de logging pour l'application.

    Args:
        log_dir: Répertoire où sauvegarder les fichiers de log
        log_level: Niveau de logging (par défaut INFO)
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "trading_workflow.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Désactiver les logs trop verbeux pour certaines librairies
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
