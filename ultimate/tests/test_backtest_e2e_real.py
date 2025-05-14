import pytest
import pandas as pd
from pathlib import Path
import os
import logging
import sys

# Ajouter le répertoire racine du projet au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from backtesting.backtest_engine import BacktestEngine
from utils.config_loader import load_config  # Assumant que vous avez un loader de config

# Configuration du logger pour ce fichier de test
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Définir le chemin vers le fichier de configuration principal
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
MODEL_DIR_PATH = BASE_DIR / "models"  # Ajustez si votre modèle est ailleurs
DATA_DIR_PATH = BASE_DIR / "data" / "processed"


@pytest.fixture(scope="module")
def config():
    """Charge la configuration principale."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def backtest_config(config):
    """Extrait la configuration spécifique au backtesting."""
    return config.get("backtesting", {})


@pytest.fixture(scope="module")
def realistic_data():
    """Charge un jeu de données plus réaliste pour les tests E2E."""
    # Assurez-vous que ce fichier existe et contient environ 1000 lignes de données pertinentes
    data_file = DATA_DIR_PATH / "eth_usdt_1h_1000_lines.parquet"
    if not data_file.exists():
        pytest.skip(f"Jeu de données réaliste non trouvé : {data_file}")
    df = pd.read_parquet(data_file)
    # S'assurer que le DataFrame a les colonnes attendues (au minimum 'timestamp', 'open', 'high', 'low', 'close', 'volume')
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        pytest.fail(f"Le jeu de données {data_file} ne contient pas toutes les colonnes requises: {required_cols}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(by="timestamp")


@pytest.fixture(scope="module")
def trained_model_path(config):
    """Retourne le chemin vers un modèle entraîné."""
    # Utiliser le chemin du modèle depuis la config ou un chemin par défaut
    model_config = config.get("model", {})
    model_filename = model_config.get("model_filename", "morningstar_model.h5")  # Nom par défaut

    # Essayer de trouver le modèle dans le répertoire spécifié par la config ou un répertoire par défaut
    model_save_path_str = model_config.get("save_path", str(MODEL_DIR_PATH))

    # Si save_path est un répertoire, ajouter le nom du fichier
    model_path = Path(model_save_path_str)
    if model_path.is_dir():
        model_path = model_path / model_filename

    if not model_path.exists():
        pytest.skip(f"Modèle entraîné non trouvé : {model_path}")
    return model_path


def test_backtest_e2e_realistic_data(trained_model_path, realistic_data, backtest_config, config):
    """
    Teste le moteur de backtesting de bout en bout avec un jeu de données réaliste.
    Vérifie la complétude de l'exécution et la cohérence de base des métriques.
    """
    # S'assurer que la configuration du backtest est bien passée
    engine_config = {**backtest_config, **config}  # Fusionner pour passer la config globale aussi si nécessaire

    engine = BacktestEngine(model_path=trained_model_path, config=engine_config)

    # Exécuter le backtest
    # Note: sentiment_data est optionnel, passez None si non utilisé pour ce test
    metrics = engine.run_backtest(data=realistic_data, sentiment_data=None, symbol="ETH/USDT", timeframe="1h")

    # 1. Vérifier la complétude (le backtest s'est terminé)
    assert metrics is not None, "Le backtest n'a pas retourné de métriques."
    assert isinstance(metrics, dict), "Les métriques devraient être un dictionnaire."
    assert len(engine.trades) >= 0, "La liste des trades ne devrait pas être None (peut être vide)."
    assert engine.equity_curve is not None, "La courbe d'equity ne devrait pas être None."
    assert not engine.equity_curve.empty, "La courbe d'equity ne devrait pas être vide."

    # 2. Vérifier la cohérence de base des métriques
    assert "total_return_pct" in metrics, "Le pourcentage de retour total doit être dans les métriques."
    assert "max_drawdown_pct" in metrics, "Le drawdown maximum doit être dans les métriques."
    assert (
        "sharpe_ratio" in metrics or metrics.get("annual_return_pct", 0) <= 0
    ), "Le ratio de Sharpe doit être présent si le rendement est positif."
    assert "total_trades" in metrics, "Le nombre total de trades doit être dans les métriques."
    assert (
        "win_rate_pct" in metrics or metrics["total_trades"] == 0
    ), "Le taux de victoire doit être présent si des trades ont été effectués."

    # 3. Vérifications spécifiques (exemples, à adapter selon les attentes pour les données réalistes)
    # Ces assertions sont des exemples et peuvent nécessiter des ajustements
    # en fonction de la nature des données "réalistes" et de la performance attendue du modèle.

    # Exemple : S'attendre à un ROI positif si les données sont globalement haussières
    # Ceci est une heuristique et peut échouer si le marché était baissier ou si le modèle est mauvais.
    # Il faudrait une analyse plus poussée des données pour définir une attente solide.
    # Pour l'instant, on vérifie juste que la métrique existe.
    if not realistic_data.empty:
        price_change_overall = (realistic_data["close"].iloc[-1] - realistic_data["close"].iloc[0]) / realistic_data[
            "close"
        ].iloc[0]
        if price_change_overall > 0.05:  # Si le marché a augmenté de plus de 5%
            # On pourrait s'attendre à un ROI positif, mais ce n'est pas garanti.
            # Pour un test robuste, il faudrait un benchmark ou des données spécifiques.
            logger.info(
                f"Marché haussier détecté (variation de {price_change_overall*100:.2f}%). ROI: {metrics.get('total_return_pct', 0):.2f}%"
            )
            # assert metrics.get("total_return_pct", 0) > -50, "Dans un marché haussier, le ROI ne devrait pas être extrêmement négatif."
        elif price_change_overall < -0.05:
            logger.info(
                f"Marché baissier détecté (variation de {price_change_overall*100:.2f}%). ROI: {metrics.get('total_return_pct', 0):.2f}%"
            )
        else:
            logger.info(
                f"Marché neutre/mixte détecté (variation de {price_change_overall*100:.2f}%). ROI: {metrics.get('total_return_pct', 0):.2f}%"
            )

    # Vérifier que le nombre de trades n'est pas excessif (par exemple, pas plus que le nombre de barres)
    assert metrics["total_trades"] <= len(realistic_data), "Nombre de trades excessif."

    # Vérifier que le drawdown n'est pas de 100% (sauf si le capital initial est 0)
    if engine_config.get("initial_capital", 0) > 0:
        assert metrics["max_drawdown_pct"] > -100, "Drawdown ne devrait pas être -100% (perte totale)."

    logger.info(f"Test E2E réaliste terminé. Métriques obtenues: {metrics}")
