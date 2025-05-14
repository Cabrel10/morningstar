import subprocess
import json
import pandas as pd
import pytest
from pathlib import Path
import shutil  # Ajout de l'import shutil


@pytest.fixture
def small_data_dir(tmp_path):
    """Crée les répertoires nécessaires pour les données de test et les résultats."""
    data_path = tmp_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    predictions_path = tmp_path / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)

    results_path = tmp_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    return tmp_path


@pytest.fixture
def small_enriched_parquet_file(small_data_dir):
    """
    Crée un petit fichier Parquet de données enrichies pour les tests de backtest.
    Ce fichier doit contenir les colonnes nécessaires pour predict_with_reasoning.py
    et ensuite pour run_backtest.py.
    """
    # Créer un DataFrame exemple
    # Assurez-vous que ce DataFrame contient toutes les colonnes attendues par vos scripts
    # Notamment 'timestamp', 'close', et les features utilisées par le modèle de raisonnement.
    # Et aussi les colonnes pour le backtest comme 'signal' si généré par predict_with_reasoning.

    # Exemple de données minimales (à adapter à vos besoins réels)
    data = {
        "timestamp": pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
                "2023-01-01 03:00:00",
                "2023-01-01 04:00:00",
                "2023-01-01 05:00:00",
            ]
        ),
        "close": [100.0, 101.0, 100.5, 102.0, 101.5, 103.0],
        # Ajouter d'autres colonnes de features si nécessaire pour predict_with_reasoning.py
        # Par exemple, si votre modèle utilise des features techniques :
        "feature_tech_1": [0.5, 0.6, 0.55, 0.65, 0.6, 0.7],
        "bert_0": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Exemple de feature LLM
        "mcp_1": [1, 1, 1, 1, 1, 1],  # Exemple de feature MCP
        "hmm_regime": [0, 0, 1, 1, 0, 0],  # Exemple de feature HMM
        "symbol": [
            "BTC/USDT",
            "BTC/USDT",
            "BTC/USDT",
            "BTC/USDT",
            "BTC/USDT",
            "BTC/USDT",
        ],  # Ajout de la colonne symbol
    }
    df = pd.DataFrame(data)  # Ne pas mettre timestamp comme index pour ce test

    # Chemin du fichier de fixture (source)
    # Pour ce test, nous allons créer le fichier directement dans le tmp_path
    # au lieu de le copier depuis tests/fixtures pour simplifier.
    # Si vous avez un fichier de fixture réel, utilisez shutil.copy comme dans l'exemple.

    fixture_file_path = small_data_dir / "data" / "small_enriched_test.parquet"
    df.to_parquet(fixture_file_path)
    return fixture_file_path


def test_backtest_performance(small_enriched_parquet_file, small_data_dir, tmp_path):
    """
    Teste le pipeline de prédiction et de backtesting de bout en bout.
    """
    # tmp_path est déjà fourni par pytest, small_data_dir est basé sur tmp_path

    # 1) Génération des prédictions
    # Le script predict_with_reasoning.py s'attend à ce que le fichier d'entrée
    # soit dans un sous-répertoire 'data' et que la sortie soit dans 'predictions'.
    # Le nom du fichier de sortie est codé en dur dans predict_with_reasoning.py
    # comme 'trading_predictions.csv'.

    # Chemin vers le script predict_with_reasoning.py
    predict_script_path = Path("predict_with_reasoning.py").resolve()
    # Chemin vers le répertoire contenant les données d'entrée (small_enriched_parquet_file)
    input_data_dir_for_predict = small_enriched_parquet_file.parent.parent  # Remonte à tmp_path
    # Chemin vers le répertoire de sortie pour les prédictions
    predictions_output_dir = small_data_dir / "predictions"

    # Assurez-vous que le modèle de raisonnement est disponible ou mockez son chargement si nécessaire.
    # Pour ce test, nous supposons qu'un modèle par défaut est utilisé ou que le script peut
    # fonctionner avec un modèle minimal si aucun n'est spécifié.

    predict_cmd = [
        "python",
        str(predict_script_path),
        "--data-path",
        str(small_enriched_parquet_file),  # Passer le chemin du fichier directement
        "--output-dir",
        str(predictions_output_dir),
        # Ajoutez d'autres arguments nécessaires pour predict_with_reasoning.py,
        # par exemple --model-path si un modèle spécifique doit être chargé.
        # Pour l'instant, on suppose qu'il peut utiliser un modèle par défaut ou un placeholder.
    ]
    print(f"Running prediction command: {' '.join(predict_cmd)}")
    subprocess.run(predict_cmd, check=True, capture_output=True, text=True)

    # Vérifier que le fichier de prédictions a été créé
    predictions_file = predictions_output_dir / "trading_predictions.csv"
    assert predictions_file.exists(), f"Le fichier de prédictions {predictions_file} n'a pas été créé."

    # 2) Exécution du backtest
    # Le script run_backtest.py prend le fichier de prédictions en entrée.
    backtest_script_path = Path("run_backtest.py").resolve()
    results_output_dir = small_data_dir / "results"

    backtest_cmd = [
        "python",
        str(backtest_script_path),
        "--predictions-file",
        str(predictions_file),  # Chemin corrigé
        "--output-dir",
        str(results_output_dir),
        # Ajoutez d'autres arguments nécessaires pour run_backtest.py
        # Par exemple, le fichier de données de prix si ce n'est pas géré par le fichier de prédictions.
        # Pour l'instant, on suppose que les prédictions contiennent assez d'infos ou que
        # run_backtest.py a un moyen de récupérer les prix (ex: via config).
        # Si run_backtest.py a besoin du fichier de données original, il faudra le passer.
        # Supposons qu'il utilise le même 'data_path' que predict_with_reasoning
        "--data-path",
        str(input_data_dir_for_predict / "data" / small_enriched_parquet_file.name),
        "--pair",
        "BTC/USDT",  # Ajout de l'argument --pair
    ]
    print(f"Running backtest command: {' '.join(backtest_cmd)}")
    subprocess.run(backtest_cmd, check=True, capture_output=True, text=True)

    # 3) Vérification des métriques
    metrics_file = results_output_dir / "backtest_metrics.json"  # Nom de fichier typique
    assert metrics_file.exists(), f"Le fichier de métriques {metrics_file} n'a pas été créé."

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Les assertions de performance dépendent fortement des données et de la stratégie.
    # Pour un test initial, on peut vérifier que les clés existent et que les valeurs sont numériques.
    assert "total_return_pct" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown_pct" in metrics

    # Exemples d'assertions plus strictes (à ajuster après avoir vu les résultats réels)
    # assert metrics["sharpe_ratio"] > -5 # Un Sharpe très négatif peut indiquer un problème
    # assert metrics["max_drawdown_pct"] < 100 # Le drawdown ne doit pas être de 100%
    # assert isinstance(metrics["total_return_pct"], (int, float))

    print(f"Backtest metrics: {metrics}")


# TODO: Ajouter des tests pour le scoring du classifieur de régime de marché.
# TODO: Ajouter des tests de robustesse (données manquantes, outliers).
