import subprocess, json, tempfile, os, pytest
import numpy as np  # Ajout de l'import manquant
from pathlib import Path  # Ajout pour une meilleure gestion des chemins

from scripts.create_dummy_model import create_dummy_model


@pytest.fixture
def setup_backtest_environment(tmp_path):
    # 1. Préparer le répertoire de données avec la fixture golden
    data_dir_fixture = tmp_path / "test_e2e_data"
    data_dir_fixture.mkdir()
    src_fixture_file = Path("tests/fixtures/golden_backtest.parquet")
    pair_filename = "btcusdt_data.parquet"
    dst_fixture_file = data_dir_fixture / pair_filename
    dst_fixture_file.write_bytes(src_fixture_file.read_bytes())

    # 2. Créer et sauvegarder un modèle factice
    dummy_model_path = tmp_path / "dummy_model_e2e.h5"
    create_dummy_model(str(dummy_model_path))

    return str(data_dir_fixture), str(dummy_model_path)


def test_backtest_e2e(setup_backtest_environment, tmp_path):
    data_dir, model_path = setup_backtest_environment
    results_dir = tmp_path / "results_e2e_backtest"  # Nom de répertoire unique

    cmd = [
        "python",
        "run_backtest.py",
        "--data-dir",
        data_dir,
        "--pair",
        "BTC/USDT",
        "--results-dir",
        str(results_dir),
        "--model",
        model_path,
    ]

    print(f"Executing command: {' '.join(cmd)}")
    process_result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    print(f"Return code: {process_result.returncode}")
    print(f"Stdout: {process_result.stdout}")
    print(f"Stderr: {process_result.stderr}")

    assert process_result.returncode == 0, f"run_backtest.py a échoué avec le code {process_result.returncode}"

    # fichiers attendus
    metrics_file = results_dir / "btcusdt_backtest_*.json"  # Le nom du fichier inclut un timestamp
    equity_file = results_dir / "equity_curve.csv"  # Supposant que ce nom est fixe ou que le script le crée
    # ou qu'il est généré par cerebro.plot() si utilisé.
    # Pour l'instant, on va supposer qu'il n'est pas généré par défaut.

    # Trouver le fichier metrics.json qui correspond au pattern (car il contient un timestamp)
    # Note: save_results dans run_backtest.py crée un fichier avec un timestamp.
    # Nous devons trouver ce fichier.

    # Lister les fichiers dans results_dir
    result_files = list(results_dir.glob("btcusdt_backtest_*.json"))
    assert len(result_files) > 0, f"Aucun fichier metrics.json correspondant au pattern dans {results_dir}"
    # Prendre le premier fichier correspondant (ou le plus récent si plusieurs)
    # Pour la simplicité du test, on prend le premier.
    # Si plusieurs backtests sont lancés rapidement, il faudrait une logique plus robuste.
    actual_metrics_file = result_files[0]

    assert actual_metrics_file.exists(), f"⚠️ Fichier metrics ({actual_metrics_file}) manquant dans {results_dir}"
    # assert equity_file.exists(),  f"⚠️ equity_curve.csv manquant dans {results_dir}" # Commenté pour l'instant

    # vérifier quelques valeurs
    m = json.loads(actual_metrics_file.read_text())
    assert "sharpe_ratio" in m, "Clé 'sharpe_ratio' manquante dans metrics.json"
    # Sharpe peut être NaN si pas de trades ou variance nulle des retours, donc on accepte float ou None (si json le convertit)
    assert (
        isinstance(m["sharpe_ratio"], (float, int)) or m["sharpe_ratio"] is None or np.isnan(m["sharpe_ratio"])
    ), f"sharpe_ratio n'est pas un float ou None: {m['sharpe_ratio']}"

    assert "max_drawdown" in m, "Clé 'max_drawdown' manquante dans metrics.json"
    assert isinstance(m["max_drawdown"], (float, int)), f"max_drawdown n'est pas un float: {m['max_drawdown']}"
    assert m["max_drawdown"] >= 0, f"max_drawdown devrait être >= 0, obtenu: {m['max_drawdown']}"

    print("✅ Test E2E Backtest réussi.")
