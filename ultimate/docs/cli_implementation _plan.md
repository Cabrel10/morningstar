## Plan d'implémentation de la CLI pour Morningstar Crypto Trading

Ce document décrit comment construire une interface en ligne de commande (CLI) modulaire, interactive et maintenable pour piloter l'ensemble du projet Morningstar.

### 1. Dépendances

- **typer[all]** : framework de CLI, inclut Rich pour l'affichage
- **python-dateutil** : parsing et validation des dates

> **Action** : Mettre à jour `requirements.txt` en ajoutant `typer[all]` et `python-dateutil`.

### 2. Structure Générale

Fichier principal : `cli.py` à la racine du projet

```python
import typer
from rich import print

app = typer.Typer(help="CLI pour le projet Morningstar Crypto Trading")
```

##### Sous-applications Typer

| Groupe    | Description                           |
|-----------|---------------------------------------|
| `data`    | Ingestion et préparation des données  |
| `train`   | Entraînement du modèle                |
| `backtest`| Simulation et backtesting             |
| `live`    | Trading en direct                     |
| `report`  | Génération de rapports métriques      |
| `utils`   | Commandes utilitaires (clean, doc)    |

> **Action** : Dans `cli.py`, créer une Typer app pour chaque groupe et les lier à l'app principale.

### 3. Commandes et Options

Pour chaque groupe, définir des commandes et options claires :

#### `data`  
- `ingest` : importer les données brutes depuis un exchange ou fichier
- `pipeline` : exécuter le pré-traitement complet
- **Options communes** : `--asset`, `--start`, `--end`, `--config`

#### `train`  
- `local` : lancer entraînement local
- `colab` : générer un notebook prêt pour Colab
- `evaluate` : évaluer un modèle sauvegardé
- **Options** : `--epochs`, `--batch-size`, `--dry-run`

#### `backtest`  
- `run` : exécuter un backtest complet
- `walkforward` : analyse walk-forward
- **Options** : `--from`, `--to`, `--initial-capital`

#### `live`  
- `start` : démarrer le trading live
- `stop`  : arrêter proprement
- **Options** : `--exchange`, `--symbol`, `--config`

#### `report`  
- `metrics` : afficher une synthèse des performances
- `plot`    : générer des graphiques

#### `utils`  
- `clean-cache` : purger les caches
- `gen-docs`   : mettre à jour la documentation

> **Action** : Utiliser `typer.Argument` et `typer.Option` pour déclarer les paramètres.

### 4. Integration de Rich

- **Progress Bars** : pour `pipeline`, `train`, `backtest` (via `rich.progress`)
- **Tables**        : pour afficher les résultats de backtest (`rich.table`)
- **Prompts**       : confirmations avant actions sensibles (`rich.prompt`)
- **Logs colorés**  : configurer `rich.logging.RichHandler`

> **Action** : Injecter les objets Rich (Progress, Console) dans les fonctions métier.

### 5. Entrypoint et Packaging

- Ajouter dans `setup.py` ou `pyproject.toml` :
  ```toml
  [tool.poetry.scripts]
  crypto-robot = "cli:app"
  ```
- Après `pip install -e .`, la commande `crypto-robot` utilisera notre CLI.

> **Action** : Vérifier le packaging pour que `crypto-robot` soit disponible post-installation.

---

*Ce plan sera mis en œuvre dans le fichier `cli.py`. Les modules existants (scripts, notebooks, utilitaires) seront appelés depuis chaque commande. Des tests unitaires seront ajoutés dans `tests/test_cli.py` pour valider la syntaxe et les enchaînements de commandes.*

