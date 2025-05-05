# cli.py

# import typer # Supprimé pour l'interface interactive
import questionary # Ajouté pour l'interface interactive
import yaml # Ajouté pour charger la config
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
from datetime import date as date_type  # Renommer pour éviter conflit
import os
import signal
import subprocess

# Importer Rich pour l'UX
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress # Pour les barres de progression

# Importer les fonctions métier nécessaires
# (Assurez-vous que ces imports fonctionnent depuis la racine)
try:
    # Essayer d'abord l'import relatif (fonctionne quand exécuté en tant que module)
    try:
        from .utils.api_manager import fetch_ohlcv_data, save_data, format_symbol, verify_downloaded_file
    except ImportError:
        # Fallback à l'import absolu si l'import relatif échoue
        from utils.api_manager import fetch_ohlcv_data, save_data, format_symbol, verify_downloaded_file
    
    # Ajout de l'import manquant
    try:
        from data.data_loader import load_data  # Si ce module existe
    except ImportError:
        load_data = None
    
    # Importer les fonctions du pipeline de données
    try:
        from data.pipelines.data_pipeline import run_pipeline, validate_processed_data
    except ImportError:
        run_pipeline = None
        validate_processed_data = None
    
    # Importer la fonction d'entraînement local
    try:
        from model.training.training_script import run_local_training
        if run_local_training is None:
            raise ImportError
    except ImportError:
        from rich.console import Console
        console = Console()
        console.print("[bold yellow]Avertissement: La fonction run_local_training n'est pas disponible[/]")
        run_local_training = None
    
    # Importer les fonctions de backtest
    try:
        from workflows.backtest_workflow import run_walk_forward, run_simple_backtest
    except ImportError:
        run_walk_forward = None
        run_simple_backtest = None
    # Importer d'autres fonctions métier au besoin
    import webbrowser # Pour ouvrir l'URL Colab
    # Importer psutil pour la gestion des processus (optionnel mais recommandé)
    try:
        import psutil
    except ImportError:
        psutil = None
        print("Avertissement: Le module 'psutil' n'est pas installé. Les commandes 'live stop' et 'live status' pourraient être moins robustes.")
    # Importer les fonctions de reporting (hypothétique)
    try:
        from reporting.reporting_utils import load_and_display_metrics, export_run_report
    except ImportError:
        load_and_display_metrics = None
        export_run_report = None
        print("Avertissement: Module 'reporting.reporting_utils' non trouvé. Les commandes 'report' ne fonctionneront pas.")
    # Importer les fonctions utilitaires (hypothétique)
    try:
        from utils.maintenance import clean_all_caches
    except ImportError:
        clean_all_caches = None
        print("Avertissement: Module 'utils.maintenance' non trouvé. La commande 'utils clean-cache' ne fonctionnera pas.")
    # ... etc ...
except ImportError as e:
    # Gérer le cas où le script est exécuté avant que PYTHONPATH soit bien configuré
    print(f"Erreur d'import: {e}. Assurez-vous d'exécuter depuis la racine du projet ou que PYTHONPATH est configuré.")
    # On peut choisir de continuer avec des placeholders ou d'arrêter
    fetch_ohlcv_data = None
    save_data = None
    format_symbol = None
    verify_downloaded_file = None
    run_pipeline = None
    validate_processed_data = None
    run_local_training = None
    webbrowser = None # Définir à None si l'import échoue
    # Définir d'autres fonctions importées à None pour éviter les erreurs plus loin


# --- Constantes ---
PID_FILE = Path("run/live.pid") # Chemin vers le fichier PID

# Configuration minimale
# app = typer.Typer(help="Morningstar Trading CLI") # Supprimé
console = Console() # Initialisation standard de Rich Console

# --- Configuration Globale (sera chargée au début) ---
CONFIG_PATH = Path("config/config.yaml")
config_data: Dict[str, Any] = {}
global_settings: Dict[str, Any] = {
    "verbose": False,
    "dry_run": False,
    "config_path": CONFIG_PATH
}

# --- Fonctions Helper ---
def load_config(config_path: Path) -> Dict[str, Any]:
    """Charge le fichier de configuration YAML."""
    if not config_path.exists():
        console.print(f"[bold red]Erreur: Fichier de configuration non trouvé à {config_path}[/]")
        # Retourner un dict vide ou lever une exception ? Pour l'instant, dict vide.
        return {}
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            if data is None: # Cas d'un fichier vide
                return {}
            return data
    except yaml.YAMLError as e:
        console.print(f"[bold red]Erreur lors de la lecture du fichier YAML {config_path}: {e}[/]")
        return {}
    except Exception as e:
        console.print(f"[bold red]Erreur inattendue lors du chargement de {config_path}: {e}[/]")
        return {}

def setup_logging(verbose: bool):
    """Configure le logging avec RichHandler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose, console=console)]
    )
    logging.info(f"Niveau de log réglé sur: {logging.getLevelName(log_level)}")

# --- Fonctions de Validation pour Questionary ---
def is_valid_date(text):
    """Valide si le texte est une date YYYY-MM-DD."""
    try:
        date_type.fromisoformat(text)
        return True
    except ValueError:
        return "Format de date invalide. Utilisez YYYY-MM-DD."

def is_valid_asset(text):
    """Valide le format de l'asset (simple vérification)."""
    if not text or '/' not in text: # Ex: BTC/USDT
        # Pourrait être affiné pour vérifier contre une liste ou un pattern
        # return "Format d'asset invalide (ex: BTC/USDT)."
        pass # Accepter aussi les formats simples comme BTC pour l'instant
    return True

def is_valid_timeframe(text):
    """Valide le format du timeframe (simple vérification)."""
    # Pourrait vérifier contre une liste: ['1m', '5m', '1h', '4h', '1d']
    if not text:
        return "Le timeframe ne peut pas être vide."
    return True

def handle_live_menu():
    """Gère le sous-menu interactif pour le trading en direct."""
    console.print("\n[bold green]--- Live Trading ---[/]")
    
    # Lire les valeurs par défaut depuis la config
    defaults = config_data.get('live_trading', {})
    current_exchange = defaults.get('exchange', 'binance')
    current_symbol = defaults.get('symbol', 'BTC/USDT')
    current_strategy = defaults.get('strategy', 'live_strategy')
    
    while True:
        menu_choice = questionary.select(
            "Choisissez une action:",
            choices=[
                f"1. Exchange    : [{current_exchange}]",
                f"2. Symbol      : [{current_symbol}]",
                f"3. Strategy    : [{current_strategy}]",
                "--- Actions ---",
                "4. Start Live Trading",
                "5. Stop Live Trading",
                "6. Check Live Status",
                "7. ← Back to Main Menu"
            ],
            qmark="📡",
            pointer="→"
        ).ask()

        if menu_choice is None or "7" in menu_choice:
            break

        action = menu_choice.split(".")[0].strip()

        try:
            if action == '1':
                new_exchange = questionary.text(
                    "Nouvel Exchange (ex: binance, kucoin):",
                    default=current_exchange
                ).ask()
                if new_exchange:
                    current_exchange = new_exchange
            
            elif action == '2':
                new_symbol = questionary.text(
                    "Nouveau Symbole (ex: ETH/USDT):",
                    default=current_symbol,
                    validate=is_valid_asset
                ).ask()
                if new_symbol:
                    current_symbol = new_symbol.upper()
            
            elif action == '3':
                new_strategy = questionary.text(
                    "Nouvelle stratégie (ex: live_strategy, scalping):",
                    default=current_strategy
                ).ask()
                if new_strategy:
                    current_strategy = new_strategy
            
            elif action == '4': # Start Live Trading
                console.print(f"\n[bold blue]Starting live trading on {current_exchange} for {current_symbol} using {current_strategy} strategy...[/]")
                # Implémentation à compléter
                console.print("[bold green]Live trading démarré avec succès![/]")
            
            elif action == '5': # Stop Live Trading
                console.print("\n[bold blue]Stopping live trading...[/]")
                # Implémentation à compléter
                console.print("[bold green]Live trading arrêté avec succès![/]")
            
            elif action == '6': # Check Live Status
                console.print("\n[bold blue]Checking live trading status...[/]")
                # Implémentation à compléter
                console.print("[bold green]Statut vérifié avec succès![/]")
            
            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            questionary.press_any_key_to_continue().ask()

def handle_data_menu():
    """Gère le sous-menu interactif pour la gestion des données et datasets."""
    console.print("\n[bold green]--- Data Management ---[/]")

    # Lire les valeurs par défaut depuis la config globale ou utiliser des fallbacks
    defaults = config_data.get('data', {})
    current_asset = defaults.get('assets', ['BTC/USDT'])[0] # Prend le premier par défaut
    current_tf = defaults.get('timeframe', '1h')
    # Pour les dates, utiliser une période récente par défaut si non défini
    default_end = date_type.today()
    default_start = date_type(default_end.year - 1, default_end.month, default_end.day)
    current_start = defaults.get('start_date', default_start.isoformat())
    current_end = defaults.get('end_date', default_end.isoformat())
    current_cache = defaults.get('cache_path', 'data/processed/') # Pas modifiable ici pour l'instant

    while True:
        menu_choice = questionary.select(
            "Choisissez une action ou modifiez un paramètre:",
            choices=[
                f"1. Asset       : [{current_asset}]",
                f"2. Timeframe   : [{current_tf}]",
                f"3. Start Date  : [{current_start}]",
                f"4. End Date    : [{current_end}]",
                "--- Actions ---",
                "5. Download historical data",
                "6. Run data pipeline",
                "7. View dataset statistics",
                "8. Split dataset (train/val/test)",
                "9. ← Back to Main Menu",
            ],
            qmark="⚙️",
            pointer="→"
        ).ask()

        if menu_choice is None or menu_choice == "9. ← Back to Main Menu":
            break # Retour au menu principal

        action = menu_choice.split(".")[0].strip() # Extrait le numéro

        try:
            if action == '1':
                new_asset = questionary.text(
                    "Nouvel Asset (ex: ETH/USDT):",
                    default=current_asset,
                    validate=is_valid_asset
                ).ask()
                if new_asset: current_asset = new_asset.upper()
            elif action == '2':
                new_tf = questionary.text(
                    "Nouveau Timeframe (ex: 4h):",
                    default=current_tf,
                    validate=is_valid_timeframe
                ).ask()
                if new_tf: current_tf = new_tf.lower()
            elif action == '3':
                new_start = questionary.text(
                    "Nouvelle Date de Début (YYYY-MM-DD):",
                    default=current_start,
                    validate=is_valid_date
                ).ask()
                if new_start: current_start = new_start
            elif action == '4':
                new_end = questionary.text(
                    "Nouvelle Date de Fin (YYYY-MM-DD):",
                    default=current_end,
                    validate=is_valid_date
                ).ask()
                if new_end: current_end = new_end
            elif action == '5': # Download
                console.print(f"\nPréparation du téléchargement pour {current_asset}...")
                exchange_id = config_data.get('live_trading', {}).get('exchange', 'binance') # Exemple
                asset_part = current_asset.replace('/', '_').lower()
                output_path = Path(f"data/raw/{asset_part}_{exchange_id}_{current_tf}.csv")
                output_path.parent.mkdir(parents=True, exist_ok=True)

                success = data_ingest(
                    asset=current_asset,
                    exchange=exchange_id,
                    timeframe=current_tf,
                    start=current_start,
                    end=current_end,
                    output=output_path,
                    dry_run=global_settings['dry_run']
                )
                if success:
                    console.print("[bold green]Téléchargement terminé.[/]")
                else:
                    console.print("[bold red]Le téléchargement a échoué.[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '6': # Pipeline
                console.print(f"\nLancement du pipeline pour {current_asset}...")
                limit_str = questionary.text(
                    "Limiter aux N premières lignes ? (laisser vide pour tout traiter):",
                    validate=lambda text: text.isdigit() or text == "" or "Doit être un nombre ou vide."
                ).ask()
                limit = int(limit_str) if limit_str else None

                success = data_pipeline(
                    asset=current_asset,
                    limit=limit,
                    dry_run=global_settings['dry_run']
                )
                if success:
                    console.print("[bold green]Pipeline terminé.[/]")
                else:
                    console.print("[bold red]Le pipeline a échoué.[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '7': # Stats
                console.print("\n[bold blue]Statistiques du dataset...[/]")
                try:
                    from data.data_loader import get_dataset_stats
                    stats = get_dataset_stats(asset=current_asset)
                    console.print(f"Taille du dataset: {stats['size']} échantillons")
                    console.print(f"Répartition des classes: {stats['class_distribution']}")
                    console.print(f"Période couverte: {stats['date_range']}")
                except ImportError:
                    console.print("[bold red]Module de statistiques non disponible[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '8': # Split dataset
                console.print("\n[bold blue]Split du dataset en train/val/test...[/]")
                try:
                    from data.data_loader import split_dataset
                    split_ratios = questionary.text(
                        "Ratios de split (train/val/test) séparés par des virgules (ex: 70,15,15):",
                        validate=lambda text: all([r.isdigit() for r in text.split(",")]) and sum(map(int, text.split(","))) == 100
                    ).ask()
                    train_ratio, val_ratio, test_ratio = map(int, split_ratios.split(","))
                    success = split_dataset(
                        asset=current_asset,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio
                    )
                    if success:
                        console.print("[bold green]Split terminé avec succès![/]")
                    else:
                        console.print("[bold red]Le split a échoué.[/]")
                except ImportError:
                    console.print("[bold red]Module de split non disponible[/]")
                questionary.press_any_key_to_continue().ask()

        except KeyboardInterrupt:
             console.print("\n[yellow]Action annulée. Retour au menu Data Management.[/]")
        except Exception as e:
             logging.exception("Erreur dans le sous-menu Data Management:")
             console.print(f"[bold red]Une erreur inattendue est survenue: {e}[/]")
             questionary.press_any_key_to_continue().ask()


def handle_backtest_menu():
    """Gère le sous-menu interactif pour le backtesting."""
    console.print("\n[bold green]--- Backtesting ---[/]")
    
    # Lire les valeurs par défaut depuis la config
    defaults = config_data.get('backtest', {})
    current_asset = defaults.get('asset', 'BTC/USDT')
    current_start = defaults.get('start_date', '2023-01-01')
    current_end = defaults.get('end_date', '2023-12-31')
    current_strategy = defaults.get('strategy', 'simple')
    
    while True:
        menu_choice = questionary.select(
            "Choisissez une action:",
            choices=[
                f"1. Asset       : [{current_asset}]",
                f"2. Start Date  : [{current_start}]",
                f"3. End Date    : [{current_end}]",
                f"4. Strategy    : [{current_strategy}]",
                "--- Actions ---",
                "5. Run Backtest",
                "6. ← Back to Main Menu"
            ],
            qmark="📈",
            pointer="→"
        ).ask()

        if menu_choice is None or "6" in menu_choice:
            break

        action = menu_choice.split(".")[0].strip()

        try:
            if action == '1':
                new_asset = questionary.text(
                    "Nouvel Asset (ex: ETH/USDT):",
                    default=current_asset,
                    validate=is_valid_asset
                ).ask()
                if new_asset:
                    current_asset = new_asset.upper()
            
            elif action == '2':
                new_start = questionary.text(
                    "Nouvelle Date de Début (YYYY-MM-DD):",
                    default=current_start,
                    validate=is_valid_date
                ).ask()
                if new_start:
                    current_start = new_start
            
            elif action == '3':
                new_end = questionary.text(
                    "Nouvelle Date de Fin (YYYY-MM-DD):",
                    default=current_end,
                    validate=is_valid_date
                ).ask()
                if new_end:
                    current_end = new_end
            
            elif action == '4':
                new_strategy = questionary.text(
                    "Nouvelle stratégie (ex: simple, advanced):",
                    default=current_strategy
                ).ask()
                if new_strategy:
                    current_strategy = new_strategy
            
            elif action == '5': # Run Backtest
                console.print(f"\n[bold blue]Running backtest for {current_asset} from {current_start} to {current_end} using {current_strategy} strategy...[/]")
                # Implémentation à compléter
                console.print("[bold green]Backtest terminé avec succès![/]")
            
            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            questionary.press_any_key_to_continue().ask()

def handle_eval_menu():
    """Gère le sous-menu interactif pour l'évaluation des modèles."""
    console.print("\n[bold green]--- Evaluation & Metrics ---[/]")
    
    # Lire les valeurs par défaut depuis la config
    defaults = config_data.get('eval', {})
    current_test_size = str(defaults.get('test_size', 15)) + "%"
    
    while True:
        menu_choice = questionary.select(
            "Choisissez une action:",
            choices=[
                f"1. Test Set Size: [{current_test_size}]",
                "2. Run Evaluation",
                "3. Generate Report",
                "4. Compare Models",
                "5. ← Back to Main Menu"
            ],
            qmark="📊",
            pointer="→"
        ).ask()

        if menu_choice is None or "5" in menu_choice:
            break

        action = menu_choice.split(".")[0].strip()

        try:
            if action == '1':
                new_size = questionary.text(
                    "Taille du set de test (%):",
                    default=current_test_size.replace("%",""),
                    validate=lambda x: x.isdigit() and 0 < int(x) < 100
                ).ask()
                if new_size:
                    current_test_size = f"{new_size}%"
            
            elif action == '2': # Run Evaluation
                console.print("\n[bold blue]Running evaluation...[/]")
                # Implémentation à compléter
            
            elif action == '3': # Generate Report
                console.print("\n[bold blue]Generating evaluation report...[/]")
                # Implémentation à compléter
                
            elif action == '4': # Compare Models
                console.print("\n[bold blue]Comparing model versions...[/]")
                # Implémentation à compléter

            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            questionary.press_any_key_to_continue().ask()


def handle_train_menu():
    """Gère le sous-menu interactif pour l'entraînement du modèle."""
    console.print("\n[bold green]--- Model Training ---[/]")

    # Lire les valeurs par défaut depuis la config globale ou utiliser des fallbacks
    defaults = config_data.get('train', {})
    current_epochs = str(defaults.get('epochs', 50))
    current_batch = str(defaults.get('batch_size', 32))
    current_gpu = defaults.get('use_gpu', True)

    while True:
        menu_choice = questionary.select(
            "Choisissez une action ou modifiez un paramètre:",
            choices=[
                f"1. Epochs      : [{current_epochs}]",
                f"2. Batch Size  : [{current_batch}]",
                f"3. GPU         : [{'enabled' if current_gpu else 'disabled'}]",
                "--- Actions ---",
                "4. Train locally",
                "5. Train on Colab (instructions)",
                "6. Hyperparameter tuning",
                "7. View training curves",
                "8. ← Back to Main Menu",
            ],
            qmark="🧠",
            pointer="→"
        ).ask()

        if menu_choice is None or menu_choice == "8. ← Back to Main Menu":
            break # Retour au menu principal

        action = menu_choice.split(".")[0].strip() # Extrait le numéro

        try:
            if action == '1':
                new_epochs = questionary.text(
                    "Nombre d'époques:",
                    default=current_epochs,
                    validate=lambda text: text.isdigit() and int(text) > 0
                ).ask()
                if new_epochs: current_epochs = new_epochs
            elif action == '2':
                new_batch = questionary.text(
                    "Taille du batch:",
                    default=current_batch,
                    validate=lambda text: text.isdigit() and int(text) > 0
                ).ask()
                if new_batch: current_batch = new_batch
            elif action == '3':
                current_gpu = not current_gpu # Toggle
            elif action == '4': # Train local
                console.print(f"\nLancement de l'entraînement local pour {current_epochs} époques...")
                success = train_local(
                    epochs=int(current_epochs),
                    config_path=global_settings['config_path'],
                    dry_run=global_settings['dry_run']
                )
                if success:
                    console.print("[bold green]Entraînement terminé.[/]")
                else:
                    console.print("[bold red]L'entraînement a échoué.[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '5': # Colab
                notebook_path = Path("notebooks/training_on_colab.ipynb")
                success = train_colab(notebook_path)
                if not success:
                    console.print("[bold red]Impossible d'afficher les instructions Colab.[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '6': # Hyperparameter tuning
                console.print("\n[bold blue]Démarrage du tuning des hyperparamètres...[/]")
                try:
                    from model.optimization.optimization_module import run_hyperparameter_tuning
                    success = run_hyperparameter_tuning(
                        config_path=global_settings['config_path'],
                        dry_run=global_settings['dry_run']
                    )
                    if success:
                        console.print("[bold green]Tuning terminé avec succès![/]")
                    else:
                        console.print("[bold red]Le tuning a échoué[/]")
                except ImportError:
                    console.print("[bold red]Module d'optimisation non disponible[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '7': # View training curves
                console.print("\n[bold blue]Affichage des courbes d'entraînement...[/]")
                try:
                    from model.training.evaluation import plot_training_curves
                    plot_training_curves(
                        config_path=global_settings['config_path']
                    )
                except ImportError:
                    console.print("[bold red]Module de visualisation non disponible[/]")
                questionary.press_any_key_to_continue().ask()
            elif action == '8': # Back
                break

        except KeyboardInterrupt:
             console.print("\n[yellow]Action annulée. Retour au menu Model Training.[/]")
        except Exception as e:
             logging.exception("Erreur dans le sous-menu Model Training:")
             console.print(f"[bold red]Une erreur inattendue est survenue: {e}[/]")
             questionary.press_any_key_to_continue().ask()


# --- Fonctions Métier (Anciennes commandes Typer refactorisées) ---

# Groupe DATA
# @data_app.command("ingest") # Supprimé
def data_ingest(
    # ctx: typer.Context, # Supprimé
    asset: str, # = typer.Argument(..., help="Ticker de l'actif (ex: BTC/USDT ou BTC)."), # Modifié
    exchange: str, # = typer.Option(..., "--exchange", "-e", help="ID de l'exchange ccxt (ex: binance, kucoin)."), # Modifié
    timeframe: str, # = typer.Option(..., "--timeframe", "-t", help="Timeframe des données (ex: 1m, 5m, 1h, 1d)."), # Modifié
    start: str, # = typer.Option(..., help="Date de début (YYYY-MM-DD)."), # Modifié
    end: str, # = typer.Option(..., help="Date de fin (YYYY-MM-DD)."), # Modifié
    output: Path, # = typer.Option(..., "--output", "-o", help="Chemin du fichier CSV de sortie (ex: data/raw/btc_binance_1h.csv).") # Modifié
    dry_run: bool # Ajouté pour passer l'option globale
):
    """Télécharge les données OHLCV brutes via CCXT et sauvegarde en CSV."""
    console.print(f"[bold blue]Démarrage de l'ingestion pour {asset} sur {exchange} ({timeframe})[/]")
    # Convertir d'abord les dates string en objets date avant d'appeler isoformat()
    try:
        start_date = date_type.fromisoformat(start)
        end_date = date_type.fromisoformat(end)
        console.print(f"Période: {start_date.isoformat()} à {end_date.isoformat()}")
    except ValueError as e:
        console.print(f"[bold red]Format de date invalide: {e}[/]")
        # Dans un mode interactif, on pourrait redemander la date
        return False # Indique l'échec
    console.print(f"Sortie prévue: {output}")

    # Vérifier si les fonctions métier sont disponibles
    if fetch_ohlcv_data is None or save_data is None or format_symbol is None:
         console.print("[bold red]Erreur: Les fonctions nécessaires depuis utils.api_manager n'ont pas pu être importées.[/]")
         return False # Indique l'échec

    # Les dates ont été validées au début. On peut maintenant formater.
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # Formater le symbole (gère BTC vs BTC/USDT)
    # La fonction format_symbol vient de api_manager.py
    formatted_asset = format_symbol(asset, exchange)
    console.print(f"Formatage du symbole: '{asset}' -> '{formatted_asset}' pour {exchange}")

    if dry_run: # Utilisation directe du paramètre
        console.print("[yellow]DRY-RUN: Simulation du téléchargement et de la sauvegarde.[/]")
        # Simuler un succès pour permettre de tester le reste du flux
        console.print(f"[green]DRY-RUN: Téléchargement et sauvegarde simulés pour {output}.[/]")
        return True # Indique le succès (simulé)

    try:
        # Utiliser Rich Progress pour montrer l'activité
        with Progress(console=console) as progress:
            task = progress.add_task(f"[cyan]Téléchargement {formatted_asset}...", total=None) # Total indéfini

            # Appel à la fonction de téléchargement
            # Note: fetch_ohlcv_data gère déjà le logging interne
            df_data = fetch_ohlcv_data(exchange, formatted_asset, timeframe, start_str, end_str)

            progress.update(task, completed=100, description=f"[cyan]Téléchargement {formatted_asset} terminé.") # Marquer comme terminé

        if df_data is not None and not df_data.empty:
            console.print(f"[green]Téléchargement réussi: {len(df_data)} lignes récupérées.[/]")
            console.print(f"Sauvegarde des données dans {output}...")

            # Appel à la fonction de sauvegarde
            if save_data(df_data, str(output)):
                console.print(f"[bold green]Données sauvegardées avec succès dans: {output}[/]")
                # Optionnel: Vérifier le fichier sauvegardé
                if verify_downloaded_file and verify_downloaded_file(str(output)):
                     console.print("[green]Vérification du fichier téléchargé réussie.[/]")
                else:
                     console.print("[yellow]Avertissement: La vérification post-sauvegarde a échoué ou n'est pas disponible.[/]")
            else:
                console.print(f"[bold red]Erreur lors de la sauvegarde du fichier {output}.[/]")
                return False # Indique l'échec
        elif df_data is None:
             console.print(f"[bold red]Échec du téléchargement. La fonction fetch_ohlcv_data a retourné None. Vérifiez les logs pour plus de détails.[/]")
             return False # Indique l'échec
        else: # df_data is empty
            console.print(f"[bold yellow]Aucune donnée retournée par l'exchange pour {formatted_asset} sur cette période/timeframe.[/]")
            # Pas nécessairement une erreur, mais on retourne True car l'opération s'est terminée
            return True

    except Exception as e:
        logging.exception(f"Une erreur inattendue est survenue lors de l'ingestion:")
        console.print(f"[bold red]Erreur lors de l'ingestion: {e}[/]")
        return False # Indique l'échec

# ... (le reste des commandes reste avec les placeholders pour l'instant) ...

# Groupe DATA (suite)
# @data_app.command("pipeline") # Supprimé
def data_pipeline(
    # ctx: typer.Context, # Supprimé
    asset: str, # = typer.Argument(..., help="Ticker de l'actif (ex: BTC/USDT)"), # Modifié
    limit: Optional[int], # = typer.Option(None, "--limit", "-l", help="Limiter le traitement aux N premières lignes du fichier brut (pour tests)."), # Modifié
    # output: Path = typer.Option(..., "--output", help="Chemin du fichier Parquet de sortie.") # Optionnel si défini dans config
    dry_run: bool # Ajouté
):
    """Exécute le pipeline de données complet (features, labels, embeddings)."""
    console.print(f"[bold blue]Démarrage du pipeline de données pour {asset}{f' (limité à {limit} lignes)' if limit else ''}[/]")

    if run_pipeline is None:
        console.print("[bold red]Erreur: La fonction 'run_pipeline' n'a pas pu être importée depuis data.pipelines.data_pipeline.[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de l'exécution du pipeline pour {asset}.[/]")
        return True # Indique le succès (simulé)

    try:
        # Utiliser Rich Progress si la fonction run_pipeline est longue
        with Progress(console=console) as progress:
            task = progress.add_task(f"[cyan]Exécution du pipeline {asset}...", total=None)

            # Appel à la fonction du pipeline
            # Assumer que run_pipeline gère son propre logging détaillé
            # et retourne un booléen ou lève une exception en cas d'échec.
            # Déterminer les chemins d'entrée et de sortie dynamiquement
            # TODO: Idéalement, lire ces conventions depuis la config ou une fonction helper
            asset_filename_part = asset.replace('/', '_').lower() # ex: btc_usdt
            # Supposer que l'exchange et timeframe sont implicites ou définis ailleurs pour le pipeline
            # Pour cet exemple, on se base sur l'étape précédente
            input_file = Path(f"data/raw/{asset_filename_part}_binance_4h.csv")
            output_file = Path(f"data/processed/{asset_filename_part}_binance_4h.parquet")

            # S'assurer que le répertoire de sortie existe
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Vérifier si le fichier d'entrée existe avant d'appeler le pipeline
            if not input_file.exists():
                console.print(f"[bold red]Erreur: Fichier d'entrée non trouvé pour le pipeline: {input_file}[/]")
                console.print(f"[bold yellow]Assurez-vous d'avoir exécuté 'data ingest' pour {asset} avec les bons paramètres.[/]")
                return False # Indique l'échec

            console.print(f"Utilisation du fichier d'entrée: {input_file}")
            console.print(f"Fichier de sortie prévu: {output_file}")

            success = run_pipeline(
                input_path=str(input_file),
                output_path=str(output_file),
                limit=limit # Passer la valeur de l'option limit
            )

            progress.update(task, completed=100, description=f"[cyan]Pipeline {asset}{f' (limité à {limit} lignes)' if limit else ''} terminé.")

        if success:
            console.print(f"[bold green]Pipeline de données pour {asset} exécuté avec succès.[/]")
        else:
            console.print(f"[bold red]Le pipeline de données pour {asset} a échoué. Vérifiez les logs.[/]")
            return False # Indique l'échec

    except Exception as e:
        logging.exception(f"Une erreur inattendue est survenue lors de l'exécution du pipeline:")
        console.print(f"[bold red]Erreur lors de l'exécution du pipeline: {e}[/]")
        return False # Indique l'échec


# @data_app.command("validate") # Supprimé
def data_validate(
    # ctx: typer.Context, # Supprimé
    asset: str, # = typer.Argument(..., help="Ticker de l'actif (ex: BTC/USDT)"), # Modifié
    config_path: Path, # Ajouté car utilisé dans la fonction originale
    dry_run: bool # Ajouté
):
    """Valide le fichier Parquet traité (NaN, variance, nb colonnes)."""
    console.print(f"[bold blue]Démarrage de la validation des données traitées pour {asset}[/]")

    if validate_processed_data is None:
        console.print("[bold red]Erreur: La fonction 'validate_processed_data' n'a pas pu être importée depuis data.pipelines.data_pipeline.[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de la validation des données pour {asset}.[/]")
        return True # Indique le succès (simulé)

    try:
        # Appel à la fonction de validation
        # Supposer qu'elle retourne True si valide, False ou lève une exception sinon
        # et qu'elle logue les détails des erreurs éventuelles.
        is_valid = validate_processed_data(
            asset=asset,
            config_path=config_path # Utilisation directe
        )

        if is_valid:
            console.print(f"[bold green]Validation des données traitées pour {asset} réussie.[/]")
            return True
        else:
            console.print(f"[bold red]Validation des données traitées pour {asset} échouée. Vérifiez les logs.[/]")
            return False # Indique l'échec

    except FileNotFoundError:
        # Cas spécifique où le fichier traité n'existe pas
        logging.error(f"Le fichier de données traitées pour l'asset '{asset}' n'a pas été trouvé.")
        console.print(f"[bold red]Erreur: Fichier de données traitées pour {asset} non trouvé. Exécutez d'abord 'data pipeline'.[/]")
        return False # Indique l'échec
    except Exception as e:
        logging.exception(f"Une erreur inattendue est survenue lors de la validation des données:")
        console.print(f"[bold red]Erreur lors de la validation des données: {e}[/]")
        return False # Indique l'échec


# Groupe TRAIN
# @train_app.command("local") # Supprimé
def train_local(
    # ctx: typer.Context, # Supprimé
    epochs: int, # = typer.Option(50, help="Nombre d'époques d'entraînement.") # Modifié
    config_path: Path, # Ajouté
    dry_run: bool # Ajouté
):
    """Entraîne le modèle en local en utilisant le script ou notebook."""
    console.print(f"[bold blue]Démarrage de l'entraînement local pour {epochs} époques[/]")

    if run_local_training is None:
        console.print("[bold red]Erreur: La fonction 'run_local_training' n'a pas pu être importée depuis model.training.training_script.[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de l'entraînement local pour {epochs} époques.[/]")
        return True # Indique le succès (simulé)

    try:
        # Utiliser Rich Progress si l'entraînement est long et fournit des callbacks
        # Sinon, simple message avant/après
        console.print("[cyan]Lancement de l'entraînement... (cela peut prendre du temps)[/]")

        # Appel à la fonction d'entraînement
        # Supposer qu'elle gère son propre logging et retourne True/False ou lève une exception
        success = run_local_training(
            config_path=config_path, # Utilisation directe
            epochs=epochs,
            # dry_run=dry_run # La fonction métier doit gérer dry_run
        )

        if success:
            console.print(f"[bold green]Entraînement local terminé avec succès ({epochs} époques).[/]")
            return True
        else:
            console.print(f"[bold red]L'entraînement local a échoué. Vérifiez les logs.[/]")
            return False # Indique l'échec

    except Exception as e:
        logging.exception(f"Une erreur inattendue est survenue lors de l'entraînement local:")
        console.print(f"[bold red]Erreur lors de l'entraînement local: {e}[/]")
        return False # Indique l'échec


# @train_app.command("colab") # Supprimé
def train_colab(
    # ctx: typer.Context, # Supprimé
    notebook: Path # = typer.Option("notebooks/training_on_colab.ipynb", help="Chemin vers le notebook Colab.") # Modifié
):
    """Affiche les instructions pour lancer l'entraînement sur Google Colab."""
    console.print(f"[bold blue]Préparation de l'entraînement sur Google Colab avec le notebook:[/]")
    console.print(f"[cyan]{notebook}[/]")

    if not notebook.exists():
        console.print(f"[bold red]Erreur: Le fichier notebook '{notebook}' n'a pas été trouvé.[/]")
        return False # Indique l'échec

    # Essayer de générer une URL Colab (fonctionne mieux si le repo est public sur GitHub)
    # Pour un fichier local, l'upload manuel est souvent nécessaire.
    # Exemple d'URL pour un repo GitHub:
    # colab_url = f"https://colab.research.google.com/github/VOTRE_USER/VOTRE_REPO/blob/main/{notebook.relative_to(Path.cwd())}"
    # Pour un fichier local, on ne peut pas générer d'URL directe facilement.

    console.print("\n[bold yellow]Instructions pour lancer l'entraînement sur Colab:[/]")
    console.print("1. [bold]Uploadez le notebook[/] (si nécessaire) et l'ensemble du projet (ou au moins les données et le code requis) sur votre Google Drive.")
    console.print(f"2. [bold]Ouvrez Google Colab[/] et chargez le notebook : [cyan]{notebook.name}[/]")
    console.print("3. [bold]Montez votre Google Drive[/] dans le notebook en exécutant la cellule appropriée (généralement la première).")
    console.print("   ```python")
    # Utiliser console.print pour que Rich gère l'affichage
    console.print("   from google.colab import drive")
    console.print("   drive.mount('/content/drive')")
    console.print("   ```")
    console.print("4. [bold]Ajustez les chemins[/] dans le notebook pour pointer vers les fichiers sur votre Drive (ex: `/content/drive/MyDrive/crypto_robot/...`).")
    console.print("5. [bold]Installez les dépendances[/] en exécutant la cellule :")
    console.print("   ```bash")
    # Utiliser console.print
    console.print("   !pip install -r /content/drive/MyDrive/chemin/vers/requirements.txt")
    console.print("   ```")
    console.print("6. [bold]Exécutez les cellules[/] du notebook pour lancer l'entraînement.")
    console.print("7. Assurez-vous que l'environnement d'exécution Colab utilise un [bold]GPU[/] pour accélérer l'entraînement (Menu 'Exécution' -> 'Modifier le type d'exécution').")

    # Optionnel: Tenter d'ouvrir Colab dans le navigateur
    if webbrowser:
        try:
            console.print("\n[cyan]Tentative d'ouverture de Google Colab dans votre navigateur...[/]")
            webbrowser.open("https://colab.research.google.com/", new=2)
        except Exception as e:
            logging.warning(f"Impossible d'ouvrir le navigateur: {e}")
    else:
        console.print("\n[yellow]Module 'webbrowser' non disponible pour ouvrir Colab automatiquement.[/]")


# Groupe BACKTEST
# @backtest_app.command("run") # Supprimé
def backtest_run(
    # ctx: typer.Context, # Supprimé
    asset: str, # = typer.Argument(..., help="Ticker de l'actif (ex: BTC/USDT)"), # Modifié
    start: str, # = typer.Option(..., help="Date de début (YYYY-MM-DD)"), # Modifié
    end: str, # = typer.Option(..., help="Date de fin (YYYY-MM-DD)") # Modifié
    config_path: Path, # Ajouté
    dry_run: bool # Ajouté
):
    """Exécute un backtest simple sur la période donnée."""
    console.print(f"[bold blue]Démarrage du backtest simple pour {asset}[/]")
    # Conversion des dates string en objets date
    try:
        start_date = date_type.fromisoformat(start)
        end_date = date_type.fromisoformat(end)
        console.print(f"Période: {start_date.isoformat()} à {end_date.isoformat()}")
    except ValueError as e:
        console.print(f"[bold red]Format de date invalide: {e}[/]")
        # Dans un mode interactif, on pourrait redemander la date
        # Ici, on arrête l'action en cours
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print("[yellow]DRY-RUN: Simulation du backtest simple.[/]")
        return True # Indique le succès (simulé)

    # Le bloc try précédent (lignes 458-466) a déjà validé les dates.
    # Maintenant, le try pour l'exécution de la logique métier.
    try:
        # Vérifier si la fonction métier existe
        if run_simple_backtest is None:
            console.print("[bold red]Erreur: La fonction 'run_simple_backtest' n'a pas pu être importée.[/]")
            return False

        success = run_simple_backtest(
            asset=asset,
            start_date=start_date, # Utiliser la date convertie
            end_date=end_date, # Utiliser la date convertie
            config_path=config_path # Utilisation directe
        )

        if success:
            console.print("[bold green]Backtest simple terminé avec succès.[/]")
            return True
        else:
            console.print("[bold red]Le backtest simple a échoué. Vérifiez les logs.[/]")
            return False

    except Exception as e:
        logging.exception("Erreur lors de l'exécution du backtest simple:")
        console.print(f"[bold red]Erreur: {e}[/]")
        return False

# @backtest_app.command("wf") # Supprimé
def backtest_wf(
    # ctx: typer.Context, # Supprimé
    asset: str, # = typer.Argument(..., help="Ticker de l'actif (ex: BTC/USDT)"), # Modifié
    train_months: int, # = typer.Option(6, help="Nombre de mois pour la fenêtre d'entraînement."), # Modifié
    test_months: int, # = typer.Option(1, help="Nombre de mois pour la fenêtre de test (horizon).") # Modifié
    config_path: Path, # Ajouté
    dry_run: bool # Ajouté
):
    """Exécute un backtest en walk-forward."""
    console.print(f"[bold blue]Démarrage du walk-forward pour {asset}[/]")
    console.print(f"Configuration: {train_months} mois train / {test_months} mois test")

    if dry_run: # Utilisation directe
        console.print("[yellow]DRY-RUN: Simulation du walk-forward.[/]")
        return True # Indique le succès (simulé)

    try:
        # Vérifier si la fonction métier existe
        if run_walk_forward is None:
            console.print("[bold red]Erreur: La fonction 'run_walk_forward' n'a pas pu être importée.[/]")
            return False

        success = run_walk_forward(
            asset=asset,
            train_months=train_months,
            test_months=test_months,
            config_path=config_path # Utilisation directe
        )

        if success:
            console.print("[bold green]Walk-forward terminé avec succès.[/]")
            return True
        else:
            console.print("[bold red]Le walk-forward a échoué. Vérifiez les logs.[/]")
            return False

    except Exception as e:
        logging.exception("Erreur lors de l'exécution du walk-forward:")
        console.print(f"[bold red]Erreur: {e}[/]")
        return False


# Groupe LIVE
# @live_app.command("start") # Supprimé
# Renommée pour éviter conflit avec le mot-clé 'start'
def start_live_process(
    # ctx: typer.Context, # Supprimé
    config_path: Path, # Ajouté
    exchange: Optional[str], # = typer.Option(None, help="ID de l'exchange (ex: binance), surcharge la config."), # Modifié
    symbol: Optional[str], # = typer.Option(None, help="Symbole à trader (ex: BTC/USDT), surcharge la config.") # Modifié
    dry_run: bool # Ajouté
):
    """Démarre le processus de trading live."""
    logging.info(f"Appel à live start (Exchange: {exchange or 'config'}, Symbol: {symbol or 'config'})")

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text())
            if psutil and psutil.pid_exists(pid):
                 console.print(f"[bold yellow]Un processus live semble déjà en cours (PID: {pid}). Utilisez 'live status' ou 'live stop'.[/]")
                 return False # Indique l'échec (processus déjà en cours)
            else:
                 console.print(f"[yellow]Ancien fichier PID trouvé ({PID_FILE}). Nettoyage...[/]")
                 PID_FILE.unlink()
        except ValueError:
             console.print(f"[yellow]Fichier PID invalide trouvé ({PID_FILE}). Nettoyage...[/]")
             PID_FILE.unlink()
        except Exception as e:
             logging.warning(f"Erreur lors de la vérification du fichier PID existant: {e}")
             # Continuer malgré l'erreur de vérification ? Ou arrêter ? Pour l'instant on continue.

    console.print(f"[yellow]Lancement du processus live...[/]")
    if dry_run: # Utilisation directe
        console.print("[yellow]DRY-RUN: Simulation du lancement du processus live et de la création du fichier PID.[/]")
        # Créer un faux fichier PID pour tester stop/status en dry-run
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text("0") # PID 0 n'existe jamais
        console.print(f"[yellow]DRY-RUN: Fichier PID simulé créé: {PID_FILE}[/]")
        return True # Indique le succès (simulé)

    # Construire la commande
    cmd = ["python", "run_live.py", "-c", str(config_path)] # Utilisation directe
    if exchange:
        cmd.extend(["--exchange", exchange])
    if symbol: # Ajouter le symbole s'il est fourni
        cmd.extend(["--symbol", symbol])

    try:
        console.print(f"Exécution de: {' '.join(cmd)}")
        # Utiliser Popen pour ne pas bloquer la CLI
        # Rediriger stdout/stderr vers des fichiers logs dédiés serait mieux en production
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Créer le répertoire 'run' s'il n'existe pas
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Écrire le PID dans le fichier
        PID_FILE.write_text(str(process.pid))

        console.print(f"[bold green]Processus live démarré (PID: {process.pid}). Fichier PID créé: {PID_FILE}[/]")
        console.print("[yellow]Utilisez 'live stop' pour l'arrêter proprement.[/]")

        return True # Indique le succès

    except FileNotFoundError:
         logging.error("Erreur: 'python' ou 'run_live.py' non trouvé. Assurez-vous d'être dans le bon environnement/répertoire.")
         console.print("[bold red]Erreur: Impossible de trouver 'python' ou 'run_live.py'.[/]")
         return False # Indique l'échec
    except Exception as e:
        logging.exception("Erreur lors du lancement de run_live.py")
        console.print(f"[bold red]Erreur lors du lancement du processus live: {e}[/]")
        # Essayer de nettoyer le fichier PID s'il a été créé par erreur
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
        return False # Indique l'échec


# @live_app.command("stop") # Supprimé
# Renommée pour éviter conflit
def stop_live_process(dry_run: bool): # Ajouté dry_run
    """Arrête proprement le processus de trading live."""
    logging.info("Appel à live stop")

    if not PID_FILE.exists():
        console.print("[yellow]Aucun fichier PID trouvé. Le processus live n'est probablement pas en cours.[/]")
        return True # Pas une erreur fatale, on considère que c'est "arrêté"

    try:
        pid_str = PID_FILE.read_text()
        pid = int(pid_str)
    except ValueError:
        console.print(f"[bold red]Erreur: Fichier PID invalide ({PID_FILE}). Contenu: '{pid_str}'. Nettoyage manuel requis.[/]")
        return False # Indique l'échec
    except Exception as e:
        logging.exception(f"Erreur lors de la lecture du fichier PID {PID_FILE}")
        console.print(f"[bold red]Erreur lors de la lecture du fichier PID: {e}[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de l'arrêt du processus avec PID {pid} et suppression de {PID_FILE}.[/]")
        # Supprimer le faux fichier PID créé par start --dry-run
        if pid == 0:
             PID_FILE.unlink(missing_ok=True)
             console.print(f"[yellow]DRY-RUN: Fichier PID simulé supprimé: {PID_FILE}[/]")
        return True # Indique le succès (simulé)

    if psutil is None:
        console.print("[yellow]Module 'psutil' non disponible. Tentative d'arrêt via os.kill uniquement.[/]")
        try:
            os.kill(pid, signal.SIGTERM) # Envoyer SIGTERM
            console.print(f"Signal SIGTERM envoyé au processus {pid}. Vérifiez manuellement s'il s'est arrêté.")
            # Supprimer le fichier PID même si on ne peut pas confirmer l'arrêt
            PID_FILE.unlink(missing_ok=True)
            console.print(f"Fichier PID supprimé: {PID_FILE}")
        except ProcessLookupError:
            console.print(f"[yellow]Processus avec PID {pid} non trouvé. Il s'est peut-être déjà arrêté.[/]")
            PID_FILE.unlink(missing_ok=True) # Nettoyer le fichier PID
            console.print(f"Fichier PID supprimé: {PID_FILE}")
        except Exception as e:
            logging.exception(f"Erreur lors de l'envoi du signal SIGTERM au PID {pid}")
            console.print(f"[bold red]Erreur lors de la tentative d'arrêt du processus {pid}: {e}[/]")
            # Ne pas supprimer le fichier PID en cas d'erreur inconnue
            return False # Indique l'échec
        return True # Indique le succès (signal envoyé ou processus déjà arrêté)

    # --- Logique avec psutil ---
    try:
        if not psutil.pid_exists(pid):
            console.print(f"[yellow]Processus avec PID {pid} non trouvé. Il s'est peut-être déjà arrêté.[/]")
            PID_FILE.unlink(missing_ok=True) # Nettoyer le fichier PID
            console.print(f"Fichier PID nettoyé: {PID_FILE}")
            return True # Indique le succès (processus déjà arrêté)

        p = psutil.Process(pid)
        console.print(f"Envoi du signal SIGTERM au processus {pid} ({p.name()})...")
        p.terminate() # Envoie SIGTERM

        # Attendre un peu que le processus se termine proprement
        try:
            gone, alive = psutil.wait_procs([p], timeout=10)
            if gone:
                console.print(f"[bold green]Processus {pid} arrêté avec succès.[/]")
            elif alive:
                console.print(f"[yellow]Le processus {pid} n'a pas répondu au SIGTERM après 10s. Envoi de SIGKILL...[/]")
                p.kill() # Envoie SIGKILL
                gone, alive = psutil.wait_procs([p], timeout=5)
                if gone:
                     console.print(f"[bold green]Processus {pid} arrêté avec SIGKILL.[/]")
                else:
                     console.print(f"[bold red]Impossible d'arrêter le processus {pid} même avec SIGKILL.[/]")
                     # Laisser le fichier PID en place dans ce cas ? Ou le supprimer ?
                     # Pour l'instant, on le supprime pour permettre de relancer.
                     # Laisser le fichier PID en place dans ce cas ? Ou le supprimer ?
                     # Pour l'instant, on le supprime pour permettre de relancer.
                     # PID_FILE.unlink(missing_ok=True) # Déplacé dans finally
                     return False # Indique l'échec

        except psutil.NoSuchProcess:
             console.print(f"[yellow]Le processus {pid} s'est terminé pendant l'attente.[/]")
        except psutil.TimeoutExpired: # Devrait être géré par wait_procs, mais par sécurité
             console.print(f"[bold red]Timeout inattendu lors de l'attente de l'arrêt du processus {pid}.[/]")
             return False # Indique l'échec

    except psutil.NoSuchProcess:
        console.print(f"[yellow]Processus avec PID {pid} non trouvé au moment de l'arrêt.[/]")
    except psutil.AccessDenied:
        logging.error(f"Accès refusé pour arrêter le processus {pid}. Vérifiez les permissions.")
        console.print(f"[bold red]Erreur: Accès refusé pour arrêter le processus {pid}. Exécutez avec les permissions appropriées.[/]")
        return False # Indique l'échec
    except Exception as e:
        logging.exception(f"Erreur inattendue lors de l'arrêt du processus {pid}")
        console.print(f"[bold red]Erreur inattendue lors de l'arrêt: {e}[/]")
        return False # Indique l'échec
    finally:
        # Toujours essayer de supprimer le fichier PID si le processus n'existe plus
        if not (psutil and psutil.pid_exists(pid)):
             if PID_FILE.exists():
                  PID_FILE.unlink(missing_ok=True)
                  logging.info(f"Fichier PID {PID_FILE} supprimé.")
                  # Ne pas afficher de message ici si déjà fait plus haut


# @live_app.command("status") # Supprimé
# Renommée pour éviter conflit
def get_live_status(dry_run: bool): # Ajouté dry_run
    """Affiche l'état actuel du trading live (si actif)."""
    logging.info("Appel à live status")

    if not PID_FILE.exists():
        console.print("[green]Statut: Le processus live n'est pas en cours (aucun fichier PID trouvé).[/]")
        return # On ne retourne rien, l'affichage suffit

    try:
        pid_str = PID_FILE.read_text()
        pid = int(pid_str)
    except ValueError:
        console.print(f"[bold red]Erreur: Fichier PID invalide ({PID_FILE}). Contenu: '{pid_str}'. Utilisez 'live stop' pour nettoyer si possible.[/]")
        return # Affichage de l'erreur suffit
    except Exception as e:
        logging.exception(f"Erreur lors de la lecture du fichier PID {PID_FILE}")
        console.print(f"[bold red]Erreur lors de la lecture du fichier PID: {e}[/]")
        return # Affichage de l'erreur suffit

    if dry_run: # Utilisation directe
         # En dry-run, on vérifie juste si le fichier PID simulé existe
         if pid == 0 and PID_FILE.exists():
              console.print("[yellow]DRY-RUN: Le processus live est simulé comme étant 'en cours' (fichier PID simulé trouvé).[/]")
         else:
              console.print("[yellow]DRY-RUN: Le processus live est simulé comme étant 'arrêté' (pas de fichier PID simulé).[/]")
         return # Affichage suffit

    if psutil is None:
        console.print("[yellow]Module 'psutil' non disponible. Impossible de vérifier l'état réel du processus.[/]")
        console.print(f"Fichier PID trouvé ({PID_FILE}) avec PID {pid}. Le processus est [bold]probablement[/] en cours.")
        return # Affichage suffit

    try:
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            status = p.status()
            create_time = datetime.fromtimestamp(p.create_time()).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"[bold green]Statut: Live trading EN COURS[/]")
            console.print(f"  PID: {pid}")
            console.print(f"  Nom: {p.name()}")
            console.print(f"  Statut: {status}")
            console.print(f"  Démarré le: {create_time}")
            # Ajouter d'autres infos si pertinent (utilisation CPU/mémoire)
            # console.print(f"  CPU: {p.cpu_percent(interval=0.1)}%")
            # console.print(f"  Mémoire: {p.memory_info().rss / (1024 * 1024):.2f} MB")
        else:
            console.print(f"[yellow]Statut: Le processus live (PID {pid}) n'est PLUS en cours, mais le fichier PID existe.[/]")
            console.print(f"[yellow]Nettoyage du fichier PID obsolète: {PID_FILE}[/]")
            PID_FILE.unlink(missing_ok=True)

    except psutil.NoSuchProcess:
         console.print(f"[yellow]Statut: Le processus live (PID {pid}) n'existe plus (NoSuchProcess).[/]")
         console.print(f"[yellow]Nettoyage du fichier PID obsolète: {PID_FILE}[/]")
         PID_FILE.unlink(missing_ok=True)
    except psutil.AccessDenied:
         logging.warning(f"Accès refusé pour obtenir le statut du processus {pid}.")
         console.print(f"[yellow]Statut: Fichier PID trouvé (PID {pid}), mais accès refusé pour vérifier l'état réel.[/]")
         console.print("[yellow]Le processus est probablement en cours, mais impossible de confirmer.")
    except Exception as e:
         logging.exception(f"Erreur inattendue lors de la vérification du statut du processus {pid}")
         console.print(f"[bold red]Erreur inattendue lors de la vérification du statut: {e}[/]")
         # Ne pas supprimer le fichier PID en cas d'erreur inconnue
         return # Affichage de l'erreur suffit


# Groupe REPORT
# @report_app.command("metrics") # Supprimé
def report_metrics(
    # ctx: typer.Context, # Supprimé
    run_id: str, # = typer.Argument(..., help="ID du run d'entraînement ou de backtest.") # Modifié
    config_path: Path, # Ajouté
    dry_run: bool # Ajouté
):
    """Affiche les métriques clés pour un run spécifique."""
    console.print(f"[bold blue]Affichage des métriques pour le run ID: {run_id}[/]")

    if load_and_display_metrics is None:
        console.print("[bold red]Erreur: La fonction 'load_and_display_metrics' n'a pas pu être importée.[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de l'affichage des métriques pour {run_id}.[/]")
        return True # Indique le succès (simulé)

    try:
        # Supposer que la fonction affiche directement les métriques via console/logging
        # et retourne True/False ou lève une exception (ex: RunNotFoundError)
        success = load_and_display_metrics(
            run_id=run_id,
            config_path=config_path # Utilisation directe
        )

        if not success:
             console.print(f"[bold red]Impossible d'afficher les métriques pour le run {run_id}. Vérifiez les logs.[/]")
             return False # Indique l'échec
        # Si success est True, on suppose que l'affichage a été fait dans la fonction
        return True

    except FileNotFoundError: # Ou une exception custom comme RunNotFoundError
         logging.error(f"Run ID '{run_id}' non trouvé.")
         console.print(f"[bold red]Erreur: Run ID '{run_id}' non trouvé.[/]")
         return False # Indique l'échec
    except Exception as e:
        logging.exception(f"Erreur lors de l'affichage des métriques pour {run_id}:")
        console.print(f"[bold red]Erreur: {e}[/]")
        return False # Indique l'échec


# @report_app.command("export") # Supprimé
def report_export(
    # ctx: typer.Context, # Supprimé
    run_id: str, # = typer.Argument(..., help="ID du run d'entraînement ou de backtest."), # Modifié
    format: str, # = typer.Option("csv", "--format", "-f", help="Format d'export (csv, md, json).") # Modifié
    config_path: Path, # Ajouté
    dry_run: bool # Ajouté
):
    """Exporte les résultats et métriques d'un run."""
    console.print(f"[bold blue]Exportation du rapport pour le run ID: {run_id} au format {format}[/]")

    if export_run_report is None:
        console.print("[bold red]Erreur: La fonction 'export_run_report' n'a pas pu être importée.[/]")
        return False # Indique l'échec

    # Valider le format (optionnel mais recommandé)
    allowed_formats = ["csv", "md", "json"]
    if format.lower() not in allowed_formats:
        console.print(f"[bold red]Erreur: Format d'export '{format}' non supporté. Formats valides: {', '.join(allowed_formats)}[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print(f"[yellow]DRY-RUN: Simulation de l'export du rapport pour {run_id} au format {format}.[/]")
        return True # Indique le succès (simulé)

    try:
        # Supposer que la fonction gère la création du fichier et retourne le chemin ou True/False
        result_path = export_run_report(
            run_id=run_id,
            format=format.lower(),
            config_path=config_path # Utilisation directe
        )

        if result_path: # Si la fonction retourne le chemin du fichier créé
             console.print(f"[bold green]Rapport exporté avec succès : {result_path}[/]")
        elif result_path is True: # Si la fonction retourne juste un booléen
             console.print(f"[bold green]Rapport exporté avec succès (chemin non spécifié).[/]")
        else:
             console.print(f"[bold red]Impossible d'exporter le rapport pour le run {run_id}. Vérifiez les logs.[/]")
             return False # Indique l'échec

    except FileNotFoundError: # Ou RunNotFoundError
         logging.error(f"Run ID '{run_id}' non trouvé pour l'export.")
         console.print(f"[bold red]Erreur: Run ID '{run_id}' non trouvé.[/]")
         return False # Indique l'échec
    except Exception as e:
        logging.exception(f"Erreur lors de l'export du rapport pour {run_id}:")
        console.print(f"[bold red]Erreur: {e}[/]")
        return False # Indique l'échec
    return True # Indique le succès si aucune exception n'est levée


# Groupe UTILS (Exemple)
# Note: Typer convertit automatiquement clean_cache en clean-cache dans la CLI
# @utils_app.command("clean-cache") # Supprimé
def utils_clean_cache(dry_run: bool): # Ajouté dry_run
    """Nettoie les caches divers (LLM, temporaires, etc.)."""
    console.print("[bold blue]Nettoyage des caches...[/]")

    if clean_all_caches is None:
        console.print("[bold red]Erreur: La fonction 'clean_all_caches' n'a pas pu être importée depuis utils.maintenance.[/]")
        return False # Indique l'échec

    if dry_run: # Utilisation directe
        console.print("[yellow]DRY-RUN: Simulation du nettoyage des caches.[/]")
        # Appeler la fonction en mode dry_run si elle le supporte
        try:
             # Assumons que la fonction métier gère le dry_run
             clean_all_caches(dry_run=True)
             console.print("[yellow]DRY-RUN: Simulation terminée.[/]")
        except Exception as e:
             logging.error(f"Erreur lors de la simulation du nettoyage des caches: {e}")
             console.print(f"[bold red]Erreur lors de la simulation: {e}[/]")
        return True # Indique le succès (simulé)

    try:
        # Supposer que la fonction retourne True/False ou lève une exception
        success = clean_all_caches(dry_run=False) # Appeler explicitement sans dry_run

        if success:
            console.print("[bold green]Nettoyage des caches terminé avec succès.[/]")
        else:
            console.print("[bold red]Le nettoyage des caches a échoué. Vérifiez les logs.[/]")
            return False # Indique l'échec

    except Exception as e:
        logging.exception("Erreur lors du nettoyage des caches:")
        console.print(f"[bold red]Erreur: {e}[/]")
        return False # Indique l'échec
    return True # Indique le succès


# --- Point d'entrée ---
# --- Interface Interactive Principale ---

def handle_utils_menu():
    """Gère le sous-menu interactif pour les utilitaires."""
    console.print("\n[bold green]--- Utilities ---[/]")
    
    while True:
        menu_choice = questionary.select(
            "Choisissez une action:",
            choices=[
                "1. Clean Cache",
                "2. ← Back to Main Menu"
            ],
            qmark="🛠️",
            pointer="→"
        ).ask()

        if menu_choice is None or "2" in menu_choice:
            break

        action = menu_choice.split(".")[0].strip()

        try:
            if action == '1': # Clean Cache
                console.print("\n[bold blue]Nettoyage des caches en cours...[/]")
                success = utils_clean_cache(dry_run=global_settings['dry_run'])
                if success:
                    console.print("[bold green]Caches nettoyés avec succès![/]")
                else:
                    console.print("[bold red]Le nettoyage des caches a échoué.[/]")
            
            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            questionary.press_any_key_to_continue().ask()


def handle_options_menu():
    """Gère le sous-menu interactif pour les options globales."""
    console.print("\n[bold green]--- Options ---[/]")
    
    while True:
        menu_choice = questionary.select(
            "Modifiez les options globales:",
            choices=[
                f"1. Verbose Mode : [{'ON' if global_settings['verbose'] else 'OFF'}]",
                f"2. Dry Run Mode : [{'ON' if global_settings['dry_run'] else 'OFF'}]",
                "3. ← Back to Main Menu"
            ],
            qmark="⚙️",
            pointer="→"
        ).ask()

        if menu_choice is None or "3" in menu_choice:
            break

        action = menu_choice.split(".")[0].strip()

        try:
            if action == '1': # Toggle Verbose
                global_settings['verbose'] = not global_settings['verbose']
                setup_logging(global_settings['verbose'])
                console.print(f"[bold blue]Verbose Mode {'activé' if global_settings['verbose'] else 'désactivé'}[/]")
            
            elif action == '2': # Toggle Dry Run
                global_settings['dry_run'] = not global_settings['dry_run']
                console.print(f"[bold blue]Dry Run Mode {'activé' if global_settings['dry_run'] else 'désactivé'}[/]")
            
            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            console.print(f"[bold red]Error: {e}[/]")
            questionary.press_any_key_to_continue().ask()

def run_interactive_menu():
    """Lance la boucle principale du menu interactif."""
    global config_data, global_settings # Utiliser les variables globales

    # Charger la configuration au démarrage
    config_data = load_config(global_settings["config_path"])
    if not config_data:
        console.print("[bold yellow]Avertissement: Impossible de charger la configuration. Les valeurs par défaut pourraient être incorrectes.[/]")

    # Configurer le logging initial (peut être modifié par les options)
    setup_logging(global_settings["verbose"])

    # TODO: Ajouter ici un prompt initial pour configurer verbose/dry_run si souhaité

    while True:
        # Effacer la console pour une meilleure lisibilité (optionnel)
        # console.clear() # Peut être déroutant, à tester

        console.print("\n[bold cyan]--- Morningstar Interactive CLI ---[/]")
        try:
            choice = questionary.select(
                "Menu Principal:",
                choices=[
                    "1) Data Management",
                    "2) Model Training",
                    "3) Evaluation & Metrics",
                    "4) Backtesting",
                    "5) Live Trading",
                    "6) Utilities",
                    "7) Options", # Ajouté pour gérer verbose/dry-run
                    "8) Exit",
                ],
                qmark="▶",
                pointer="→"
            ).ask()

            if choice is None or choice == "8) Exit": # Gère Ctrl+C et la sélection explicite
                console.print("[bold yellow]Au revoir ![/]")
                break
            elif choice == "1) Data Management":
                handle_data_menu() # Appel de la fonction du sous-menu
            elif choice == "2) Model Training":
                handle_train_menu() # Appel de la fonction du sous-menu
            elif choice == "3) Evaluation & Metrics":
                handle_eval_menu()
            elif choice == "4) Backtesting":
                handle_backtest_menu()
            elif choice == "5) Live Trading":
                handle_live_menu()
            elif choice == "6) Utilities":
                handle_utils_menu()
            elif choice == "7) Options":
                handle_options_menu()

        except KeyboardInterrupt: # Gérer Ctrl+C proprement
             console.print("\n[bold yellow]Interruption détectée. Au revoir ![/]")
             break
        except Exception as e: # Gérer les erreurs inattendues dans la boucle principale
             logging.exception("Erreur inattendue dans la boucle du menu principal:")
             console.print(f"[bold red]Une erreur inattendue est survenue: {e}[/]")
             console.print("[yellow]Retour au menu principal...[/]")
             questionary.press_any_key_to_continue().ask()


# --- Point d'entrée ---
if __name__ == "__main__":
    # Lancer l'interface interactive
    run_interactive_menu()
