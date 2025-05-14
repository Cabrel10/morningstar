# Plan d'Implémentation : Interface CLI Interactive (Remplacement Complet)

Ce document détaille le plan pour remplacer l'interface CLI actuelle basée sur `typer` par une nouvelle interface entièrement interactive utilisant la bibliothèque `questionary`.

## Objectif

Remplacer l'interface en ligne de commande existante par une interface pilotée par menus textuels pour améliorer l'ergonomie et la facilité d'utilisation, tout en conservant la logique métier sous-jacente.

## Plan d'Implémentation

1.  **Choisir une Bibliothèque Interactive :**
    *   Utiliser `questionary` pour créer les menus, les prompts de sélection (listes, cases à cocher, texte), et la validation des entrées.

2.  **Refactoriser `cli.py` :**
    *   Supprimer toute la structure basée sur `typer` (`@app.command`, `@data_app.command`, `typer.Typer()`, etc.).
    *   Conserver les imports essentiels des modules métier (`utils`, `data`, `model`, `live`, `workflows`).
    *   Conserver la configuration du logging avec `RichHandler`.
    *   Conserver la logique de gestion du fichier PID (`run/live.pid`) et l'utilisation de `psutil` (ou `os.kill`) pour le contrôle des processus live.
    *   Modifier le point d'entrée principal (`if __name__ == "__main__":`) pour qu'il lance directement la boucle du menu interactif principal.

3.  **Implémenter la Logique des Menus Interactifs :**
    *   Créer une fonction principale (ex: `run_interactive_menu()`) qui affiche le menu principal et gère la navigation entre les sous-menus. Utiliser une boucle `while` pour maintenir l'interface active jusqu'à ce que l'utilisateur choisisse de quitter.
    *   Développer des fonctions Python dédiées pour chaque sous-menu (Data, Train, Eval, Backtest, Live, Utils).
    *   Utiliser les différents types de prompts de `questionary` (`select`, `checkbox`, `text`, `confirm`, etc.) pour afficher les options, les paramètres et les actions.
    *   **Lecture de la Configuration :** Implémenter une fonction helper (ex: `load_config()`) pour charger `config.yaml` au démarrage. Les valeurs par défaut des prompts seront lues depuis cet objet de configuration.
    *   **Gestion de l'État :** Maintenir l'état des paramètres modifiés par l'utilisateur au sein de la session interactive. Un dictionnaire ou un objet de contexte passé entre les fonctions de menu peut être utilisé.

4.  **Connecter les Actions aux Fonctions Métier :**
    *   Lorsqu'une action est sélectionnée dans un menu :
        *   Récupérer les paramètres nécessaires depuis l'état actuel du menu (valeurs par défaut de la config ou valeurs modifiées par l'utilisateur).
        *   Appeler **directement** la fonction métier correspondante importée (ex: `fetch_ohlcv_data`, `run_pipeline`, `run_local_training`, `run_simple_backtest`, etc.).
        *   Passer les arguments requis (paramètres du menu, chemin de config, état `dry_run`) à ces fonctions.
        *   Utiliser `rich.console` pour afficher les messages de statut et `rich.progress` pour les tâches longues, en s'inspirant de l'implémentation existante dans `cli.py`.

5.  **Adapter la Gestion Live :**
    *   Créer des fonctions wrapper dédiées (ex: `start_live_process`, `stop_live_process`, `get_live_status`) dans `cli.py` ou un module utilitaire.
    *   Ces wrappers encapsuleront la logique de :
        *   Lancement de `run_live.py` via `subprocess.Popen`.
        *   Création et lecture du fichier `run/live.pid`.
        *   Vérification de l'existence du processus et envoi de signaux (SIGTERM, SIGKILL) via `psutil` ou `os.kill`.
        *   Nettoyage du fichier PID.
    *   Les actions correspondantes dans le menu "Live Trading" appelleront ces fonctions wrapper.

6.  **Gérer les Utilitaires :**
    *   Connecter les actions du menu "Utilities" (ex: "Clear cache", "Validate configuration") aux fonctions métier appropriées (ex: `utils.maintenance.clean_all_caches`, une nouvelle fonction de validation de config).

7.  **Gestion des Options Globales (Verbose, Dry-Run) :**
    *   Au démarrage de `run_interactive_menu()` ou via un sous-menu "Options", proposer à l'utilisateur de définir ces modes.
    *   Le mode `verbose` ajustera le niveau de logging global (`logging.basicConfig(level=...)`).
    *   Le mode `dry_run` sera stocké dans l'état de la session et passé aux fonctions métier ou utilisé pour conditionner leur appel.

8.  **Tests et Raffinements :**
    *   Tester de manière exhaustive :
        *   La navigation dans tous les menus et sous-menus.
        *   La sélection et la modification des paramètres.
        *   L'exécution de chaque action et la vérification des résultats attendus.
        *   La gestion des erreurs (entrées invalides, fichiers manquants, échecs des fonctions métier).
        *   Le fonctionnement des modes `verbose` et `dry_run`.
        *   L'expérience utilisateur globale (clarté des prompts, feedback).

## Diagramme Mermaid (Flux Simplifié - Remplacement Complet)

```mermaid
graph TD
    A[Lancer `python cli.py`] --> B{Chargement Config & Initialisation};
    B --> C{Menu Principal Interactif};
    C -- Choix 1 --> D{Menu Data Management};
    D -- Sélection Params (Asset, TF...) --> D;
    D -- Action 1 (Download) --> E[Appel direct `fetch_ohlcv_data` & `save_data`];
    E --> F[Affichage Résultat/Progression];
    F --> D;
    D -- Action 4 (Back) --> C;
    C -- Choix 5 --> G{Menu Live Trading};
    G -- Action 1 (Start) --> H[Appel `start_live_process` (utilise subprocess, PID)];
    H --> I[Affichage Résultat];
    I --> G;
    G -- Action 3 (Stop) --> J[Appel `stop_live_process` (utilise PID, psutil/os.kill)];
    J --> K[Affichage Résultat];
    K --> G;
    G -- Action 4 (Back) --> C;
    C -- Choix 7 (Exit) --> L[Quitter];

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px