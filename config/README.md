# Documentation de la Configuration - Morningstar V2

Ce dossier contient les fichiers nécessaires à la configuration du robot de trading Morningstar V2.

## Fichiers

*   **`config.yaml`**:
    *   **Rôle**: Fichier principal de configuration au format YAML. Contient tous les paramètres non sensibles nécessaires au fonctionnement du système.
    *   **Contenu Typique**:
        *   Chemins vers les données, modèles, logs, secrets.
        *   Liste des exchanges actifs (`active_exchanges`).
        *   Liste des paires de trading (`trading_pairs`).
        *   Paramètres de préparation des données (`data_preparation`): indicateurs à calculer, fenêtres, méthode de nettoyage...
        *   Paramètres du modèle (`model_params`): `time_window`, `lstm_units`, têtes actives...
        *   Paramètres d'entraînement (`training_params`): `epochs`, `batch_size`, `learning_rate`, poids des pertes...
        *   Paramètres du workflow (`workflow_params`): fréquence d'exécution, mode (`live`/`backtest`), seuils de décision...
        *   Paramètres de gestion de risque (`risk_management`): profil de risque, taille de position max/par défaut, drawdown max...
        *   Paramètres de l'API LLM (`llm_params`): modèle à utiliser, température...
        *   Paramètres de logging (`logging_params`): niveau de log, format...
    *   **Gestion**: Ce fichier **peut être versionné** (commit Git) car il ne contient pas d'informations sensibles.

*   **`secrets.env`**:
    *   **Rôle**: Fichier contenant toutes les informations sensibles (clés API, mots de passe API, etc.) sous forme de variables d'environnement.
    *   **Format**: Clé=Valeur, une par ligne.
        ```dotenv
        # Example secrets.env
        BINANCE_API_KEY=VotreCleBinanceApiKey
        BINANCE_SECRET_KEY=VotreCleBinanceSecretKey

        KUCOIN_API_KEY=VotreCleKucoinApiKey
        KUCOIN_SECRET_KEY=VotreCleKucoinSecretKey
        KUCOIN_API_PASSWORD=VotreMotDePasseApiKucoin

        BITGET_API_KEY=VotreCleBitgetApiKey
        BITGET_SECRET_KEY=VotreCleBitgetSecretKey
        BITGET_API_PASSWORD=VotreMotDePasseApiBitget

        OPENAI_API_KEY=VotreCleOpenAIApiKey

        # Autres clés (NewsAPI, Twitter...)
        # NEWSAPI_KEY=...
        ```
    *   **Gestion**: Ce fichier **NE DOIT JAMAIS ÊTRE VERSIONNÉ** (commit Git). Il doit être listé dans le fichier `.gitignore` à la racine du projet. Chaque utilisateur doit créer son propre fichier `secrets.env` localement.

*   **`secrets.env.example`** (À créer manuellement):
    *   **Rôle**: Fichier modèle (template) pour `secrets.env`. Il liste toutes les variables d'environnement attendues mais avec des valeurs vides ou des placeholders.
    *   **Gestion**: Ce fichier **doit être versionné** pour que les autres utilisateurs sachent quelles variables définir dans leur propre `secrets.env`.
    *   **Exemple `secrets.env.example`**:
        ```dotenv
        # Template for secrets.env - DO NOT COMMIT secrets.env itself!
        BINANCE_API_KEY=
        BINANCE_SECRET_KEY=

        KUCOIN_API_KEY=
        KUCOIN_SECRET_KEY=
        KUCOIN_API_PASSWORD=

        BITGET_API_KEY=
        BITGET_SECRET_KEY=
        BITGET_API_PASSWORD=

        OPENAI_API_KEY=

        # NEWSAPI_KEY=
        ```

## Utilisation

1.  **Copier le Modèle**: Copiez `config/secrets.env.example` vers `config/secrets.env`.
2.  **Remplir les Secrets**: Ouvrez `config/secrets.env` et remplissez-le avec vos propres clés API et informations sensibles.
3.  **Configurer les Paramètres**: Modifiez `config/config.yaml` pour ajuster les paramètres du robot à vos besoins.
4.  **Chargement**: Le module `utils/api_manager.py` utilisera `python-dotenv` pour charger automatiquement les variables depuis `config/secrets.env`. Les autres modules chargeront `config/config.yaml` (ex: avec `PyYAML`) pour accéder aux paramètres généraux.

## Sécurité

*   **Ne jamais partager votre fichier `secrets.env`**.
*   **Ne jamais commiter `secrets.env` dans Git**. Assurez-vous qu'il est bien présent dans `.gitignore`.
*   Utilisez des clés API avec des permissions restreintes si possible (ex: permissions de trading activées, mais retraits désactivés).
*   Envisagez des solutions plus sécurisées pour la gestion des secrets en production (ex: HashiCorp Vault, gestionnaires de secrets cloud AWS/GCP/Azure).
