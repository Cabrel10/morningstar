## Handover Document for New Development Service

### 1. Project Overview
**Nom du projet**: CryptoRobot - Morningstar

Une plateforme de trading algorithmique qui combine des indicateurs techniques et des embeddings LLM pour prédire:
- Signaux de trading (signal)
- Quantiles de volatilité (volatility_quantiles)
- Régime de volatilité (volatility_regime)
- Régime de marché (market_regime)
- Points stop-loss et take-profit (sl_tp)

Le projet utilise TensorFlow/Keras pour le modèle hybride, un pipeline de données en Python pour préparer les features et labels, et des notebooks (local + Colab) pour l'entraînement et l'évaluation.

---

### 2. Architecture & Composants Clés

1. **Data Pipeline** (`data/pipelines/data_pipeline.py`)
   - Chargement et nettoyage des données brutes CSV
   - Feature engineering: calcul de 19 indicateurs techniques (SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastics)
   - Labellisation: création de colonnes target et placeholders
   - Complétion à 38 features techniques via colonnes factices
   - (Simulation) Génération de 768 embeddings LLM factices pour chaque point temporel
   - Sauvegarde finale en Parquet (`data/processed/*.parquet`)

2. **Data Loader** (`model/training/data_loader.py`)
   - Lecture des fichiers Parquet
   - Sélection des 38 features techniques et 768 embeddings LLM
   - Mapping des labels textuels (`market_regime`, `signal`) en entiers
   - Split des données en train/validation
   - Support `as_tensor=True` pour fournir des tenseurs TF directement

3. **Architecture du Modèle** (`model/architecture/enhanced_hybrid_model.py`)
   - Modèle Keras à deux branches (technique & LLM)
   - Fusion multimodale suivie de têtes de prédiction (5 sorties)
   - Wrapper `MorningstarModel` pour initialiser, compiler, prédire, sauvegarder/recharger

4. **Pipeline d’Entraînement**
   - Script Python (`model/training/training_script.py`) et notebook local (`notebooks/training_local.ipynb`)
   - Notebook Colab (`notebooks/training_on_colab.ipynb`) monté sur Google Drive
   - Callbacks: `ModelCheckpoint`, `EarlyStopping`

5. **Intégration LLM** (`utils/llm_integration.py`)
   - Module hybride: priorise Google AI API / Gemini
   - Fallback sur OpenRouter
   - Cache multi-niveaux (Redis + fichiers locaux)
   - Fonctions: `get_embeddings()`, `get_llm_analysis()`

6. **Tests & Validation** (`tests/`)
   - **Unitaires**: Data pipeline, Feature engineering, Data loader, Modèle, Workflow, LLM Integration
   - **End-to-end**: Pipeline complet avec mock data
   - 100% des tests passent localement sur `pytest` et `unittest`

---

### 3. Environnement & Pré-requis

- **Python** 3.9+
- **TensorFlow** 2.x
- **Redis** (pour cache LLM)
- **Clés API**:
  - `GOOGLE_API_KEY` (Gemini)
  - `OPENROUTER_API_KEY`
- **Fichiers de configuration**:
  - `config/config.yaml`
  - `config/secrets.env` (à remplir)
- **Notebooks**: Jupyter local & Google Colab

---

### 4. Points d’Attention Immédiats

1. **Mismatch Features**: `data_pipeline.py` doit fournir exactement 38 colonnes techniques avant embeddings. Corriger l’algorithme d’ajout de colonnes factices.
2. **Gestion Embeddings Réels**: Remplacer la simulation par une intégration LLM réelle (via `utils/llm_integration.py`) dans `data_pipeline.py`.
3. **Alignement Labels**: Vérifier la correspondance stricte entre noms de colonnes, clés de loss/metrics et sorties du modèle.
4. **Notebooks**: Adapter les chemins (local vs Drive), vérifier l’utilisation de `MorningstarModel` et la gestion des deux inputs.

---

### 5. Prochaines Étapes & Livrables

1. **Finaliser Data Pipeline**: Ajouter source textuelle, générer embeddings LLM réels.
2. **Exécuter Entraînement**: Tester sur Colab, ajuster hyperparamètres, analyser résultats.
3. **Évaluer & Fine-tuner**: Implémenter pipeline de fine-tuning, explorer différentes architectures (heads, layers).
4. **Déploiement**: Préparer un service REST (FastAPI ou Flask) pour servir les prédictions.
5. **Monitoring & Logging**: Mettre en place des métriques (prometheus), logs structurés, alerting.

---

### 6. Exigences au Nouveau Service

- **Compréhension**: Maîtrise des pipelines de données Pandas & Parquet, TensorFlow/Keras, intégration API externes.
- **Expérience**: API LLM (Google AI, OpenRouter), cache Redis, déploiement notebook & service.
- **Qualité**: Couverture tests, CI/CD (pytest, flake8), documentation à jour.

Merci de démarrer vos travaux à partir de cette base et de valider chaque étape par des tests automatisés et des revues de code.

