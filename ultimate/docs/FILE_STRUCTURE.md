# Structure des Fichiers - Morningstar V3 (Avec Intégration LLM)

## Nouveautés de la V3
- **Double entrée modèle** : Données techniques (38 features) + Embeddings LLM (768 dim)
- **Cache LLM** : Dossier `data/llm_cache/` pour stocker les embeddings
- **Multi-tâches** : 5 sorties (signal, régimes, sl/tp)

## Structure Principale

```
data/
├── llm_cache/            # Cache des embeddings LLM (JSON)
├── processed/            # Données avec features techniques + embeddings LLM
└── pipelines/
    └── data_pipeline.py  # Génère maintenant les 2 types de features

model/
├── architecture/
│   └── enhanced_hybrid_model.py  # Modèle à 2 entrées (tech + LLM)
└── training/
    └── data_loader.py    # Charge les paires (X_technical, X_llm)

utils/
└── llm_integration.py    # Nouveau module pour générer les embeddings
