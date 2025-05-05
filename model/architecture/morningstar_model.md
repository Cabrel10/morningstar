# Documentation de l'Interface du Modèle Morningstar (`MorningstarModel`)

## Rôle

La classe `MorningstarModel` sert de **wrapper** (interface) pour le modèle d'architecture complexe `MorningstarHybridModel` (défini dans `enhanced_hybrid_model.py`). Elle simplifie l'utilisation du modèle dans le workflow de trading en fournissant des méthodes standardisées pour l'initialisation, la prédiction, le chargement/sauvegarde des poids, etc.

L'architecture sous-jacente (`MorningstarHybridModel`) combine des données techniques et des embeddings LLM via un module de fusion, puis utilise plusieurs têtes de prédiction spécialisées. Voir `enhanced_hybrid_model.md` pour les détails de l'architecture interne.

## Interface Publique (`MorningstarModel`)

### Méthodes Principales

1.  **`__init__(model_config=None)`**
    - Initialise le wrapper. Prend un dictionnaire de configuration optionnel pour paramétrer le `MorningstarHybridModel` sous-jacent.

2.  **`initialize_model()`**
    - Construit l'architecture du modèle `MorningstarHybridModel` en utilisant la configuration fournie. Doit être appelée avant `predict` ou `load_weights` si le modèle n'est pas déjà initialisé.

3.  **`predict(technical_data, llm_embeddings)`**
    - **Entrées**:
        - `technical_data`: Array numpy de shape `[batch_size, num_technical_features]` (ex: `[10, 38]`)
        - `llm_embeddings`: Array numpy de shape `[batch_size, llm_embedding_dim]` (ex: `[10, 768]`)
    - **Sorties**: Un dictionnaire indiquant le statut de l'opération et le résultat (ou un message d'erreur).
      - **En cas de succès :**
        ```python
        {
            'status': 'success',
            'result': {
                'signal': np.ndarray[batch_size, 5],           # Probabilités (softmax) pour Strong Sell, Sell, Hold, Buy, Strong Buy
                'volatility_quantiles': np.ndarray[batch_size, 3],  # Prédiction des quantiles (ex: 0.1, 0.5, 0.9)
                'volatility_regime': np.ndarray[batch_size, 3],     # Probabilités (softmax) pour les régimes de volatilité (ex: Low, Medium, High)
                'market_regime': np.ndarray[batch_size, 4],    # Probabilités (softmax) pour les régimes de marché (ex: Bullish, Bearish, Lateral, Volatile)
                'sl_tp': np.ndarray[batch_size, 2]             # Valeurs prédites pour Stop Loss / Take Profit (placeholder RL)
            }
        }
        ```
      - **En cas d'erreur :**
        ```python
        {
            'status': 'error',
            'message': 'Description de l'erreur rencontrée (ex: shape incorrecte, modèle non initialisé...)'
        }
        ```

4.  **`save_weights(filepath)`**
    - Sauvegarde les poids du modèle entraîné au format HDF5 (`.h5`).

5.  **`load_weights(filepath)`**
    - Charge les poids d'un modèle pré-entraîné depuis un fichier HDF5. Initialise le modèle si nécessaire avant le chargement.

6.  **`get_model_summary()`**
    - Retourne un résumé textuel de l'architecture du modèle (`MorningstarHybridModel`). Utile pour le débogage.

7.  **`prepare_for_inference()`**
    - Applique des optimisations potentielles au modèle pour accélérer l'inférence (actuellement, effectue une recompilation simple). À appeler avant le déploiement en production.

## Intégration avec le Workflow (`trading_workflow.py`)

Le `TradingWorkflow` utilise cette classe `MorningstarModel` pour interagir avec le modèle de prédiction :

1.  Instanciation : `model_interface = MorningstarModel(config)`
2.  Chargement des poids : `model_interface.load_weights('path/to/saved_model.h5')`
3.  Préparation (optionnel mais recommandé) : `model_interface.prepare_for_inference()`
4.  Prédiction sur de nouvelles données : `predictions = model_interface.predict(tech_data, llm_data)`
5.  Utilisation des `predictions` (dictionnaire) pour la prise de décision.

## Tests Unitaires

Les tests vérifient:

- La cohérence des shapes de sortie
- La gestion des erreurs sur inputs invalides
- La sérialisation/deserialisation des poids
- La stabilité numérique des prédictions

Exemple de test:

```python
def test_predict_output_shapes():
    # Utilise la configuration par défaut
    model_wrapper = MorningstarModel() 
    model_wrapper.initialize_model() # Important: Initialiser avant predict
    
    batch_size = 10
    num_tech_features = model_wrapper.config['num_technical_features']
    llm_dim = model_wrapper.config['llm_embedding_dim']
    
    technical = np.random.rand(batch_size, num_tech_features)
    llm = np.random.rand(batch_size, llm_dim)
    
    preds = model_wrapper.predict(technical, llm)
    
    assert isinstance(preds, dict)
    assert preds['signal'].shape == (batch_size, model_wrapper.config['num_signal_classes'])
    assert preds['volatility_quantiles'].shape == (batch_size, 3) # Assumant 3 quantiles
    assert preds['volatility_regime'].shape == (batch_size, model_wrapper.config['num_volatility_regimes'])
    assert preds['market_regime'].shape == (batch_size, model_wrapper.config['num_market_regimes'])
    assert preds['sl_tp'].shape == (batch_size, 2) # Assumant 2 valeurs SL/TP
```

## Bonnes Pratiques

1. Toujours appeler `prepare_for_inference()` avant de déployer
2. Vérifier les shapes des inputs avant prédiction
3. Monitorer les performances via les logs:

```log
INFO:MorningstarModel:Prédiction effectuée - temps: 45ms
INFO:MorningstarModel:Volatilité détectée: HIGH (prob: 0.87)
