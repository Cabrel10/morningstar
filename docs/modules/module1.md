# Module 1: CNN+LSTM pour l'extraction de caractéristiques

## Vue d'ensemble

Le Module 1 du framework DECoT-RL-GA implémente un modèle hybride CNN+LSTM pour l'extraction de caractéristiques à partir des données de marché. Ce modèle est conçu pour capturer à la fois les motifs spatiaux (via CNN) et les dépendances temporelles (via LSTM) dans les données de trading.

## Fonctionnalités

- **Architecture CNN+LSTM** : Combine des couches convolutives pour l'extraction de caractéristiques locales et des couches LSTM pour capturer les dépendances temporelles
- **Prétraitement des données** : Normalisation et préparation des données de marché pour l'entraînement
- **Extraction de caractéristiques** : Génération de représentations de haut niveau des données de marché

## Implémentation

Le module est implémenté dans les fichiers suivants :
- `model/architecture/cnn_lstm.py` : Définition de l'architecture du modèle hybride
- `model/training/data_loader.py` : Chargement et prétraitement des données

## Utilisation

```python
from model.architecture.cnn_lstm import create_cnn_lstm_model
from model.training.data_loader import load_market_data

# Charger les données
data = load_market_data(data_path, window_size=30)

# Créer le modèle
model = create_cnn_lstm_model(
    input_shape=(30, 20),  # (window_size, features)
    cnn_filters=64,
    cnn_kernel_size=3,
    lstm_units=128,
    dropout_rate=0.2
)

# Utiliser le modèle pour l'extraction de caractéristiques
features = model.predict(data)
```

## Paramètres optimisables

Les hyperparamètres suivants peuvent être optimisés par le Module 4 (GA) :
- `cnn_filters` : Nombre de filtres dans les couches convolutives
- `cnn_kernel_size` : Taille du noyau des filtres convolutifs
- `lstm_units` : Nombre d'unités dans les couches LSTM
- `dropout_rate` : Taux de dropout pour la régularisation

## Intégration avec les autres modules

Le Module 1 fournit les caractéristiques extraites au Module 2 (CoT) pour le raisonnement explicite et au Module 3 (RL) pour l'apprentissage des stratégies de trading.
