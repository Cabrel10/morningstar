# Module 2: Chain-of-Thought (CoT) pour le raisonnement explicite

## Vue d'ensemble

Le Module 2 du framework DECoT-RL-GA implémente un mécanisme de Chain-of-Thought (CoT) qui permet au système de trading de générer des explications claires et logiques pour ses décisions. Ce module transforme le système d'une "boîte noire" en un système transparent qui peut expliquer son raisonnement.

## Fonctionnalités

- **Génération d'explications** : Produit des explications textuelles pour chaque décision de trading
- **Analyse multi-facteurs** : Intègre l'analyse technique, les indicateurs de marché et les caractéristiques extraites
- **Raisonnement structuré** : Suit une séquence logique d'étapes de raisonnement pour arriver à une décision
- **Justification des actions** : Explique pourquoi une action spécifique (achat, vente, maintien) est recommandée

## Implémentation

Le module est implémenté dans les fichiers suivants :
- `model/reasoning/chain_of_thought.py` : Implémentation du mécanisme de raisonnement
- `generate_trading_explanations.py` : Génération d'explications pour les décisions de trading

## Utilisation

```python
from model.reasoning.chain_of_thought import ChainOfThoughtReasoning

# Initialiser le module CoT
cot = ChainOfThoughtReasoning()

# Générer une explication pour une décision de trading
explanation = cot.generate_explanation(
    market_data=current_data,
    technical_indicators=indicators,
    extracted_features=features,
    action="buy"
)

print(explanation)
# Exemple de sortie:
# "J'ai décidé d'acheter BTC/USDT pour les raisons suivantes:
# 1. Le RSI est à 32, indiquant une condition de survente
# 2. Le MACD montre un croisement haussier
# 3. Le volume a augmenté de 15% par rapport à la moyenne
# 4. Le modèle CNN+LSTM détecte un motif de retournement haussier
# En conclusion, ces facteurs suggèrent une forte probabilité de mouvement haussier à court terme."
```

## Paramètres optimisables

Les hyperparamètres suivants peuvent être optimisés par le Module 4 (GA) :
- `reasoning_depth` : Profondeur du raisonnement (nombre d'étapes de raisonnement)
- `factors_weight` : Poids attribués à différents facteurs dans le raisonnement

## Intégration avec les autres modules

Le Module 2 reçoit les caractéristiques extraites du Module 1 (CNN+LSTM) et fournit des explications pour les actions recommandées par le Module 3 (RL). Ces explications aident à comprendre les décisions de l'agent RL et à renforcer la confiance dans le système.
