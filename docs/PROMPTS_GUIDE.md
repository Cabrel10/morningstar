# Guide des Prompts LLM

## Prompts pour Embeddings
```python
prompt_embedding = '''
Analyze this market context and generate a dense vector representation 
that captures: sentiment, key events, and market regime indicators.

Context: {text_input}
'''
```

## Prompts pour Analyse
```python
prompt_analysis = '''
Analyze this trading scenario and return JSON with:
- sentiment (positive/neutral/negative)
- confidence_score (0-1) 
- potential_actions (list)
- market_conditions (text)

Scenario: {market_situation}
'''
```

## Bonnes Pratiques
1. **Reproductibilité** :
   - Versionner les prompts avec git
   - Sauvegarder les templates dans ce fichier

2. **Sécurité** :
   - Ne pas inclure d'informations sensibles
   - Utiliser des placeholders pour les variables

3. **Optimisation** :
   - Tester différents formulations
   - Documenter les performances
