# Checklist d'Implémentation - Version LLM

## Pré-requis
- [ ] Clé API Google Generative AI
- [ ] Compte OpenRouter (backup)
- [ ] Redis configuré pour le cache (optionnel)

## Étapes d'Intégration LLM

1. **Génération des embeddings**:
   - [x] Configurer le cache (Redis ou fichiers locaux)
   - [x] Tester la connexion aux APIs LLM
   - [x] Générer les embeddings pour l'historique
   - [x] Valider la dimension (768)

2. **Modèle**:
   - [ ] Vérifier les shapes d'entrée (38 + 768)
   - [ ] Tester avec embeddings simulés
   - [ ] Tester avec vrais embeddings
   - [ ] Valider les 5 sorties

3. **Workflow**:
   - [ ] Mettre à jour data_pipeline.py
   - [ ] Adapter les notebooks Colab/local
   - [ ] Configurer le monitoring des embeddings

## Bonnes Pratiques
- [ ] Versionner les embeddings avec les données
- [ ] Documenter les prompts utilisés
- [ ] Prévoir un fallback (embeddings simulés)
