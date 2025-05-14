# Plan de Backtesting Initial (Étape 1)

Ce document décrit le plan pour la mise en place de la première étape de backtesting du modèle Morningstar, conformément à la demande initiale.

## Objectif

Créer un script de backtest dédié (`backtest.py`) utilisant `backtrader` pour valider la stratégie basée sur les signaux du modèle Morningstar, en calculant les métriques de performance clés et en tenant compte des coûts de transaction et du slippage.

## Analyse Préliminaire

L'analyse du projet a révélé :
*   Une structure de projet organisée.
*   Un modèle hybride complexe (`enhanced_hybrid_model`) utilisé dans le workflow de trading live (`workflows/trading_workflow.py`).
*   Un workflow d'entraînement séparé (`Morningstar/workflows/training_workflow.py`) pour un `MorningstarModel`.
*   Un fichier `backtest.py` existant utilisant `backtrader` mais contenant du code superflu lié à `ccxt`.
*   Un notebook `backtest.ipynb` utilisant une approche de backtesting vectorisée avec Pandas et contenant des implémentations pour le stress testing et la walk-forward analysis (mais utilisant des signaux factices).

## Plan d'Action Détaillé

1.  **Nettoyer `backtest.py` :**
    *   Supprimer la section de code liée à `ccxt` (approximativement lignes 158-351) qui concerne l'exécution live et est gérée ailleurs (`utils/api_manager.py`, `workflows/trading_workflow.py`).
    *   Conserver et se concentrer sur la partie utilisant la bibliothèque `backtrader`.

2.  **Améliorer le Script `backtrader` dans `backtest.py` :**
    *   **Ajouter les Analyseurs `backtrader` :** Intégrer les analyseurs suivants dans l'instance `Cerebro` :
        *   `bt.analyzers.TradeAnalyzer`: Pour obtenir le P&L total et moyen par trade, le nombre de trades gagnants/perdants, etc. (permet de calculer le taux de réussite).
        *   `bt.analyzers.SharpeRatio`: Pour calculer le ratio de Sharpe (en spécifiant `riskfreerate=0.0` si nécessaire).
        *   `bt.analyzers.DrawDown`: Pour obtenir le drawdown maximum du portefeuille.
    *   **Afficher les Résultats des Analyseurs :** Modifier la section `if __name__ == '__main__':` après `cerebro.run()` pour :
        *   Récupérer les résultats de chaque analyseur ajouté (ex: `results = cerebro.run()`, puis `results[0].analyzers.tradeanalyzer.get_analysis()`).
        *   Extraire les métriques spécifiques (P&L, Max Drawdown, Sharpe Ratio, Win Rate) des dictionnaires retournés par les analyseurs.
        *   Afficher ces métriques de manière claire dans la console.
    *   **Confirmer les Coûts/Slippage :** S'assurer que les appels `cerebro.broker.setcommission(commission=TRANSACTION_COST)` et `cerebro.broker.set_slippage_perc(perc=SLIPPAGE)` sont présents et utilisent les constantes définies.

## Prochaines Étapes (Après Implémentation)

Une fois ce script de backtest initial fonctionnel et validé :
1.  Implémenter la Walk-Forward Analysis (potentiellement en s'inspirant de `backtest.ipynb` et en l'adaptant à `backtrader`).
2.  Implémenter le Stress-Testing (potentiellement en s'inspirant de `backtest.ipynb` et en l'adaptant à `backtrader`).

## Diagramme Mermaid du Plan

```mermaid
graph LR
    A[Analyser Projet] --> B(Nettoyer backtest.py);
    B --> C{Améliorer backtest.py};
    C --> D[Ajouter Analyseurs Backtrader];
    C --> E[Afficher Résultats Analyseurs];
    C --> F[Vérifier Coûts/Slippage];
    A --> G(Examiner backtest.ipynb);
    D & E & F & G --> H{Validation Script Backtest};
    H --> I[Prochaines Étapes: Walk-Forward & Stress-Test];