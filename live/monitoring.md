# Spécification : Module de Monitoring et Supervision

**Objectif :** Définir les fonctionnalités, les composants et le comportement attendu du module de monitoring (`live/monitoring.py`), incluant le dashboard de suivi et l'intégration de la supervision par LLM.

---

## 1. Vue d'ensemble

Le module de monitoring est crucial pour la surveillance en temps réel des performances, de l'état du système et des décisions prises par le robot de trading Morningstar V2. Il intègre également l'interaction avec le LLM pour fournir une couche de supervision intelligente.

Il peut être composé de deux parties principales :
1.  **Collecte et Stockage de Métriques**: Enregistrement continu des données pertinentes.
2.  **Visualisation et Interaction**: Un dashboard (ex: Streamlit, Dash) et/ou des scripts pour afficher les informations et interagir avec le LLM.

---

## 2. Fonctionnalités Clés

*   **Suivi des Performances**:
    *   Calcul et affichage du P&L (Profit and Loss) global et par paire/stratégie (réalisé et non réalisé).
    *   Calcul et affichage de métriques clés : Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor (calculées sur une fenêtre glissante ou depuis le début).
    *   Visualisation de la courbe d'équité (Equity Curve).
*   **État du Système**:
    *   Affichage de l'état des connexions aux exchanges.
    *   Suivi des ressources système (CPU, mémoire, GPU si utilisé).
    *   Affichage des derniers logs d'erreurs ou d'avertissements importants.
*   **Suivi des Positions et Ordres**:
    *   Affichage des positions actuellement ouvertes (taille, prix d'entrée, P&L non réalisé, SL/TP actuels).
    *   Historique récent des ordres passés et de leur statut.
*   **Visualisation des Prédictions**:
    *   Affichage des dernières prédictions du modèle `EnhancedHybridModel` (signal, volatilité, régime...).
    *   (Optionnel) Visualisation des features importantes ou de l'attention du modèle si des techniques XAI sont intégrées.
*   **Supervision par LLM**:
    *   Affichage des dernières analyses contextuelles ou évaluations de cohérence fournies par le LLM (suite aux prompts envoyés par le workflow).
    *   Génération d'alertes basées sur les réponses du LLM (ex: risque élevé détecté, signal incohérent).
    *   (Optionnel) Interface permettant à l'utilisateur d'envoyer manuellement des prompts de diagnostic au LLM via le dashboard.
*   **Alerting**:
    *   Configuration d'alertes (email, Telegram, etc.) pour des événements critiques : erreurs système, drawdown important, alertes LLM spécifiques.

---

## 3. Composants Techniques Possibles

*   **Backend / Collecte**:
    *   Le `trading_workflow.py` pousse les informations pertinentes (logs, décisions, P&L, prédictions, réponses LLM) vers une base de données (ex: InfluxDB, Prometheus, simple fichier log structuré, base de données SQL) ou un système de messagerie (ex: Redis, Kafka).
    *   Le module `live/monitoring.py` peut contenir des fonctions pour lire ces données stockées.
*   **Frontend / Dashboard**:
    *   **Streamlit** ou **Plotly Dash**: Frameworks Python populaires pour créer rapidement des dashboards interactifs.
    *   Bibliothèques de visualisation : **Plotly**, **Matplotlib**, **Seaborn**.
*   **Interaction LLM**:
    *   Utilisation de `utils/api_manager.py` pour communiquer avec l'API du LLM.
    *   Gestion des prompts définis dans `docs/PROMPTS_GUIDE.md`.

---

## 4. Interface Attendue (Exemple Fonctions/Classe)

```python
# Exemple de structure (pseudo-code)

# Potentiellement dans monitoring.py ou un script de dashboard séparé

# Fonctions pour récupérer les données (depuis DB, logs, état interne du workflow...)
def get_latest_pnl() -> float: ...
def get_equity_curve() -> pd.DataFrame: ...
def get_open_positions() -> list: ...
def get_recent_trades() -> list: ...
def get_last_model_predictions() -> dict: ...
def get_last_llm_analysis() -> dict: ...
def get_system_status() -> dict: ...

# Si utilisation de Streamlit (dans monitoring.py ou app_dashboard.py)
import streamlit as st

st.title("Morningstar V2 - Monitoring Dashboard")

# Affichage P&L et Métriques
pnl = get_latest_pnl()
st.metric("Profit & Loss Actuel", f"{pnl:.2f} USDT")
# ... afficher autres métriques ...

# Affichage Courbe d'Équité
equity_df = get_equity_curve()
st.line_chart(equity_df)

# Affichage Positions Ouvertes
st.subheader("Positions Ouvertes")
positions = get_open_positions()
st.dataframe(positions)

# Affichage Dernières Prédictions Modèle
st.subheader("Dernières Prédictions Modèle")
preds = get_last_model_predictions()
st.json(preds)

# Affichage Dernière Analyse LLM
st.subheader("Dernière Analyse LLM")
llm_analysis = get_last_llm_analysis()
st.info(llm_analysis.get("evaluation", "N/A"))
st.caption(llm_analysis.get("justification", ""))

# Affichage État Système
st.sidebar.subheader("État Système")
status = get_system_status()
st.sidebar.json(status)

# (Optionnel) Section pour interaction manuelle LLM
st.subheader("Diagnostic LLM Manuel")
user_prompt = st.text_area("Entrez votre question pour le LLM:")
if st.button("Envoyer au LLM"):
    # Utiliser api_manager pour envoyer user_prompt
    # Afficher la réponse
    pass

```

---

## 5. Interaction avec les Autres Modules

*   **Workflow -> Monitoring**: Le workflow fournit les données en temps réel (directement ou via stockage intermédiaire).
*   **Monitoring -> ApiManager**: Pour interagir avec le LLM (si interaction manuelle).
*   **Monitoring -> Logging**: Peut lire les logs pour afficher les erreurs.
*   **Monitoring -> Executor**: Peut lire l'état des positions/ordres via l'Executor ou une source de données partagée.

---

## 6. Considérations

*   **Performance du Dashboard**: S'assurer que la récupération et l'affichage des données sont efficaces pour ne pas ralentir le système.
*   **Fréquence de Mise à Jour**: Définir à quelle fréquence les données du dashboard doivent être rafraîchies.
*   **Sécurité**: Si le dashboard est exposé sur un réseau, sécuriser l'accès.
*   **Complexité**: Commencer par un dashboard simple affichant les métriques essentielles et l'enrichir progressivement.
*   **Stockage des Données**: Choisir une solution de stockage adaptée pour les métriques et l'historique si un suivi à long terme est nécessaire (simple logs vs base de données time-series).

---

Cette spécification guide l'implémentation de `live/monitoring.py` et du dashboard associé.
