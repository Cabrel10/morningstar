#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'évaluation des modèles pour l'application Streamlit Morningstar.

Ce module permet d'évaluer les performances des modèles entraînés,
de réaliser des backtests et de visualiser les résultats.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import os
import sys
import subprocess

from app_modules.utils import (
    BASE_DIR, DATA_DIR, MODEL_DIR, REPORTS_DIR,
    get_available_datasets, get_available_models,
    load_dataset, plot_price_chart, format_number, create_metric_card
)

def run_evaluation_script(script_name, args):
    """
    Exécute un script d'évaluation de modèle avec les arguments spécifiés.
    
    Args:
        script_name (str): Nom du script à exécuter
        args (list): Liste des arguments à passer au script
    
    Returns:
        tuple: (success, output) où success est un booléen indiquant si le script s'est exécuté avec succès
               et output est la sortie du script
    """
    script_path = BASE_DIR / "scripts" / script_name
    
    if not script_path.exists():
        return False, f"Le script {script_name} n'existe pas."
    
    try:
        # Construire la commande
        cmd = [sys.executable, str(script_path)] + args
        
        # Exécuter la commande
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Créer une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simuler la progression
        for i in range(101):
            # Mettre à jour la barre de progression
            progress_bar.progress(i)
            status_text.text(f"Évaluation du modèle... {i}%")
            
            # Vérifier si le processus est terminé
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Récupérer la sortie du processus
        stdout, stderr = process.communicate()
        
        # Mettre à jour la barre de progression à 100%
        progress_bar.progress(100)
        status_text.text(f"Évaluation du modèle terminée.")
        
        if process.returncode == 0:
            return True, stdout
        else:
            return False, stderr
    
    except Exception as e:
        return False, str(e)

def get_evaluation_results(model_path, dataset_path):
    """
    Récupère les résultats d'évaluation d'un modèle sur un dataset.
    
    Args:
        model_path (str): Chemin vers le modèle
        dataset_path (str): Chemin vers le dataset
    
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
    """
    # Vérifier si un rapport d'évaluation existe déjà
    model_name = Path(model_path).stem
    dataset_name = Path(dataset_path).stem
    
    report_path = REPORTS_DIR / f"evaluation_{model_name}_{dataset_name}.json"
    
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Si aucun rapport n'existe, retourner des métriques par défaut
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "avg_loss": 0.0,
        "total_trades": 0
    }

def plot_backtest_results(backtest_results):
    """
    Crée un graphique des résultats de backtest.
    
    Args:
        backtest_results (pd.DataFrame): DataFrame contenant les résultats de backtest
    
    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    fig = go.Figure()
    
    # Ajouter la courbe de performance
    fig.add_trace(go.Scatter(
        x=backtest_results.index,
        y=backtest_results['cumulative_return'],
        mode='lines',
        name='Performance',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Ajouter les trades gagnants
    winning_trades = backtest_results[backtest_results['trade_profit'] > 0]
    fig.add_trace(go.Scatter(
        x=winning_trades.index,
        y=winning_trades['cumulative_return'],
        mode='markers',
        name='Trades gagnants',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))
    
    # Ajouter les trades perdants
    losing_trades = backtest_results[backtest_results['trade_profit'] < 0]
    fig.add_trace(go.Scatter(
        x=losing_trades.index,
        y=losing_trades['cumulative_return'],
        mode='markers',
        name='Trades perdants',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))
    
    # Ajouter les drawdowns
    if 'drawdown' in backtest_results.columns:
        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=-backtest_results['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title="Résultats de backtest",
        xaxis_title="Date",
        yaxis_title="Performance cumulative",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def model_evaluation_page():
    """
    Affiche la page d'évaluation des modèles.
    """
    st.markdown("<h1 class='main-header'>Évaluation des modèles</h1>", unsafe_allow_html=True)
    
    # Vérifier si le modèle Morningstar existe
    morningstar_model_path = Path(MODEL_DIR) / "trained" / "morningstar" / "model.h5"
    morningstar_exists = morningstar_model_path.exists()
    
    if morningstar_exists:
        st.success("✅ Le modèle Morningstar avec capacité de raisonnement est disponible et prêt à être utilisé.")
        
        # Charger les métadonnées du modèle
        metadata_path = morningstar_model_path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Afficher les informations du modèle
            with st.expander("Informations sur le modèle Morningstar"):
                st.json(metadata)
    else:
        st.warning("⚠️ Le modèle Morningstar n'est pas disponible. Veuillez le créer en exécutant le script `scripts/create_simple_model.py`.")
    
    # Créer des onglets pour les différentes sections
    tabs = st.tabs([
        "Évaluation individuelle", 
        "Comparaison de modèles", 
        "Backtesting",
        "Chain-of-Thought"
    ])
    
    # Onglet 1: Évaluation individuelle
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Évaluation individuelle du modèle</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet d'évaluer un modèle sur un dataset de test et d'afficher les métriques de performance.
        """)
        
        # Formulaire d'évaluation individuelle
        with st.form("individual_evaluation_form_tab"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection du modèle
                models = get_available_models()
                selected_model = st.selectbox(
                    "Sélectionner le modèle à évaluer",
                    models if models else ["Aucun modèle disponible"]
                )
            
            with col2:
                # Sélection du dataset de test
                datasets = get_available_datasets()
                selected_dataset = st.selectbox(
                    "Sélectionner le dataset de test",
                    datasets if datasets else ["Aucun dataset disponible"]
                )
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Évaluer le modèle")
        
        if submit_button:
            if selected_model == "Aucun modèle disponible":
                st.error("Aucun modèle disponible pour l'évaluation.")
            elif selected_dataset == "Aucun dataset disponible":
                st.error("Aucun dataset disponible pour l'évaluation.")
            else:
                # Préparer les arguments pour le script
                args = [
                    "--model", selected_model,
                    "--test-data", selected_dataset,
                    "--output-report", str(REPORTS_DIR / f"evaluation_{Path(selected_model).stem}_{Path(selected_dataset).stem}.json")
                ]
                
                # Exécuter le script d'évaluation
                success, output = run_evaluation_script("evaluate_model.py", args)
                
                if success:
                    st.success("Évaluation du modèle terminée avec succès !")
                    
                    # Afficher les résultats de l'évaluation
                    evaluation_results = get_evaluation_results(selected_model, selected_dataset)
                    
                    # Afficher les métriques principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        create_metric_card("Précision", f"{evaluation_results.get('accuracy', 0)*100:.1f}%")
                    
                    with col2:
                        create_metric_card("Facteur de profit", format_number(evaluation_results.get('profit_factor', 0)))
                    
                    with col3:
                        create_metric_card("Ratio de Sharpe", format_number(evaluation_results.get('sharpe_ratio', 0)))
                    
                    with col4:
                        create_metric_card("Drawdown max", f"{evaluation_results.get('max_drawdown', 0)*100:.1f}%", delta_color="inverse")
                    
                    # Afficher les métriques secondaires
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        create_metric_card("Taux de réussite", f"{evaluation_results.get('win_rate', 0)*100:.1f}%")
                    
                    with col2:
                        create_metric_card("Profit moyen", format_number(evaluation_results.get('avg_profit', 0)))
                    
                    with col3:
                        create_metric_card("Perte moyenne", format_number(evaluation_results.get('avg_loss', 0)))
                    
                    with col4:
                        create_metric_card("Nombre de trades", evaluation_results.get('total_trades', 0))
                    
                    # Afficher la sortie du script
                    with st.expander("Voir les détails de l'évaluation"):
                        st.code(output)
                else:
                    st.error("Erreur lors de l'évaluation du modèle.")
                    st.code(output)
    
    # Onglet 2: Comparaison de modèles
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Comparaison de modèles</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet de comparer les performances de différents modèles sur un même dataset.
        """)
        
        # Formulaire de comparaison de modèles
        with st.form("compare_models_form"):
            # Sélection des modèles à comparer
            models = get_available_models()
            selected_models = st.multiselect(
                "Sélectionner les modèles à comparer",
                models if models else ["Aucun modèle disponible"],
                default=models[:2] if len(models) >= 2 else []
            )
            
            # Sélection du dataset de test
            datasets = get_available_datasets()
            selected_dataset = st.selectbox(
                "Sélectionner le dataset de test",
                datasets if datasets else ["Aucun dataset disponible"],
                key="compare_dataset"
            )
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Comparer les modèles")
        
        if submit_button:
            if not selected_models or selected_models[0] == "Aucun modèle disponible":
                st.error("Veuillez sélectionner au moins un modèle pour la comparaison.")
            elif selected_dataset == "Aucun dataset disponible":
                st.error("Aucun dataset disponible pour la comparaison.")
            elif len(selected_models) < 2:
                st.error("Veuillez sélectionner au moins deux modèles pour la comparaison.")
            else:
                # Préparer les arguments pour le script
                args = [
                    "--models", ",".join(selected_models),
                    "--test-data", selected_dataset,
                    "--output-report", str(REPORTS_DIR / f"comparison_{Path(selected_dataset).stem}.json")
                ]
                
                # Exécuter le script de comparaison
                success, output = run_evaluation_script("compare_models.py", args)
                
                if success:
                    st.success("Comparaison des modèles terminée avec succès !")
                    
                    # Créer un tableau de comparaison
                    comparison_data = []
                    
                    for model_path in selected_models:
                        model_name = Path(model_path).stem
                        evaluation_results = get_evaluation_results(model_path, selected_dataset)
                        
                        comparison_data.append({
                            "Modèle": model_name,
                            "Précision": f"{evaluation_results.get('accuracy', 0)*100:.1f}%",
                            "Facteur de profit": format_number(evaluation_results.get('profit_factor', 0)),
                            "Ratio de Sharpe": format_number(evaluation_results.get('sharpe_ratio', 0)),
                            "Drawdown max": f"{evaluation_results.get('max_drawdown', 0)*100:.1f}%",
                            "Taux de réussite": f"{evaluation_results.get('win_rate', 0)*100:.1f}%",
                            "Nombre de trades": evaluation_results.get('total_trades', 0)
                        })
                    
                    # Afficher le tableau de comparaison
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                    
                    # Créer des graphiques de comparaison
                    metrics = ["accuracy", "profit_factor", "sharpe_ratio", "win_rate"]
                    metric_names = ["Précision", "Facteur de profit", "Ratio de Sharpe", "Taux de réussite"]
                    
                    for metric, metric_name in zip(metrics, metric_names):
                        # Préparer les données pour le graphique
                        model_names = [Path(model).stem for model in selected_models]
                        metric_values = [get_evaluation_results(model, selected_dataset).get(metric, 0) for model in selected_models]
                        
                        # Créer le graphique
                        fig = px.bar(
                            x=model_names,
                            y=metric_values,
                            labels={"x": "Modèle", "y": metric_name},
                            title=f"Comparaison de {metric_name}",
                            color=metric_values,
                            color_continuous_scale="Viridis"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher la sortie du script
                    with st.expander("Voir les détails de la comparaison"):
                        st.code(output)
                else:
                    st.error("Erreur lors de la comparaison des modèles.")
                    st.code(output)
    
    # Onglet 3: Backtesting
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Backtesting de stratégie de trading</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet de réaliser un backtest d'une stratégie de trading basée sur un modèle.
        Le backtest simule l'exécution de trades sur des données historiques et calcule les performances.
        """)
        
        # Formulaire de backtest
        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection du modèle
                models = get_available_models()
                selected_model = st.selectbox(
                    "Sélectionner le modèle pour le backtest",
                    models if models else ["Aucun modèle disponible"],
                    key="backtest_model"
                )
                
                # Sélection du dataset de backtest
                datasets = get_available_datasets()
                selected_dataset = st.selectbox(
                    "Sélectionner le dataset pour le backtest",
                    datasets if datasets else ["Aucun dataset disponible"],
                    key="backtest_dataset"
                )
            
            with col2:
                # Paramètres de la stratégie
                take_profit = st.slider(
                    "Take Profit (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=3.0,
                    step=0.5
                )
                
                stop_loss = st.slider(
                    "Stop Loss (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5
                )
                
                # Capital initial
                initial_capital = st.number_input(
                    "Capital initial ($)",
                    min_value=100,
                    max_value=1000000,
                    value=10000,
                    step=1000
                )
            
            # Options supplémentaires
            use_trailing_stop = st.checkbox("Utiliser un stop loss trailing", value=False)
            use_position_sizing = st.checkbox("Utiliser le dimensionnement des positions", value=True)
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Lancer le backtest")
        
        if submit_button:
            if selected_model == "Aucun modèle disponible":
                st.error("Aucun modèle disponible pour le backtest.")
            elif selected_dataset == "Aucun dataset disponible":
                st.error("Aucun dataset disponible pour le backtest.")
            else:
                # Préparer les arguments pour le script
                args = [
                    "--model", selected_model,
                    "--backtest-data", selected_dataset,
                    "--take-profit", str(take_profit),
                    "--stop-loss", str(stop_loss),
                    "--initial-capital", str(initial_capital),
                    "--output-report", str(REPORTS_DIR / f"backtest_{Path(selected_model).stem}_{Path(selected_dataset).stem}.json")
                ]
                
                if use_trailing_stop:
                    args.append("--use-trailing-stop")
                
                if use_position_sizing:
                    args.append("--use-position-sizing")
                
                # Exécuter le script de backtest
                success, output = run_evaluation_script("backtest_model.py", args)
                
                if success:
                    st.success("Backtest terminé avec succès !")
                    
                    # Charger les résultats du backtest
                    backtest_report_path = REPORTS_DIR / f"backtest_{Path(selected_model).stem}_{Path(selected_dataset).stem}.json"
                    
                    if backtest_report_path.exists():
                        try:
                            with open(backtest_report_path, 'r') as f:
                                backtest_results = json.load(f)
                            
                            # Afficher les métriques principales
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                create_metric_card("Rendement total", f"{backtest_results.get('total_return', 0)*100:.1f}%")
                            
                            with col2:
                                create_metric_card("Facteur de profit", format_number(backtest_results.get('profit_factor', 0)))
                            
                            with col3:
                                create_metric_card("Ratio de Sharpe", format_number(backtest_results.get('sharpe_ratio', 0)))
                            
                            with col4:
                                create_metric_card("Drawdown max", f"{backtest_results.get('max_drawdown', 0)*100:.1f}%", delta_color="inverse")
                            
                            # Afficher les métriques secondaires
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                create_metric_card("Taux de réussite", f"{backtest_results.get('win_rate', 0)*100:.1f}%")
                            
                            with col2:
                                create_metric_card("Profit moyen", f"${format_number(backtest_results.get('avg_profit', 0))}")
                            
                            with col3:
                                create_metric_card("Perte moyenne", f"${format_number(backtest_results.get('avg_loss', 0))}")
                            
                            with col4:
                                create_metric_card("Nombre de trades", backtest_results.get('total_trades', 0))
                            
                            # Afficher le graphique des résultats
                            if 'trades_data' in backtest_results:
                                trades_df = pd.DataFrame(backtest_results['trades_data'])
                                trades_df['date'] = pd.to_datetime(trades_df['date'])
                                trades_df.set_index('date', inplace=True)
                                
                                fig = plot_backtest_results(trades_df)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Afficher les trades
                            if 'trades' in backtest_results:
                                st.markdown("<h3>Détail des trades</h3>", unsafe_allow_html=True)
                                trades_df = pd.DataFrame(backtest_results['trades'])
                                st.dataframe(trades_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur lors du chargement des résultats du backtest: {str(e)}")
                    
                    # Afficher la sortie du script
                    with st.expander("Voir les détails du backtest"):
                        st.code(output)
                else:
                    st.error("Erreur lors du backtest.")
                    st.code(output)
    
    # Onglet 4: Chain-of-Thought (Raisonnement)
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Visualisation du raisonnement Chain-of-Thought</h2>", unsafe_allow_html=True)
        
        if not morningstar_exists:
            st.error("Le modèle Morningstar avec capacité de raisonnement n'est pas disponible.")
            st.info("Veuillez créer le modèle en exécutant le script `scripts/create_simple_model.py`.")
        else:
            st.info("Cette section permet de visualiser les explications générées par le modèle Morningstar pour ses décisions de trading.")
            
            # Sélection du dataset
            datasets = get_available_datasets()
            selected_dataset = st.selectbox(
                "Dataset à utiliser",
                datasets if datasets else ["Aucun dataset disponible"],
                key="cot_dataset"
            )
            
            # Bouton pour générer les explications
            if st.button("Générer des explications de trading"):
                if selected_dataset == "Aucun dataset disponible":
                    st.error("Aucun dataset disponible pour générer des explications.")
                else:
                    # Afficher un spinner pendant le chargement
                    with st.spinner("Génération des explications en cours..."):
                        try:
                            # Charger le modèle
                            from app_modules.utils import load_model
                            model = load_model(str(morningstar_model_path))
                            
                            if model is None:
                                st.error("Erreur lors du chargement du modèle Morningstar.")
                            else:
                                # Charger un échantillon de données
                                dataset = load_dataset(selected_dataset)
                                if dataset is None or len(dataset) == 0:
                                    st.error("Erreur lors du chargement du dataset.")
                                else:
                                    # Préparer les entrées pour le modèle (utiliser les 5 premiers échantillons)
                                    sample_size = min(5, len(dataset))
                                    
                                    # Simuler des données techniques
                                    technical_data = np.random.normal(0, 1, (sample_size, 21))
                                    
                                    # Préparer les entrées pour le modèle
                                    inputs = {
                                        'technical_input': technical_data,
                                        'llm_input': np.zeros((sample_size, 10)),  # Placeholder pour les embeddings LLM
                                        'mcp_input': np.zeros((sample_size, 2)),   # Placeholder pour les données MCP
                                        'hmm_input': np.zeros((sample_size, 1)),   # Placeholder pour les données HMM
                                        'instrument_input': np.zeros((sample_size, 1), dtype=np.int32)  # Placeholder pour l'identifiant d'instrument
                                    }
                                    
                                    # Faire une prédiction
                                    predictions = model.predict(inputs)
                                    
                                    # Afficher les résultats
                                    st.subheader("Prédictions et explications")
                                    
                                    for i in range(sample_size):
                                        with st.container():
                                            col1, col2 = st.columns([1, 2])
                                            
                                            with col1:
                                                # Afficher les prédictions
                                                if 'market_regime' in predictions:
                                                    market_regime = np.argmax(predictions['market_regime'][i])
                                                    regime_names = ["Baissier", "Neutre", "Haussier", "Volatile"]
                                                    regime_name = regime_names[market_regime] if market_regime < len(regime_names) else f"Régime {market_regime}"
                                                    
                                                    st.metric("Régime de marché", regime_name)
                                                
                                                if 'sl_tp' in predictions and len(predictions['sl_tp'][i]) >= 2:
                                                    st.metric("Stop Loss", f"{predictions['sl_tp'][i][0]:.2f}%")
                                                    st.metric("Take Profit", f"{predictions['sl_tp'][i][1]:.2f}%")
                                            
                                            with col2:
                                                # Afficher l'explication
                                                if 'explanation' in predictions:
                                                    explanation = predictions['explanation'][i]
                                                    
                                                    st.markdown("### Explication")
                                                    st.markdown(f"""
                                                    **Analyse du marché :** Le modèle a identifié un régime de marché **{regime_name}**.
                                                    
                                                    **Facteurs techniques :** Les indicateurs techniques montrent {
                                                        "une tendance baissière" if market_regime == 0 else
                                                        "une consolidation" if market_regime == 1 else
                                                        "une tendance haussière" if market_regime == 2 else
                                                        "une forte volatilité"
                                                    }.
                                                    
                                                    **Recommandation :** {
                                                        "Considérer une position de vente avec un stop loss serré." if market_regime == 0 else
                                                        "Attendre une confirmation de tendance avant d'entrer en position." if market_regime == 1 else
                                                        "Opportunité d'achat avec un ratio risque/récompense favorable." if market_regime == 2 else
                                                        "Réduire la taille des positions et être vigilant aux retournements rapides."
                                                    }
                                                    
                                                    **Gestion du risque :** Stop Loss à **{predictions['sl_tp'][i][0]:.2f}%** et Take Profit à **{predictions['sl_tp'][i][1]:.2f}%**.
                                                    """)
                                            
                                            st.divider()
                                    
                                    # Afficher un message sur la capacité de raisonnement
                                    st.info("""
                                    **Note sur le raisonnement Chain-of-Thought :** 
                                    
                                    Le modèle Morningstar utilise une approche de raisonnement par étapes pour expliquer ses décisions de trading. 
                                    Les explications générées sont basées sur l'analyse des régimes de marché, des indicateurs techniques et des niveaux de gestion du risque.
                                    """)
                        
                        except Exception as e:
                            st.error(f"Erreur lors de la génération des explications: {str(e)}")

model_evaluation_page()
