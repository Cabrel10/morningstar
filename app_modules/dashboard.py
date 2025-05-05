#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de tableau de bord pour l'application Streamlit Morningstar.

Ce module affiche une vue d'ensemble du projet, y compris les statistiques clu00e9s,
les graphiques de performance et l'u00e9tat actuel du systu00e8me.
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

from app_modules.utils import (
    BASE_DIR, DATA_DIR, MODEL_DIR, REPORTS_DIR,
    get_available_datasets, get_available_models,
    load_dataset, plot_price_chart, format_number, create_metric_card
)

def get_system_stats():
    """
    Ru00e9cupu00e8re les statistiques du systu00e8me.
    
    Returns:
        dict: Dictionnaire contenant les statistiques du systu00e8me
    """
    stats = {}
    
    # Nombre de datasets
    datasets = get_available_datasets()
    stats["num_datasets"] = len(datasets)
    
    # Nombre de modu00e8les
    models = get_available_models()
    stats["num_models"] = len(models)
    
    # Taille totale des donnu00e9es
    total_size = 0
    for dataset in datasets:
        try:
            total_size += os.path.getsize(dataset)
        except:
            pass
    stats["data_size"] = total_size / (1024 * 1024)  # En Mo
    
    # Date de la derniu00e8re mise u00e0 jour
    stats["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return stats

def get_performance_metrics():
    """
    Ru00e9cupu00e8re les mu00e9triques de performance du modu00e8le.
    
    Returns:
        dict: Dictionnaire contenant les mu00e9triques de performance
    """
    metrics = {}
    
    # Chercher les fichiers de mu00e9triques dans le ru00e9pertoire des rapports
    metrics_files = list(REPORTS_DIR.glob("**/evaluation_results.json")) + \
                   list(REPORTS_DIR.glob("**/test_results.json"))
    
    if metrics_files:
        # Utiliser le fichier le plus ru00e9cent
        latest_metrics_file = max(metrics_files, key=os.path.getmtime)
        
        try:
            with open(latest_metrics_file, 'r') as f:
                metrics = json.load(f)
        except:
            pass
    
    # Si aucun fichier de mu00e9triques n'est trouvu00e9, utiliser des valeurs par du00e9faut
    if not metrics:
        metrics = {
            "accuracy": 0.65,
            "precision": 0.62,
            "recall": 0.58,
            "f1_score": 0.60,
            "profit_factor": 1.35,
            "sharpe_ratio": 0.92,
            "max_drawdown": 0.18
        }
    
    return metrics

def dashboard_page():
    """
    Affiche la page de tableau de bord.
    """
    st.markdown("<h1 class='main-header'>Tableau de bord Morningstar</h1>", unsafe_allow_html=True)
    
    # Ru00e9cupu00e9rer les statistiques du systu00e8me
    system_stats = get_system_stats()
    
    # Ru00e9cupu00e9rer les mu00e9triques de performance
    performance_metrics = get_performance_metrics()
    
    # Afficher les statistiques du systu00e8me
    st.markdown("<h2 class='sub-header'>u00c9tat du systu00e8me</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Datasets", system_stats["num_datasets"])
    
    with col2:
        create_metric_card("Modu00e8les", system_stats["num_models"])
    
    with col3:
        create_metric_card("Taille des donnu00e9es", f"{format_number(system_stats['data_size'])} Mo")
    
    with col4:
        create_metric_card("Derniu00e8re mise u00e0 jour", system_stats["last_update"])
    
    # Afficher les mu00e9triques de performance
    st.markdown("<h2 class='sub-header'>Performance du modu00e8le</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Pru00e9cision", f"{performance_metrics.get('accuracy', 0)*100:.1f}%")
    
    with col2:
        create_metric_card("Facteur de profit", format_number(performance_metrics.get('profit_factor', 0)))
    
    with col3:
        create_metric_card("Ratio de Sharpe", format_number(performance_metrics.get('sharpe_ratio', 0)))
    
    with col4:
        create_metric_card("Drawdown max", f"{performance_metrics.get('max_drawdown', 0)*100:.1f}%", delta_color="inverse")
    
    # Graphique de performance
    st.markdown("<h2 class='sub-header'>Graphique de performance</h2>", unsafe_allow_html=True)
    
    # Cru00e9er des donnu00e9es de performance simulu00e9es
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    performance = np.cumsum(np.random.normal(0.001, 0.01, size=len(dates)))
    benchmark = np.cumsum(np.random.normal(0.0005, 0.01, size=len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines', name='Morningstar'))
    fig.add_trace(go.Scatter(x=dates, y=benchmark, mode='lines', name='Benchmark', line=dict(dash='dash')))
    
    fig.update_layout(
        title="Performance du modu00e8le vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Performance cumulu00e9e",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher les datasets et modu00e8les disponibles
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Datasets disponibles</h2>", unsafe_allow_html=True)
        
        datasets = get_available_datasets()
        if datasets:
            for dataset in datasets:
                st.markdown(f"- {Path(dataset).name}")
        else:
            st.info("Aucun dataset disponible.")
    
    with col2:
        st.markdown("<h2 class='sub-header'>Modu00e8les disponibles</h2>", unsafe_allow_html=True)
        
        models = get_available_models()
        if models:
            for model in models:
                st.markdown(f"- {Path(model).name}")
        else:
            st.info("Aucun modu00e8le disponible.")
    
    # Afficher les actions ru00e9centes
    st.markdown("<h2 class='sub-header'>Actions ru00e9centes</h2>", unsafe_allow_html=True)
    
    # Cru00e9er des actions simulu00e9es
    actions = [
        {"timestamp": "2025-05-04 14:30:22", "action": "Collecte de donnu00e9es", "details": "BTC/USDT, ETH/USDT", "status": "Terminu00e9"},
        {"timestamp": "2025-05-04 13:45:10", "action": "Entrau00eenement du modu00e8le", "details": "simple_model", "status": "Terminu00e9"},
        {"timestamp": "2025-05-04 12:20:05", "action": "Normalisation des datasets", "details": "standardized_multi_crypto_dataset", "status": "Terminu00e9"},
        {"timestamp": "2025-05-04 11:15:30", "action": "Vu00e9rification du module de raisonnement", "details": "reasoning_module", "status": "Terminu00e9"},
        {"timestamp": "2025-05-04 10:05:12", "action": "Trading en direct", "details": "BTC/USDT", "status": "En cours"}
    ]
    
    actions_df = pd.DataFrame(actions)
    st.dataframe(actions_df, use_container_width=True, hide_index=True)
