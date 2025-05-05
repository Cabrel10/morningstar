#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de collecte de donnu00e9es pour l'application Streamlit Morningstar.

Ce module permet de collecter des donnu00e9es de marchu00e9, des actualitu00e9s crypto,
des sentiments et d'autres informations pertinentes pour le trading.
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
    get_available_datasets, load_dataset, plot_price_chart
)

# Liste des paires de crypto-monnaies disponibles
AVAILABLE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT", 
    "DOGE/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "BCH/USDT"
]

def run_data_collection_script(script_name, args):
    """
    Exu00e9cute un script de collecte de donnu00e9es avec les arguments spu00e9cifiu00e9s.
    
    Args:
        script_name (str): Nom du script u00e0 exu00e9cuter
        args (list): Liste des arguments u00e0 passer au script
    
    Returns:
        tuple: (success, output) ou00f9 success est un boolu00e9en indiquant si le script s'est exu00e9cutu00e9 avec succu00e8s
               et output est la sortie du script
    """
    script_path = BASE_DIR / "scripts" / script_name
    
    if not script_path.exists():
        return False, f"Le script {script_name} n'existe pas."
    
    try:
        # Construire la commande
        cmd = [sys.executable, str(script_path)] + args
        
        # Exu00e9cuter la commande
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Cru00e9er une barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simuler la progression
        for i in range(101):
            # Mettre u00e0 jour la barre de progression
            progress_bar.progress(i)
            status_text.text(f"Exu00e9cution du script {script_name}... {i}%")
            
            # Vu00e9rifier si le processus est terminu00e9
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Ru00e9cupu00e9rer la sortie du processus
        stdout, stderr = process.communicate()
        
        # Mettre u00e0 jour la barre de progression u00e0 100%
        progress_bar.progress(100)
        status_text.text(f"Exu00e9cution du script {script_name} terminu00e9e.")
        
        if process.returncode == 0:
            return True, stdout
        else:
            return False, stderr
    
    except Exception as e:
        return False, str(e)

def data_collection_page():
    """
    Affiche la page de collecte de donnu00e9es.
    """
    st.markdown("<h1 class='main-header'>Collecte de donnu00e9es</h1>", unsafe_allow_html=True)
    
    # Créer des onglets pour les différents types de collecte de données
    tabs = st.tabs([
        "Collecte de données complètes", 
        "Visualisation des données"
    ])
    
    # Onglet 1: Collecte de données complètes
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Collecte de données complètes</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet de collecter un ensemble complet de données pour l'entraînement du modèle Morningstar.
        Les données collectées incluent:
        - Données de marché (OHLCV) via l'API CCXT
        - Actualités et sentiment via l'API de news
        - Embeddings CryptoBERT pour l'analyse de texte
        - Détection de régimes de marché via HMM
        - Informations de marché complémentaires
        
        Le dataset généré sera complet et prêt à être utilisé pour l'entraînement du modèle Morningstar.
        """)
        
        # Formulaire de collecte de données complètes
        with st.form("complete_data_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection des symboles
                selected_symbols = st.multiselect(
                    "Sélectionner les symboles",
                    AVAILABLE_SYMBOLS,
                    default=["BTC/USDT", "ETH/USDT"]
                )
                
                # Période de collecte
                start_date = st.date_input(
                    "Date de début",
                    value=datetime.now() - timedelta(days=365),
                    key="complete_start_date"
                )
                
                end_date = st.date_input(
                    "Date de fin",
                    value=datetime.now(),
                    key="complete_end_date"
                )
            
            with col2:
                # Options de collecte
                st.markdown("#### Options de collecte")
                
                use_sentiment = st.checkbox("Collecter le sentiment", value=True)
                use_news = st.checkbox("Collecter les actualités", value=True)
                use_hmm = st.checkbox("Détecter les régimes HMM", value=True)
                use_cryptobert = st.checkbox("Utiliser CryptoBERT", value=True)
                use_market_info = st.checkbox("Collecter les infos de marché", value=True)
                
                # Nombre de workers
                max_workers = st.slider(
                    "Nombre de workers",
                    min_value=1,
                    max_value=8,
                    value=2,
                    key="complete_max_workers"
                )
                
                # Utiliser le cache
                use_cache = st.checkbox("Utiliser le cache", value=True, help="Utiliser les données déjà collectées si disponibles")
            
            # Nom du dataset
            dataset_name = st.text_input(
                "Nom du dataset",
                value="complete_dataset",
                help="Nom du fichier de sortie (sans extension)"
            )
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Collecter les données complètes")
        
        if submit_button:
            if not selected_symbols:
                st.error("Veuillez sélectionner au moins un symbole.")
            else:
                # Préparer les arguments pour le script
                args = [
                    "--symbols", ",".join(selected_symbols),
                    "--start-date", start_date.strftime("%Y-%m-%d"),
                    "--end-date", end_date.strftime("%Y-%m-%d"),
                    "--max-workers", str(max_workers),
                    "--output", f"{dataset_name}.parquet"
                ]
                
                if use_sentiment:
                    args.append("--use-sentiment")
                
                if use_news:
                    args.append("--use-news")
                
                if use_hmm:
                    args.append("--use-hmm")
                
                if use_cryptobert:
                    args.append("--use-cryptobert")
                
                if use_market_info:
                    args.append("--use-market-info")
                
                if use_cache:
                    args.append("--use-cache")
                
                # Exécuter le script de collecte de données complètes
                success, output = run_data_collection_script("collect_optimized_data.py", args)
                
                if success:
                    st.success("Collecte de données complètes terminée avec succès !")
                    st.code(output)
                else:
                    st.error("Erreur lors de la collecte de données complètes.")
                    st.code(output)
    
    # Onglet 2: Visualisation des données
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Visualisation des données</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet de visualiser les datasets disponibles et d'analyser les prix des actifs.
        """)
        
        # Afficher les datasets disponibles
        datasets = get_available_datasets()
        if datasets:
            selected_dataset = st.selectbox(
                "Sélectionner un dataset pour l'aperçu",
                datasets
            )
            
            if selected_dataset:
                df = load_dataset(selected_dataset)
                
                if df is not None:
                    # Informations sur le dataset
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nombre de lignes", f"{len(df):,}")
                    with col2:
                        st.metric("Nombre de colonnes", f"{len(df.columns):,}")
                    with col3:
                        if 'timestamp' in df.columns:
                            date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} à {df['timestamp'].max().strftime('%Y-%m-%d')}"
                            st.metric("Période", date_range)
                    
                    # Afficher les premières lignes du dataset
                    with st.expander("Aperçu des données"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Afficher un graphique des prix si les colonnes nécessaires existent
                    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        st.markdown("### Graphique des prix")
                        
                        # Sélectionner le symbole pour le graphique
                        if 'symbol' in df.columns:
                            symbols = df['symbol'].unique()
                            selected_symbol = st.selectbox(
                                "Sélectionner un symbole pour le graphique",
                                symbols
                            )
                            
                            # Filtrer les données pour le symbole sélectionné
                            symbol_df = df[df['symbol'] == selected_symbol].copy()
                        else:
                            selected_symbol = "Unknown"
                            symbol_df = df.copy()
                        
                        # Sélectionner la période à afficher
                        if 'timestamp' in symbol_df.columns:
                            min_date = symbol_df['timestamp'].min().date()
                            max_date = symbol_df['timestamp'].max().date()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input(
                                    "Date de début du graphique",
                                    value=min_date,
                                    min_value=min_date,
                                    max_value=max_date
                                )
                            with col2:
                                end_date = st.date_input(
                                    "Date de fin du graphique",
                                    value=max_date,
                                    min_value=min_date,
                                    max_value=max_date
                                )
                            
                            # Convertir les dates en datetime pour le filtrage
                            start_datetime = pd.Timestamp(start_date)
                            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            
                            # Filtrer les données par date
                            symbol_df = symbol_df[(symbol_df['timestamp'] >= start_datetime) & 
                                                 (symbol_df['timestamp'] <= end_datetime)]
                        
                        # Créer le graphique des prix avec une taille plus grande
                        if not symbol_df.empty:
                            fig = go.Figure()
                            
                            # Ajouter la chandelle
                            fig.add_trace(go.Candlestick(
                                x=symbol_df['timestamp'] if 'timestamp' in symbol_df.columns else symbol_df.index,
                                open=symbol_df['open'],
                                high=symbol_df['high'],
                                low=symbol_df['low'],
                                close=symbol_df['close'],
                                name=selected_symbol
                            ))
                            
                            # Ajouter le volume en bas
                            if 'volume' in symbol_df.columns:
                                # Créer une figure séparée pour le volume
                                volume_colors = ['red' if row['close'] < row['open'] else 'green' 
                                               for _, row in symbol_df.iterrows()]
                                
                                fig.add_trace(go.Bar(
                                    x=symbol_df['timestamp'] if 'timestamp' in symbol_df.columns else symbol_df.index,
                                    y=symbol_df['volume'],
                                    marker_color=volume_colors,
                                    name='Volume',
                                    yaxis='y2'
                                ))
                                
                                # Mettre à jour la mise en page pour inclure un axe y secondaire pour le volume
                                fig.update_layout(
                                    yaxis2=dict(
                                        title="Volume",
                                        overlaying="y",
                                        side="right",
                                        showgrid=False
                                    )
                                )
                            
                            # Mise en page du graphique
                            fig.update_layout(
                                title=f"Prix de {selected_symbol}",
                                xaxis_title="Date",
                                yaxis_title="Prix",
                                height=600,
                                xaxis_rangeslider_visible=False,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                template="plotly_dark"
                            )
                            
                            # Afficher le graphique
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Afficher les statistiques de prix
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                price_change = (symbol_df['close'].iloc[-1] - symbol_df['close'].iloc[0]) / symbol_df['close'].iloc[0] * 100
                                st.metric("Variation de prix", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
                            
                            with col2:
                                highest_price = symbol_df['high'].max()
                                st.metric("Prix le plus haut", f"{highest_price:.2f}")
                            
                            with col3:
                                lowest_price = symbol_df['low'].min()
                                st.metric("Prix le plus bas", f"{lowest_price:.2f}")
                            
                            with col4:
                                volatility = symbol_df['high'].pct_change().std() * 100
                                st.metric("Volatilité", f"{volatility:.2f}%")
                        else:
                            st.warning("Aucune donnée disponible pour la période sélectionnée.")
                    else:
                        st.warning("Le dataset ne contient pas les colonnes OHLC nécessaires pour afficher un graphique des prix.")
        else:
            st.info("Aucun dataset disponible. Utilisez l'onglet 'Collecte de données complètes' pour collecter des données.")
