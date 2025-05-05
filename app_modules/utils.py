#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'utilitaires pour l'application Streamlit Morningstar.

Ce module contient des fonctions utilitaires communes utilisées dans toute l'application.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import tensorflow as tf

# Chemins importants
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"

# Créer les répertoires s'ils n'existent pas
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_css():
    """
    Charge le CSS personnalisé pour l'application Streamlit.
    """
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .live-indicator {
        color: #4CAF50;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def get_available_datasets():
    """
    Récupère la liste des datasets disponibles dans le projet.
    
    Returns:
        list: Liste des chemins vers les datasets confirmés
    """
    datasets = []
    
    # Ne retourner que les datasets confirmés
    confirmed_datasets = [
        "standardized_multi_crypto_dataset.parquet",
        "enriched_dataset.parquet",
        "final_dataset.parquet"
    ]
    
    # Parcourir les répertoires de données pour trouver les datasets confirmés
    for data_type in ["standardized", "enriched", "real"]:
        data_dir = DATA_DIR / data_type
        if data_dir.exists():
            for dataset_name in confirmed_datasets:
                for file in data_dir.glob(f"*{dataset_name}"):
                    datasets.append(str(file))
    
    return datasets

def get_available_models():
    """
    Récupère la liste des modèles disponibles dans le projet.
    
    Returns:
        list: Liste des chemins vers le modèle Morningstar
    """
    models = []
    
    # Ne retourner que le modèle Morningstar
    morningstar_model_path = MODEL_DIR / "trained" / "morningstar" / "model.h5"
    if morningstar_model_path.exists():
        models.append(str(morningstar_model_path))
    
    return models

def load_dataset(dataset_path):
    """
    Charge un dataset à partir de son chemin.
    
    Args:
        dataset_path (str): Chemin vers le dataset
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du dataset
    """
    path = Path(dataset_path)
    
    if not path.exists():
        st.error(f"Le dataset {path} n'existe pas.")
        return None
    
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            st.error(f"Format de fichier non pris en charge: {path.suffix}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du dataset: {e}")
        return None

def load_model(model_path):
    """
    Charge un modèle à partir de son chemin.
    
    Args:
        model_path (str): Chemin vers le modèle
    
    Returns:
        tf.keras.Model: Modèle chargé
    """
    path = Path(model_path)
    
    if not path.exists():
        st.error(f"Le modèle {path} n'existe pas.")
        return None
    
    try:
        if path.is_dir() and (path / "saved_model.pb").exists():
            return tf.keras.models.load_model(path)
        elif path.suffix == ".h5":
            return tf.keras.models.load_model(path)
        else:
            st.error(f"Format de modèle non pris en charge: {path}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def plot_price_chart(df, symbol):
    """
    Crée un graphique des prix pour un symbole donné.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données de prix
        symbol (str): Symbole à afficher
    
    Returns:
        plotly.graph_objects.Figure: Figure Plotly contenant le graphique
    """
    # Filtrer les données pour le symbole spécifié
    if 'symbol' in df.columns:
        df_filtered = df[df['symbol'] == symbol].copy()
    else:
        df_filtered = df.copy()
    
    # S'assurer que les colonnes nécessaires existent
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df_filtered.columns for col in required_columns):
        st.error(f"Les colonnes {required_columns} sont nécessaires pour afficher le graphique des prix.")
        return None
    
    # Créer le graphique des prix
    fig = go.Figure()
    
    # Ajouter les chandeliers
    # Convertir range en liste pour éviter l'erreur de Plotly
    x_values = df_filtered.index if df_filtered.index.name == 'timestamp' else list(range(len(df_filtered)))
    
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=df_filtered['open'],
            high=df_filtered['high'],
            low=df_filtered['low'],
            close=df_filtered['close'],
            name=symbol
        )
    )
    
    # Personnaliser le graphique
    fig.update_layout(
        title=f"Graphique des prix pour {symbol}",
        xaxis_title="Date",
        yaxis_title="Prix",
        template="plotly_white",
        height=600
    )
    
    return fig

def format_number(number, precision=2):
    """
    Formate un nombre avec séparateurs de milliers et précision spécifiée.
    
    Args:
        number (float): Nombre à formater
        precision (int): Nombre de décimales
    
    Returns:
        str: Nombre formaté
    """
    if number is None:
        return "N/A"
    
    if isinstance(number, (int, float)):
        return f"{number:,.{precision}f}".replace(",", " ").replace(".", ",")
    
    return str(number)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """
    Crée une carte métrique pour afficher une valeur avec un titre et éventuellement une variation.
    
    Args:
        title (str): Titre de la métrique
        value (str): Valeur à afficher
        delta (str, optional): Variation à afficher. Defaults to None.
        delta_color (str, optional): Couleur de la variation ("normal", "inverse", "off"). Defaults to "normal".
    """
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)
