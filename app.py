#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Streamlit pour le projet Morningstar.

Cette application fournit une interface utilisateur interactive pour toutes les fonctionnalités
du projet Morningstar, y compris la collecte de données, l'entraînement des modèles,
l'évaluation des performances et le trading en direct.
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
import threading

# Configuration de la page Streamlit - DOIT ÊTRE LE PREMIER APPEL STREAMLIT
st.set_page_config(
    page_title="Morningstar Crypto Trading",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(BASE_DIR))

# Importer les modules de l'application
from app_modules.data_collection import data_collection_page
from app_modules.model_training import model_training_page
from app_modules.model_evaluation import model_evaluation_page
from app_modules.live_trading import show_live_trading
from app_modules.dashboard import dashboard_page
from app_modules.utils import load_css, MODEL_DIR

# Charger le CSS personnalisé
load_css()

def main():
    """
    Fonction principale qui gère la navigation et l'affichage des différentes pages de l'application.
    """
    # Vérifier si le modèle Morningstar existe
    morningstar_model_path = Path(MODEL_DIR) / "trained" / "morningstar" / "model.h5"
    morningstar_exists = morningstar_model_path.exists()
    
    # Sidebar pour la navigation
    with st.sidebar:
        st.image(str(BASE_DIR / "app_modules" / "assets" / "morningstar_logo.png"), width=200)
        st.title("Morningstar Trading Platform")
        st.markdown("---")
        
        # Afficher le statut du modèle Morningstar
        if morningstar_exists:
            st.success("✅ Modèle Morningstar disponible")
        else:
            st.warning("⚠️ Modèle Morningstar non disponible")
        
        # Menu de navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Collecte de données", "Entraînement du modèle", "Évaluation du modèle", "Trading en direct"],
            index=0
        )
        
        st.markdown("---")
        st.caption(" Morningstar Trading Platform")
        st.caption(f"Version: 1.1.0 | {datetime.now().strftime('%Y-%m-%d')}")
    
    # Afficher la page sélectionnée
    if page == "Dashboard":
        # Ajouter une notification sur le modèle Morningstar
        if morningstar_exists:
            st.success("✅ Le modèle Morningstar avec capacité de raisonnement est disponible et prêt à être utilisé.")
            
            # Charger les métadonnées du modèle
            metadata_path = morningstar_model_path.parent / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Créer un expander pour les informations du modèle
                    with st.expander("Informations sur le modèle Morningstar"):
                        st.json(metadata)
                except Exception as e:
                    st.error(f"Erreur lors de la lecture des métadonnées: {str(e)}")
        else:
            st.warning("⚠️ Le modèle Morningstar n'est pas disponible. Vous pouvez le créer en exécutant le script `scripts/create_simple_model.py`.")
        
        dashboard_page()
    elif page == "Collecte de données":
        data_collection_page()
    elif page == "Entraînement du modèle":
        model_training_page()
    elif page == "Évaluation du modèle":
        model_evaluation_page()
    elif page == "Trading en direct":
        show_live_trading()

if __name__ == "__main__":
    main()
