#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'entraînement des modèles pour l'application Streamlit Morningstar.

Ce module permet d'entraîner différents types de modèles pour la prédiction
des mouvements de prix des crypto-monnaies et la génération de signaux de trading.
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
    load_dataset
)

def run_training_script(script_name, args):
    """
    Exécute un script d'entraînement de modèle avec les arguments spécifiés.
    
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
            status_text.text(f"Entraînement du modèle... {i}%")
            
            # Vérifier si le processus est terminé
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Récupérer la sortie du processus
        stdout, stderr = process.communicate()
        
        # Mettre à jour la barre de progression à 100%
        progress_bar.progress(100)
        status_text.text(f"Entraînement du modèle terminé.")
        
        if process.returncode == 0:
            return True, stdout
        else:
            return False, stderr
    
    except Exception as e:
        return False, str(e)

def model_training_page():
    """
    Affiche la page d'entraînement du modèle Morningstar.
    """
    st.markdown("<h1 class='main-header'>Entraînement du modèle Morningstar</h1>", unsafe_allow_html=True)
    
    # Créer des onglets pour les différentes méthodes d'entraînement
    tabs = st.tabs([
        "Entraînement standard", 
        "Optimisation génétique"
    ])
    
    # Onglet 1: Entraînement standard
    with tabs[0]:
        st.markdown("""
        Cette section permet d'entraîner le modèle Morningstar, qui combine plusieurs sources de données pour générer des signaux de trading précis.
        
        Le modèle Morningstar utilise une architecture hybride avec plusieurs entrées:
        - Données techniques (OHLCV et indicateurs)
        - Embeddings LLM/CryptoBERT
        - Données du Market Context Processor (MCP)
        - Régimes de marché détectés par HMM
        - Type d'instrument
        
        Et produit plusieurs sorties:
        - Signal de trading (5 classes)
        - Régime de marché (4 classes)
        - Quantiles de volatilité (3 valeurs)
        - Stop Loss et Take Profit (2 valeurs)
        
        Le modèle intègre également une capacité de raisonnement Chain-of-Thought pour expliquer ses décisions.
        """)
        
        # Formulaire d'entraînement du modèle Morningstar
        with st.form("morningstar_model_form"):
            # Sélection du dataset d'entraînement
            datasets = get_available_datasets()
            selected_dataset = st.selectbox(
                "Sélectionner le dataset d'entraînement",
                datasets if datasets else ["Aucun dataset disponible"],
                key="morningstar_dataset"
            )
            
            # Paramètres d'architecture
            st.markdown("### Architecture du modèle")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Paramètres des entrées
                st.markdown("#### Paramètres des entrées")
                
                tech_input_dim = st.number_input(
                    "Dimension des entrées techniques",
                    min_value=10,
                    max_value=50,
                    value=21,
                    step=1,
                    help="Nombre de features techniques (OHLCV + indicateurs)"
                )
                
                llm_embedding_dim = st.number_input(
                    "Dimension des embeddings LLM",
                    min_value=8,
                    max_value=768,
                    value=10,
                    step=8,
                    help="Dimension des embeddings CryptoBERT"
                )
                
                mcp_input_dim = st.number_input(
                    "Dimension des entrées MCP",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    help="Dimension des données du Market Context Processor"
                )
                
                hmm_input_dim = st.number_input(
                    "Dimension des entrées HMM",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    help="Dimension des données de régime HMM"
                )
            
            with col2:
                # Paramètres des sorties
                st.markdown("#### Paramètres des sorties")
                
                num_trading_classes = st.number_input(
                    "Nombre de classes de trading",
                    min_value=3,
                    max_value=7,
                    value=5,
                    step=1,
                    help="Nombre de classes pour les signaux de trading"
                )
                
                num_market_regime_classes = st.number_input(
                    "Nombre de classes de régime de marché",
                    min_value=2,
                    max_value=6,
                    value=4,
                    step=1,
                    help="Nombre de classes pour les régimes de marché"
                )
                
                num_volatility_quantiles = st.number_input(
                    "Nombre de quantiles de volatilité",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1,
                    help="Nombre de quantiles pour la prédiction de volatilité"
                )
                
                num_sl_tp_outputs = st.number_input(
                    "Nombre de sorties SL/TP",
                    min_value=2,
                    max_value=4,
                    value=2,
                    step=1,
                    help="Nombre de sorties pour Stop Loss et Take Profit"
                )
            
            # Paramètres d'entraînement
            st.markdown("### Paramètres d'entraînement")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                epochs = st.slider(
                    "Nombre d'époques",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Nombre total d'époques d'entraînement"
                )
                
                batch_size = st.select_slider(
                    "Taille du batch",
                    options=[16, 32, 64, 128, 256, 512],
                    value=64,
                    help="Nombre d'échantillons par batch"
                )
            
            with col2:
                learning_rate = st.select_slider(
                    "Taux d'apprentissage",
                    options=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
                    value=0.0005,
                    format_func=lambda x: f"{x:.4f}",
                    help="Taux d'apprentissage pour l'optimiseur"
                )
                
                optimizer = st.selectbox(
                    "Optimiseur",
                    ["Adam", "RMSprop", "SGD", "AdamW"],
                    index=0,
                    help="Algorithme d'optimisation"
                )
            
            with col3:
                validation_split = st.slider(
                    "Split de validation",
                    min_value=0.1,
                    max_value=0.3,
                    value=0.2,
                    step=0.05,
                    format="%.2f",
                    help="Proportion des données utilisée pour la validation"
                )
                
                early_stopping_patience = st.slider(
                    "Patience pour early stopping",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Nombre d'époques sans amélioration avant d'arrêter l'entraînement"
                )
            
            # Paramètres avancés
            with st.expander("Paramètres avancés"):
                col1, col2 = st.columns(2)
                
                with col1:
                    dropout_rate = st.slider(
                        "Taux de dropout",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.3,
                        step=0.05,
                        format="%.2f",
                        help="Taux de dropout pour la régularisation"
                    )
                    
                    l2_reg = st.slider(
                        "Régularisation L2",
                        min_value=0.0,
                        max_value=0.01,
                        value=0.001,
                        step=0.001,
                        format="%.3f",
                        help="Coefficient de régularisation L2"
                    )
                    
                    use_batch_norm = st.checkbox(
                        "Utiliser Batch Normalization",
                        value=True,
                        help="Activer la normalisation par batch"
                    )
                
                with col2:
                    # Paramètres de raisonnement
                    num_reasoning_steps = st.slider(
                        "Nombre d'étapes de raisonnement",
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        help="Nombre d'étapes pour le raisonnement Chain-of-Thought"
                    )
                    
                    num_attention_heads = st.slider(
                        "Nombre de têtes d'attention",
                        min_value=1,
                        max_value=8,
                        value=4,
                        step=1,
                        help="Nombre de têtes pour le mécanisme d'attention"
                    )
                    
                    use_chain_of_thought = st.checkbox(
                        "Activer Chain-of-Thought",
                        value=True,
                        help="Activer le raisonnement Chain-of-Thought"
                    )
            
            # Options de sauvegarde
            st.markdown("### Options de sauvegarde")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input(
                    "Nom du modèle",
                    value="morningstar",
                    help="Nom du modèle à sauvegarder"
                )
                
                save_best_only = st.checkbox(
                    "Sauvegarder uniquement le meilleur modèle",
                    value=True,
                    help="Ne sauvegarder que la version avec les meilleures performances"
                )
            
            with col2:
                save_weights_only = st.checkbox(
                    "Sauvegarder uniquement les poids",
                    value=False,
                    help="Sauvegarder uniquement les poids du modèle, pas l'architecture"
                )
                
                save_metadata = st.checkbox(
                    "Sauvegarder les métadonnées",
                    value=True,
                    help="Sauvegarder les métadonnées du modèle (scalers, configuration, etc.)"
                )
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Entraîner le modèle Morningstar")
    
    if submit_button:
        if selected_dataset == "Aucun dataset disponible":
            st.error("Aucun dataset disponible pour l'entraînement.")
        else:
            # Préparer les arguments pour le script
            args = [
                "--dataset", selected_dataset,
                "--model-name", model_name,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate),
                "--optimizer", optimizer.lower(),
                "--validation-split", str(validation_split),
                "--early-stopping-patience", str(early_stopping_patience),
                "--dropout-rate", str(dropout_rate),
                "--l2-reg", str(l2_reg),
                "--tech-input-dim", str(tech_input_dim),
                "--llm-embedding-dim", str(llm_embedding_dim),
                "--mcp-input-dim", str(mcp_input_dim),
                "--hmm-input-dim", str(hmm_input_dim),
                "--num-trading-classes", str(num_trading_classes),
                "--num-market-regime-classes", str(num_market_regime_classes),
                "--num-volatility-quantiles", str(num_volatility_quantiles),
                "--num-sl-tp-outputs", str(num_sl_tp_outputs),
                "--num-reasoning-steps", str(num_reasoning_steps),
                "--num-attention-heads", str(num_attention_heads)
            ]
            
            if use_batch_norm:
                args.append("--use-batch-norm")
            
            if use_chain_of_thought:
                args.append("--use-chain-of-thought")
            
            if save_best_only:
                args.append("--save-best-only")
            
            if save_weights_only:
                args.append("--save-weights-only")
            
            if save_metadata:
                args.append("--save-metadata")
            
            # Exécuter le script d'entraînement
            with st.spinner("Entraînement du modèle Morningstar en cours..."):
                success, output = run_training_script("train_morningstar_model.py", args)
                
                if success:
                    st.success("Entraînement du modèle Morningstar terminé avec succès !")
                    
                    # Afficher les métriques d'entraînement
                    try:
                        # Essayer de parser les métriques de sortie
                        metrics_lines = [line for line in output.split('\n') if 'val_loss' in line]
                        if metrics_lines:
                            final_metrics = metrics_lines[-1].split(' - ')
                            
                            # Créer un tableau de métriques
                            metrics_data = {}
                            for metric in final_metrics:
                                if ':' in metric:
                                    key, value = metric.strip().split(':', 1)
                                    metrics_data[key] = float(value)
                            
                            # Afficher les métriques dans des cartes
                            if metrics_data:
                                cols = st.columns(len(metrics_data))
                                for i, (key, value) in enumerate(metrics_data.items()):
                                    with cols[i]:
                                        st.metric(label=key, value=f"{value:.4f}")
                    except:
                        pass
                    
                    # Afficher la sortie complète
                    with st.expander("Voir les détails de l'entraînement"):
                        st.code(output)
                else:
                    st.error("Erreur lors de l'entraînement du modèle Morningstar.")
                    st.code(output)
    
    # Onglet 2: Optimisation génétique
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Optimisation génétique des hyperparamètres</h2>", unsafe_allow_html=True)
        st.markdown("""
        Cette section permet d'optimiser les hyperparamètres du modèle Morningstar en utilisant un algorithme génétique.
        
        L'algorithme génétique explore l'espace des hyperparamètres pour trouver la combinaison optimale qui maximise les performances du modèle.
        Les hyperparamètres optimisés incluent:
        - Taux de régularisation L2
        - Taux de dropout
        - Taux d'apprentissage
        - Taille du batch
        - Utilisation de la normalisation par batch
        """)
        
        # Formulaire d'optimisation génétique
        with st.form("genetic_optimization_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection du dataset d'entraînement
                datasets = get_available_datasets()
                selected_dataset = st.selectbox(
                    "Sélectionner le dataset d'entraînement",
                    datasets if datasets else ["Aucun dataset disponible"],
                    key="genetic_dataset"
                )
                
                # Paramètres de l'algorithme génétique
                population_size = st.slider(
                    "Taille de la population",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Nombre d'individus dans la population"
                )
                
                generations = st.slider(
                    "Nombre de générations",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    help="Nombre de générations à exécuter"
                )
            
            with col2:
                # Paramètres avancés de l'algorithme génétique
                crossover_rate = st.slider(
                    "Taux de croisement",
                    min_value=0.5,
                    max_value=0.9,
                    value=0.7,
                    step=0.1,
                    help="Probabilité de croisement entre individus"
                )
                
                mutation_rate = st.slider(
                    "Taux de mutation",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.1,
                    help="Probabilité de mutation des individus"
                )
                
                tournament_size = st.slider(
                    "Taille du tournoi",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1,
                    help="Nombre d'individus dans chaque tournoi de sélection"
                )
            
            # Plages de recherche pour les hyperparamètres
            st.markdown("### Plages de recherche des hyperparamètres")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                l2_reg_min = st.number_input(
                    "L2 régularisation (min)",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.0001,
                    format="%.4f"
                )
                
                l2_reg_max = st.number_input(
                    "L2 régularisation (max)",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.01,
                    format="%.4f"
                )
            
            with col2:
                dropout_min = st.number_input(
                    "Dropout (min)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.1,
                    step=0.1
                )
                
                dropout_max = st.number_input(
                    "Dropout (max)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.5,
                    step=0.1
                )
            
            with col3:
                lr_min = st.number_input(
                    "Taux d'apprentissage (min)",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.0001,
                    format="%.4f"
                )
                
                lr_max = st.number_input(
                    "Taux d'apprentissage (max)",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.01,
                    format="%.4f"
                )
            
            # Options supplémentaires
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size_min = st.select_slider(
                    "Taille du batch (min)",
                    options=[16, 32, 64, 128],
                    value=16
                )
                
                batch_size_max = st.select_slider(
                    "Taille du batch (max)",
                    options=[16, 32, 64, 128],
                    value=128
                )
            
            with col2:
                # Nom du modèle optimisé
                model_name = st.text_input(
                    "Nom du modèle optimisé",
                    value="morningstar_optimized",
                    help="Nom du modèle à sauvegarder après optimisation"
                )
                
                # Nombre d'époques pour l'évaluation
                eval_epochs = st.slider(
                    "Époques d'évaluation",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Nombre d'époques pour évaluer chaque combinaison d'hyperparamètres"
                )
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Lancer l'optimisation génétique")
        
        if submit_button:
            if selected_dataset == "Aucun dataset disponible":
                st.error("Aucun dataset disponible pour l'optimisation.")
            else:
                # Vérifier que les plages sont cohérentes
                if l2_reg_min >= l2_reg_max:
                    st.error("La valeur minimale de régularisation L2 doit être inférieure à la valeur maximale.")
                elif dropout_min >= dropout_max:
                    st.error("La valeur minimale de dropout doit être inférieure à la valeur maximale.")
                elif lr_min >= lr_max:
                    st.error("La valeur minimale du taux d'apprentissage doit être inférieure à la valeur maximale.")
                elif batch_size_min >= batch_size_max:
                    st.error("La valeur minimale de la taille du batch doit être inférieure à la valeur maximale.")
                else:
                    # Préparer les arguments pour le script
                    args = [
                        "--data-path", selected_dataset,
                        "--output-dir", str(MODEL_DIR / "trained" / "morningstar"),
                        "--population-size", str(population_size),
                        "--generations", str(generations),
                        "--crossover-rate", str(crossover_rate),
                        "--mutation-rate", str(mutation_rate),
                        "--tournament-size", str(tournament_size),
                        "--l2-reg-min", str(l2_reg_min),
                        "--l2-reg-max", str(l2_reg_max),
                        "--dropout-min", str(dropout_min),
                        "--dropout-max", str(dropout_max),
                        "--lr-min", str(lr_min),
                        "--lr-max", str(lr_max),
                        "--batch-size-min", str(batch_size_min),
                        "--batch-size-max", str(batch_size_max),
                        "--eval-epochs", str(eval_epochs),
                        "--model-name", model_name
                    ]
                    
                    # Exécuter le script d'optimisation génétique
                    with st.spinner("Optimisation génétique en cours... Cela peut prendre un certain temps."):
                        success, output = run_training_script("model/training/genetic_optimizer.py", args)
                        
                        if success:
                            st.success("Optimisation génétique terminée avec succès !")
                            
                            # Essayer de charger les résultats de l'optimisation
                            results_path = MODEL_DIR / "trained" / "morningstar" / "genetic_optimization_results.json"
                            if results_path.exists():
                                with open(results_path, 'r') as f:
                                    results = json.load(f)
                                
                                # Afficher les meilleurs hyperparamètres
                                st.markdown("### Meilleurs hyperparamètres trouvés")
                                
                                best_hyperparams = results.get('best_hyperparams', {})
                                cols = st.columns(len(best_hyperparams))
                                
                                for i, (param, value) in enumerate(best_hyperparams.items()):
                                    with cols[i]:
                                        if param == 'batch_size':
                                            value = 2 ** int(value)  # Convertir l'exposant en taille réelle
                                        elif param == 'use_batch_norm':
                                            value = "Oui" if int(value) == 1 else "Non"
                                        elif isinstance(value, float):
                                            value = f"{value:.4f}"
                                        
                                        st.metric(
                                            label=param.replace('_', ' ').title(),
                                            value=value
                                        )
                                
                                # Afficher l'évolution de l'optimisation
                                if 'logbook' in results:
                                    logbook = results['logbook']
                                    generations = list(range(len(logbook)))
                                    
                                    # Créer un DataFrame pour le graphique
                                    df = pd.DataFrame({
                                        'Génération': generations,
                                        'Fitness moyen': [gen['avg'] for gen in logbook],
                                        'Fitness max': [gen['max'] for gen in logbook]
                                    })
                                    
                                    # Créer le graphique
                                    fig = px.line(
                                        df, 
                                        x='Génération', 
                                        y=['Fitness moyen', 'Fitness max'],
                                        title='Évolution de l\'optimisation génétique',
                                        labels={'value': 'Fitness', 'variable': 'Métrique'}
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Afficher la sortie complète
                            with st.expander("Voir les détails de l'optimisation"):
                                st.code(output)
                        else:
                            st.error("Erreur lors de l'optimisation génétique.")
                            st.code(output)
    
    # Afficher les modèles disponibles
    st.markdown("<h2 class='sub-header'>Modèles disponibles</h2>", unsafe_allow_html=True)
    
    models = get_available_models()
    if models:
        # Créer un tableau avec les informations des modèles
        model_data = []
        
        for model_path in models:
            model_name = Path(model_path).name
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # En Mo
            model_date = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M:%S")
            
            model_data.append({
                "Nom": model_name,
                "Taille": f"{model_size:.2f} Mo",
                "Date de création": model_date
            })
        
        # Afficher le tableau des modèles
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun modèle disponible. Utilisez le formulaire ci-dessus pour entraîner le modèle Morningstar.")
