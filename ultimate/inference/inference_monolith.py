#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'inférence pour le modèle monolithique Morningstar.

Ce script permet de:
1. Charger un modèle monolithique entraîné
2. Préparer les données d'entrée
3. Effectuer des prédictions
4. Interpréter et visualiser les résultats
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import argparse
from datetime import datetime

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inference_monolith")

# Importer le modèle monolithique
try:
    from ultimate.model.monolith_model import MonolithModel
except ImportError:
    # Fallback pour les tests ou les exécutions directes
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ultimate.model.monolith_model import MonolithModel


def load_model_with_metadata(model_path: str) -> Tuple[MonolithModel, Dict]:
    """
    Charge le modèle monolithique et ses métadonnées associées.
    
    Args:
        model_path: Chemin vers le fichier modèle
        
    Returns:
        Tuple (modèle, métadonnées)
    """
    logger.info(f"Chargement du modèle depuis {model_path}")
    
    # Dossier contenant le modèle et les métadonnées
    model_dir = os.path.dirname(model_path)
    
    # Charger le modèle
    model = MonolithModel.load(model_path)
    
    # Chercher le fichier de métadonnées
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(model_dir, os.path.splitext(os.path.basename(model_path))[0] + "_metadata.json")
    
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Métadonnées chargées depuis {metadata_path}")
        except json.JSONDecodeError:
            logger.warning(f"Erreur lors du chargement des métadonnées depuis {metadata_path}")
    else:
        logger.warning("Aucun fichier de métadonnées trouvé")
    
    # Chercher les scalers
    scalers_path = os.path.join(model_dir, "scalers.pkl")
    if os.path.exists(scalers_path):
        try:
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            metadata["scalers"] = scalers
            logger.info(f"Scalers chargés depuis {scalers_path}")
        except (pickle.PickleError, EOFError):
            logger.warning(f"Erreur lors du chargement des scalers depuis {scalers_path}")
    
    return model, metadata


def prepare_inference_data(
    df: pd.DataFrame,
    metadata: Dict,
    tech_cols: Optional[List[str]] = None,
    embedding_cols: Optional[List[str]] = None,
    mcp_cols: Optional[List[str]] = None,
    instrument_col: Optional[str] = None,
    embedding_dim: int = 768
) -> Dict[str, np.ndarray]:
    """
    Prépare les données pour l'inférence.
    
    Args:
        df: DataFrame contenant les données
        metadata: Métadonnées du modèle
        tech_cols: Liste des colonnes techniques à utiliser (si None, utilise metadata)
        embedding_cols: Liste des colonnes d'embeddings à utiliser (si None, utilise metadata)
        mcp_cols: Liste des colonnes MCP à utiliser (si None, utilise metadata)
        instrument_col: Nom de la colonne contenant l'identifiant de l'instrument
        embedding_dim: Dimension des embeddings (par défaut 768 pour modèles de type BERT)
        
    Returns:
        Dictionnaire des inputs pour le modèle
    """
    logger.info("Préparation des données pour l'inférence")
    
    # Utiliser les colonnes des métadonnées si non spécifiées
    if tech_cols is None and "tech_cols" in metadata:
        tech_cols = metadata["tech_cols"]
    
    if embedding_cols is None and "llm_cols" in metadata:
        embedding_cols = metadata["llm_cols"]
    elif embedding_cols is None and "embedding_cols" in metadata:
        embedding_cols = metadata["embedding_cols"]
    
    if mcp_cols is None and "mcp_cols" in metadata:
        mcp_cols = metadata["mcp_cols"]
    
    if instrument_col is None and "instrument_col" in metadata:
        instrument_col = metadata["instrument_col"]
    elif instrument_col is None:
        instrument_col = "symbol"
    
    # Vérifier que les colonnes existent
    missing_cols = []
    if tech_cols:
        missing_tech = [col for col in tech_cols if col not in df.columns]
        if missing_tech:
            logger.warning(f"Colonnes techniques manquantes: {missing_tech}")
            tech_cols = [col for col in tech_cols if col in df.columns]
            missing_cols.extend(missing_tech)
    
    if embedding_cols:
        missing_emb = [col for col in embedding_cols if col not in df.columns]
        if missing_emb:
            logger.warning(f"Colonnes d'embeddings manquantes: {missing_emb}")
            embedding_cols = [col for col in embedding_cols if col in df.columns]
            missing_cols.extend(missing_emb)
    
    if mcp_cols:
        missing_mcp = [col for col in mcp_cols if col not in df.columns]
        if missing_mcp:
            logger.warning(f"Colonnes MCP manquantes: {missing_mcp}")
            mcp_cols = [col for col in mcp_cols if col in df.columns]
            missing_cols.extend(missing_mcp)
    
    if instrument_col not in df.columns:
        logger.warning(f"Colonne d'instrument '{instrument_col}' manquante. Création d'une colonne par défaut.")
        df[instrument_col] = "default"
    
    if missing_cols:
        logger.warning(f"Total de {len(missing_cols)} colonnes manquantes")
    
    # Dimensions d'entrée du modèle à partir des métadonnées ou des valeurs par défaut
    tech_input_dim = metadata.get("tech_input_shape", [38])[0] if isinstance(metadata.get("tech_input_shape"), list) else 38
    embeddings_input_dim = metadata.get("embeddings_input_shape", embedding_dim)
    mcp_input_dim = metadata.get("mcp_input_shape", 128)
    
    # Préparation des features techniques
    if tech_cols:
        X_tech = df[tech_cols].values
        
        # Vérifier la dimension des entrées techniques
        if X_tech.shape[1] != tech_input_dim:
            logger.warning(f"Dimension des entrées techniques ({X_tech.shape[1]}) ne correspond pas à celle attendue ({tech_input_dim})")
            
            if X_tech.shape[1] < tech_input_dim:
                # Padding avec des zéros
                padding = np.zeros((X_tech.shape[0], tech_input_dim - X_tech.shape[1]))
                X_tech = np.concatenate([X_tech, padding], axis=1)
                logger.info(f"Ajout de padding pour atteindre la dimension attendue")
            else:
                # Tronquer
                X_tech = X_tech[:, :tech_input_dim]
                logger.info(f"Troncature des entrées techniques à la dimension attendue")
        
        # Appliquer le scaler si disponible
        if "scalers" in metadata and "tech" in metadata["scalers"]:
            try:
                X_tech = metadata["scalers"]["tech"].transform(X_tech)
                logger.info("Données techniques normalisées avec le scaler")
            except Exception as e:
                logger.warning(f"Erreur lors de l'application du scaler technique: {e}")
    else:
        logger.warning(f"Aucune colonne technique spécifiée ou trouvée. Utilisation de zéros ({tech_input_dim} dimensions).")
        X_tech = np.zeros((len(df), tech_input_dim))
    
    # Préparation des embeddings
    if embedding_cols and len(embedding_cols) > 0:
        try:
            # Essai d'extraction comme vecteurs (si stockés sous forme de liste/tableau)
            if isinstance(df[embedding_cols[0]].iloc[0], (list, np.ndarray)):
                X_emb = np.stack(df[embedding_cols[0]].values)
            # Si stockés sous forme de chaîne JSON
            elif isinstance(df[embedding_cols[0]].iloc[0], str):
                try:
                    X_emb = np.array([json.loads(emb) for emb in df[embedding_cols[0]].values])
                except json.JSONDecodeError:
                    logger.warning("Les embeddings sont en texte mais pas au format JSON. Utilisation de zéros.")
                    X_emb = np.zeros((len(df), embeddings_input_dim))
            else:
                logger.warning(f"Format d'embedding non reconnu: {type(df[embedding_cols[0]].iloc[0])}. Utilisation de zéros.")
                X_emb = np.zeros((len(df), embeddings_input_dim))
                
            # Vérifier la dimension
            if X_emb.shape[1] != embeddings_input_dim:
                logger.warning(f"Dimension des embeddings ({X_emb.shape[1]}) ne correspond pas à celle attendue ({embeddings_input_dim})")
                if X_emb.shape[1] < embeddings_input_dim:
                    padding = np.zeros((X_emb.shape[0], embeddings_input_dim - X_emb.shape[1]))
                    X_emb = np.concatenate([X_emb, padding], axis=1)
                else:
                    X_emb = X_emb[:, :embeddings_input_dim]
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Impossible d'extraire les embeddings: {e}. Utilisation de zéros.")
            X_emb = np.zeros((len(df), embeddings_input_dim))
    else:
        logger.warning(f"Aucune colonne d'embeddings spécifiée ou trouvée. Utilisation de zéros ({embeddings_input_dim} dimensions).")
        X_emb = np.zeros((len(df), embeddings_input_dim))
    
    # Préparation des features MCP
    if mcp_cols:
        X_mcp = df[mcp_cols].values
        
        # Vérifier la dimension
        if X_mcp.shape[1] != mcp_input_dim:
            logger.warning(f"Dimension des entrées MCP ({X_mcp.shape[1]}) ne correspond pas à celle attendue ({mcp_input_dim})")
            if X_mcp.shape[1] < mcp_input_dim:
                padding = np.zeros((X_mcp.shape[0], mcp_input_dim - X_mcp.shape[1]))
                X_mcp = np.concatenate([X_mcp, padding], axis=1)
            else:
                X_mcp = X_mcp[:, :mcp_input_dim]
        
        # Appliquer le scaler si disponible
        if "scalers" in metadata and "mcp" in metadata["scalers"]:
            try:
                X_mcp = metadata["scalers"]["mcp"].transform(X_mcp)
                logger.info("Données MCP normalisées avec le scaler")
            except Exception as e:
                logger.warning(f"Erreur lors de l'application du scaler MCP: {e}")
    else:
        logger.warning(f"Aucune colonne MCP spécifiée ou trouvée. Utilisation de zéros ({mcp_input_dim} dimensions).")
        X_mcp = np.zeros((len(df), mcp_input_dim))
    
    # Préparation des identifiants d'instruments
    if "instrument_map" in metadata:
        instrument_map = metadata["instrument_map"]
        instruments = df[instrument_col].values
        
        # Convertir les instruments en indices
        instrument_indices = []
        for inst in instruments:
            inst_str = str(inst)
            if inst_str in instrument_map:
                instrument_indices.append(instrument_map[inst_str])
            else:
                logger.warning(f"Instrument inconnu: {inst}. Utilisation de l'indice 0.")
                instrument_indices.append(0)
        
        X_inst = np.array(instrument_indices).reshape(-1, 1)
    else:
        logger.warning("Aucune carte d'instruments trouvée dans les métadonnées. Utilisation de zéros.")
        X_inst = np.zeros((len(df), 1), dtype=np.int32)
    
    # Créer le dictionnaire d'entrées
    inputs = {
        "technical_input": X_tech,
        "embeddings_input": X_emb,
        "mcp_input": X_mcp,
        "instrument_input": X_inst.astype(np.int32)
    }
    
    # Ajouter CoT si présent dans le modèle
    if "cot_input_shape" in metadata:
        cot_dim = metadata["cot_input_shape"]
        inputs["cot_input"] = np.zeros((len(df), cot_dim))
        logger.info(f"Ajout d'une entrée CoT de dimension {cot_dim}")
    
    # Vérifier la présence de séquence
    if "sequence_length" in metadata and metadata["sequence_length"] is not None:
        logger.info(f"Le modèle attend des séquences de longueur {metadata['sequence_length']}")
        
        # Reshape si nécessaire et possible
        if len(X_tech.shape) == 2:  # Non séquentiel → séquentiel
            seq_len = metadata["sequence_length"]
            if len(df) >= seq_len:
                # Transformer en séquences
                logger.info(f"Conversion des données en séquences de longueur {seq_len}")
                # Cette logique devrait être adaptée selon le besoin réel
            else:
                logger.warning(f"Pas assez de données ({len(df)}) pour former une séquence de longueur {seq_len}")
    
    logger.info(f"Données préparées: {len(df)} échantillons")
    return inputs


def run_inference(
    model: MonolithModel,
    inputs: Dict[str, np.ndarray],
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Exécute l'inférence avec le modèle monolithique.
    
    Args:
        model: Instance du modèle monolithique
        inputs: Dictionnaire des inputs pour le modèle
        batch_size: Taille de batch pour l'inférence
        
    Returns:
        Dictionnaire des prédictions
    """
    logger.info("Exécution de l'inférence...")
    
    predictions = model.predict(inputs)
    
    logger.info(f"Inférence terminée. Sorties: {list(predictions.keys())}")
    return predictions


def interpret_predictions(
    predictions: Dict[str, np.ndarray],
    signal_threshold: float = 0.5,
    return_classes: bool = False
) -> Dict[str, np.ndarray]:
    """
    Interprète les prédictions du modèle.
    
    Args:
        predictions: Dictionnaire des prédictions du modèle
        signal_threshold: Seuil pour la conversion en classe
        return_classes: Si True, retourne les classes plutôt que les probabilités
        
    Returns:
        Dictionnaire des prédictions interprétées
    """
    logger.info("Interprétation des prédictions...")
    
    interpreted = {}
    
    # Interpréter les signaux (probas de classe -> classe)
    if "signal_output" in predictions:
        signal_probs = predictions["signal_output"]
        
        if return_classes:
            # Convertir en classes (0: Sell, 1: Neutral, 2: Buy)
            signal_classes = np.argmax(signal_probs, axis=1)
            interpreted["signal"] = signal_classes
        else:
            # Garder les probabilités
            interpreted["signal_proba"] = signal_probs
            
            # Calculer un score continu (-1 à 1) pour faciliter la visualisation
            # -1 = forte probabilité de vente, 1 = forte probabilité d'achat
            signal_score = signal_probs[:, 2] - signal_probs[:, 0]
            interpreted["signal_score"] = signal_score
    
    # Interpréter les SL/TP
    if "sl_tp_output" in predictions:
        sl_tp_values = predictions["sl_tp_output"]
        
        # Séparer SL et TP
        if sl_tp_values.shape[1] >= 2:
            interpreted["stop_loss"] = sl_tp_values[:, 0]
            interpreted["take_profit"] = sl_tp_values[:, 1]
    
    logger.info("Interprétation terminée")
    return interpreted


def save_predictions(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    interpreted: Dict[str, np.ndarray],
    output_path: str
) -> pd.DataFrame:
    """
    Sauvegarde les prédictions dans un fichier.
    
    Args:
        df: DataFrame d'origine
        predictions: Prédictions brutes du modèle
        interpreted: Prédictions interprétées
        output_path: Chemin de sauvegarde
        
    Returns:
        DataFrame avec les prédictions ajoutées
    """
    # Créer une copie du DataFrame
    result_df = df.copy()
    
    # Ajouter les prédictions
    for key, values in interpreted.items():
        result_df[f"pred_{key}"] = values
    
    # Pour les probabilités de signal, ajouter chaque classe
    if "signal_output" in predictions:
        signal_probs = predictions["signal_output"]
        for i in range(signal_probs.shape[1]):
            class_name = ["sell", "neutral", "buy"][i] if i < 3 else f"class_{i}"
            result_df[f"pred_signal_proba_{class_name}"] = signal_probs[:, i]
    
    # Sauvegarder
    if output_path.endswith('.parquet'):
        result_df.to_parquet(output_path)
    else:
        result_df.to_csv(output_path)
    
    logger.info(f"Prédictions sauvegardées dans {output_path}")
    return result_df


def visualize_predictions(
    df: pd.DataFrame,
    interpreted: Dict[str, np.ndarray],
    output_path: str = None,
    show_plot: bool = True
):
    """
    Visualise les prédictions.
    
    Args:
        df: DataFrame avec les données et les prédictions
        interpreted: Prédictions interprétées
        output_path: Chemin pour sauvegarder la visualisation (optionnel)
        show_plot: Si True, affiche le graphique
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logger.error("Matplotlib non disponible. Impossible de visualiser les prédictions.")
        return
    
    logger.info("Création de la visualisation...")
    
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Sous-graphique 1: Prix et SL/TP
    ax1 = axs[0]
    if 'close' in df.columns:
        ax1.plot(df.index, df['close'], label='Prix de clôture', color='black')
        
        # Ajouter SL/TP si disponibles
        if 'stop_loss' in interpreted:
            ax1.plot(df.index, interpreted['stop_loss'], label='Stop Loss', color='red', linestyle='--', alpha=0.7)
        
        if 'take_profit' in interpreted:
            ax1.plot(df.index, interpreted['take_profit'], label='Take Profit', color='green', linestyle='--', alpha=0.7)
        
        ax1.set_title('Prix et niveaux SL/TP')
        ax1.set_ylabel('Prix')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
    
    # Sous-graphique 2: Score de signal
    ax2 = axs[1]
    if 'signal_score' in interpreted:
        signal_scores = interpreted['signal_score']
        ax2.plot(df.index, signal_scores, label='Score de signal', color='blue')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.fill_between(df.index, 0, signal_scores, where=(signal_scores > 0), 
                         color='green', alpha=0.3, label='Achat')
        ax2.fill_between(df.index, 0, signal_scores, where=(signal_scores < 0), 
                         color='red', alpha=0.3, label='Vente')
        
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_title('Score de signal (-1: Vente, 1: Achat)')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
    
    # Sous-graphique 3: Probabilités de classe
    ax3 = axs[2]
    if 'signal_output' in df.columns or any(col.startswith('pred_signal_proba_') for col in df.columns):
        proba_cols = [col for col in df.columns if col.startswith('pred_signal_proba_')]
        if len(proba_cols) >= 3:
            ax3.stackplot(df.index, 
                         df[proba_cols[0]].values, 
                         df[proba_cols[1]].values, 
                         df[proba_cols[2]].values,
                         labels=['Vente', 'Neutre', 'Achat'],
                         colors=['red', 'gray', 'green'],
                         alpha=0.7)
            
            ax3.set_ylim(0, 1)
            ax3.set_title('Probabilités de classe')
            ax3.set_ylabel('Probabilité')
            ax3.set_xlabel('Date')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
    
    # Formater l'axe des x pour les dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(MaxNLocator(10))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualisation sauvegardée dans {output_path}")
    
    # Afficher si demandé
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Inférence avec modèle monolithique Morningstar")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le fichier modèle")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers le fichier de données")
    parser.add_argument("--output", type=str, help="Chemin pour sauvegarder les prédictions")
    parser.add_argument("--visualize", action="store_true", help="Générer une visualisation")
    parser.add_argument("--vis-output", type=str, help="Chemin pour sauvegarder la visualisation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil pour la conversion en classe")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille de batch pour l'inférence")
    
    args = parser.parse_args()
    
    # Définir le chemin de sortie par défaut si non spécifié
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(args.data) if os.path.dirname(args.data) else "."
        base_name = os.path.splitext(os.path.basename(args.data))[0]
        args.output = os.path.join(output_dir, f"{base_name}_predictions_{timestamp}.parquet")
    
    # Charger le modèle et les métadonnées
    model, metadata = load_model_with_metadata(args.model)
    
    # Charger les données
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Format de fichier non supporté: {args.data}")
    
    logger.info(f"Données chargées: {len(df)} échantillons, {len(df.columns)} colonnes")
    
    # Préparer les données pour l'inférence
    inputs = prepare_inference_data(df, metadata)
    
    # Exécuter l'inférence
    predictions = run_inference(model, inputs, args.batch_size)
    
    # Interpréter les prédictions
    interpreted = interpret_predictions(predictions, args.threshold)
    
    # Sauvegarder les prédictions
    result_df = save_predictions(df, predictions, interpreted, args.output)
    
    # Visualiser les prédictions si demandé
    if args.visualize:
        if not args.vis_output:
            vis_output = os.path.splitext(args.output)[0] + ".png"
        else:
            vis_output = args.vis_output
        
        visualize_predictions(result_df, interpreted, vis_output)
    
    logger.info("Inférence terminée avec succès!")


if __name__ == "__main__":
    main() 