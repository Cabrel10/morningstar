#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Architecture de modèle avec capacité de raisonnement pour Morningstar.
Ce modèle combine l'architecture simplifiée avec un module de raisonnement
pour expliquer les décisions de trading.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Embedding, Flatten, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import Tuple, Dict, Optional, List, Any, Union

from model.reasoning.reasoning_module import ReasoningModule


def build_reasoning_model(
    tech_input_shape: Optional[Tuple[int]] = None,  # Rendu optionnel
    # llm_embedding_dim: int = 1, # Supprimé, remplacé par cryptobert
    mcp_input_dim: int = 0,  # Défaut 0
    hmm_input_dim: int = 0,  # Défaut 0
    sentiment_input_dim: int = 0,  # Défaut 0
    cryptobert_input_dim: int = 0,  # Défaut 0
    market_input_dim: int = 0,  # Défaut 0
    instrument_vocab_size: int = 0,  # Défaut 0
    instrument_embedding_dim: int = 8,
    num_market_regime_classes: int = 2,
    num_sl_tp_outputs: int = 2,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    num_reasoning_steps: int = 3,
    reasoning_units: int = 128,
    num_attention_heads: int = 4,
    attention_key_dim: int = 64,
    sl_tp_initial_bias: Optional[List[float]] = None,
    active_outputs: List[str] = None,
    feature_names: Optional[List[str]] = None,
    use_chain_of_thought: bool = True,
):
    """
    Construit un modèle Morningstar avec capacité de raisonnement.

    Args:
        tech_input_shape: Shape des features techniques (si > 0)
        # llm_embedding_dim: Supprimé
        mcp_input_dim: Dimension des features MCP (si > 0)
        hmm_input_dim: Dimension des features HMM (si > 0)
        sentiment_input_dim: Dimension des features de sentiment (si > 0)
        cryptobert_input_dim: Dimension des embeddings CryptoBERT (si > 0)
        market_input_dim: Dimension des features de marché (si > 0)
        instrument_vocab_size: Taille du vocabulaire des instruments
        instrument_embedding_dim: Dimension de l'embedding des instruments
        num_market_regime_classes: Nombre de classes pour le régime de marché
        num_sl_tp_outputs: Nombre de sorties pour SL/TP
        l2_reg: Coefficient de régularisation L2
        dropout_rate: Taux de dropout
        use_batch_norm: Utiliser la normalisation par batch
        num_reasoning_steps: Nombre d'étapes de raisonnement
        reasoning_units: Nombre d'unités dans les couches de raisonnement
        num_attention_heads: Nombre de têtes d'attention
        attention_key_dim: Dimension des clefs d'attention
        sl_tp_initial_bias: Valeurs d'initialisation du biais pour la couche SL/TP
        active_outputs: Liste des noms des sorties à construire
        feature_names: Noms des features pour le décodage des explications
        use_chain_of_thought: Activer le raisonnement Chain-of-Thought

    Returns:
        Le modèle Keras compilé
    """
    # Définir les valeurs par défaut
    if active_outputs is None:
        active_outputs = ["market_regime", "sl_tp", "reasoning"]

    # 1. Définir les entrées
    inputs_dict = {}
    features_to_merge = []

    # --- Technical Features ---
    x_technical = None
    if tech_input_shape and tech_input_shape[0] > 0:
        technical_input = Input(shape=tech_input_shape, name="technical_input")
        inputs_dict["technical_input"] = technical_input
        x_technical = Dense(
            64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="technical_dense_1"
        )(technical_input)
        if use_batch_norm:
            x_technical = BatchNormalization(name="technical_bn_1")(x_technical)
        x_technical = Dropout(dropout_rate, name="technical_dropout_1")(x_technical)
        x_technical = Dense(
            32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="technical_dense_2"
        )(x_technical)
        features_to_merge.append(x_technical)
    else:
        print("INFO: build_reasoning_model - Skipping technical input branch as tech_input_shape is invalid or zero.")

    # --- LLM Features (Supprimé, remplacé par CryptoBERT) ---

    # --- MCP Features ---
    x_mcp = None
    if mcp_input_dim and mcp_input_dim > 0:
        mcp_input = Input(shape=(mcp_input_dim,), name="mcp_input")
        inputs_dict["mcp_input"] = mcp_input
        x_mcp = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="mcp_dense_1")(mcp_input)
        if use_batch_norm:
            x_mcp = BatchNormalization(name="mcp_bn_1")(x_mcp)
        x_mcp = Dropout(dropout_rate, name="mcp_dropout_1")(x_mcp)
        features_to_merge.append(x_mcp)

    # --- HMM Features ---
    x_hmm = None
    if hmm_input_dim and hmm_input_dim > 0:
        hmm_input = Input(shape=(hmm_input_dim,), name="hmm_input")
        inputs_dict["hmm_input"] = hmm_input
        x_hmm = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="hmm_dense_1")(hmm_input)
        if use_batch_norm:
            x_hmm = BatchNormalization(name="hmm_bn_1")(x_hmm)
        x_hmm = Dropout(dropout_rate, name="hmm_dropout_1")(x_hmm)
        features_to_merge.append(x_hmm)

    # --- Instrument Features ---
    instrument_embedding = None
    if instrument_vocab_size and instrument_vocab_size > 0 and instrument_embedding_dim > 0:
        instrument_input = Input(shape=(1,), name="instrument_input", dtype="int32")
        inputs_dict["instrument_input"] = instrument_input
        instrument_embedding = Embedding(
            input_dim=instrument_vocab_size, output_dim=instrument_embedding_dim, name="instrument_embedding"
        )(instrument_input)
        instrument_embedding = Flatten(name="instrument_flatten")(instrument_embedding)
        features_to_merge.append(instrument_embedding)

    # --- Sentiment Features ---
    x_sentiment = None
    if sentiment_input_dim and sentiment_input_dim > 0:
        sentiment_input = Input(shape=(sentiment_input_dim,), name="sentiment_input")
        inputs_dict["sentiment_input"] = sentiment_input
        x_sentiment = Dense(
            16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="sentiment_dense_1"
        )(sentiment_input)
        if use_batch_norm:
            x_sentiment = BatchNormalization(name="sentiment_bn_1")(x_sentiment)
        x_sentiment = Dropout(dropout_rate, name="sentiment_dropout_1")(x_sentiment)
        features_to_merge.append(x_sentiment)

    # --- CryptoBERT Features ---
    x_cryptobert = None
    if cryptobert_input_dim and cryptobert_input_dim > 0:
        # Utilise le nom d'input correspondant à la clé dans X_features
        cryptobert_input = Input(shape=(cryptobert_input_dim,), name="cryptobert_input")
        inputs_dict["cryptobert_input"] = cryptobert_input
        # Traitement des features CryptoBERT
        x_cryptobert = Dense(
            64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="cryptobert_dense_1"
        )(cryptobert_input)
        if use_batch_norm:
            x_cryptobert = BatchNormalization(name="cryptobert_bn_1")(x_cryptobert)
        x_cryptobert = Dropout(dropout_rate, name="cryptobert_dropout_1")(x_cryptobert)
        x_cryptobert = Dense(
            32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="cryptobert_dense_2"
        )(x_cryptobert)
        features_to_merge.append(x_cryptobert)

    # --- Market Features ---
    x_market = None
    if market_input_dim and market_input_dim > 0:
        market_input = Input(shape=(market_input_dim,), name="market_input")
        inputs_dict["market_input"] = market_input
        x_market = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="market_dense_1")(
            market_input
        )
        if use_batch_norm:
            x_market = BatchNormalization(name="market_bn_1")(x_market)
        x_market = Dropout(dropout_rate, name="market_dropout_1")(x_market)
        features_to_merge.append(x_market)

    # 2. Traitement des features (déjà fait ci-dessus)

    # 3. Fusion des features
    if len(features_to_merge) == 0:
        raise ValueError("Aucune feature à fusionner. Vérifiez les dimensions d'entrée dans la configuration.")
    elif len(features_to_merge) == 1:
        merged_features = features_to_merge[0]
    else:
        merged_features = Concatenate(name="merged_features")(features_to_merge)

    # 4. Couches partagées
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shared_dense1")(merged_features)
    if use_batch_norm:
        x = BatchNormalization(name="shared_bn1")(x)
    x = Dropout(dropout_rate, name="shared_dropout1")(x)

    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shared_dense2")(x)
    if use_batch_norm:
        x = BatchNormalization(name="shared_bn2")(x)
    x = Dropout(dropout_rate, name="shared_dropout2")(x)

    # 5. Module de raisonnement
    reasoning_outputs = None
    if "reasoning" in active_outputs:
        # Créer un module de raisonnement
        reasoning_module = ReasoningModule(
            num_reasoning_steps=num_reasoning_steps,
            reasoning_units=reasoning_units,
            num_attention_heads=num_attention_heads,
            attention_key_dim=attention_key_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            num_market_regimes=num_market_regime_classes,
            name="reasoning_module",
        )

        # Appliquer le module de raisonnement aux features combinées
        reasoning_inputs = {
            "features": x,
            "market_regime": x if "market_regime" in active_outputs else None,
            "sl_tp": x if "sl_tp" in active_outputs else None,
        }

        reasoning_outputs = reasoning_module(reasoning_inputs, training=True)

        # Ajouter les sorties de raisonnement au dictionnaire de sorties
        outputs = {
            "final_reasoning": reasoning_outputs["final_reasoning"],
            "attention_scores": reasoning_outputs["attention_scores"],
        }

        # Ajouter les étapes de raisonnement comme sorties si Chain-of-Thought est activé
        if use_chain_of_thought:
            for i, step in enumerate(reasoning_outputs["reasoning_steps"]):
                outputs[f"reasoning_step_{i}"] = step

            # L'explication textuelle sera générée post-entraînement via ExplanationDecoder
            # outputs['cot_explanation'] = tf.zeros((1, 1), name='cot_explanation') # Ligne supprimée
    else:
        outputs = {}

    # 6. Têtes de sortie
    if "market_regime" in active_outputs:
        market_regime_output = Dense(num_market_regime_classes, activation="softmax", name="market_regime")(x)
        outputs["market_regime"] = market_regime_output

        # Ajouter l'explication du régime de marché si le raisonnement est actif
        if reasoning_outputs is not None:
            outputs["market_regime_explanation"] = reasoning_outputs["market_regime_explanation"]

    if "sl_tp" in active_outputs:
        sl_tp_output = Dense(
            num_sl_tp_outputs,
            activation="linear",
            name="sl_tp",
            bias_initializer=(
                "zeros" if sl_tp_initial_bias is None else tf.keras.initializers.Constant(sl_tp_initial_bias)
            ),
        )(x)
        outputs["sl_tp"] = sl_tp_output

        # Ajouter les explications SL/TP si le raisonnement est actif
        if reasoning_outputs is not None:
            outputs["sl_explanation"] = reasoning_outputs["sl_explanation"]
            outputs["tp_explanation"] = reasoning_outputs["tp_explanation"]

    # 7. Création du modèle
    model = Model(
        inputs=inputs_dict,  # Utiliser le dictionnaire d'inputs créé dynamiquement
        outputs=outputs,
        name="morningstar_reasoning_model",
    )

    return model


# Fonction pour compiler le modèle avec des paramètres optimaux
def compile_reasoning_model(model, learning_rate=0.001, active_outputs=None):
    """
    Compile le modèle avec des paramètres optimaux.

    Args:
        model: Modèle Keras à compiler
        learning_rate: Taux d'apprentissage
        active_outputs: Liste des noms des sorties actives

    Returns:
        Le modèle compilé
    """
    if active_outputs is None:
        active_outputs = ["market_regime", "sl_tp", "reasoning"]

    # Définir les pertes et métriques
    losses = {}
    metrics = {}
    loss_weights = {}

    if "market_regime" in active_outputs:
        losses["market_regime"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics["market_regime"] = ["accuracy"]
        loss_weights["market_regime"] = 1.0

        # Ajouter une perte pour l'explication du régime de marché si présente
        if "market_regime_explanation" in model.output_names:
            # Perte symbolique pour l'explication (pas vraiment utilisée pour l'entraînement)
            losses["market_regime_explanation"] = tf.keras.losses.MeanSquaredError()
            loss_weights["market_regime_explanation"] = 0.0  # Poids nul

    if "sl_tp" in active_outputs:
        losses["sl_tp"] = tf.keras.losses.Huber(delta=1.0)
        metrics["sl_tp"] = [
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ]
        loss_weights["sl_tp"] = 1.0

        # Ajouter des pertes pour les explications SL/TP si présentes
        if "sl_explanation" in model.output_names:
            losses["sl_explanation"] = tf.keras.losses.MeanSquaredError()
            loss_weights["sl_explanation"] = 0.0  # Poids nul

        if "tp_explanation" in model.output_names:
            losses["tp_explanation"] = tf.keras.losses.MeanSquaredError()
            loss_weights["tp_explanation"] = 0.0  # Poids nul

    # Ajouter des pertes pour les sorties de raisonnement si présentes
    if "final_reasoning" in model.output_names:
        losses["final_reasoning"] = tf.keras.losses.MeanSquaredError()
        loss_weights["final_reasoning"] = 0.1  # Poids faible

    if "attention_scores" in model.output_names:
        losses["attention_scores"] = tf.keras.losses.MeanSquaredError()
        loss_weights["attention_scores"] = 0.0  # Poids nul

    # Ajouter des pertes pour les étapes de raisonnement si présentes
    for output_name in model.output_names:
        if output_name.startswith("reasoning_step_"):
            losses[output_name] = tf.keras.losses.MeanSquaredError()
            loss_weights[output_name] = 0.05  # Poids très faible

    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
    )

    return model
