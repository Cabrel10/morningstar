#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Architecture simplifiu00e9e et mieux ru00e9gularisu00e9e pour le modu00e8le Morningstar.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Embedding, Flatten, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import Tuple, Dict, Optional, List


def build_simplified_model(
    tech_input_shape: Tuple[int] = (44,),
    llm_embedding_dim: int = 768,
    mcp_input_dim: int = 128,
    hmm_input_dim: int = 4,
    instrument_vocab_size: int = 10,
    instrument_embedding_dim: int = 8,
    num_market_regime_classes: int = 4,
    num_sl_tp_outputs: int = 2,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    sl_tp_initial_bias: Optional[List[float]] = None,
    active_outputs: List[str] = None,
):
    """
    Construit un modu00e8le simplifiu00e9 et mieux ru00e9gularisu00e9 pour Morningstar.

    Args:
        tech_input_shape: Shape des features techniques (44)
        llm_embedding_dim: Dimension des embeddings LLM (768)
        mcp_input_dim: Dimension des features MCP (128)
        hmm_input_dim: Dimension des features HMM (ru00e9gime + probas) (4)
        instrument_vocab_size: Taille du vocabulaire des instruments
        instrument_embedding_dim: Dimension de l'embedding des instruments
        num_market_regime_classes: Nombre de classes pour le ru00e9gime de marchu00e9
        num_sl_tp_outputs: Nombre de sorties pour SL/TP
        l2_reg: Coefficient de ru00e9gularisation L2
        dropout_rate: Taux de dropout
        use_batch_norm: Utiliser la normalisation par batch
        sl_tp_initial_bias: Valeurs d'initialisation du biais pour la couche SL/TP
        active_outputs: Liste des noms des sorties u00e0 construire

    Returns:
        Le modu00e8le Keras compilu00e9
    """
    # Du00e9finir les valeurs par du00e9faut
    if active_outputs is None:
        active_outputs = ["market_regime", "sl_tp"]

    # 1. Du00e9finir les entru00e9es
    technical_input = Input(shape=tech_input_shape, name="technical_input")
    llm_input = Input(shape=(llm_embedding_dim,), name="llm_input")
    mcp_input = Input(shape=(mcp_input_dim,), name="mcp_input")
    hmm_input = Input(shape=(hmm_input_dim,), name="hmm_input")
    instrument_input = Input(shape=(1,), dtype="int32", name="instrument_input")

    # 2. Traitement des features techniques
    x_tech = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="tech_dense")(
        technical_input
    )
    if use_batch_norm:
        x_tech = BatchNormalization(name="tech_bn")(x_tech)
    x_tech = Dropout(dropout_rate, name="tech_dropout")(x_tech)

    # 3. Traitement des features LLM
    x_llm = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="llm_dense")(llm_input)
    if use_batch_norm:
        x_llm = BatchNormalization(name="llm_bn")(x_llm)
    x_llm = Dropout(dropout_rate, name="llm_dropout")(x_llm)

    # 4. Traitement des features MCP
    x_mcp = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="mcp_dense")(mcp_input)
    if use_batch_norm:
        x_mcp = BatchNormalization(name="mcp_bn")(x_mcp)
    x_mcp = Dropout(dropout_rate, name="mcp_dropout")(x_mcp)

    # 5. Traitement des features HMM
    x_hmm = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="hmm_dense")(hmm_input)
    if use_batch_norm:
        x_hmm = BatchNormalization(name="hmm_bn")(x_hmm)
    x_hmm = Dropout(dropout_rate, name="hmm_dropout")(x_hmm)

    # 6. Traitement de l'instrument
    x_instrument = Embedding(instrument_vocab_size, instrument_embedding_dim, name="instrument_embedding")(
        instrument_input
    )
    x_instrument = Flatten(name="instrument_flatten")(x_instrument)

    # 7. Fusion des features
    fusion = Concatenate(name="fusion_concat")([x_tech, x_llm, x_mcp, x_hmm, x_instrument])

    # 8. Couches partagu00e9es
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shared_dense1")(fusion)
    if use_batch_norm:
        x = BatchNormalization(name="shared_bn1")(x)
    x = Dropout(dropout_rate, name="shared_dropout1")(x)

    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="shared_dense2")(x)
    if use_batch_norm:
        x = BatchNormalization(name="shared_bn2")(x)
    x = Dropout(dropout_rate, name="shared_dropout2")(x)

    # 9. Tu00eates de sortie
    outputs = {}

    if "market_regime" in active_outputs:
        market_regime_output = Dense(num_market_regime_classes, activation="softmax", name="market_regime")(x)
        outputs["market_regime"] = market_regime_output

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

    # 10. Cru00e9er le modu00e8le
    model = Model(
        inputs={
            "technical_input": technical_input,
            "llm_input": llm_input,
            "mcp_input": mcp_input,
            "hmm_input": hmm_input,
            "instrument_input": instrument_input,
        },
        outputs=outputs,
        name="simplified_morningstar",
    )

    return model


# Fonction pour compiler le modu00e8le avec des paramu00e8tres optimaux
def compile_model(model, learning_rate=0.001, active_outputs=None):
    """
    Compile le modu00e8le avec des paramu00e8tres optimaux.

    Args:
        model: Modu00e8le Keras u00e0 compiler
        learning_rate: Taux d'apprentissage
        active_outputs: Liste des noms des sorties actives

    Returns:
        Le modu00e8le compilu00e9
    """
    if active_outputs is None:
        active_outputs = ["market_regime", "sl_tp"]

    # Du00e9finir les pertes et mu00e9triques
    losses = {}
    metrics = {}
    loss_weights = {}

    if "market_regime" in active_outputs:
        losses["market_regime"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics["market_regime"] = ["accuracy"]
        loss_weights["market_regime"] = 1.0

    if "sl_tp" in active_outputs:
        losses["sl_tp"] = tf.keras.losses.Huber(delta=1.0)
        metrics["sl_tp"] = [
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ]
        loss_weights["sl_tp"] = 1.0

    # Compiler le modu00e8le
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights,
    )

    return model
