#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modèle Monolithique Morningstar
-------------------------------

Architecture unifiée qui combine toutes les entrées et fonctionnalités dans un seul modèle Keras.
Ce modèle remplace les architectures séparées précédentes (simplified, morningstar, enhanced_hybrid, reasoning)
avec une approche unifiée.

Le modèle ingère:
- Données techniques (OHLCV, indicateurs)
- Embeddings de LLM/texte
- Données MCP (Market Context Processor)
- Identifiants d'instruments
- Données Chain-of-Thought (optionnel)

Architecture:
1. Backbone partagé (Dense → LSTM → Transformer)
2. Trois têtes spécialisées:
   - Signal: Classification pour {Sell, Neutral, Buy}
   - SL: Régression pour niveau de stop-loss
   - TP: Régression pour niveau de take-profit
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization,
    Embedding, Flatten, GlobalAveragePooling1D, MultiHeadAttention,
    LayerNormalization, LSTM, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import Dict, Optional, List, Tuple, Any, Union
import logging
import numpy as np


class TransformerBlock(tf.keras.layers.Layer):
    """Bloc Transformer avec multi-head attention et feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Initialise le bloc Transformer.
        
        Args:
            embed_dim: Dimension d'embedding pour l'attention multi-tête
            num_heads: Nombre de têtes d'attention
            ff_dim: Dimension de la couche feed-forward
            rate: Taux de dropout
        """
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        """
        Passe les entrées à travers le bloc Transformer.
        
        Args:
            inputs: Tenseurs d'entrée pour le bloc
            training: Flag d'entraînement pour le dropout
            
        Returns:
            Sortie du bloc Transformer
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        """Retourne la configuration du bloc pour sérialisation."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def build_monolith_model(
    tech_input_shape: Tuple[int] = (38,),
    embeddings_input_shape: int = 768,
    mcp_input_shape: int = 128,
    instrument_vocab_size: int = 10,
    instrument_embedding_dim: int = 8,
    cot_input_shape: Optional[int] = None,
    sequence_length: Optional[int] = None,
    backbone_config: Optional[Dict[str, Any]] = None,
    head_config: Optional[Dict[str, Any]] = None,
    use_lstm: bool = True,
    use_transformer: bool = True,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    active_outputs: List[str] = None
) -> tf.keras.Model:
    """
    Construit le modèle monolithique Morningstar.
    
    Args:
        tech_input_shape: Shape des features techniques (default: 38)
        embeddings_input_shape: Dimension des embeddings LLM (default: 768)
        mcp_input_shape: Dimension des features MCP (default: 128)
        instrument_vocab_size: Taille du vocabulaire d'instruments (default: 10)
        instrument_embedding_dim: Dimension de l'embedding d'instrument (default: 8)
        cot_input_shape: Dimension de l'entrée Chain-of-Thought (optionnel)
        sequence_length: Longueur de séquence pour les entrées (si None, modèle non-séquentiel)
        backbone_config: Configuration du backbone (unités, blocs, etc.)
        head_config: Configuration des têtes de sortie
        use_lstm: Utiliser une couche LSTM dans le backbone
        use_transformer: Utiliser des blocs Transformer dans le backbone
        l2_reg: Coefficient de régularisation L2
        dropout_rate: Taux de dropout
        use_batch_norm: Utiliser la normalisation par lot
        active_outputs: Liste des sorties actives (signal, sl, tp)
        
    Returns:
        Modèle Keras compilé
    """
    # Configuration par défaut
    if backbone_config is None:
        backbone_config = {
            "dense_units": 128,
            "lstm_units": 64,
            "transformer_blocks": 2,
            "transformer_heads": 4,
            "transformer_dim": 64,
            "ff_dim": 128
        }
    
    if head_config is None:
        head_config = {
            "signal": {"units": [32], "classes": 3},
            "sl_tp": {"units": [32], "outputs": 2}
        }
    
    if active_outputs is None:
        active_outputs = ["signal", "sl", "tp"]
    
    # --- Entrées du modèle ---
    inputs_dict = {}
    
    # Entrée technique
    if sequence_length is not None:
        tech_input = Input(shape=(sequence_length, tech_input_shape[0]), name="technical_input")
    else:
        tech_input = Input(shape=tech_input_shape, name="technical_input")
    inputs_dict["technical_input"] = tech_input
    
    # Entrée embeddings
    embeddings_input = Input(shape=(embeddings_input_shape,), name="embeddings_input")
    inputs_dict["embeddings_input"] = embeddings_input
    
    # Entrée MCP
    mcp_input = Input(shape=(mcp_input_shape,), name="mcp_input")
    inputs_dict["mcp_input"] = mcp_input
    
    # Entrée instrument
    instrument_input = Input(shape=(1,), dtype=tf.int32, name="instrument_input")
    inputs_dict["instrument_input"] = instrument_input
    
    # Entrée Chain-of-Thought (optionnelle)
    cot_input = None
    if cot_input_shape is not None:
        cot_input = Input(shape=(cot_input_shape,), name="cot_input")
        inputs_dict["cot_input"] = cot_input
    
    # --- Traitement des entrées ---
    
    # Traitement des features techniques
    x_tech = tech_input
    
    if sequence_length is None:
        # Cas non-séquentiel: Dense layers
        x_tech = Dense(
            backbone_config["dense_units"], 
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="tech_dense1"
        )(x_tech)
        
        if use_batch_norm:
            x_tech = BatchNormalization(name="tech_bn1")(x_tech)
        
        x_tech = Dropout(dropout_rate, name="tech_dropout1")(x_tech)
        
        # Reshape pour LSTM si nécessaire
        if use_lstm:
            # Ajouter dimension de séquence (traiter comme séquence de longueur 1)
            x_tech = Reshape((1, backbone_config["dense_units"]))(x_tech)
    
    # LSTM (séquentiel ou non)
    if use_lstm:
        x_tech = LSTM(
            backbone_config["lstm_units"],
            return_sequences=use_transformer,  # Retourner séquences si Transformer suit
            name="lstm_layer"
        )(x_tech)
    
    # Blocs Transformer (si activés)
    if use_transformer and (use_lstm or sequence_length is not None):
        for i in range(backbone_config["transformer_blocks"]):
            x_tech = TransformerBlock(
                embed_dim=backbone_config["transformer_dim"],
                num_heads=backbone_config["transformer_heads"],
                ff_dim=backbone_config["ff_dim"],
                rate=dropout_rate
            )(x_tech)
        
        # Pooling global pour réduire à un vecteur
        x_tech = GlobalAveragePooling1D(name="global_pooling")(x_tech)
    
    # Traitement des embeddings
    x_embeddings = Dense(
        64, 
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="embeddings_dense"
    )(embeddings_input)
    
    if use_batch_norm:
        x_embeddings = BatchNormalization(name="embeddings_bn")(x_embeddings)
    
    x_embeddings = Dropout(dropout_rate, name="embeddings_dropout")(x_embeddings)
    
    # Traitement MCP
    x_mcp = Dense(
        32, 
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="mcp_dense"
    )(mcp_input)
    
    if use_batch_norm:
        x_mcp = BatchNormalization(name="mcp_bn")(x_mcp)
    
    x_mcp = Dropout(dropout_rate, name="mcp_dropout")(x_mcp)
    
    # Traitement de l'instrument
    x_instrument = Embedding(
        instrument_vocab_size,
        instrument_embedding_dim,
        name="instrument_embedding"
    )(instrument_input)
    
    x_instrument = Flatten(name="instrument_flatten")(x_instrument)
    
    # Traitement CoT (si présent)
    x_cot = None
    if cot_input is not None:
        x_cot = Dense(
            32, 
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="cot_dense"
        )(cot_input)
        
        if use_batch_norm:
            x_cot = BatchNormalization(name="cot_bn")(x_cot)
        
        x_cot = Dropout(dropout_rate, name="cot_dropout")(x_cot)
    
    # --- Fusion des features ---
    features_to_concat = [x_tech, x_embeddings, x_mcp, x_instrument]
    if x_cot is not None:
        features_to_concat.append(x_cot)
    
    x = Concatenate(name="fusion")(features_to_concat)
    
    # --- Couche partagée finale ---
    x = Dense(
        128, 
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="shared_dense"
    )(x)
    
    if use_batch_norm:
        x = BatchNormalization(name="shared_bn")(x)
    
    x = Dropout(dropout_rate, name="shared_dropout")(x)
    
    # --- Têtes de sortie ---
    outputs_dict = {}
    
    # Tête Signal
    if "signal" in active_outputs:
        signal_head = x
        for i, units in enumerate(head_config["signal"]["units"]):
            signal_head = Dense(
                units, 
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"signal_dense_{i+1}"
            )(signal_head)
        
        signal_output = Dense(
            head_config["signal"]["classes"],
            activation="softmax",
            name="signal_output"
        )(signal_head)
        
        outputs_dict["signal_output"] = signal_output
    
    # Têtes SL et TP
    if "sl" in active_outputs or "tp" in active_outputs:
        sl_tp_head = x
        for i, units in enumerate(head_config["sl_tp"]["units"]):
            sl_tp_head = Dense(
                units, 
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f"sl_tp_dense_{i+1}"
            )(sl_tp_head)
        
        sl_tp_output = Dense(
            head_config["sl_tp"]["outputs"],
            activation="linear",
            name="sl_tp_output"
        )(sl_tp_head)
        
        outputs_dict["sl_tp_output"] = sl_tp_output
    
    # Créer le modèle
    model = Model(inputs=inputs_dict, outputs=outputs_dict, name="MonolithModel")
    
    # Compiler le modèle
    losses = {}
    metrics = {}
    
    if "signal_output" in outputs_dict:
        losses["signal_output"] = "categorical_crossentropy"
        metrics["signal_output"] = ["accuracy"]
    
    if "sl_tp_output" in outputs_dict:
        losses["sl_tp_output"] = "mse"
        metrics["sl_tp_output"] = ["mae"]
    
    model.compile(
        optimizer="adam",
        loss=losses,
        metrics=metrics
    )
    
    return model


def load_monolith_model(filepath: str, custom_objects: Optional[Dict] = None) -> tf.keras.Model:
    """
    Charge un modèle monolithique sauvegardé.
    
    Args:
        filepath: Chemin vers le fichier modèle sauvegardé
        custom_objects: Objets personnalisés requis pour le chargement
        
    Returns:
        Modèle Keras chargé
    """
    if custom_objects is None:
        custom_objects = {"TransformerBlock": TransformerBlock}
    else:
        custom_objects["TransformerBlock"] = TransformerBlock
    
    return tf.keras.models.load_model(filepath, custom_objects=custom_objects)


class MonolithModel:
    """
    Classe wrapper pour le modèle monolithique Morningstar.
    Facilite l'utilisation du modèle avec diverses fonctionnalités.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[tf.keras.Model] = None
    ):
        """
        Initialise le wrapper du modèle monolithique.
        
        Args:
            config: Configuration du modèle
            model: Modèle Keras préexistant (optionnel)
        """
        self.logger = logging.getLogger("MonolithModel")
        
        # Configuration par défaut
        self.default_config = {
            "tech_input_shape": (38,),
            "embeddings_input_shape": 768,
            "mcp_input_shape": 128,
            "instrument_vocab_size": 10,
            "instrument_embedding_dim": 8,
            "sequence_length": None,
            "use_lstm": True,
            "use_transformer": True,
            "backbone_config": {
                "dense_units": 128,
                "lstm_units": 64,
                "transformer_blocks": 2,
                "transformer_heads": 4,
                "transformer_dim": 64,
                "ff_dim": 128
            },
            "head_config": {
                "signal": {"units": [32], "classes": 3},
                "sl_tp": {"units": [32], "outputs": 2}
            },
            "l2_reg": 0.001,
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "active_outputs": ["signal", "sl", "tp"]
        }
        
        # Fusionner avec la configuration fournie
        self.config = self.default_config.copy()
        if config is not None:
            self._update_nested_dict(self.config, config)
        
        # Créer ou utiliser le modèle fourni
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Met à jour un dictionnaire imbriqué avec un autre."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _build_model(self) -> tf.keras.Model:
        """Construit le modèle à partir de la configuration."""
        return build_monolith_model(
            tech_input_shape=self.config["tech_input_shape"],
            embeddings_input_shape=self.config["embeddings_input_shape"],
            mcp_input_shape=self.config["mcp_input_shape"],
            instrument_vocab_size=self.config["instrument_vocab_size"],
            instrument_embedding_dim=self.config["instrument_embedding_dim"],
            sequence_length=self.config["sequence_length"],
            backbone_config=self.config["backbone_config"],
            head_config=self.config["head_config"],
            use_lstm=self.config["use_lstm"],
            use_transformer=self.config["use_transformer"],
            l2_reg=self.config["l2_reg"],
            dropout_rate=self.config["dropout_rate"],
            use_batch_norm=self.config["use_batch_norm"],
            active_outputs=self.config["active_outputs"]
        )
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            inputs: Dictionnaire d'entrées correspondant aux noms du modèle
        
        Returns:
            Dictionnaire des sorties prédites
        """
        return self.model.predict(inputs)
    
    def save(self, filepath: str, save_config: bool = True) -> None:
        """
        Sauvegarde le modèle et éventuellement sa configuration.
        
        Args:
            filepath: Chemin de sauvegarde du modèle
            save_config: Si True, sauvegarde aussi la configuration
        """
        self.model.save(filepath)
        
        if save_config:
            import json
            import os
            
            config_path = os.path.splitext(filepath)[0] + "_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, load_config: bool = True, custom_objects: Optional[Dict] = None) -> "MonolithModel":
        """
        Charge un modèle sauvegardé et sa configuration.
        
        Args:
            filepath: Chemin du modèle sauvegardé
            load_config: Si True, charge aussi la configuration
            custom_objects: Objets personnalisés pour le chargement
            
        Returns:
            Instance MonolithModel avec le modèle chargé
        """
        # Charger le modèle
        model = load_monolith_model(filepath, custom_objects)
        
        # Charger la configuration si demandé
        config = None
        if load_config:
            import json
            import os
            
            config_path = os.path.splitext(filepath)[0] + "_config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
        
        # Créer l'instance
        return cls(config=config, model=model)
    
    def summary(self) -> None:
        """Affiche un résumé du modèle."""
        self.model.summary()


# Test simple pour vérifier l'architecture
if __name__ == "__main__":
    # Créer un modèle avec configuration par défaut
    model = MonolithModel()
    model.summary()
    
    # Teste avec un mini-batch d'entrées aléatoires
    import numpy as np
    
    batch_size = 2
    test_inputs = {
        "technical_input": np.random.randn(batch_size, 38),
        "embeddings_input": np.random.randn(batch_size, 768),
        "mcp_input": np.random.randn(batch_size, 128),
        "instrument_input": np.random.randint(0, 10, (batch_size, 1))
    }
    
    # Prédiction
    outputs = model.predict(test_inputs)
    print("\nSorties du modèle:")
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}") 