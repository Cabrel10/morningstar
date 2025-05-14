#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modèle Monolithique Morningstar
-------------------------------

Architecture unifiée qui combine toutes les entrées et fonctionnalités dans un seul modèle Keras.
Ce modèle remplace les architectures séparées précédentes avec une approche unifiée.

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
    LayerNormalization, LSTM, Reshape, RepeatVector
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import Dict, Optional, List, Tuple, Any, Union
import logging
import numpy as np
import json
import os


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
            Dense(ff_dim, activation="relu", kernel_initializer="he_normal"),
            Dense(embed_dim, kernel_initializer="he_normal"),
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
    active_outputs: List[str] = None,
    learning_rate: float = 1e-3
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
        active_outputs: Liste des sorties actives (signal, sl_tp)
        learning_rate: Taux d'apprentissage pour l'optimiseur Adam
        
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
        active_outputs = ["signal", "sl_tp"]
    
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
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
            name="tech_dense1"
        )(x_tech)
        
        if use_batch_norm:
            x_tech = BatchNormalization(name="tech_bn1")(x_tech)
        
        x_tech = Dropout(dropout_rate, name="tech_dropout1")(x_tech)
        
        # Reshape pour LSTM si nécessaire
        if use_lstm:
            # Ajouter dimension de séquence (traiter comme séquence de longueur 1)
            x_tech = Reshape((1, backbone_config["dense_units"]), name="tech_reshape")(x_tech)
            
            # Si on utilise le Transformer sans séquence d'entrée, on duplique la timestep
            if use_transformer:
                x_tech = RepeatVector(3, name="tech_repeat")(x_tech[:, 0, :])
    
    # LSTM (séquentiel ou non)
    if use_lstm:
        x_tech = LSTM(
            backbone_config["lstm_units"],
            return_sequences=use_transformer,  # Retourner séquences si Transformer suit
            kernel_initializer="orthogonal",
            recurrent_initializer="orthogonal",
            name="lstm_layer"
        )(x_tech)
    
    # Blocs Transformer (si activés et seulement si on a une séquence)
    if use_transformer and ((use_lstm and sequence_length is not None) or (use_lstm and sequence_length is None)):
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
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        bias_regularizer=regularizers.l2(l2_reg),
        name="embeddings_dense"
    )(embeddings_input)
    
    if use_batch_norm:
        x_embeddings = BatchNormalization(name="embeddings_bn")(x_embeddings)
    
    x_embeddings = Dropout(dropout_rate, name="embeddings_dropout")(x_embeddings)
    
    # Traitement MCP
    x_mcp = Dense(
        32, 
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        bias_regularizer=regularizers.l2(l2_reg),
        name="mcp_dense"
    )(mcp_input)
    
    if use_batch_norm:
        x_mcp = BatchNormalization(name="mcp_bn")(x_mcp)
    
    x_mcp = Dropout(dropout_rate, name="mcp_dropout")(x_mcp)
    
    # Traitement de l'instrument
    x_instrument = Embedding(
        instrument_vocab_size,
        instrument_embedding_dim,
        embeddings_initializer="uniform",
        name="instrument_embedding"
    )(instrument_input)
    
    x_instrument = Flatten(name="instrument_flatten")(x_instrument)
    
    # Traitement CoT (si présent)
    x_cot = None
    if cot_input is not None:
        x_cot = Dense(
            32, 
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
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
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        bias_regularizer=regularizers.l2(l2_reg),
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
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
                name=f"signal_dense_{i+1}"
            )(signal_head)
        
        signal_output = Dense(
            head_config["signal"]["classes"],
            activation="softmax",
            kernel_initializer="glorot_uniform",
            name="signal_output"
        )(signal_head)
        
        outputs_dict["signal_output"] = signal_output
    
    # Tête SL/TP unifiée
    if "sl_tp" in active_outputs:
        sl_tp_head = x
        for i, units in enumerate(head_config["sl_tp"]["units"]):
            sl_tp_head = Dense(
                units, 
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_reg),
                bias_regularizer=regularizers.l2(l2_reg),
                name=f"sl_tp_dense_{i+1}"
            )(sl_tp_head)
        
        sl_tp_output = Dense(
            head_config["sl_tp"]["outputs"],
            activation="linear",
            kernel_initializer="glorot_uniform",
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
        # Définir des métriques personnalisées pour SL et TP séparément
        metrics["sl_tp_output"] = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanAbsoluteError(name="mae_sl", subset_weights=[True, False]),
            tf.keras.metrics.MeanAbsoluteError(name="mae_tp", subset_weights=[False, True])
        ]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
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
        # Configurer le logger
        self.logger = logging.getLogger("MonolithModel")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
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
            "active_outputs": ["signal", "sl_tp"],
            "learning_rate": 1e-3
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
            cot_input_shape=self.config.get("cot_input_shape"),
            sequence_length=self.config["sequence_length"],
            backbone_config=self.config["backbone_config"],
            head_config=self.config["head_config"],
            use_lstm=self.config["use_lstm"],
            use_transformer=self.config["use_transformer"],
            l2_reg=self.config["l2_reg"],
            dropout_rate=self.config["dropout_rate"],
            use_batch_norm=self.config["use_batch_norm"],
            active_outputs=self.config["active_outputs"],
            learning_rate=self.config.get("learning_rate", 1e-3)
        )
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Effectue une prédiction avec le modèle.
        
        Args:
            inputs: Dictionnaire d'entrées correspondant aux noms du modèle
        
        Returns:
            Dictionnaire des sorties prédites
        """
        predictions = self.model.predict(inputs)
        
        # Standardiser la gestion des prédictions
        if isinstance(predictions, dict):
            return {k: v for k, v in predictions.items()}
        elif isinstance(predictions, list):
            return dict(zip([o.name.split("/")[0] for o in self.model.outputs], predictions))
        else:
            # Si une seule sortie
            return {self.model.output_names[0]: predictions}
    
    def train(
        self, 
        inputs: Dict[str, np.ndarray], 
        outputs: Dict[str, np.ndarray], 
        validation_data: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Entraîne le modèle monolithique.
        
        Args:
            inputs: Dictionnaire des entrées d'entraînement
            outputs: Dictionnaire des sorties d'entraînement
            validation_data: Tuple optionnel (inputs_val, outputs_val) pour la validation
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille de batch
            callbacks: Liste de callbacks Keras
            verbose: Niveau de verbosité (0, 1, ou 2)
            
        Returns:
            Historique d'entraînement
        """
        self.logger.info(f"Démarrage de l'entraînement sur {len(next(iter(inputs.values())))} échantillons")
        
        # Préparer les données de validation si fournies
        validation_data_formatted = None
        if validation_data is not None:
            val_inputs, val_outputs = validation_data
            validation_data_formatted = (val_inputs, val_outputs)
            self.logger.info(f"Validation sur {len(next(iter(val_inputs.values())))} échantillons")
        
        # Entraîner le modèle
        history = self.model.fit(
            inputs,
            outputs,
            validation_data=validation_data_formatted,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.logger.info("Entraînement terminé")
        return history
    
    def evaluate(
        self,
        inputs: Dict[str, np.ndarray],
        outputs: Dict[str, np.ndarray],
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur des données de test.
        
        Args:
            inputs: Dictionnaire des entrées de test
            outputs: Dictionnaire des sorties de test (vérité terrain)
            batch_size: Taille de batch pour l'évaluation
            verbose: Niveau de verbosité
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        self.logger.info(f"Évaluation du modèle sur {len(next(iter(inputs.values())))} échantillons")
        
        # Évaluer le modèle
        results = self.model.evaluate(
            inputs,
            outputs,
            batch_size=batch_size,
            verbose=verbose,
            return_dict=True
        )
        
        # Afficher les résultats
        for metric_name, value in results.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
        
        return results
    
    def save(self, filepath: str, save_config: bool = True) -> None:
        """
        Sauvegarde le modèle et éventuellement sa configuration.
        
        Args:
            filepath: Chemin de sauvegarde du modèle
            save_config: Si True, sauvegarde aussi la configuration
        """
        self.model.save(filepath)
        self.logger.info(f"Modèle sauvegardé à {filepath}")
        
        if save_config:
            config_path = os.path.splitext(filepath)[0] + "_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration sauvegardée à {config_path}")
    
    @classmethod
    def load(cls, filepath: str, load_config: bool = True, custom_objects: Optional[Dict] = None) -> "MonolithModel":
        """
        Charge un modèle monolithique sauvegardé.
        
        Args:
            filepath: Chemin vers le fichier modèle
            load_config: Si True, essaie de charger la configuration
            custom_objects: Objets personnalisés pour le chargement
            
        Returns:
            Instance de MonolithModel avec le modèle chargé
        """
        model = load_monolith_model(filepath, custom_objects)
        
        config = None
        if load_config:
            config_path = os.path.splitext(filepath)[0] + "_config.json"
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Impossible de charger la configuration du modèle: {e}")
                config = None
        
        return cls(config=config, model=model)
    
    def summary(self) -> None:
        """Affiche le résumé du modèle."""
        self.model.summary() 