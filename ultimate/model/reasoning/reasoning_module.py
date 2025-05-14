#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de raisonnement pour le modu00e8le Morningstar.
Ce module permet au modu00e8le d'expliquer ses du00e9cisions de trading en utilisant
des techniques de Chain-of-Thought (CoT) et d'attention interprétable.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Concatenate,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Reshape,
    Embedding,
    Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging # Ajout de l'import logging

# Initialisation du logger pour ce module
logger = logging.getLogger(__name__)


class ReasoningStep(Layer):
    """
    Couche de raisonnement qui implémente une étape de Chain-of-Thought.
    Cette couche prend des features en entrée et produit un raisonnement intermédiaire.
    """

    def __init__(self, units: int, dropout_rate: float = 0.2, l2_reg: float = 0.001, name: str = None):
        super(ReasoningStep, self).__init__(name=name)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Couches pour le raisonnement
        self.dense1 = Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_dense1" if name else None,
        )
        self.norm1 = LayerNormalization(name=f"{name}_norm1" if name else None)
        self.dropout1 = Dropout(dropout_rate, name=f"{name}_dropout1" if name else None)

        self.dense2 = Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_dense2" if name else None,
        )
        self.norm2 = LayerNormalization(name=f"{name}_norm2" if name else None)
        self.dropout2 = Dropout(dropout_rate, name=f"{name}_dropout2" if name else None)

        # Couche de sortie pour le raisonnement
        self.reasoning_output = Dense(
            units,
            activation="tanh",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_reasoning_output" if name else None,
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.norm2(x)
        x = self.dropout2(x, training=training)

        reasoning = self.reasoning_output(x)

        return reasoning

    def get_config(self):
        config = super(ReasoningStep, self).get_config()
        config.update({"units": self.units, "dropout_rate": self.dropout_rate, "l2_reg": self.l2_reg})
        return config


class InterpretableAttention(Layer):
    """
    Couche d'attention interprétable qui permet de comprendre quelles features
    ont le plus d'influence sur les décisions du modèle.
    """

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        return_attention_scores: bool = True,
        name: str = None,
    ):
        super(InterpretableAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.return_attention_scores = return_attention_scores

        # Utiliser les paramètres compatibles avec la version actuelle de TensorFlow
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate, name=f"{name}_mha" if name else None
        )
        self.norm = LayerNormalization(name=f"{name}_norm" if name else None)

    def call(self, inputs, training=False):
        # Ajouter une dimension temporelle si nécessaire
        if len(inputs.shape) == 2:
            expanded_inputs = tf.expand_dims(inputs, axis=1)
        else:
            expanded_inputs = inputs

        # Appliquer l'attention
        if self.return_attention_scores:
            # Avec la version actuelle de TensorFlow, nous devons gérer les scores d'attention différemment
            attention_output = self.mha(expanded_inputs, expanded_inputs, return_attention_scores=True)
            # Dans les versions récentes, attention_output est un tuple (output, attention_scores)
            attention_output, attention_scores = attention_output
        else:
            attention_output = self.mha(expanded_inputs, expanded_inputs)
            attention_scores = None

        # Normalisation
        attention_output = self.norm(attention_output)

        # Réduire la dimension temporelle si elle a été ajoutée
        if len(inputs.shape) == 2:
            attention_output = tf.squeeze(attention_output, axis=1)

        if self.return_attention_scores:
            return attention_output, attention_scores
        else:
            return attention_output

    def get_config(self):
        config = super(InterpretableAttention, self).get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "return_attention_scores": self.return_attention_scores,
            }
        )
        return config


class ReasoningModule(Layer):
    """
    Module de raisonnement complet qui combine plusieurs étapes de raisonnement
    et un mécanisme d'attention interprétable pour expliquer les décisions de trading.
    """

    def __init__(
        self,
        num_reasoning_steps: int = 3,
        reasoning_units: int = 128,
        num_attention_heads: int = 4,
        attention_key_dim: int = 64,
        dropout_rate: float = 0.2,
        l2_reg: float = 0.001,
        num_market_regimes: int = 3,
        name: str = None,
    ):
        super(ReasoningModule, self).__init__(name=name)
        self.num_reasoning_steps = num_reasoning_steps
        self.reasoning_units = reasoning_units
        self.num_attention_heads = num_attention_heads
        self.attention_key_dim = attention_key_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.num_market_regimes = num_market_regimes

        # Étapes de raisonnement
        self.reasoning_steps = [
            ReasoningStep(
                units=reasoning_units,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg,
                name=f"{name}_reasoning_step_{i}" if name else None,
            )
            for i in range(num_reasoning_steps)
        ]

        # Attention interprétable
        self.attention = InterpretableAttention(
            num_heads=num_attention_heads,
            key_dim=attention_key_dim,
            dropout_rate=dropout_rate,
            return_attention_scores=True,
            name=f"{name}_attention" if name else None,
        )

        # Couches pour générer les explications
        self.explanation_dense = Dense(
            reasoning_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_explanation_dense" if name else None,
        )

        # Couches pour les différents types d'explications
        self.market_regime_explanation = Dense(
            num_market_regimes * reasoning_units,
            activation="linear",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_market_regime_explanation" if name else None,
        )

        self.sl_explanation = Dense(
            reasoning_units,
            activation="linear",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_sl_explanation" if name else None,
        )

        self.tp_explanation = Dense(
            reasoning_units,
            activation="linear",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_tp_explanation" if name else None,
        )

        # Couche de fusion pour combiner les raisonnements
        self.fusion = Dense(
            reasoning_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_fusion" if name else None,
        )

        # Couche de sortie pour le raisonnement final
        self.reasoning_output = Dense(
            reasoning_units,
            activation="tanh",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_reasoning_output" if name else None,
        )

    def call(self, inputs: Dict[str, tf.Tensor], training=False):
        # Appliquer les étapes de raisonnement séquentiellement
        reasoning_outputs = []

        # L'entrée initiale pour la première étape de raisonnement est 'features'
        current_reasoning_input = inputs["features"]

        # Si d'autres entrées contextuelles sont fournies (market_regime, sl_tp),
        # elles pourraient être concaténées ou traitées spécifiquement ici si nécessaire.
        # Pour l'instant, nous supposons que chaque ReasoningStep opère sur la sortie du précédent,
        # en commençant par les features principales.

        for step_module in self.reasoning_steps:
            current_reasoning_input = step_module(current_reasoning_input, training=training)
            reasoning_outputs.append(current_reasoning_input)

        # Le reste du code utilise la dernière sortie de raisonnement ou la collection
        x = current_reasoning_input  # La sortie de la dernière étape

        # Concaténer tous les raisonnements intermédiaires
        all_reasoning = tf.stack(reasoning_outputs, axis=1)

        # Appliquer l'attention pour identifier les étapes de raisonnement importantes
        attention_output, attention_scores = self.attention(all_reasoning, training=training)

        # Calculer la moyenne sur la dimension temporelle
        attention_output = tf.reduce_mean(attention_output, axis=1)

        # Générer les explications pour chaque type de prédiction
        explanation_base = self.explanation_dense(attention_output)

        market_regime_explanation = self.market_regime_explanation(explanation_base)
        market_regime_explanation = tf.reshape(
            market_regime_explanation, [-1, self.num_market_regimes, self.reasoning_units]
        )

        sl_explanation = self.sl_explanation(explanation_base)
        tp_explanation = self.tp_explanation(explanation_base)

        # Fusionner les raisonnements
        final_reasoning = self.fusion(attention_output)
        final_reasoning = self.reasoning_output(final_reasoning)

        return {
            "final_reasoning": final_reasoning,
            "market_regime_explanation": market_regime_explanation,
            "sl_explanation": sl_explanation,
            "tp_explanation": tp_explanation,
            "attention_scores": attention_scores,
            "reasoning_steps": reasoning_outputs,
        }

    def get_config(self):
        config = super(ReasoningModule, self).get_config()
        config.update(
            {
                "num_reasoning_steps": self.num_reasoning_steps,
                "reasoning_units": self.reasoning_units,
                "num_attention_heads": self.num_attention_heads,
                "attention_key_dim": self.attention_key_dim,
                "dropout_rate": self.dropout_rate,
                "l2_reg": self.l2_reg,
                "num_market_regimes": self.num_market_regimes,
            }
        )
        return config


class ExplanationDecoder:
    """
    Décodeur d'explications qui convertit les représentations vectorielles
    en explications textuelles compréhensibles.
    """

    def __init__(
        self,
        feature_names: List[str],
        market_regime_names: List[str] = None,
        explanation_templates: Dict[str, str] = None,
    ):
        self.feature_names = feature_names
        self.market_regime_names = market_regime_names or ["Baissier", "Neutre", "Haussier", "Volatil"]

        # Templates par défaut pour les explications
        default_templates = {
            "market_regime": "Le régime de marché prédit est {regime}. Raisons principales: {reasons}",
            "feature_importance": "{feature} (importance: {importance:.1f}%)",
            "sl": "Stop Loss recommandé à {sl_value}. Basé sur: {reasons}",
            "tp": "Take Profit recommandé à {tp_value}. Basé sur: {reasons}",
            "reasoning_step": "Étape {step}: {reasoning}",
            "signal": "Signal de trading: {signal}. Confiance: {confidence:.1f}%",
            "volatility": "Volatilité prévue: {volatility}. Impact sur le trading: {impact}",
            "market_context": "Contexte de marché: {context}. Implications: {implications}",
            "technical_analysis": "Analyse technique: {analysis}. Indicateurs clés: {indicators}",
            "risk_assessment": "Évaluation du risque: {risk_level}. Facteurs: {factors}",
        }

        # Utiliser les templates fournis ou les templates par défaut
        self.explanation_templates = explanation_templates or default_templates

    def decode_market_regime_explanation(
        self,
        market_regime_pred_idx: int, # Index du régime prédit
        market_regime_explanation_vec: np.ndarray, # Vecteur d'explication pour le régime prédit (shape: [reasoning_units])
                                                # Ou tous les vecteurs (shape: [num_regimes, reasoning_units])
        attention_scores_vec: Optional[np.ndarray] = None, # Scores d'attention (shape: [num_heads, seq_len_q, seq_len_k])
        top_k: int = 3,
    ) -> str:
        """
        Décode l'explication pour la prédiction du régime de marché.
        Utilise market_regime_explanation_vec si fourni, sinon se base sur attention_scores_vec.
        """
        regime_name = self.market_regime_names[market_regime_pred_idx]
        reasons_parts = []

        # TODO: Implémenter une logique pour interpréter market_regime_explanation_vec.
        # Pour l'instant, si attention_scores_vec est fourni, on l'utilise pour l'importance des features.
        if attention_scores_vec is not None and self.feature_names:
            # Calculer l'importance des features (simplifié, suppose que attention_scores_vec peut être moyenné sur les features)
            # La forme réelle de attention_scores_vec dépend de comment il est généré et stocké.
            # Supposons qu'il puisse être moyenné pour obtenir une importance par feature d'entrée.
            # Ceci est une simplification majeure. Une vraie interprétation des scores d'attention
            # dépend de ce sur quoi l'attention a été appliquée (features brutes, étapes de raisonnement, etc.)
            if attention_scores_vec.ndim >= 2: # Exemple: (num_heads, num_features_attended_to)
                 # Moyenne sur les têtes, puis prendre les scores pour les features
                feature_importance_scores = np.mean(attention_scores_vec, axis=0)
                if len(feature_importance_scores) == len(self.feature_names):
                    feature_importance = (feature_importance_scores / np.sum(feature_importance_scores)) * 100
                    top_indices = np.argsort(feature_importance)[-top_k:]
                    for idx in top_indices:
                        reasons_parts.append(
                            self.explanation_templates["feature_importance"].format(
                                feature=self.feature_names[idx], importance=feature_importance[idx]
                            )
                        )
        
        if not reasons_parts:
            reasons_parts.append("analyse interne du modèle") # Fallback

        return self.explanation_templates["market_regime"].format(regime=regime_name, reasons=", ".join(reasons_parts))

    def decode_sl_tp_explanation(
        self,
        sl_value: float,
        tp_value: float,
        sl_explanation_vec: Optional[np.ndarray] = None,
        tp_explanation_vec: Optional[np.ndarray] = None,
        attention_scores_vec: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> Tuple[str, str]:
        """
        Décode les explications pour les prédictions de SL et TP.
        """
        sl_reasons_parts = []
        tp_reasons_parts = []

        # TODO: Logique similaire à decode_market_regime_explanation pour interpréter
        # sl_explanation_vec, tp_explanation_vec ou attention_scores_vec.
        if attention_scores_vec is not None and self.feature_names:
            # Logique d'importance des features (simplifiée et répétée pour l'exemple)
            if attention_scores_vec.ndim >=2:
                feature_importance_scores = np.mean(attention_scores_vec, axis=0)
                if len(feature_importance_scores) == len(self.feature_names):
                    feature_importance = (feature_importance_scores / np.sum(feature_importance_scores)) * 100
                    top_indices = np.argsort(feature_importance)[-top_k:]
                    for idx in top_indices:
                        reason_part = self.explanation_templates["feature_importance"].format(
                            feature=self.feature_names[idx], importance=feature_importance[idx]
                        )
                        sl_reasons_parts.append(reason_part)
                        tp_reasons_parts.append(reason_part)

        if not sl_reasons_parts: sl_reasons_parts.append("analyse interne du modèle")
        if not tp_reasons_parts: tp_reasons_parts.append("analyse interne du modèle")

        sl_explanation_text = self.explanation_templates["sl"].format(
            sl_value=f"{sl_value:.4f}", reasons=", ".join(sl_reasons_parts) # Précision augmentée
        )
        tp_explanation_text = self.explanation_templates["tp"].format(
            tp_value=f"{tp_value:.4f}", reasons=", ".join(tp_reasons_parts) # Précision augmentée
        )
        return sl_explanation_text, tp_explanation_text

    def decode_reasoning_steps(self, reasoning_steps_vecs: List[np.ndarray]) -> List[str]:
        """
        Décode les étapes de raisonnement intermédiaires.
        reasoning_steps_vecs: Liste des vecteurs de raisonnement (un par étape).
        """
        explanations = []
        if not reasoning_steps_vecs:
            return ["Aucune étape de raisonnement fournie."]
            
        for i, step_vec in enumerate(reasoning_steps_vecs):
            # TODO: Implémenter une logique plus fine pour interpréter step_vec.
            # Par exemple, trouver les activations les plus fortes, ou si les unités ont une sémantique.
            # Pour l'instant, on donne une description générique.
            # On pourrait aussi afficher la norme du vecteur, ou des statistiques simples.
            norm = np.linalg.norm(step_vec)
            explanations.append(
                self.explanation_templates["reasoning_step"].format(
                    step=i + 1, reasoning=f"Activation (norme {norm:.2f}). État interne du modèle après l'étape {i+1}."
                )
            )
        return explanations

    def generate_chain_of_thought_explanation(
        self,
        market_data: Dict[str, float], # Données brutes OHLCV pour l'instance
        predictions: Dict[str, np.ndarray], # Prédictions principales (market_regime, sl_tp, etc.) pour l'instance
        # Vecteurs d'explication dédiés du ReasoningModule
        final_reasoning_vec: Optional[np.ndarray] = None,
        market_regime_expl_vec: Optional[np.ndarray] = None, 
        sl_expl_vec: Optional[np.ndarray] = None,
        tp_expl_vec: Optional[np.ndarray] = None,
        reasoning_steps_vecs: List[np.ndarray] = None, 
        attention_scores_vec: Optional[np.ndarray] = None,
        top_k: int = 3, # top_k pour l'importance des features
    ) -> str:
        """
        Génère une explication complète Chain-of-Thought.
        """
        explanation_parts = []

        # 1. Contexte de marché (inchangé)
        if "close" in market_data and "open" in market_data and market_data["open"] != 0:
            price_change = ((market_data["close"] - market_data["open"]) / market_data["open"]) * 100
            context = f"Le prix est {'en hausse' if price_change > 0 else 'en baisse'} de {abs(price_change):.2f}% sur la période."
            implications = ("Tendance à court terme potentiellement maintenue." if abs(price_change) > 0.5 else "Mouvement de prix modéré.") # Seuil ajusté
            explanation_parts.append(self.explanation_templates["market_context"].format(context=context, implications=implications))

        # 2. Analyse technique (utilise attention_scores_vec si disponible)
        if attention_scores_vec is not None and self.feature_names:
            try:
                # attention_scores_vec a une forme comme (1, num_heads, seq_len_q, seq_len_k)
                # On enlève la dimension batch si elle est de 1
                squeezed_scores = np.squeeze(attention_scores_vec, axis=0) if attention_scores_vec.shape[0] == 1 else attention_scores_vec
                
                # Si la forme est (num_heads, seq_len_q, seq_len_k)
                # On moyenne sur les têtes (axis=0) et sur les queries (axis=0 après la première moyenne, donc axis=1 sur l'original)
                # pour obtenir un score par key/feature.
                if squeezed_scores.ndim == 3:
                    # Moyenne sur les têtes
                    scores_by_query_key = np.mean(squeezed_scores, axis=0) # Shape (seq_len_q, seq_len_k)
                    # Moyenne sur les queries pour obtenir un score par key (feature)
                    feature_importance_raw = np.mean(scores_by_query_key, axis=0) # Shape (seq_len_k,)
                elif squeezed_scores.ndim == 2: # Peut-être (num_heads, num_features) ou (query_len, key_len)
                    # Si c'est (num_heads, num_features), moyenner sur les têtes
                    if squeezed_scores.shape[1] == len(self.feature_names): # Supposons (num_heads, num_features)
                         feature_importance_raw = np.mean(squeezed_scores, axis=0)
                    # Si c'est (query_len, key_len) et key_len est num_features
                    elif squeezed_scores.shape[0] > 1 and squeezed_scores.shape[1] == len(self.feature_names):
                         feature_importance_raw = np.mean(squeezed_scores, axis=0)
                    else: # Forme inattendue
                        raise ValueError(f"Forme d'attention_scores_vec inattendue après squeeze: {squeezed_scores.shape}")
                elif squeezed_scores.ndim == 1 and len(squeezed_scores) == len(self.feature_names):
                    # Déjà un score par feature
                    feature_importance_raw = squeezed_scores
                else:
                    raise ValueError(f"Forme d'attention_scores_vec inattendue: {attention_scores_vec.shape}")

                if len(feature_importance_raw) == len(self.feature_names):
                    feature_importance_percent = (feature_importance_raw / (np.sum(feature_importance_raw) + 1e-9)) * 100
                    
                    # Trier les features par importance (décroissant)
                    sorted_indices = np.argsort(feature_importance_percent)[::-1]
                    
                    top_features_parts = []
                    for idx in sorted_indices[:top_k]: # Prendre les top_k features
                        feature_name = self.feature_names[idx]
                        importance = feature_importance_percent[idx]
                        if importance > 1: # Seuil pour afficher (éviter le bruit)
                            top_features_parts.append(
                                self.explanation_templates["feature_importance"].format(
                                    feature=feature_name, importance=importance
                                )
                            )
                    
                    if top_features_parts:
                        explanation_parts.append(f"Analyse technique (features influentes): {', '.join(top_features_parts)}.")
                    else:
                        explanation_parts.append("Analyse technique: Aucune feature avec une influence significative détectée par l'attention.")
                else:
                    explanation_parts.append(
                        f"Analyse technique: Incohérence entre le nombre de scores d'attention ({len(feature_importance_raw)}) et le nombre de features ({len(self.feature_names)})."
                    )
            except Exception as e:
                logger.error(f"Erreur lors du calcul de l'importance des features par attention: {e}", exc_info=True)
                explanation_parts.append(f"Analyse technique: Erreur lors du calcul de l'importance.")

        # 3. Signal de trading (inchangé)
        if "signal" in predictions: # Supposons que 'signal' soit une sortie du modèle principal
            signal_pred_idx = np.argmax(predictions["signal"].flatten()) # Aplatir si shape (1, N)
            signal_conf = np.max(predictions["signal"].flatten()) * 100
            signal_names = ["Vente Forte", "Vente", "Neutre", "Achat", "Achat Fort"] # Assurer 5 classes
            if signal_pred_idx < len(signal_names):
                signal_name = signal_names[signal_pred_idx]
                explanation_parts.append(self.explanation_templates["signal"].format(signal=signal_name, confidence=signal_conf))

        # 4. Régime de marché (utilisation de decode_market_regime_explanation)
        if "market_regime" in predictions:
            market_regime_pred_idx = np.argmax(predictions["market_regime"].flatten()) # Aplatir
            # market_regime_explanation_vec est la sortie du ReasoningModule pour le régime prédit.
            # La sortie 'market_regime_explanation' du modèle est (batch, num_regimes, reasoning_units)
            # On doit sélectionner le vecteur pour le régime prédit.
            current_market_regime_expl_vec = None
            if market_regime_expl_vec is not None and market_regime_expl_vec.shape[0] == len(self.market_regime_names):
                 current_market_regime_expl_vec = market_regime_expl_vec[market_regime_pred_idx]

            explanation_parts.append(self.decode_market_regime_explanation(
                market_regime_pred_idx,
                current_market_regime_expl_vec, # Passer le vecteur spécifique au régime prédit
                attention_scores_vec, # Ou une autre source d'importance si market_regime_expl_vec est utilisé différemment
                top_k=top_k
            ))
        
        # 5. Volatilité (inchangé)
        if "volatility_quantiles" in predictions: # Supposons que 'volatility_quantiles' soit une sortie
            vol_pred_idx = np.argmax(predictions["volatility_quantiles"].flatten())
            vol_names = ["Faible", "Moyenne", "Élevée"]
            if vol_pred_idx < len(vol_names):
                vol_name = vol_names[vol_pred_idx]
                vol_impact = "Conditions de marché stables." if vol_pred_idx == 0 else "Volatilité modérée attendue." if vol_pred_idx == 1 else "Forte volatilité attendue, prudence."
                explanation_parts.append(self.explanation_templates["volatility"].format(volatility=vol_name, impact=vol_impact))

        # 6. Stop Loss / Take Profit (utilisation de decode_sl_tp_explanation)
        if "sl_tp" in predictions:
            # sl_tp_pred_values est la sortie de la tête de prédiction SL/TP, shape (1,2) ou (2,)
            sl_tp_pred_values = predictions["sl_tp"].flatten() 
            sl_val, tp_val = sl_tp_pred_values[0], sl_tp_pred_values[1]
            
            sl_text, tp_text = self.decode_sl_tp_explanation(
                sl_val, tp_val,
                sl_expl_vec, # Vecteur d'explication SL du ReasoningModule
                tp_expl_vec, # Vecteur d'explication TP du ReasoningModule
                attention_scores_vec, # Ou une autre source d'importance
                top_k=top_k
            )
            explanation_parts.append(sl_text)
            explanation_parts.append(tp_text)

            if sl_val != 0:
                risk_reward = abs(tp_val / sl_val) if sl_val != 0 else float('inf')
                risk_level = "Faible (R/R > 3)" if risk_reward > 3 else "Moyen (R/R 1.5-3)" if risk_reward > 1.5 else "Élevé (R/R < 1.5)"
                explanation_parts.append(self.explanation_templates["risk_assessment"].format(risk_level=risk_level, factors=f"Ratio Risque/Récompense de {risk_reward:.2f}"))

        # 7. Étapes de raisonnement (utilisation de decode_reasoning_steps)
        if reasoning_steps_vecs:
            explanation_parts.append("\nÉtapes de Raisonnement Interne:")
            step_explanations = self.decode_reasoning_steps(reasoning_steps_vecs)
            explanation_parts.extend(step_explanations)
            
        # 8. Conclusion basée sur final_reasoning_vec (si disponible)
        if final_reasoning_vec is not None:
            # TODO: Implémenter une logique pour interpréter final_reasoning_vec.
            # Par exemple, similarité cosinus avec des "concepts de raisonnement" prototypiques,
            # ou simplement un résumé basé sur ses activations.
            explanation_parts.append(f"Conclusion du raisonnement interne (norme: {np.linalg.norm(final_reasoning_vec):.2f}).")

        full_explanation = "\n".join(part for part in explanation_parts if part) # Filtrer les None ou vides
        return full_explanation
