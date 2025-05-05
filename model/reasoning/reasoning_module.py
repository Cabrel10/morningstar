#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de raisonnement pour le modu00e8le Morningstar.
Ce module permet au modu00e8le d'expliquer ses du00e9cisions de trading en utilisant
des techniques de Chain-of-Thought (CoT) et d'attention interprétable.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Concatenate, MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D, Reshape, Embedding, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class ReasoningStep(Layer):
    """
    Couche de raisonnement qui implémente une étape de Chain-of-Thought.
    Cette couche prend des features en entrée et produit un raisonnement intermédiaire.
    """
    def __init__(self, 
                 units: int, 
                 dropout_rate: float = 0.2, 
                 l2_reg: float = 0.001, 
                 name: str = None):
        super(ReasoningStep, self).__init__(name=name)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Couches pour le raisonnement
        self.dense1 = Dense(
            units, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_dense1" if name else None
        )
        self.norm1 = LayerNormalization(name=f"{name}_norm1" if name else None)
        self.dropout1 = Dropout(dropout_rate, name=f"{name}_dropout1" if name else None)
        
        self.dense2 = Dense(
            units, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_dense2" if name else None
        )
        self.norm2 = LayerNormalization(name=f"{name}_norm2" if name else None)
        self.dropout2 = Dropout(dropout_rate, name=f"{name}_dropout2" if name else None)
        
        # Couche de sortie pour le raisonnement
        self.reasoning_output = Dense(
            units, activation='tanh',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_reasoning_output" if name else None
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
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config


class InterpretableAttention(Layer):
    """
    Couche d'attention interprétable qui permet de comprendre quelles features
    ont le plus d'influence sur les décisions du modèle.
    """
    def __init__(self, 
                 num_heads: int = 4, 
                 key_dim: int = 64, 
                 dropout_rate: float = 0.1, 
                 return_attention_scores: bool = True,
                 name: str = None):
        super(InterpretableAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.return_attention_scores = return_attention_scores
        
        # Utiliser les paramètres compatibles avec la version actuelle de TensorFlow
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name=f"{name}_mha" if name else None
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
            attention_output = self.mha(
                expanded_inputs, expanded_inputs, return_attention_scores=True
            )
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
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
            'return_attention_scores': self.return_attention_scores
        })
        return config


class ReasoningModule(Layer):
    """
    Module de raisonnement complet qui combine plusieurs étapes de raisonnement
    et un mécanisme d'attention interprétable pour expliquer les décisions de trading.
    """
    def __init__(self, 
                 num_reasoning_steps: int = 3,
                 reasoning_units: int = 128,
                 num_attention_heads: int = 4,
                 attention_key_dim: int = 64,
                 dropout_rate: float = 0.2,
                 l2_reg: float = 0.001,
                 num_market_regimes: int = 3,
                 name: str = None):
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
                name=f"{name}_reasoning_step_{i}" if name else None
            ) for i in range(num_reasoning_steps)
        ]
        
        # Attention interprétable
        self.attention = InterpretableAttention(
            num_heads=num_attention_heads,
            key_dim=attention_key_dim,
            dropout_rate=dropout_rate,
            return_attention_scores=True,
            name=f"{name}_attention" if name else None
        )
        
        # Couches pour générer les explications
        self.explanation_dense = Dense(
            reasoning_units, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_explanation_dense" if name else None
        )
        
        # Couches pour les différents types d'explications
        self.market_regime_explanation = Dense(
            num_market_regimes * reasoning_units, activation='linear',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_market_regime_explanation" if name else None
        )
        
        self.sl_explanation = Dense(
            reasoning_units, activation='linear',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_sl_explanation" if name else None
        )
        
        self.tp_explanation = Dense(
            reasoning_units, activation='linear',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_tp_explanation" if name else None
        )
        
        # Couche de fusion pour combiner les raisonnements
        self.fusion = Dense(
            reasoning_units, activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_fusion" if name else None
        )
        
        # Couche de sortie pour le raisonnement final
        self.reasoning_output = Dense(
            reasoning_units, activation='tanh',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name}_reasoning_output" if name else None
        )
    
    def call(self, inputs, training=False):
        # Appliquer les étapes de raisonnement séquentiellement
        reasoning_outputs = []
        x = inputs
        
        for step in self.reasoning_steps:
            x = step(x, training=training)
            reasoning_outputs.append(x)
        
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
            market_regime_explanation, 
            [-1, self.num_market_regimes, self.reasoning_units]
        )
        
        sl_explanation = self.sl_explanation(explanation_base)
        tp_explanation = self.tp_explanation(explanation_base)
        
        # Fusionner les raisonnements
        final_reasoning = self.fusion(attention_output)
        final_reasoning = self.reasoning_output(final_reasoning)
        
        return {
            'final_reasoning': final_reasoning,
            'market_regime_explanation': market_regime_explanation,
            'sl_explanation': sl_explanation,
            'tp_explanation': tp_explanation,
            'attention_scores': attention_scores,
            'reasoning_steps': reasoning_outputs
        }
    
    def get_config(self):
        config = super(ReasoningModule, self).get_config()
        config.update({
            'num_reasoning_steps': self.num_reasoning_steps,
            'reasoning_units': self.reasoning_units,
            'num_attention_heads': self.num_attention_heads,
            'attention_key_dim': self.attention_key_dim,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'num_market_regimes': self.num_market_regimes
        })
        return config


class ExplanationDecoder:
    """
    Décodeur d'explications qui convertit les représentations vectorielles
    en explications textuelles compréhensibles.
    """
    def __init__(self, 
                 feature_names: List[str],
                 market_regime_names: List[str] = None,
                 explanation_templates: Dict[str, str] = None):
        self.feature_names = feature_names
        self.market_regime_names = market_regime_names or ["Baissier", "Neutre", "Haussier", "Volatil"]
        
        # Templates par défaut pour les explications
        default_templates = {
            'market_regime': "Le régime de marché prédit est {regime}. Raisons principales: {reasons}",
            'feature_importance': "{feature} (importance: {importance:.1f}%)",
            'sl': "Stop Loss recommandé à {sl_value}. Basé sur: {reasons}",
            'tp': "Take Profit recommandé à {tp_value}. Basé sur: {reasons}",
            'reasoning_step': "Étape {step}: {reasoning}",
            'signal': "Signal de trading: {signal}. Confiance: {confidence:.1f}%",
            'volatility': "Volatilité prévue: {volatility}. Impact sur le trading: {impact}",
            'market_context': "Contexte de marché: {context}. Implications: {implications}",
            'technical_analysis': "Analyse technique: {analysis}. Indicateurs clés: {indicators}",
            'risk_assessment': "Évaluation du risque: {risk_level}. Facteurs: {factors}"
        }
        
        # Utiliser les templates fournis ou les templates par défaut
        self.explanation_templates = explanation_templates or default_templates

    def decode_market_regime_explanation(self, 
                                        market_regime_pred: int,
                                        market_regime_explanation: np.ndarray,
                                        attention_scores: np.ndarray,
                                        top_k: int = 3) -> str:
        """
        Décode l'explication pour la prédiction du régime de marché.
        
        Args:
            market_regime_pred: Indice du régime de marché prédit
            market_regime_explanation: Vecteur d'explication pour chaque régime
            attention_scores: Scores d'attention pour les features
            top_k: Nombre de features importantes à inclure
            
        Returns:
            Explication textuelle
        """
        # Obtenir l'explication pour le régime prédit
        regime_explanation = market_regime_explanation[market_regime_pred]
        
        # Calculer l'importance des features pour ce régime
        feature_importance = np.mean(attention_scores, axis=(0, 1))  # Moyenne sur les têtes d'attention
        feature_importance = feature_importance / np.sum(feature_importance) * 100  # En pourcentage
        
        # Trouver les features les plus importantes
        top_indices = np.argsort(feature_importance)[-top_k:]
        top_features = [self.feature_names[i] for i in top_indices if i < len(self.feature_names)]
        top_importances = [feature_importance[i] for i in top_indices if i < len(feature_importance)]
        
        # Générer les raisons
        reasons = []
        for feature, importance in zip(top_features, top_importances):
            reasons.append(
                self.explanation_templates['feature_importance'].format(
                    feature=feature, importance=importance
                )
            )
        
        # Générer l'explication complète
        regime_name = self.market_regime_names[market_regime_pred]
        explanation = self.explanation_templates['market_regime'].format(
            regime=regime_name,
            reasons=" ".join(reasons)
        )
        
        return explanation
    
    def decode_sl_tp_explanation(self, 
                               sl_value: float,
                               tp_value: float,
                               sl_explanation: np.ndarray,
                               tp_explanation: np.ndarray,
                               attention_scores: np.ndarray,
                               top_k: int = 3) -> Tuple[str, str]:
        """
        Décode les explications pour les prédictions de SL et TP.
        
        Args:
            sl_value: Valeur prédite pour le stop loss
            tp_value: Valeur prédite pour le take profit
            sl_explanation: Vecteur d'explication pour le SL
            tp_explanation: Vecteur d'explication pour le TP
            attention_scores: Scores d'attention pour les features
            top_k: Nombre de features importantes à inclure
            
        Returns:
            Tuple d'explications textuelles (SL, TP)
        """
        # Calculer l'importance des features
        feature_importance = np.mean(attention_scores, axis=(0, 1))  # Moyenne sur les têtes d'attention
        feature_importance = feature_importance / np.sum(feature_importance) * 100  # En pourcentage
        
        # Trouver les features les plus importantes
        top_indices = np.argsort(feature_importance)[-top_k:]
        top_features = [self.feature_names[i] for i in top_indices if i < len(self.feature_names)]
        top_importances = [feature_importance[i] for i in top_indices if i < len(feature_importance)]
        
        # Générer les raisons
        reasons = []
        for feature, importance in zip(top_features, top_importances):
            reasons.append(
                self.explanation_templates['feature_importance'].format(
                    feature=feature, importance=importance
                )
            )
        
        # Générer les explications complètes
        sl_explanation_text = self.explanation_templates['sl'].format(
            sl_value=f"{sl_value:.2f}",
            reasons=" ".join(reasons)
        )
        
        tp_explanation_text = self.explanation_templates['tp'].format(
            tp_value=f"{tp_value:.2f}",
            reasons=" ".join(reasons)
        )
        
        return sl_explanation_text, tp_explanation_text
    
    def decode_reasoning_steps(self, 
                              reasoning_steps: List[np.ndarray]) -> List[str]:
        """
        Décode les étapes de raisonnement intermédiaires.
        
        Args:
            reasoning_steps: Liste des vecteurs de raisonnement
            
        Returns:
            Liste d'explications textuelles pour chaque étape
        """
        # Pour l'instant, nous retournons simplement les indices des étapes
        # Dans une implémentation plus avancée, on pourrait utiliser un modèle de langage
        # pour générer des explications textuelles à partir des vecteurs
        explanations = []
        for i, step in enumerate(reasoning_steps):
            explanations.append(
                self.explanation_templates['reasoning_step'].format(
                    step=i+1, reasoning=f"Analyse des patterns de marché (étape {i+1})"
                )
            )
        
        return explanations
        
    def generate_chain_of_thought_explanation(self,
                                             market_data: Dict[str, float],
                                             predictions: Dict[str, np.ndarray],
                                             reasoning_steps: List[np.ndarray] = None,
                                             attention_scores: np.ndarray = None,
                                             top_k: int = 5) -> str:
        """
        Génère une explication complète Chain-of-Thought pour les prédictions du modèle.
        
        Args:
            market_data: Dictionnaire des données de marché (prix, volumes, etc.)
            predictions: Dictionnaire des prédictions du modèle
            reasoning_steps: Liste des vecteurs de raisonnement intermédiaires
            attention_scores: Scores d'attention pour les features
            top_k: Nombre de features importantes à inclure
            
        Returns:
            Explication textuelle complète avec raisonnement étape par étape
        """
        # Initialiser l'explication
        explanation_parts = []
        
        # 1. Contexte de marché
        if 'close' in market_data and 'open' in market_data:
            price_change = ((market_data['close'] - market_data['open']) / market_data['open']) * 100
            context = f"Le prix est {'en hausse' if price_change > 0 else 'en baisse'} de {abs(price_change):.2f}%"
            implications = "Tendance à court terme potentiellement maintenue" if abs(price_change) > 1 else "Mouvement de prix modéré"
            
            explanation_parts.append(self.explanation_templates['market_context'].format(
                context=context, implications=implications
            ))
        
        # 2. Analyse technique
        if attention_scores is not None:
            # Calculer l'importance des features
            feature_importance = np.mean(attention_scores, axis=(0, 1))
            feature_importance = feature_importance / np.sum(feature_importance) * 100
            
            # Trouver les features techniques les plus importantes
            tech_indices = [i for i, name in enumerate(self.feature_names) 
                          if any(indicator in name.lower() for indicator in 
                                ['rsi', 'macd', 'ema', 'sma', 'atr', 'bbands', 'volume'])]
            
            if tech_indices:
                tech_importance = feature_importance[tech_indices]
                top_tech_indices = np.argsort(tech_importance)[-min(3, len(tech_importance)):]
                top_tech_features = [self.feature_names[tech_indices[i]] for i in top_tech_indices]
                
                tech_analysis = "Signaux mixtes" if len(top_tech_features) < 2 else "Signaux convergents"
                explanation_parts.append(self.explanation_templates['technical_analysis'].format(
                    analysis=tech_analysis, indicators=", ".join(top_tech_features)
                ))
        
        # 3. Signal de trading
        if 'signal' in predictions:
            signal_pred = np.argmax(predictions['signal'])
            signal_conf = np.max(predictions['signal']) * 100
            signal_names = ["Vente forte", "Vente", "Neutre", "Achat", "Achat fort"]
            
            if signal_pred < len(signal_names):
                signal_name = signal_names[signal_pred]
                explanation_parts.append(self.explanation_templates['signal'].format(
                    signal=signal_name, confidence=signal_conf
                ))
        
        # 4. Régime de marché
        if 'market_regime' in predictions:
            regime_pred = np.argmax(predictions['market_regime'])
            if regime_pred < len(self.market_regime_names):
                regime_name = self.market_regime_names[regime_pred]
                explanation_parts.append(f"Régime de marché: {regime_name}")
        
        # 5. Volatilité
        if 'volatility_quantiles' in predictions:
            vol_pred = np.argmax(predictions['volatility_quantiles'])
            vol_names = ["Faible", "Moyenne", "Élevée"]
            
            if vol_pred < len(vol_names):
                vol_name = vol_names[vol_pred]
                vol_impact = "Opportunité pour des positions à faible risque" if vol_pred == 0 else \
                           "Ajustement modéré des stop loss" if vol_pred == 1 else \
                           "Réduction de la taille des positions recommandée"
                
                explanation_parts.append(self.explanation_templates['volatility'].format(
                    volatility=vol_name, impact=vol_impact
                ))
        
        # 6. Stop Loss / Take Profit
        if 'sl_tp' in predictions and len(predictions['sl_tp']) > 1:
            sl_value = predictions['sl_tp'][0]
            tp_value = predictions['sl_tp'][1]
            
            explanation_parts.append(f"Stop Loss recommandé: {sl_value:.2f}%")
            explanation_parts.append(f"Take Profit recommandé: {tp_value:.2f}%")
            
            # Calculer le ratio risque/récompense
            if sl_value != 0:
                risk_reward = abs(tp_value / sl_value)
                risk_level = "Faible" if risk_reward > 3 else "Moyen" if risk_reward > 1.5 else "Élevé"
                
                explanation_parts.append(self.explanation_templates['risk_assessment'].format(
                    risk_level=risk_level, 
                    factors=f"Ratio risque/récompense de {risk_reward:.2f}"
                ))
        
        # 7. Étapes de raisonnement (si disponibles)
        if reasoning_steps:
            explanation_parts.append("\nRaisonnement étape par étape:")
            step_explanations = self.decode_reasoning_steps(reasoning_steps)
            explanation_parts.extend(step_explanations)
        
        # Joindre toutes les parties de l'explication
        full_explanation = "\n".join(explanation_parts)
        
        return full_explanation
