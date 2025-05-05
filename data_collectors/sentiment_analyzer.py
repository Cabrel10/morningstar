#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyseur de sentiment pour le module Morningstar.
Ce module utilise l'API Gemini 2.5 Pro pour analyser le sentiment des actualités crypto.
"""

import logging
import json
import time
import random
from typing import List, Dict, Any, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiSentimentAnalyzer:
    """
    Classe pour analyser le sentiment des actualités crypto en utilisant l'API Gemini 2.5 Pro.
    """
    def __init__(self, api_keys=None, model_name="gemini-2.5-pro-preview-03-25", max_retries=3, retry_delay=2):
        """
        Initialise l'analyseur de sentiment Gemini.
        
        Args:
            api_keys: Liste des clés API Gemini
            model_name: Nom du modèle Gemini à utiliser
            max_retries: Nombre maximum de tentatives en cas d'échec
            retry_delay: Délai entre les tentatives en secondes
        """
        # Si aucune clé n'est fournie, essayer de récupérer les clés depuis les variables d'environnement
        if not api_keys:
            import os
            api_keys = []
            # Chercher toutes les clés GOOGLE_API_KEY_* et GEMINI_API_KEY dans l'environnement
            for key, value in os.environ.items():
                if key.startswith('GOOGLE_API_KEY_') or key == 'GEMINI_API_KEY':
                    if value and value not in api_keys:
                        api_keys.append(value)
        
        self.api_keys = api_keys or []
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.clients = []
        self.key_usage = {key: 0 for key in self.api_keys}  # Compteur d'utilisation pour chaque clé
        self.key_errors = {key: 0 for key in self.api_keys}  # Compteur d'erreurs pour chaque clé
        
        if not self.api_keys:
            logger.warning("Aucune clé API Gemini fournie. L'analyse de sentiment ne fonctionnera pas.")
        else:
            logger.info(f"Initialisation de {len(self.api_keys)} clients Gemini")
            self._initialize_clients()
    
    def _initialize_clients(self):
        """
        Initialise les clients Gemini pour toutes les clés API disponibles.
        """
        self.clients = []
        for api_key in self.api_keys:
            try:
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel(model_name=self.model_name)
                # Tester le client avec une requête simple pour vérifier qu'il fonctionne
                test_response = client.generate_content("Hello")
                if test_response:
                    self.clients.append((api_key, client))
                    logger.info(f"Client Gemini initialisé avec succès pour la clé se terminant par ...{api_key[-4:]}")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client Gemini pour la clé se terminant par ...{api_key[-4:]}: {e}")
                self.key_errors[api_key] += 1
    
    def get_client(self):
        """
        Récupère un client Gemini en utilisant une stratégie de rotation des clés.
        Privilégie les clés les moins utilisées et évite celles qui ont généré des erreurs.
        
        Returns:
            Tuple (clé API, client Gemini) ou (None, None) si aucun client n'est disponible
        """
        if not self.clients:
            logger.error("Aucun client Gemini disponible")
            return None, None
        
        # Trier les clients par nombre d'utilisations (privilégier les moins utilisés)
        sorted_clients = sorted(self.clients, key=lambda x: (self.key_errors[x[0]], self.key_usage[x[0]]))
        
        # Sélectionner le client avec le moins d'utilisations et d'erreurs
        api_key, client = sorted_clients[0]
        self.key_usage[api_key] += 1
        
        return api_key, client
    
    def analyze_sentiment(self, news_articles: List[Dict[str, str]], crypto_name: str) -> Dict[str, Any]:
        """
        Analyse le sentiment d'une liste d'articles d'actualités pour une crypto-monnaie donnée.
        
        Args:
            news_articles: Liste d'articles d'actualités (chaque article est un dictionnaire avec 'title' et 'content')
            crypto_name: Nom de la crypto-monnaie (ex: 'Bitcoin', 'BTC')
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse de sentiment
        """
        api_key, client = self.get_client()
        if not client or not news_articles:
            return {
                "sentiment_score": 0,
                "sentiment_magnitude": 0,
                "bullish_probability": 0.5,
                "bearish_probability": 0.5,
                "summary": "",
                "key_events": []
            }
        
        # Préparer les articles pour l'analyse
        articles_text = ""
        for i, article in enumerate(news_articles[:10]):  # Limiter à 10 articles pour éviter de dépasser les limites
            title = article.get('title', f"Article {i+1}")
            content = article.get('content', "")
            articles_text += f"Article {i+1}: {title}\n{content[:500]}...\n\n"  # Limiter la taille du contenu
        
        # Créer le prompt pour l'analyse de sentiment
        prompt = f"""
        Analyse le sentiment des actualités suivantes concernant {crypto_name}. 

        {articles_text}

        Fournis une analyse structurée au format JSON avec les éléments suivants:
        1. sentiment_score: un score de -1.0 (très négatif) à 1.0 (très positif)
        2. sentiment_magnitude: l'intensité du sentiment de 0.0 à 1.0
        3. bullish_probability: probabilité que le marché soit haussier (0.0 à 1.0)
        4. bearish_probability: probabilité que le marché soit baissier (0.0 à 1.0)
        5. summary: un résumé concis des actualités et de leur impact potentiel sur le prix
        6. key_events: liste des événements clés qui pourraient influencer le prix

        Réponds uniquement avec le JSON, sans texte additionnel.
        """
        
        # Analyser le sentiment avec Gemini
        for attempt in range(self.max_retries):
            try:
                response = client.generate_content(prompt)
                response_text = response.text
                
                # Extraire le JSON de la réponse
                try:
                    # Nettoyer la réponse si nécessaire
                    if response_text.startswith("```json"):
                        response_text = response_text.replace("```json", "").replace("```", "")
                    
                    result = json.loads(response_text)
                    logger.info(f"Analyse de sentiment réussie pour {crypto_name} avec la clé se terminant par ...{api_key[-4:]}")
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de décodage JSON: {e}. Réponse: {response_text}")
                    self.key_errors[api_key] += 1
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse de sentiment (tentative {attempt+1}/{self.max_retries}) avec la clé se terminant par ...{api_key[-4:]}: {e}")
                self.key_errors[api_key] += 1
                
                # Si la clé semble épuisée ou invalide, essayer une autre clé
                if "quota" in str(e).lower() or "limit" in str(e).lower() or "invalid" in str(e).lower():
                    logger.warning(f"La clé API se terminant par ...{api_key[-4:]} semble épuisée ou invalide. Essai d'une autre clé.")
                    api_key, client = self.get_client()
                    if not client:
                        break
                else:
                    time.sleep(self.retry_delay)
        
        # Valeurs par défaut en cas d'échec
        return {
            "sentiment_score": 0,
            "sentiment_magnitude": 0,
            "bullish_probability": 0.5,
            "bearish_probability": 0.5,
            "summary": f"Analyse de sentiment non disponible pour {crypto_name}",
            "key_events": []
        }
    
    def analyze_market_context(self, price_data: Dict[str, Any], news_summary: str, crypto_name: str) -> Dict[str, Any]:
        """
        Analyse le contexte de marché en combinant les données de prix et les actualités.
        
        Args:
            price_data: Dictionnaire contenant les données de prix récentes
            news_summary: Résumé des actualités récentes
            crypto_name: Nom de la crypto-monnaie
            
        Returns:
            Dictionnaire contenant l'analyse du contexte de marché
        """
        api_key, client = self.get_client()
        if not client:
            return {
                "market_regime": "unknown",
                "trend_strength": 0,
                "support_levels": [],
                "resistance_levels": [],
                "key_factors": []
            }
        
        # Préparer les données de prix pour l'analyse
        price_summary = f"Prix actuel: {price_data.get('current_price', 'N/A')}\n"
        price_summary += f"Variation sur 24h: {price_data.get('price_change_24h', 'N/A')}%\n"
        price_summary += f"Volume sur 24h: {price_data.get('volume_24h', 'N/A')}\n"
        price_summary += f"RSI: {price_data.get('rsi', 'N/A')}\n"
        price_summary += f"MACD: {price_data.get('macd', 'N/A')}\n"
        
        # Créer le prompt pour l'analyse du contexte de marché
        prompt = f"""
        Analyse le contexte de marché actuel pour {crypto_name} en te basant sur les données de prix et les actualités suivantes.

        Données de prix:
        {price_summary}

        Actualités récentes:
        {news_summary}

        Fournis une analyse structurée au format JSON avec les éléments suivants:
        1. market_regime: le régime de marché actuel ("sideways", "bullish", "bearish", "volatile")
        2. trend_strength: force de la tendance de 0.0 à 1.0
        3. support_levels: liste des niveaux de support importants
        4. resistance_levels: liste des niveaux de résistance importants
        5. key_factors: liste des facteurs clés influençant le marché actuellement

        Réponds uniquement avec le JSON, sans texte additionnel.
        """
        
        # Analyser le contexte de marché avec Gemini
        for attempt in range(self.max_retries):
            try:
                response = client.generate_content(prompt)
                response_text = response.text
                
                # Extraire le JSON de la réponse
                try:
                    # Nettoyer la réponse si nécessaire
                    if response_text.startswith("```json"):
                        response_text = response_text.replace("```json", "").replace("```", "")
                    
                    result = json.loads(response_text)
                    logger.info(f"Analyse du contexte de marché réussie pour {crypto_name}")
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de décodage JSON: {e}. Réponse: {response_text}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du contexte de marché (tentative {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        # Valeurs par défaut en cas d'échec
        return {
            "market_regime": "unknown",
            "trend_strength": 0,
            "support_levels": [],
            "resistance_levels": [],
            "key_factors": []
        }
    
    def generate_trading_insights(self, market_data: Dict[str, Any], sentiment_data: Dict[str, Any], crypto_name: str) -> Dict[str, Any]:
        """
        Génère des insights de trading en combinant les données de marché et de sentiment.
        
        Args:
            market_data: Dictionnaire contenant les données de marché
            sentiment_data: Dictionnaire contenant les données de sentiment
            crypto_name: Nom de la crypto-monnaie
            
        Returns:
            Dictionnaire contenant les insights de trading
        """
        api_key, client = self.get_client()
        if not client:
            return {
                "trade_recommendation": "neutral",
                "confidence": 0,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "reasoning": ""
            }
        
        # Préparer les données pour l'analyse
        market_summary = f"Prix actuel: {market_data.get('current_price', 'N/A')}\n"
        market_summary += f"Régime de marché: {market_data.get('market_regime', 'unknown')}\n"
        market_summary += f"Force de la tendance: {market_data.get('trend_strength', 0)}\n"
        market_summary += f"Niveaux de support: {', '.join(map(str, market_data.get('support_levels', [])))}\n"
        market_summary += f"Niveaux de résistance: {', '.join(map(str, market_data.get('resistance_levels', [])))}\n"
        
        sentiment_summary = f"Score de sentiment: {sentiment_data.get('sentiment_score', 0)}\n"
        sentiment_summary += f"Probabilité haussière: {sentiment_data.get('bullish_probability', 0.5)}\n"
        sentiment_summary += f"Probabilité baissière: {sentiment_data.get('bearish_probability', 0.5)}\n"
        sentiment_summary += f"Résumé des actualités: {sentiment_data.get('summary', '')}\n"
        
        # Créer le prompt pour les insights de trading
        prompt = f"""
        En tant qu'expert en trading de crypto-monnaies, génère des recommandations de trading pour {crypto_name} en te basant sur les données de marché et de sentiment suivantes.

        Données de marché:
        {market_summary}

        Données de sentiment:
        {sentiment_summary}

        Fournis une analyse structurée au format JSON avec les éléments suivants:
        1. trade_recommendation: recommandation de trading ("buy", "sell", "neutral")
        2. confidence: niveau de confiance de 0.0 à 1.0
        3. entry_price: prix d'entrée recommandé (ou null si pas de recommandation)
        4. stop_loss: niveau de stop loss recommandé (ou null si pas de recommandation)
        5. take_profit: niveau de take profit recommandé (ou null si pas de recommandation)
        6. reasoning: explication détaillée du raisonnement derrière cette recommandation

        Réponds uniquement avec le JSON, sans texte additionnel.
        """
        
        # Générer les insights de trading avec Gemini
        for attempt in range(self.max_retries):
            try:
                response = client.generate_content(prompt)
                response_text = response.text
                
                # Extraire le JSON de la réponse
                try:
                    # Nettoyer la réponse si nécessaire
                    if response_text.startswith("```json"):
                        response_text = response_text.replace("```json", "").replace("```", "")
                    
                    result = json.loads(response_text)
                    logger.info(f"Insights de trading générés avec succès pour {crypto_name}")
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de décodage JSON: {e}. Réponse: {response_text}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la génération des insights de trading (tentative {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        # Valeurs par défaut en cas d'échec
        return {
            "trade_recommendation": "neutral",
            "confidence": 0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "reasoning": f"Insights de trading non disponibles pour {crypto_name}"
        }
