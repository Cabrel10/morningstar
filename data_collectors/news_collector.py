#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collecteur d'actualitu00e9s crypto pour le modu00e8le Morningstar.
Ce module collecte des actualitu00e9s sur les crypto-monnaies u00e0 partir de diverses sources.
"""

import logging
import requests
import json
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class CryptoNewsCollector:
    """
    Classe pour collecter des actualitu00e9s sur les crypto-monnaies u00e0 partir de diverses sources.
    """
    def __init__(self, max_retries=3, retry_delay=2):
        """
        Initialise le collecteur d'actualitu00e9s crypto.
        
        Args:
            max_retries: Nombre maximum de tentatives en cas d'u00e9chec
            retry_delay: Du00e9lai entre les tentatives en secondes
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Sources d'actualitu00e9s crypto
        self.news_sources = [
            {
                'name': 'CoinDesk',
                'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'type': 'rss'
            },
            {
                'name': 'CoinTelegraph',
                'url': 'https://cointelegraph.com/rss',
                'type': 'rss'
            },
            {
                'name': 'CryptoNews',
                'url': 'https://cryptonews.com/news/feed/',
                'type': 'rss'
            },
            {
                'name': 'Bitcoin Magazine',
                'url': 'https://bitcoinmagazine.com/feed',
                'type': 'rss'
            },
            {
                'name': 'Decrypt',
                'url': 'https://decrypt.co/feed',
                'type': 'rss'
            }
        ]
    
    def fetch_crypto_news(self, date_str: str, crypto_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Ru00e9cupu00e8re les actualitu00e9s crypto pour une date et une crypto-monnaie donnu00e9es.
        
        Args:
            date_str: Date au format 'YYYY-MM-DD'
            crypto_name: Nom de la crypto-monnaie (optionnel)
            
        Returns:
            Liste d'articles d'actualitu00e9s (chaque article est un dictionnaire avec 'title', 'content', 'source', 'date')
        """
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        all_news = []
        
        for source in self.news_sources:
            try:
                logger.info(f"Ru00e9cupu00e9ration des actualitu00e9s depuis {source['name']}")
                
                if source['type'] == 'rss':
                    news = self._fetch_rss_news(source['url'], target_date, crypto_name)
                else:
                    news = self._fetch_api_news(source['url'], target_date, crypto_name)
                
                all_news.extend(news)
                logger.info(f"Ru00e9cupu00e9ru00e9 {len(news)} articles depuis {source['name']}")
                
                # Respecter les limites de taux
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Erreur lors de la ru00e9cupu00e9ration des actualitu00e9s depuis {source['name']}: {e}")
        
        # Filtrer par date
        filtered_news = []
        for article in all_news:
            try:
                article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                if article_date.date() == target_date.date():
                    filtered_news.append(article)
            except Exception as e:
                logger.error(f"Erreur lors du filtrage par date: {e}")
        
        logger.info(f"Total d'articles pour {date_str}: {len(filtered_news)}")
        return filtered_news
    
    def _fetch_rss_news(self, rss_url: str, target_date: datetime, crypto_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Ru00e9cupu00e8re les actualitu00e9s u00e0 partir d'un flux RSS.
        
        Args:
            rss_url: URL du flux RSS
            target_date: Date cible
            crypto_name: Nom de la crypto-monnaie (optionnel)
            
        Returns:
            Liste d'articles d'actualitu00e9s
        """
        news = []
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(rss_url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items:
                    try:
                        title = item.find('title').text
                        link = item.find('link').text
                        description = item.find('description').text
                        pub_date = item.find('pubDate').text
                        
                        # Convertir la date au format standard
                        try:
                            article_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                        except ValueError:
                            try:
                                article_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                            except ValueError:
                                article_date = datetime.now()  # Fallback
                        
                        date_str = article_date.strftime('%Y-%m-%d')
                        
                        # Filtrer par crypto-monnaie si spu00e9cifiu00e9e
                        if crypto_name and crypto_name.lower() not in title.lower() and crypto_name.lower() not in description.lower():
                            continue
                        
                        # Ru00e9cupu00e9rer le contenu complet de l'article
                        content = self._fetch_article_content(link)
                        
                        news.append({
                            'title': title,
                            'content': content or description,
                            'source': rss_url,
                            'date': date_str,
                            'url': link
                        })
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement d'un article RSS: {e}")
                
                break  # Sortir de la boucle si ru00e9ussi
                
            except Exception as e:
                logger.error(f"Erreur lors de la ru00e9cupu00e9ration du flux RSS (tentative {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        return news
    
    def _fetch_api_news(self, api_url: str, target_date: datetime, crypto_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Ru00e9cupu00e8re les actualitu00e9s u00e0 partir d'une API.
        
        Args:
            api_url: URL de l'API
            target_date: Date cible
            crypto_name: Nom de la crypto-monnaie (optionnel)
            
        Returns:
            Liste d'articles d'actualitu00e9s
        """
        news = []
        
        # Construire les paramu00e8tres de la requ00eate
        params = {
            'date': target_date.strftime('%Y-%m-%d')
        }
        
        if crypto_name:
            params['q'] = crypto_name
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(api_url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Traiter les donnu00e9es selon le format de l'API
                if 'articles' in data:
                    articles = data['articles']
                elif 'data' in data:
                    articles = data['data']
                else:
                    articles = data
                
                for article in articles:
                    try:
                        title = article.get('title', '')
                        content = article.get('content', article.get('description', ''))
                        source = article.get('source', {}).get('name', api_url)
                        pub_date = article.get('publishedAt', article.get('date', target_date.strftime('%Y-%m-%d')))
                        url = article.get('url', '')
                        
                        # Convertir la date au format standard si nu00e9cessaire
                        if isinstance(pub_date, str) and 'T' in pub_date:
                            try:
                                article_date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ')
                                date_str = article_date.strftime('%Y-%m-%d')
                            except ValueError:
                                date_str = target_date.strftime('%Y-%m-%d')
                        else:
                            date_str = pub_date if isinstance(pub_date, str) else target_date.strftime('%Y-%m-%d')
                        
                        news.append({
                            'title': title,
                            'content': content,
                            'source': source,
                            'date': date_str,
                            'url': url
                        })
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement d'un article API: {e}")
                
                break  # Sortir de la boucle si ru00e9ussi
                
            except Exception as e:
                logger.error(f"Erreur lors de la ru00e9cupu00e9ration des actualitu00e9s API (tentative {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        return news
    
    def _fetch_article_content(self, url: str) -> Optional[str]:
        """
        Ru00e9cupu00e8re le contenu complet d'un article u00e0 partir de son URL.
        
        Args:
            url: URL de l'article
            
        Returns:
            Contenu de l'article ou None en cas d'u00e9chec
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Essayer diffu00e9rentes strateu00e9gies pour extraire le contenu
                # 1. Chercher les balises d'article
                article = soup.find('article')
                if article:
                    paragraphs = article.find_all('p')
                    content = ' '.join([p.text for p in paragraphs])
                    return content
                
                # 2. Chercher les balises de contenu principales
                content_div = soup.find('div', class_=['content', 'article-content', 'post-content', 'entry-content'])
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = ' '.join([p.text for p in paragraphs])
                    return content
                
                # 3. Fallback: prendre tous les paragraphes
                paragraphs = soup.find_all('p')
                content = ' '.join([p.text for p in paragraphs])
                return content
                
            except Exception as e:
                logger.error(f"Erreur lors de la ru00e9cupu00e9ration du contenu de l'article (tentative {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        return None
    
    def search_crypto_news(self, query: str, days_back: int = 7) -> List[Dict[str, str]]:
        """
        Recherche des actualitu00e9s crypto correspondant u00e0 une requ00eate.
        
        Args:
            query: Terme de recherche
            days_back: Nombre de jours en arriu00e8re pour la recherche
            
        Returns:
            Liste d'articles d'actualitu00e9s correspondant u00e0 la requ00eate
        """
        all_news = []
        today = datetime.now()
        
        for i in range(days_back):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            news = self.fetch_crypto_news(date_str, query)
            all_news.extend(news)
        
        logger.info(f"Recherche d'actualitu00e9s pour '{query}': {len(all_news)} ru00e9sultats")
        return all_news
