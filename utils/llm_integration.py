import json
import logging
import os
import requests
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Configuration des chemins de cache
CACHE_BASE_DIR = Path("data/llm_cache")
NEWS_CACHE_DIR = CACHE_BASE_DIR / "news"
INSTRUMENTS_CACHE_DIR = CACHE_BASE_DIR / "instruments"

logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    Gère l'intégration avec les modèles LLM (Gemini, CryptoBERT) et le chargement/sauvegarde des embeddings.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialise LLMIntegration en vérifiant l'existence des répertoires de cache et des clés API."""
        self.news_cache_dir = NEWS_CACHE_DIR
        self.instruments_cache_dir = INSTRUMENTS_CACHE_DIR
        
        # Créer les répertoires de cache s'ils n'existent pas
        self.news_cache_dir.mkdir(parents=True, exist_ok=True)
        self.instruments_cache_dir.mkdir(parents=True, exist_ok=True)

        # Clé API Gemini (priorité à l'argument, puis variable d'environnement, puis config)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            try:
                from config.secrets import GEMINI_API_KEY
                self.api_key = GEMINI_API_KEY
            except (ImportError, AttributeError):
                logger.warning("Aucune clé API Gemini trouvée. Les appels à l'API ne fonctionneront pas.")
                self.api_key = None
        
        # Initialiser CryptoBERT si disponible
        self.cryptobert_available = False
        try:
            from transformers import AutoModel, AutoTokenizer
            # Vérifier si les modèles sont disponibles localement
            self.cryptobert_available = True
            logger.info("CryptoBERT disponible pour les embeddings locaux")
        except ImportError:
            logger.warning("Transformers non installé. CryptoBERT ne sera pas disponible.")

    def get_cached_embedding(self, symbol: str, date_str: str, embedding_type: str = "news") -> Optional[np.ndarray]:
        """
        Charge un embedding pré-calculé depuis le cache fichier local.

        Args:
            symbol: Le symbole de l'actif (ex: 'BTC').
            date_str: La date au format 'YYYY-MM-DD' (utilisé seulement si embedding_type='news').
            embedding_type: Le type d'embedding à charger ('news' ou 'instrument').

        Returns:
            L'embedding sous forme d'array numpy, ou None si non trouvé ou invalide.
        """
        cache_file: Optional[Path] = None

        if embedding_type == "news":
            if not self.news_cache_dir.exists():
                logger.error(
                    f"Tentative de lecture du cache news, mais le répertoire n'existe pas: {self.news_cache_dir}"
                )
                return None
            cache_file = self.news_cache_dir / f"{symbol}-{date_str}.json"
            lookup_key = f"{symbol}-{date_str}"
        elif embedding_type == "instrument":
            if not self.instruments_cache_dir.exists():
                logger.error(
                    f"Tentative de lecture du cache instruments, mais le répertoire n'existe pas: {self.instruments_cache_dir}"
                )
                return None
            # Pour les instruments, la date n'est pas pertinente, on utilise juste le symbole
            cache_file = self.instruments_cache_dir / f"{symbol}_description.json"
            lookup_key = f"{symbol}_description"
        else:
            logger.error(f"Type d'embedding non supporté demandé: '{embedding_type}'")
            return None

        if cache_file and cache_file.exists():
            logger.debug(f"Tentative de chargement depuis le cache: {cache_file}")
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    embedding_list = data.get("embedding")

                    if embedding_list and isinstance(embedding_list, list):
                        # Vérifier que la liste contient bien des nombres
                        if all(isinstance(x, (int, float)) for x in embedding_list):
                            logger.info(f"Embedding chargé avec succès depuis {cache_file}")
                            return np.array(embedding_list, dtype=np.float32)
                        else:
                            logger.warning(
                                f"Le contenu 'embedding' dans {cache_file} n'est pas une liste de nombres valides."
                            )
                            return None
                    elif embedding_list is None and "embedding" in data:
                        logger.warning(
                            f"Embedding est 'null' dans le fichier cache {cache_file} (peut-être généré hors ligne ou sans texte)."
                        )
                        return None  # Retourne None si l'embedding est explicitement null
                    else:
                        logger.warning(f"Clé 'embedding' non trouvée ou format invalide dans {cache_file}")
                        return None
            except json.JSONDecodeError:
                logger.error(f"Erreur de décodage JSON pour le fichier cache: {cache_file}")
                return None
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la lecture du fichier cache {cache_file}: {e}")
                return None
        else:
            logger.warning(f"Fichier cache non trouvé pour {embedding_type} '{lookup_key}': {cache_file}")
            return None
    
    def save_embedding_to_cache(self, symbol: str, date_str: str, embedding: np.ndarray, 
                               summary: str = "", embedding_type: str = "news") -> bool:
        """
        Sauvegarde un embedding dans le cache local.
        
        Args:
            symbol: Le symbole de l'actif (ex: 'BTC').
            date_str: La date au format 'YYYY-MM-DD'.
            embedding: L'embedding à sauvegarder (numpy array).
            summary: Résumé ou contexte associé à l'embedding.
            embedding_type: Type d'embedding ('news' ou 'instrument').
            
        Returns:
            bool: True si la sauvegarde a réussi, False sinon.
        """
        try:
            # Déterminer le répertoire et le nom de fichier
            if embedding_type == "news":
                cache_dir = self.news_cache_dir
                cache_file = cache_dir / f"{symbol}-{date_str}.json"
            elif embedding_type == "instrument":
                cache_dir = self.instruments_cache_dir
                cache_file = cache_dir / f"{symbol}_description.json"
            else:
                logger.error(f"Type d'embedding non supporté: {embedding_type}")
                return False
            
            # Créer le répertoire si nécessaire
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Convertir l'embedding numpy en liste pour JSON
            embedding_list = embedding.tolist() if embedding is not None else None
            
            # Préparer les données à sauvegarder
            data = {
                "symbol": symbol,
                "date": date_str if embedding_type == "news" else None,
                "summary": summary,
                "embedding": embedding_list,
                "embedding_type": embedding_type,
                "timestamp": time.time()
            }
            
            # Sauvegarder au format JSON
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Embedding sauvegardé avec succès dans {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'embedding: {e}")
            return False
    
    def get_gemini_embedding(self, text: str, model: str = "models/embedding-001") -> Optional[np.ndarray]:
        """
        Génère un embedding via l'API Gemini.
        
        Args:
            text: Texte à encoder
            model: Modèle d'embedding Gemini à utiliser
            
        Returns:
            np.ndarray: Embedding généré ou None en cas d'échec
        """
        if not self.api_key:
            logger.error("Aucune clé API Gemini disponible. Impossible de générer l'embedding.")
            return None
            
        try:
            # Endpoint pour les embeddings Gemini
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={self.api_key}"
            
            # Préparer la requête
            payload = {
                "content": {"parts": [{"text": text}]}
            }
            
            # Envoyer la requête
            response = requests.post(url, json=payload)
            
            # Vérifier la réponse
            if response.status_code == 200:
                data = response.json()
                if "embedding" in data and "values" in data["embedding"]:
                    embedding_values = data["embedding"]["values"]
                    return np.array(embedding_values, dtype=np.float32)
                else:
                    logger.error(f"Format de réponse Gemini inattendu: {data}")
            else:
                logger.error(f"Erreur API Gemini ({response.status_code}): {response.text}")
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API Gemini: {e}")
            return None
    
    def get_cryptobert_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Génère un embedding en utilisant CryptoBERT localement.
        
        Args:
            text: Texte à encoder
            
        Returns:
            np.ndarray: Embedding généré ou None en cas d'échec
        """
        if not self.cryptobert_available:
            logger.warning("CryptoBERT n'est pas disponible. Impossible de générer l'embedding localement.")
            return None
            
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            # Charger le modèle et le tokenizer (utiliser un modèle BERT pré-entraîné sur des données financières)
            model_name = "yiyanghkust/finbert-tone"  # Alternative: "ProsusAI/finbert" ou autre modèle financier
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Tokenization et préparation des inputs
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Génération de l'embedding
            with torch.no_grad():
                outputs = model(**inputs)
                # Utiliser la représentation du token [CLS] comme embedding du texte entier
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
            logger.info(f"Embedding CryptoBERT généré avec succès (dim: {embedding.shape})")
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding CryptoBERT: {e}")
            return None
    
    def get_embedding(self, symbol: str, date_str: str, context_text: Optional[str] = None, 
                     embedding_type: str = "news", force_refresh: bool = False) -> Dict[str, Any]:
        """
        Récupère un embedding avec cache et fallback.
        
        Stratégie:
        1. Essaie de charger depuis le cache local
        2. Si non trouvé ou force_refresh, essaie CryptoBERT
        3. Si CryptoBERT échoue, essaie Gemini API
        4. Sauvegarde le nouvel embedding dans le cache
        
        Args:
            symbol: Symbole de la crypto (ex: "BTC")
            date_str: Date au format YYYY-MM-DD
            context_text: Texte de contexte optionnel pour générer l'embedding
            embedding_type: Type d'embedding ("news" ou "instrument")
            force_refresh: Force la régénération même si présent dans le cache
            
        Returns:
            Dict avec les clés:
            - embedding: np.ndarray de l'embedding ou None
            - source: str indiquant la source ("cache", "cryptobert", "gemini", "none")
            - summary: str avec un résumé ou contexte
        """
        result = {
            "embedding": None,
            "source": "none",
            "summary": ""
        }
        
        # 1. Essayer le cache local (sauf si force_refresh)
        if not force_refresh:
            cached_embedding = self.get_cached_embedding(symbol, date_str, embedding_type)
            if cached_embedding is not None:
                result["embedding"] = cached_embedding
                result["source"] = "cache"
                result["summary"] = "Embedding chargé depuis le cache local"
                return result
        
        # Si pas de texte fourni, générer un texte de contexte par défaut
        if not context_text:
            if embedding_type == "news":
                context_text = f"Latest cryptocurrency news and market analysis for {symbol} on {date_str}."
            else:  # instrument
                context_text = f"Description and characteristics of the cryptocurrency {symbol}."
        
        # 2. Essayer CryptoBERT
        if self.cryptobert_available:
            cryptobert_embedding = self.get_cryptobert_embedding(context_text)
            if cryptobert_embedding is not None:
                result["embedding"] = cryptobert_embedding
                result["source"] = "cryptobert"
                result["summary"] = f"Embedding généré par CryptoBERT: {context_text[:100]}..."
                
                # Sauvegarder dans le cache
                self.save_embedding_to_cache(
                    symbol, date_str, cryptobert_embedding, 
                    summary=result["summary"], embedding_type=embedding_type
                )
                
                return result
        
        # 3. Essayer Gemini API
        if self.api_key:
            gemini_embedding = self.get_gemini_embedding(context_text)
            if gemini_embedding is not None:
                result["embedding"] = gemini_embedding
                result["source"] = "gemini"
                result["summary"] = f"Embedding généré par Gemini API: {context_text[:100]}..."
                
                # Sauvegarder dans le cache
                self.save_embedding_to_cache(
                    symbol, date_str, gemini_embedding, 
                    summary=result["summary"], embedding_type=embedding_type
                )
                
                return result
        
        # 4. Fallback: générer un embedding aléatoire (pour développement uniquement)
        logger.warning(f"Aucune méthode d'embedding disponible. Génération d'un embedding aléatoire pour {symbol}-{date_str}.")
        random_embedding = np.random.randn(768).astype(np.float32)  # Dimension standard pour BERT
        result["embedding"] = random_embedding
        result["source"] = "random"
        result["summary"] = "Embedding aléatoire (fallback)"
        
        # Sauvegarder l'embedding aléatoire dans le cache
        self.save_embedding_to_cache(
            symbol, date_str, random_embedding, 
            summary=result["summary"], embedding_type=embedding_type
        )
        
        return result
