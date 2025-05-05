#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processeur CryptoBERT pour le modu00e8le Morningstar.
Ce module utilise le modu00e8le CryptoBERT pour extraire des embeddings u00e0 partir de textes
relatifs aux crypto-monnaies.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class CryptoBERTProcessor:
    """
    Classe pour traiter les donnu00e9es textuelles avec CryptoBERT.
    """
    def __init__(self, model_name="ElKulako/cryptobert", max_length=512, device=None):
        """
        Initialise le processeur CryptoBERT.
        
        Args:
            model_name: Nom du modu00e8le CryptoBERT
            max_length: Longueur maximale des su00e9quences
            device: Dispositif u00e0 utiliser (None pour auto-du00e9tection)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
        try:
            logger.info(f"Chargement du tokenizer CryptoBERT {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info(f"Chargement du modu00e8le CryptoBERT {model_name} sur {self.device}")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            logger.info("CryptoBERT initialisu00e9 avec succu00e8s")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de CryptoBERT: {e}")
    
    def generate_embeddings(self, texts: List[str], pooling_strategy='mean') -> List[np.ndarray]:
        """
        Gu00e9nu00e8re des embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes
            pooling_strategy: Stratu00e9gie de pooling ('mean', 'cls', 'max')
            
        Returns:
            Liste d'embeddings
        """
        if self.tokenizer is None or self.model is None:
            logger.error("CryptoBERT n'a pas u00e9tu00e9 initialisu00e9 correctement")
            # Retourner des embeddings vides de taille 768 (taille standard de BERT)
            return [np.zeros(768) for _ in texts]
        
        embeddings = []
        batch_size = 8  # Traiter les textes par lots
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokeniser les textes
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Gu00e9nu00e9rer les embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state
                
                # Appliquer la stratu00e9gie de pooling
                if pooling_strategy == 'cls':
                    # Utiliser l'embedding du token [CLS]
                    batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
                elif pooling_strategy == 'max':
                    # Pooling max sur la dimension des tokens
                    batch_embeddings = torch.max(hidden_states, dim=1)[0].cpu().numpy()
                else:  # 'mean' par du00e9faut
                    # Pooling moyen sur la dimension des tokens
                    attention_mask = inputs['attention_mask']
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    masked_hidden = hidden_states * mask
                    sum_hidden = torch.sum(masked_hidden, dim=1)
                    sum_mask = torch.sum(mask, dim=1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    batch_embeddings = (sum_hidden / sum_mask).cpu().numpy()
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Erreur lors de la gu00e9nu00e9ration des embeddings pour le lot {i//batch_size}: {e}")
                # Ajouter des embeddings vides en cas d'erreur
                batch_embeddings = [np.zeros(768) for _ in range(len(batch_texts))]
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def classify_text(self, text: str, labels: List[str]) -> Dict[str, float]:
        """
        Classifie un texte parmi un ensemble d'u00e9tiquettes.
        
        Args:
            text: Texte u00e0 classifier
            labels: Liste des u00e9tiquettes possibles
            
        Returns:
            Dictionnaire des probabilitu00e9s pour chaque u00e9tiquette
        """
        if self.tokenizer is None or self.model is None:
            logger.error("CryptoBERT n'a pas u00e9tu00e9 initialisu00e9 correctement")
            return {label: 1.0 / len(labels) for label in labels}
        
        try:
            # Gu00e9nu00e9rer l'embedding du texte
            text_embedding = self.generate_embeddings([text])[0]
            
            # Gu00e9nu00e9rer les embeddings des u00e9tiquettes
            label_embeddings = self.generate_embeddings(labels)
            
            # Calculer les similaritu00e9s cosinus
            similarities = []
            for label_embedding in label_embeddings:
                similarity = self._cosine_similarity(text_embedding, label_embedding)
                similarities.append(similarity)
            
            # Normaliser les similaritu00e9s en probabilitu00e9s
            total = sum(similarities)
            if total > 0:
                probabilities = [sim / total for sim in similarities]
            else:
                probabilities = [1.0 / len(labels) for _ in labels]
            
            return {label: prob for label, prob in zip(labels, probabilities)}
            
        except Exception as e:
            logger.error(f"Erreur lors de la classification du texte: {e}")
            return {label: 1.0 / len(labels) for label in labels}
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extrait les mots-clu00e9s les plus importants d'un texte.
        
        Args:
            text: Texte u00e0 analyser
            top_k: Nombre de mots-clu00e9s u00e0 extraire
            
        Returns:
            Liste des mots-clu00e9s
        """
        if self.tokenizer is None or self.model is None:
            logger.error("CryptoBERT n'a pas u00e9tu00e9 initialisu00e9 correctement")
            return []
        
        try:
            # Tokeniser le texte
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Gu00e9nu00e9rer les embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
            # Calculer l'importance de chaque token
            token_importances = torch.mean(hidden_states, dim=2).squeeze().cpu().numpy()
            
            # Ru00e9cupu00e9rer les tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Cru00e9er une liste de tuples (token, importance)
            token_importance_pairs = [(token, importance) for token, importance in zip(tokens, token_importances)]
            
            # Filtrer les tokens spu00e9ciaux et les tokens de padding
            filtered_pairs = [(token, importance) for token, importance in token_importance_pairs 
                             if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]'] 
                             and not token.startswith('##')]
            
            # Trier par importance du00e9croissante
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)
            
            # Ru00e9cupu00e9rer les top_k tokens
            top_tokens = [pair[0] for pair in sorted_pairs[:top_k]]
            
            return top_tokens
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des mots-clu00e9s: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similaritu00e9 cosinus entre deux vecteurs.
        
        Args:
            a: Premier vecteur
            b: Deuxiu00e8me vecteur
            
        Returns:
            Similaritu00e9 cosinus
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return np.dot(a, b) / (norm_a * norm_b)
