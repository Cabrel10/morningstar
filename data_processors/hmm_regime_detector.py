#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Du00e9tecteur de ru00e9gime de marchu00e9 utilisant les modu00e8les de Markov cachu00e9s (HMM).
Ce module permet de du00e9tecter automatiquement les ru00e9gimes de marchu00e9 u00e0 partir des donnu00e9es de prix.
"""

import numpy as np
import pandas as pd
import logging
from hmmlearn import hmm
from typing import Tuple, List, Optional, Union

logger = logging.getLogger(__name__)

class HMMRegimeDetector:
    """
    Classe pour du00e9tecter les ru00e9gimes de marchu00e9 en utilisant les modu00e8les de Markov cachu00e9s (HMM).
    """
    def __init__(self, n_components=3, n_iter=100, random_state=42):
        """
        Initialise le du00e9tecteur de ru00e9gime HMM.
        
        Args:
            n_components: Nombre de ru00e9gimes (u00e9tats cachu00e9s)
            n_iter: Nombre maximum d'itu00e9rations pour l'entrau00eenement
            random_state: Graine alu00e9atoire pour la reproductibilitu00e9
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
    
    def fit(self, returns: np.ndarray) -> None:
        """
        Entrau00eene le modu00e8le HMM sur les donnu00e9es de rendements.
        
        Args:
            returns: Array de rendements (peut u00eatre 1D ou 2D)
        """
        # S'assurer que les donnu00e9es sont au bon format
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Remplacer les valeurs NaN par 0
        returns = np.nan_to_num(returns, nan=0.0)
        
        logger.info(f"Entrau00eenement du modu00e8le HMM avec {len(returns)} observations et {self.n_components} ru00e9gimes")
        
        try:
            # Initialiser et entrau00eener le modu00e8le HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            
            self.model.fit(returns)
            logger.info(f"Modu00e8le HMM entrau00eenu00e9 avec succu00e8s (score: {self.model.score(returns)})")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entrau00eenement du modu00e8le HMM: {e}")
            self.model = None
    
    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Pru00e9dit les ru00e9gimes de marchu00e9 pour les donnu00e9es de rendements.
        
        Args:
            returns: Array de rendements (peut u00eatre 1D ou 2D)
            
        Returns:
            Array des ru00e9gimes pru00e9dits
        """
        if self.model is None:
            logger.error("Le modu00e8le HMM n'a pas u00e9tu00e9 entrau00eenu00e9")
            return np.zeros(len(returns), dtype=int)
        
        # S'assurer que les donnu00e9es sont au bon format
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Remplacer les valeurs NaN par 0
        returns = np.nan_to_num(returns, nan=0.0)
        
        try:
            # Pru00e9dire les ru00e9gimes
            regimes = self.model.predict(returns)
            logger.info(f"Ru00e9gimes pru00e9dits pour {len(returns)} observations")
            return regimes
            
        except Exception as e:
            logger.error(f"Erreur lors de la pru00e9diction des ru00e9gimes: {e}")
            return np.zeros(len(returns), dtype=int)
    
    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """
        Pru00e9dit les probabilitu00e9s des ru00e9gimes de marchu00e9 pour les donnu00e9es de rendements.
        
        Args:
            returns: Array de rendements (peut u00eatre 1D ou 2D)
            
        Returns:
            Array des probabilitu00e9s des ru00e9gimes
        """
        if self.model is None:
            logger.error("Le modu00e8le HMM n'a pas u00e9tu00e9 entrau00eenu00e9")
            return np.zeros((len(returns), self.n_components))
        
        # S'assurer que les donnu00e9es sont au bon format
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Remplacer les valeurs NaN par 0
        returns = np.nan_to_num(returns, nan=0.0)
        
        try:
            # Calculer les probabilitu00e9s a posteriori
            _, proba = self.model.score_samples(returns)
            logger.info(f"Probabilitu00e9s des ru00e9gimes calculu00e9es pour {len(returns)} observations")
            return proba
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des probabilitu00e9s des ru00e9gimes: {e}")
            return np.zeros((len(returns), self.n_components))
    
    def detect_regimes(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Du00e9tecte les ru00e9gimes de marchu00e9 et leurs probabilitu00e9s.
        
        Args:
            returns: Array de rendements (peut u00eatre 1D ou 2D)
            
        Returns:
            Tuple (ru00e9gimes, probabilitu00e9s)
        """
        # S'assurer que les donnu00e9es sont au bon format
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        
        # Entrau00eener le modu00e8le si nu00e9cessaire
        if self.model is None:
            self.fit(returns)
        
        # Pru00e9dire les ru00e9gimes et leurs probabilitu00e9s
        regimes = self.predict(returns)
        proba = self.predict_proba(returns)
        
        return regimes, proba
    
    def interpret_regimes(self, returns: np.ndarray, regimes: np.ndarray) -> List[str]:
        """
        Interpru00e8te les ru00e9gimes de marchu00e9 du00e9tectu00e9s.
        
        Args:
            returns: Array de rendements
            regimes: Array des ru00e9gimes du00e9tectu00e9s
            
        Returns:
            Liste des interpru00e9tations des ru00e9gimes
        """
        if self.model is None:
            logger.error("Le modu00e8le HMM n'a pas u00e9tu00e9 entrau00eenu00e9")
            return ["unknown"] * self.n_components
        
        # Calculer les statistiques pour chaque ru00e9gime
        regime_stats = []
        for i in range(self.n_components):
            mask = (regimes == i)
            if np.any(mask):
                regime_returns = returns[mask]
                mean_return = np.mean(regime_returns)
                volatility = np.std(regime_returns)
                sharpe = mean_return / volatility if volatility > 0 else 0
                
                regime_stats.append({
                    'regime': i,
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'count': np.sum(mask)
                })
        
        # Trier les ru00e9gimes par rendement moyen
        regime_stats.sort(key=lambda x: x['mean_return'])
        
        # Interpru00e9ter les ru00e9gimes
        interpretations = []
        for i, stats in enumerate(regime_stats):
            regime_id = stats['regime']
            
            if i == 0:  # Ru00e9gime avec le rendement le plus bas
                interpretation = "bearish"
            elif i == len(regime_stats) - 1:  # Ru00e9gime avec le rendement le plus u00e9levu00e9
                interpretation = "bullish"
            else:  # Ru00e9gimes intermu00e9diaires
                if stats['volatility'] > np.mean([s['volatility'] for s in regime_stats]):
                    interpretation = "volatile"
                else:
                    interpretation = "sideways"
            
            interpretations.append((regime_id, interpretation))
        
        # Ru00e9organiser les interpru00e9tations par ID de ru00e9gime
        interpretations.sort(key=lambda x: x[0])
        return [interp for _, interp in interpretations]
    
    def save_model(self, filepath: str) -> bool:
        """
        Sauvegarde le modu00e8le HMM dans un fichier.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            True si la sauvegarde a ru00e9ussi, False sinon
        """
        if self.model is None:
            logger.error("Aucun modu00e8le u00e0 sauvegarder")
            return False
        
        try:
            import joblib
            joblib.dump(self.model, filepath)
            logger.info(f"Modu00e8le HMM sauvegardu00e9 dans {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modu00e8le HMM: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Charge le modu00e8le HMM depuis un fichier.
        
        Args:
            filepath: Chemin du fichier de modu00e8le
            
        Returns:
            True si le chargement a ru00e9ussi, False sinon
        """
        try:
            import joblib
            self.model = joblib.load(filepath)
            logger.info(f"Modu00e8le HMM chargu00e9 depuis {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modu00e8le HMM: {e}")
            return False
