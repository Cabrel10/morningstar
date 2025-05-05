#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de gestion de la reprise après interruption pour le trading en direct.
Permet de reprendre le trading après une interruption et de gérer les données manquantes.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import time
from datetime import datetime, timedelta
import pickle
import traceback
import threading
import signal

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecoveryManager:
    """
    Gestionnaire de reprise après interruption pour le trading en direct.
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[Path] = None):
        """
        Initialise le gestionnaire de reprise.
        
        Args:
            config: Configuration du trading
            checkpoint_dir: Répertoire des points de contrôle
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # État du trading
        self.trading_state = {
            'last_update': None,
            'active_orders': {},
            'positions': {},
            'last_signals': {},
            'last_prices': {},
            'performance': {},
            'is_trading_active': True,
            'consecutive_errors': 0,
            'last_error_time': None
        }
        
        # Intervalle de sauvegarde automatique (en secondes)
        self.autosave_interval = config.get('recovery', {}).get('autosave_interval', 300)  # 5 minutes par défaut
        
        # Configurer le gestionnaire de signaux pour la capture des interruptions
        self._setup_signal_handlers()
        
        # Charger l'état précédent s'il existe
        self._load_latest_checkpoint()
        
        # Démarrer la sauvegarde automatique si configurée
        if self.autosave_interval > 0:
            self._start_autosave_thread()
        
        logger.info(f"Gestionnaire de reprise initialisé avec checkpoint_dir: {self.checkpoint_dir}")
    
    def _setup_signal_handlers(self):
        """
        Configure les gestionnaires de signaux pour capturer les interruptions.
        """
        # Capturer SIGINT (Ctrl+C) et SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Gestionnaires de signaux configurés pour la capture des interruptions")
    
    def _signal_handler(self, sig, frame):
        """
        Gestionnaire de signaux pour sauvegarder l'état avant l'arrêt.
        
        Args:
            sig: Signal reçu
            frame: Frame d'exécution
        """
        logger.info(f"Signal {sig} reçu, sauvegarde de l'état avant l'arrêt...")
        self.save_checkpoint(force=True)
        logger.info("État sauvegardé, arrêt du programme")
        sys.exit(0)
    
    def _start_autosave_thread(self):
        """
        Démarre un thread pour la sauvegarde automatique périodique.
        """
        def autosave_worker():
            while True:
                time.sleep(self.autosave_interval)
                try:
                    self.save_checkpoint()
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde automatique: {e}")
        
        self.autosave_thread = threading.Thread(target=autosave_worker, daemon=True)
        self.autosave_thread.start()
        
        logger.info(f"Thread de sauvegarde automatique démarré avec intervalle: {self.autosave_interval}s")
    
    def _load_latest_checkpoint(self) -> bool:
        """
        Charge le dernier point de contrôle disponible.
        
        Returns:
            True si un point de contrôle a été chargé, False sinon
        """
        try:
            # Trouver le dernier fichier de checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob('trading_state_*.pkl'))
            
            if not checkpoint_files:
                logger.info("Aucun point de contrôle trouvé, utilisation d'un nouvel état")
                return False
            
            # Trier par date de modification (le plus récent en premier)
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            
            # Charger l'état
            with open(latest_checkpoint, 'rb') as f:
                loaded_state = pickle.load(f)
            
            # Mettre à jour l'état actuel
            self.trading_state.update(loaded_state)
            
            # Convertir les timestamps en objets datetime si nécessaire
            if self.trading_state['last_update'] and isinstance(self.trading_state['last_update'], str):
                self.trading_state['last_update'] = datetime.fromisoformat(self.trading_state['last_update'])
            
            if self.trading_state['last_error_time'] and isinstance(self.trading_state['last_error_time'], str):
                self.trading_state['last_error_time'] = datetime.fromisoformat(self.trading_state['last_error_time'])
            
            logger.info(f"État chargé depuis {latest_checkpoint}, dernière mise à jour: {self.trading_state['last_update']}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du point de contrôle: {e}")
            traceback.print_exc()
            return False
    
    def save_checkpoint(self, force: bool = False) -> bool:
        """
        Sauvegarde l'état actuel dans un point de contrôle.
        
        Args:
            force: Force la sauvegarde même si aucun changement n'a été détecté
            
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            # Mettre à jour l'horodatage
            current_time = datetime.now()
            
            # Vérifier s'il y a eu des changements depuis la dernière sauvegarde
            if not force and self.trading_state['last_update']:
                last_update = self.trading_state['last_update']
                if isinstance(last_update, str):
                    last_update = datetime.fromisoformat(last_update)
                
                # Si moins de 5 minutes se sont écoulées et pas de changements forcés, ignorer
                if (current_time - last_update).total_seconds() < 300:
                    return True
            
            # Mettre à jour l'horodatage
            self.trading_state['last_update'] = current_time
            
            # Créer le nom du fichier avec horodatage
            timestamp = current_time.strftime('%Y%m%d_%H%M%S')
            checkpoint_file = self.checkpoint_dir / f"trading_state_{timestamp}.pkl"
            
            # Sauvegarder l'état
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(self.trading_state, f)
            
            # Sauvegarder également une version JSON pour inspection humaine
            json_file = self.checkpoint_dir / f"trading_state_{timestamp}.json"
            json_state = self.trading_state.copy()
            
            # Convertir les objets datetime en chaînes ISO
            if isinstance(json_state['last_update'], datetime):
                json_state['last_update'] = json_state['last_update'].isoformat()
            
            if isinstance(json_state['last_error_time'], datetime):
                json_state['last_error_time'] = json_state['last_error_time'].isoformat() if json_state['last_error_time'] else None
            
            with open(json_file, 'w') as f:
                json.dump(json_state, f, indent=2, default=str)
            
            # Nettoyer les anciens fichiers de checkpoint (garder les 10 plus récents)
            self._cleanup_old_checkpoints()
            
            logger.info(f"État sauvegardé dans {checkpoint_file}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du point de contrôle: {e}")
            traceback.print_exc()
            return False
    
    def _cleanup_old_checkpoints(self, max_files: int = 10):
        """
        Nettoie les anciens fichiers de checkpoint, ne gardant que les plus récents.
        
        Args:
            max_files: Nombre maximum de fichiers à conserver
        """
        try:
            # Trouver tous les fichiers de checkpoint
            pkl_files = list(self.checkpoint_dir.glob('trading_state_*.pkl'))
            json_files = list(self.checkpoint_dir.glob('trading_state_*.json'))
            
            # Trier par date de modification (le plus ancien en premier)
            pkl_files.sort(key=os.path.getmtime)
            json_files.sort(key=os.path.getmtime)
            
            # Supprimer les fichiers les plus anciens si nécessaire
            for files in [pkl_files, json_files]:
                if len(files) > max_files:
                    for old_file in files[:-max_files]:
                        os.remove(old_file)
                        logger.debug(f"Ancien fichier de checkpoint supprimé: {old_file}")
        
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des anciens checkpoints: {e}")
    
    def update_trading_state(self, key: str, value: Any) -> None:
        """
        Met à jour une valeur spécifique dans l'état du trading.
        
        Args:
            key: Clé à mettre à jour
            value: Nouvelle valeur
        """
        self.trading_state[key] = value
    
    def update_active_orders(self, symbol: str, orders: Dict[str, Any]) -> None:
        """
        Met à jour les ordres actifs pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            orders: Dictionnaire des ordres actifs
        """
        self.trading_state['active_orders'][symbol] = orders
        self.trading_state['last_update'] = datetime.now()
    
    def update_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """
        Met à jour la position pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            position: Informations sur la position
        """
        self.trading_state['positions'][symbol] = position
        self.trading_state['last_update'] = datetime.now()
    
    def update_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """
        Met à jour le dernier signal pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            signal: Informations sur le signal
        """
        self.trading_state['last_signals'][symbol] = signal
        self.trading_state['last_update'] = datetime.now()
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Met à jour le dernier prix pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            price: Dernier prix
        """
        self.trading_state['last_prices'][symbol] = price
        self.trading_state['last_update'] = datetime.now()
    
    def update_performance(self, performance: Dict[str, Any]) -> None:
        """
        Met à jour les métriques de performance.
        
        Args:
            performance: Métriques de performance
        """
        self.trading_state['performance'] = performance
        self.trading_state['last_update'] = datetime.now()
    
    def set_trading_active(self, active: bool) -> None:
        """
        Active ou désactive le trading.
        
        Args:
            active: True pour activer, False pour désactiver
        """
        self.trading_state['is_trading_active'] = active
        self.trading_state['last_update'] = datetime.now()
        logger.info(f"Trading {'activé' if active else 'désactivé'}")
    
    def is_trading_active(self) -> bool:
        """
        Vérifie si le trading est actif.
        
        Returns:
            True si le trading est actif, False sinon
        """
        return self.trading_state.get('is_trading_active', True)
    
    def update_error_count(self, increment: bool = True) -> int:
        """
        Met à jour le compteur d'erreurs consécutives.
        
        Args:
            increment: True pour incrémenter, False pour réinitialiser
            
        Returns:
            Nombre actuel d'erreurs consécutives
        """
        if increment:
            self.trading_state['consecutive_errors'] += 1
            self.trading_state['last_error_time'] = datetime.now()
        else:
            self.trading_state['consecutive_errors'] = 0
            self.trading_state['last_error_time'] = None
        
        self.trading_state['last_update'] = datetime.now()
        return self.trading_state['consecutive_errors']
    
    def get_error_status(self) -> Dict[str, Any]:
        """
        Récupère le statut des erreurs.
        
        Returns:
            Dictionnaire avec le nombre d'erreurs consécutives et l'horodatage de la dernière erreur
        """
        return {
            'consecutive_errors': self.trading_state['consecutive_errors'],
            'last_error_time': self.trading_state['last_error_time']
        }
    
    def get_active_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les ordres actifs.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dictionnaire des ordres actifs
        """
        if symbol:
            return self.trading_state['active_orders'].get(symbol, {})
        return self.trading_state['active_orders']
    
    def get_position(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les positions.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dictionnaire des positions
        """
        if symbol:
            return self.trading_state['positions'].get(symbol, {})
        return self.trading_state['positions']
    
    def get_last_signal(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les derniers signaux.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dictionnaire des derniers signaux
        """
        if symbol:
            return self.trading_state['last_signals'].get(symbol, {})
        return self.trading_state['last_signals']
    
    def get_last_price(self, symbol: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Récupère les derniers prix.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Dernier prix ou dictionnaire des derniers prix
        """
        if symbol:
            return self.trading_state['last_prices'].get(symbol, 0.0)
        return self.trading_state['last_prices']
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Récupère les métriques de performance.
        
        Returns:
            Dictionnaire des métriques de performance
        """
        return self.trading_state['performance']
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        Récupère l'état complet du trading.
        
        Returns:
            Dictionnaire de l'état complet
        """
        return self.trading_state.copy()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestionnaire de reprise pour le trading en direct")
    parser.add_argument("--config", type=str, default="config/trading_live.yaml", help="Chemin vers le fichier de configuration")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Répertoire des points de contrôle")
    parser.add_argument("--test", action="store_true", help="Exécute un test de sauvegarde et chargement")
    
    args = parser.parse_args()
    
    # Charger la configuration
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        config = {}
    
    # Initialiser le gestionnaire de reprise
    recovery_manager = RecoveryManager(config, Path(args.checkpoint_dir))
    
    # Exécuter un test si demandé
    if args.test:
        logger.info("Exécution d'un test de sauvegarde et chargement")
        
        # Mettre à jour quelques valeurs
        recovery_manager.update_price("BTC/USDT", 50000.0)
        recovery_manager.update_signal("BTC/USDT", {"action": "buy", "confidence": 0.8})
        recovery_manager.update_position("BTC/USDT", {"size": 0.1, "entry_price": 49000.0})
        recovery_manager.update_performance({"total_profit": 1000.0, "win_rate": 0.65})
        
        # Sauvegarder l'état
        recovery_manager.save_checkpoint(force=True)
        
        # Charger l'état
        recovery_manager._load_latest_checkpoint()
        
        # Afficher l'état
        print("\nÉtat chargé:")
        for key, value in recovery_manager.get_full_state().items():
            print(f"{key}: {value}")
        
        logger.info("Test terminé")
