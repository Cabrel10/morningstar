import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import os
import datetime
import json

from live.exchange_integration import ExchangeFactory, ExchangeBase
from utils.market_regime import MarketRegimeDetector
from utils.llm_integration import LLMIntegration

# Configuration du logging
logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Moteur de trading qui utilise le modu00e8le Morningstar pour gu00e9nu00e9rer des signaux
    et exu00e9cuter des ordres sur un exchange.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le moteur de trading.
        
        Args:
            config: Dictionnaire de configuration contenant:
                - exchange: Configuration de l'exchange
                - model_path: Chemin vers le modu00e8le Morningstar
                - trading_params: Paramu00e8tres de trading (timeframe, pairs, etc.)
                - risk_params: Paramu00e8tres de gestion du risque (SL/TP, taille de position, etc.)
        """
        self.config = config
        self.exchange_config = config.get('exchange', {})
        self.trading_params = config.get('trading_params', {})
        self.risk_params = config.get('risk_params', {})
        
        # Initialiser l'exchange
        exchange_id = self.exchange_config.get('exchange_id', 'binance')
        self.exchange = ExchangeFactory.create_exchange(exchange_id, self.exchange_config)
        
        # Charger le modu00e8le Morningstar
        self.model = self._load_model(config.get('model_path'))
        
        # Initialiser le du00e9tecteur de ru00e9gimes de marchu00e9
        self.hmm_detector = self._load_hmm_detector(config.get('hmm_model_path'))
        
        # Initialiser l'intu00e9gration LLM
        self.llm = LLMIntegration()
        
        # Paires de trading
        self.pairs = self.trading_params.get('pairs', ['BTC/USDT'])
        self.timeframe = self.trading_params.get('timeframe', '1h')
        
        # Historique des signaux et des transactions
        self.signals_history = {}
        self.trades_history = []
        
        # Journal de trading
        self.log_dir = Path(config.get('log_dir', 'logs/trading'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mu00e9triques de performance
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
    
    def _load_model(self, model_path: Optional[str]) -> Any:
        """
        Charge le modu00e8le Morningstar depuis un fichier.
        
        Args:
            model_path: Chemin vers le fichier du modu00e8le
            
        Returns:
            Modu00e8le chargu00e9 ou None si le chargement u00e9choue
        """
        if not model_path:
            logger.warning("Aucun chemin de modu00e8le spu00e9cifiu00e9. Fonctionnement en mode simulation.")
            return None
            
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Modu00e8le Morningstar chargu00e9 depuis {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modu00e8le: {e}")
            return None
    
    def _load_hmm_detector(self, hmm_model_path: Optional[str]) -> Optional[MarketRegimeDetector]:
        """
        Charge le du00e9tecteur de ru00e9gimes de marchu00e9 HMM.
        
        Args:
            hmm_model_path: Chemin vers le fichier du modu00e8le HMM
            
        Returns:
            Du00e9tecteur de ru00e9gimes de marchu00e9 ou None
        """
        if not hmm_model_path:
            logger.warning("Aucun chemin de modu00e8le HMM spu00e9cifiu00e9. Ru00e9gimes de marchu00e9 non disponibles.")
            return None
            
        try:
            detector = MarketRegimeDetector.load_model(hmm_model_path)
            logger.info(f"Du00e9tecteur de ru00e9gimes de marchu00e9 HMM chargu00e9 depuis {hmm_model_path}")
            return detector
        except Exception as e:
            logger.error(f"Erreur lors du chargement du du00e9tecteur HMM: {e}")
            return None
    
    def _prepare_features(self, ohlcv_data: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """
        Pru00e9pare les features pour le modu00e8le Morningstar.
        
        Args:
            ohlcv_data: Donnu00e9es OHLCV
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire des features pru00e9paru00e9es
        """
        try:
            # Pru00e9parer les features techniques
            from utils.feature_engineering import apply_feature_pipeline
            df_features = apply_feature_pipeline(ohlcv_data)
            
            # Ru00e9cupu00e9rer les embeddings LLM pour la derniu00e8re date
            last_date = ohlcv_data.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
            llm_result = self.llm.get_embedding(symbol.split('/')[0], last_date)
            llm_embedding = llm_result['embedding']
            
            # Pru00e9parer les features HMM si disponibles
            hmm_features = None
            if self.hmm_detector:
                regimes = self.hmm_detector.predict(df_features)
                probs = self.hmm_detector.predict_proba(df_features)
                hmm_features = np.column_stack([regimes.reshape(-1, 1), probs])
            
            # Construire le dictionnaire de features
            features = {
                'technical_input': df_features.iloc[-1][df_features.columns.difference(['timestamp', 'open', 'high', 'low', 'close', 'volume'])].values.astype(np.float32),
                'llm_input': llm_embedding.astype(np.float32),
                'instrument_input': np.array([0])  # Placeholder, u00e0 adapter selon votre modu00e8le
            }
            
            # Ajouter les features HMM si disponibles
            if hmm_features is not None:
                features['hmm_input'] = hmm_features[-1].astype(np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors de la pru00e9paration des features: {e}")
            return {}
    
    def _generate_signal(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Gu00e9nu00e8re un signal de trading u00e0 partir des features.
        
        Args:
            features: Dictionnaire des features
            
        Returns:
            Signal de trading (action, sl, tp, etc.)
        """
        if not self.model or not features:
            # Mode simulation ou erreur
            return {'action': 'hold', 'confidence': 0.0, 'sl': 0.0, 'tp': 0.0}
        
        try:
            # Pru00e9parer les inputs pour le modu00e8le
            model_inputs = {}
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    # Ajouter la dimension batch
                    if value.ndim == 1:
                        model_inputs[key] = np.expand_dims(value, axis=0)
                    else:
                        model_inputs[key] = value.reshape(1, -1)
            
            # Faire la pru00e9diction
            predictions = self.model.predict(model_inputs)
            
            # Interpru00e9ter les pru00e9dictions selon la structure de votre modu00e8le
            # Exemple pour un modu00e8le avec sorties 'signal', 'sl_tp'
            signal_pred = predictions[0] if isinstance(predictions, list) else predictions
            
            # Convertir en action
            if isinstance(signal_pred, np.ndarray):
                action_idx = np.argmax(signal_pred[0])
                confidence = float(signal_pred[0][action_idx])
                action = ['sell', 'hold', 'buy'][action_idx]  # Adapter selon votre modu00e8le
            else:
                action = 'hold'
                confidence = 0.0
            
            # Ru00e9cupu00e9rer SL/TP si disponibles
            sl_tp = None
            if len(predictions) > 1 and isinstance(predictions[1], np.ndarray):
                sl_tp = predictions[1][0]
            
            return {
                'action': action,
                'confidence': confidence,
                'sl': float(sl_tp[0]) if sl_tp is not None else 0.0,
                'tp': float(sl_tp[1]) if sl_tp is not None else 0.0
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la gu00e9nu00e9ration du signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'sl': 0.0, 'tp': 0.0}
    
    def _execute_signal(self, symbol: str, signal: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Exu00e9cute un signal de trading sur l'exchange.
        
        Args:
            symbol: Symbole de la paire
            signal: Signal de trading
            current_price: Prix actuel
            
        Returns:
            Ru00e9sultat de l'exu00e9cution
        """
        # Vu00e9rifier si le signal est assez fort pour agir
        min_confidence = self.risk_params.get('min_confidence', 0.7)
        if signal['confidence'] < min_confidence:
            logger.info(f"Signal trop faible ({signal['confidence']:.2f} < {min_confidence}) pour {symbol}. Pas d'action.")
            return {'status': 'skipped', 'reason': 'low_confidence'}
        
        # Vu00e9rifier s'il y a du00e9ju00e0 une position ouverte
        open_positions = self._get_open_positions(symbol)
        has_position = len(open_positions) > 0
        
        # Calculer la taille de la position
        position_size = self._calculate_position_size(symbol, signal)
        
        # Exu00e9cuter l'action
        if signal['action'] == 'buy' and not has_position:
            # Ouvrir une position longue
            order_result = self.exchange.create_order(symbol, 'market', 'buy', position_size)
            
            # Placer les ordres SL/TP si configurés
            if self.risk_params.get('use_sl_tp', True) and signal['sl'] > 0 and signal['tp'] > 0:
                sl_price = current_price * (1 - signal['sl'])
                tp_price = current_price * (1 + signal['tp'])
                
                # Placer les ordres SL/TP
                sl_order = self.exchange.create_order(symbol, 'limit', 'sell', position_size, sl_price)
                tp_order = self.exchange.create_order(symbol, 'limit', 'sell', position_size, tp_price)
                
                order_result['sl_order'] = sl_order
                order_result['tp_order'] = tp_order
            
            # Enregistrer la transaction
            trade = {
                'symbol': symbol,
                'action': 'buy',
                'price': current_price,
                'size': position_size,
                'timestamp': datetime.datetime.now().isoformat(),
                'sl': signal['sl'],
                'tp': signal['tp'],
                'confidence': signal['confidence'],
                'order_id': order_result.get('id', 'unknown')
            }
            self.trades_history.append(trade)
            
            return {'status': 'executed', 'action': 'buy', 'order': order_result}
            
        elif signal['action'] == 'sell' and has_position:
            # Fermer une position existante
            for position in open_positions:
                order_result = self.exchange.create_order(symbol, 'market', 'sell', position['amount'])
                
                # Annuler les ordres SL/TP existants
                if position.get('sl_order_id'):
                    self.exchange.cancel_order(position['sl_order_id'], symbol)
                if position.get('tp_order_id'):
                    self.exchange.cancel_order(position['tp_order_id'], symbol)
                
                # Enregistrer la transaction
                trade = {
                    'symbol': symbol,
                    'action': 'sell',
                    'price': current_price,
                    'size': position['amount'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'confidence': signal['confidence'],
                    'order_id': order_result.get('id', 'unknown'),
                    'profit': (current_price - position['entry_price']) * position['amount']
                }
                self.trades_history.append(trade)
                
                # Mettre u00e0 jour les mu00e9triques de performance
                self._update_performance_metrics(trade)
            
            return {'status': 'executed', 'action': 'sell', 'order': order_result}
        
        else:
            # Pas d'action requise
            return {'status': 'no_action', 'reason': 'no_matching_conditions'}
    
    def _get_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Ru00e9cupu00e8re les positions ouvertes pour un symbole donnu00e9.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Liste des positions ouvertes
        """
        # Cette fonction doit u00eatre adaptu00e9e selon l'exchange et sa structure de donnu00e9es
        try:
            # Pour les exchanges spot, on peut utiliser le solde
            balance = self.exchange.fetch_balance()
            base_currency = symbol.split('/')[0]
            
            # Vu00e9rifier si on a un solde non nul dans la devise de base
            if base_currency in balance and balance[base_currency]['free'] > 0:
                # Ru00e9cupu00e9rer les ordres ouverts pour trouver les SL/TP
                open_orders = self.exchange.fetch_open_orders(symbol)
                
                # Identifier les ordres SL/TP
                sl_order = next((o for o in open_orders if o.get('type') == 'stop_loss'), None)
                tp_order = next((o for o in open_orders if o.get('type') == 'take_profit'), None)
                
                # Ru00e9cupu00e9rer le prix d'entru00e9e depuis l'historique des transactions
                entry_price = 0.0
                for trade in reversed(self.trades_history):
                    if trade['symbol'] == symbol and trade['action'] == 'buy':
                        entry_price = trade['price']
                        break
                
                return [{
                    'symbol': symbol,
                    'amount': balance[base_currency]['free'],
                    'entry_price': entry_price,
                    'sl_order_id': sl_order['id'] if sl_order else None,
                    'tp_order_id': tp_order['id'] if tp_order else None
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Erreur lors de la ru00e9cupu00e9ration des positions ouvertes: {e}")
            return []
    
    def _calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Calcule la taille de position optimale selon les paramu00e8tres de risque.
        
        Args:
            symbol: Symbole de la paire
            signal: Signal de trading
            
        Returns:
            Taille de la position
        """
        try:
            # Ru00e9cupu00e9rer le solde disponible
            balance = self.exchange.fetch_balance()
            quote_currency = symbol.split('/')[1]  # USDT pour BTC/USDT
            available_balance = balance.get(quote_currency, {}).get('free', 0.0)
            
            # Paramu00e8tres de risque
            risk_per_trade = self.risk_params.get('risk_per_trade', 0.02)  # 2% du capital par trade
            max_position_size = self.risk_params.get('max_position_size', 0.1)  # 10% du capital max
            
            # Calculer la taille de position basu00e9e sur le risque
            risk_amount = available_balance * risk_per_trade
            
            # Ajuster selon la confiance du signal
            confidence_factor = signal['confidence']  # Entre 0 et 1
            adjusted_risk = risk_amount * confidence_factor
            
            # Limiter u00e0 la taille maximale
            max_amount = available_balance * max_position_size
            position_size = min(adjusted_risk, max_amount)
            
            # Convertir en quantitu00e9 de l'actif
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            quantity = position_size / current_price
            
            logger.info(f"Taille de position calculu00e9e pour {symbol}: {quantity} ({position_size} {quote_currency})")
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.0
    
    def _update_performance_metrics(self, trade: Dict[str, Any]) -> None:
        """
        Met u00e0 jour les mu00e9triques de performance avec une nouvelle transaction.
        
        Args:
            trade: Informations sur la transaction
        """
        if trade['action'] != 'sell':
            return
            
        self.performance_metrics['total_trades'] += 1
        
        if trade.get('profit', 0) > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
            
        self.performance_metrics['total_profit'] += trade.get('profit', 0)
        
        # Calculer le win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            
        # Sauvegarder les mu00e9triques
        self._save_performance_metrics()
    
    def _save_performance_metrics(self) -> None:
        """
        Sauvegarde les mu00e9triques de performance dans un fichier.
        """
        metrics_file = self.log_dir / 'performance_metrics.json'
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des mu00e9triques de performance: {e}")
    
    def _save_trade_history(self) -> None:
        """
        Sauvegarde l'historique des transactions dans un fichier.
        """
        history_file = self.log_dir / 'trade_history.json'
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.trades_history, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique des transactions: {e}")
    
    def run_single_iteration(self) -> Dict[str, Any]:
        """
        Exu00e9cute une itu00e9ration du moteur de trading.
        
        Returns:
            Ru00e9sultats de l'itu00e9ration
        """
        results = {}
        
        for symbol in self.pairs:
            try:
                logger.info(f"Traitement de {symbol}...")
                
                # Ru00e9cupu00e9rer les donnu00e9es OHLCV
                ohlcv_data = self.exchange.fetch_ohlcv(symbol, self.timeframe, 100)
                
                if ohlcv_data.empty:
                    logger.warning(f"Aucune donnu00e9e OHLCV pour {symbol}. Passage au symbole suivant.")
                    results[symbol] = {'status': 'error', 'message': 'no_data'}
                    continue
                
                # Pru00e9parer les features
                features = self._prepare_features(ohlcv_data, symbol)
                
                if not features:
                    logger.warning(f"Impossible de pru00e9parer les features pour {symbol}. Passage au symbole suivant.")
                    results[symbol] = {'status': 'error', 'message': 'feature_preparation_failed'}
                    continue
                
                # Gu00e9nu00e9rer le signal
                signal = self._generate_signal(features)
                
                # Enregistrer le signal dans l'historique
                self.signals_history[symbol] = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'signal': signal
                }
                
                # Ru00e9cupu00e9rer le prix actuel
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Exu00e9cuter le signal si le trading live est activu00e9
                execution_result = {'status': 'simulation_only'}
                if self.trading_params.get('live_trading', False):
                    execution_result = self._execute_signal(symbol, signal, current_price)
                
                results[symbol] = {
                    'status': 'success',
                    'signal': signal,
                    'current_price': current_price,
                    'execution': execution_result
                }
                
                logger.info(f"Signal pour {symbol}: {signal['action']} (confiance: {signal['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {symbol}: {e}")
                results[symbol] = {'status': 'error', 'message': str(e)}
        
        # Sauvegarder l'historique des transactions
        self._save_trade_history()
        
        return results
    
    def run_trading_loop(self, interval_seconds: int = 3600) -> None:
        """
        Exu00e9cute la boucle principale de trading.
        
        Args:
            interval_seconds: Intervalle entre les itu00e9rations en secondes
        """
        logger.info(f"Du00e9marrage de la boucle de trading avec intervalle de {interval_seconds} secondes")
        
        try:
            while True:
                start_time = time.time()
                
                logger.info("=== Du00e9but d'une nouvelle itu00e9ration de trading ===")
                results = self.run_single_iteration()
                logger.info(f"Ru00e9sultats: {json.dumps(results, indent=2)}")
                
                # Calculer le temps d'exu00e9cution
                execution_time = time.time() - start_time
                logger.info(f"Itu00e9ration terminu00e9e en {execution_time:.2f} secondes")
                
                # Attendre jusqu'au prochain intervalle
                sleep_time = max(0, interval_seconds - execution_time)
                if sleep_time > 0:
                    logger.info(f"Attente de {sleep_time:.2f} secondes jusqu'u00e0 la prochaine itu00e9ration...")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Interruption de la boucle de trading par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur dans la boucle de trading: {e}")
        finally:
            logger.info("Fin de la boucle de trading")
            self._save_trade_history()
            self._save_performance_metrics()
