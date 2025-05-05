import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import logging
import os
from pathlib import Path

# Configuration du logging
LOG_DIR = Path('./logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = LOG_DIR / 'backtest_optimized.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class CryptoData(bt.feeds.PandasData):
    """Classe personnalisée pour charger les données avec colonne de signal."""
    lines = ("signal",)
    params = (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("signal", "trading_signal"),
    )


class MemoryOptimizedStrategy(bt.Strategy):
    """Stratégie optimisée pour la mémoire avec gestion des stop-loss et take-profit."""
    params = (
        ("hold_period", 5),  # Période de maintien des positions
        ("printlog", False),  # Activer/désactiver les logs détaillés
        ("order_size", 0.1),  # Taille de l'ordre en BTC
        ("use_sl_tp", True),  # Utiliser SL/TP
        ("stop_loss", 0.02),  # Stop loss en % (2%)
        ("take_profit", 0.03),  # Take profit en % (3%)
        ("transaction_cost", 0.001),  # 0.1% par transaction
        ("max_positions", 1),  # Nombre max de positions simultanées
    )  

    def __init__(self):
        """Initialisation de la stratégie."""
        # Stocker les lignes de données
        self.data_close = self.data.close
        self.data_signal = self.data.lines.signal
        
        # Variables de suivi des positions
        self.order = None
        self.entry_bar = 0
        self.entry_price = 0
        self.active_positions = 0
        
        # Métriques de performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def next(self):
        """Fonction principale appelée à chaque barre."""
        # Si un ordre est en attente, ne rien faire
        if self.order:
            return
        
        # Vérifier si on a atteint les limites de position
        if self.active_positions >= self.params.max_positions:
            # Vérifier si on doit fermer une position existante
            self._check_exit_conditions()
            return
            
        # Si pas de position existante, vérifier les signaux d'entrée
        if not self.position and self.data_signal[0] > 0:  # Signal d'achat (1 ou plus)
            self._enter_long()
            
        # Vérifier les conditions de sortie pour les positions existantes
        elif self.position:
            self._check_exit_conditions()
            
    def _enter_long(self):
        """Entrer en position longue avec gestion de la taille."""
        # Calculer la taille optimale
        cash = self.broker.getcash()
        price = self.data_close[0]
        size = min(self.params.order_size, cash / price)
        
        if size <= 0:
            self.log("Pas assez de capital pour ouvrir une position")
            return
            
        # Placer l'ordre d'achat
        self.log(f"BUY CREATE {price:.2f} Size: {size:.4f} BTC")
        self.order = self.buy(size=size)
        self.entry_bar = len(self)
        self.entry_price = price
        self.active_positions += 1
        
    def _check_exit_conditions(self):
        """Vérifier les conditions de sortie: SL/TP ou période de maintien."""
        # Si SL/TP est activé
        if self.params.use_sl_tp:
            # Vérifier si stop loss est atteint
            current_price = self.data_close[0]
            
            # Pour les positions longues
            if self.position.size > 0:
                # Stop Loss
                sl_price = self.entry_price * (1 - self.params.stop_loss)
                if current_price <= sl_price:
                    self.log(f"STOP LOSS EXIT {current_price:.2f}")
                    self.order = self.close()
                    return
                    
                # Take Profit
                tp_price = self.entry_price * (1 + self.params.take_profit)
                if current_price >= tp_price:
                    self.log(f"TAKE PROFIT EXIT {current_price:.2f}")
                    self.order = self.close()
                    return
        
        # Vérifier si la période de maintien est atteinte
        if (len(self) - self.entry_bar) >= self.params.hold_period:
            self.log(f"HOLD PERIOD EXIT {self.data_close[0]:.2f}")
            self.order = self.close()
            
    def notify_order(self, order):
        """Notification des ordres."""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED {order.executed.price:.2f} Size: {order.executed.size:.4f}")
            else:  # Vente
                self.log(f"SELL EXECUTED {order.executed.price:.2f}")
                
                # Calculer le profit/perte
                profit = order.executed.pnl
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                else:
                    self.losing_trades += 1
                    self.total_loss += profit
                    
                self.total_trades += 1
                self.active_positions -= 1
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.status}")
            
        self.order = None
        
    def stop(self):
        """Appelé à la fin du backtest."""
        # Calculer et afficher les statistiques
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        
        self.log(f"Backtest terminé. Résultats:", doprint=True)
        self.log(f"  Trades totaux: {self.total_trades}", doprint=True)
        self.log(f"  Trades gagnants: {self.winning_trades} ({win_rate:.2f}%)", doprint=True)
        self.log(f"  Profit total: {self.total_profit:.2f}", doprint=True)
        self.log(f"  Perte totale: {self.total_loss:.2f}", doprint=True)
        self.log(f"  Profit moyen: {avg_profit:.2f}", doprint=True)
        self.log(f"  Perte moyenne: {avg_loss:.2f}", doprint=True)
        
    def log(self, txt, dt=None, doprint=False):
        """Fonction de logging."""
        if self.params.printlog or doprint:
            dt = dt or self.data.datetime.date(0)
            logger.info(f"{dt.isoformat()}, {txt}")


def run_backtest(
    data_path="data/processed/btc_final.parquet",
    cash=10000.0,
    hold_period=5,
    use_sl_tp=True, 
    stop_loss=0.02,
    take_profit=0.03,
    commission=0.001,
    chunksize=None  # Pour le chargement en chunks
):
    """
    Exécute le backtest avec optimisation de mémoire.
    
    Args:
        data_path: Chemin vers les données
        cash: Capital initial
        hold_period: Période de maintien des positions
        use_sl_tp: Utiliser les stop loss et take profit
        stop_loss: Pourcentage de stop loss
        take_profit: Pourcentage de take profit
        commission: Frais de commission
        chunksize: Taille des chunks pour le chargement (None pour charger tout)
    """
    logger.info(f"Démarrage du backtest optimisé: {data_path}")
    
    # Créer une instance Cerebro
    cerebro = bt.Cerebro()
    
    try:
        # Chargement des données (optimisé pour la mémoire)
        logger.info("Chargement des données...")
        
        if chunksize:
            # Chargement par chunks pour les très gros fichiers
            reader = pd.read_parquet(data_path, engine='pyarrow', chunksize=chunksize)
            dfs = []
            for chunk in reader:
                dfs.append(chunk)
            df = pd.concat(dfs)
            del dfs  # Libérer la mémoire
            gc.collect()
        else:
            # Chargement standard
            df = pd.read_parquet(data_path)
        
        # S'assurer que l'index est datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.info("Conversion de l'index en DatetimeIndex...")
            df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
        
        # Vérifier si la colonne trading_signal existe
        if "trading_signal" not in df.columns:
            logger.warning("Colonne 'trading_signal' manquante, utilisation d'un signal aléatoire pour le test")
            np.random.seed(42)
            df["trading_signal"] = np.random.choice([0, 1, 2], size=len(df), p=[0.7, 0.15, 0.15])
        
        # Optimisation de la mémoire pour le DataFrame
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        
        # Créer le feed de données
        logger.info(f"Création du feed de données: {len(df)} barres")
        data = CryptoData(dataname=df)
        cerebro.adddata(data)
        
        # Configuration
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Ajouter la stratégie
        cerebro.addstrategy(
            MemoryOptimizedStrategy,
            hold_period=hold_period,
            use_sl_tp=use_sl_tp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            transaction_cost=commission
        )
        
        # Ajouter les analyseurs
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        
        # Exécution avec suivi de la mémoire
        logger.info(f"Portefeuille initial: {cerebro.broker.getvalue():.2f}")
        
        # Exécuter le backtest
        results = cerebro.run()
        
        # Obtenir les résultats
        strat = results[0]
        
        # Afficher les résultats
        logger.info(f"Portefeuille final: {cerebro.broker.getvalue():.2f}")
        
        trade_analysis = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()
        
        logger.info("=== Résultats de l'analyse ===")
        logger.info(f"Profit total: {cerebro.broker.getvalue() - cash:.2f}")
        logger.info(f"Return: {(cerebro.broker.getvalue() / cash - 1) * 100:.2f}%")
        logger.info(f"Drawdown max: {drawdown.max.drawdown:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe.get('sharperatio', 0):.4f}")
        logger.info(f"Trades total: {trade_analysis.get('total', {}).get('total', 0)}")
        
        # Nettoyer la mémoire
        del df
        gc.collect()
        
        return {
            "final_value": cerebro.broker.getvalue(),
            "profit": cerebro.broker.getvalue() - cash,
            "return_pct": (cerebro.broker.getvalue() / cash - 1) * 100,
            "max_drawdown": drawdown.max.drawdown,
            "sharpe_ratio": sharpe.get('sharperatio', 0),
            "total_trades": trade_analysis.get('total', {}).get('total', 0),
        }
        
    except Exception as e:
        logger.error(f"Erreur pendant le backtest: {e}", exc_info=True)
        return {"error": str(e)}


def multi_parameter_test(
    data_path="data/processed/btc_final.parquet",
    cash=10000.0,
    sl_range=[0.01, 0.02, 0.03],
    tp_range=[0.02, 0.03, 0.04],
    hold_periods=[3, 5, 7]
):
    """
    Effectue des tests sur plusieurs combinaisons de paramètres.
    Optimisé pour la mémoire en exécutant un test à la fois.
    """
    results = []
    
    logger.info("Démarrage du test multi-paramètres...")
    
    for sl in sl_range:
        for tp in tp_range:
            for hp in hold_periods:
                logger.info(f"Test avec SL={sl}, TP={tp}, Hold Period={hp}")
                
                # Exécuter un backtest avec ces paramètres
                result = run_backtest(
                    data_path=data_path,
                    cash=cash,
                    hold_period=hp,
                    use_sl_tp=True,
                    stop_loss=sl,
                    take_profit=tp
                )
                
                # Ajouter les paramètres au résultat
                result["stop_loss"] = sl
                result["take_profit"] = tp
                result["hold_period"] = hp
                
                # Stocker le résultat
                results.append(result)
                
                # Forcer le garbage collection
                gc.collect()
    
    # Trouver la meilleure combinaison
    if results:
        # Trier par return
        sorted_results = sorted(results, key=lambda x: x.get("return_pct", -999), reverse=True)
        best = sorted_results[0]
        
        logger.info("=== Meilleure combinaison de paramètres ===")
        logger.info(f"Stop Loss: {best['stop_loss']}")
        logger.info(f"Take Profit: {best['take_profit']}")
        logger.info(f"Hold Period: {best['hold_period']}")
        logger.info(f"Return: {best['return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {best['sharpe_ratio']:.4f}")
        
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest optimisé pour la mémoire")
    parser.add_argument('--data', type=str, default="data/processed/btc_final.parquet", help="Chemin des données")
    parser.add_argument('--cash', type=float, default=10000.0, help="Capital initial")
    parser.add_argument('--sl', type=float, default=0.02, help="Stop Loss (%)")
    parser.add_argument('--tp', type=float, default=0.03, help="Take Profit (%)")
    parser.add_argument('--hold', type=int, default=5, help="Période de maintien")
    parser.add_argument('--no-sltp', action='store_true', help="Désactiver SL/TP")
    parser.add_argument('--multi', action='store_true', help="Exécuter des tests multi-paramètres")
    parser.add_argument('--chunksize', type=int, default=None, help="Taille des chunks pour chargement par morceaux")
    
    args = parser.parse_args()
    
    # Exécuter un backtest unique ou multi-paramètres
    if args.multi:
        multi_parameter_test(
            data_path=args.data,
            cash=args.cash,
            sl_range=[0.01, 0.02, 0.03],
            tp_range=[0.02, 0.03, 0.04],
            hold_periods=[3, 5, 7]
        )
    else:
        run_backtest(
            data_path=args.data,
            cash=args.cash,
            hold_period=args.hold,
            use_sl_tp=not args.no_sltp,
            stop_loss=args.sl,
            take_profit=args.tp,
            chunksize=args.chunksize
        )