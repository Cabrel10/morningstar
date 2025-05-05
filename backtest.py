import backtrader as bt
import pandas as pd
from datetime import datetime


class CryptoData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("signal", "trading_signal"),
    )


class TestStrategy(bt.Strategy):
    params = (("hold_period", 5), ("printlog", True), ("order_size", 0.1))  # 0.1 BTC par trade

    def __init__(self):
        self.data_close = self.data.close
        self.data_signal = self.data.lines.signal
        self.order = None
        self.entry_bar = 0

    def next(self):
        if self.order:
            return

        if not self.position and self.data_signal[0] == 1:
            cash = self.broker.getcash()
            price = self.data_close[0]
            size = min(self.params.order_size, cash / price)

            self.log(f"BUY ATTEMPT {price:.2f} Size: {size:.4f} BTC")
            self.order = self.buy(size=size)
            self.entry_bar = len(self)

        elif self.position and (len(self) - self.entry_bar) >= self.params.hold_period:
            self.log(f"SELL CREATE {self.data_close[0]:.2f}")
            self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED {order.executed.price:.2f} Size: {order.executed.size:.4f}")
            else:
                self.log(f"SELL EXECUTED {order.executed.price:.2f}")

        self.order = None

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.data.datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")


if __name__ == "__main__":
    cerebro = bt.Cerebro()

    # Chargement des données
    df = pd.read_parquet("data/processed/btc_final.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

    data = CryptoData(dataname=df)
    cerebro.adddata(data)

    # Configuration
    cerebro.broker.setcash(10000.0)
    cerebro.addstrategy(TestStrategy)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Exécution
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Analyse
    trades = results[0].analyzers.trades.get_analysis()
    print(f"Total Trades: {trades.get('total', {}).get('total', 0)}")
