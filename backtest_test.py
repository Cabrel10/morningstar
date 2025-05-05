import backtrader as bt
import pandas as pd
from datetime import datetime


class TestStrategy(bt.Strategy):
    def next(self):
        if len(self) == 1:  # Premier jour
            print(f"BUY ORDER PASSED at {self.data.close[0]}")
            self.buy(size=1)
        elif len(self) == 2:  # Deuxième jour
            print(f"SELL ORDER PASSED at {self.data.close[0]}")
            self.close()


# Création de données avec index temporel valide
df = pd.DataFrame(
    {
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [101, 102, 103],
        "volume": [1000, 2000, 3000],
    },
    index=pd.date_range(start="2020-01-01", periods=3),
)

data = bt.feeds.PandasData(dataname=df)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(TestStrategy)
cerebro.broker.setcash(10000)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

print("STARTING PORTFOLIO VALUE: %.2f" % cerebro.broker.getvalue())
results = cerebro.run()
print("FINAL PORTFOLIO VALUE: %.2f" % cerebro.broker.getvalue())
print("TRADES:", results[0].analyzers.trades.get_analysis())
