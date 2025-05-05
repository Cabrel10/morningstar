import backtrader as bt
import pandas as pd
from datetime import datetime


class TestStrategy(bt.Strategy):
    def next(self):
        print(f"Bar: {len(self)}, Close: {self.data.close[0]}")
        if len(self) == 1:
            print("Achat")
            self.buy(size=1)
        elif len(self) == 2:
            print("Vente")
            self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            print(f"Order Submitted/Accepted: {order.status}")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            else:  # Sell
                print(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None


if __name__ == "__main__":
    cerebro = bt.Cerebro()

    # Load data using Pandas
    df = pd.read_parquet("data/processed/btc_final.parquet")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

    # Create a Data Feed
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    cerebro.run()
