import pandas as pd
import numpy as np
import os

# Create a sample DataFrame with required structure
date_range = pd.date_range(start="2023-01-01", periods=100, freq="H")
data = {
    "timestamp": date_range,
    "open": np.random.uniform(30000, 35000, 100),
    "high": np.random.uniform(30000, 35000, 100),
    "low": np.random.uniform(30000, 35000, 100),
    "close": np.random.uniform(30000, 35000, 100),
    "volume": np.random.uniform(100, 1000, 100),
}

df = pd.DataFrame(data)
df.set_index("timestamp", inplace=True)
df.to_csv("tests/fixtures/mock_data.csv", index=True)
