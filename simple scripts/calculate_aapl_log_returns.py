import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download AAPL historical data (daily)
ticker = "AAPL"
df = yf.download(ticker, start="1980-01-01")

# Calculate log returns
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df.dropna(inplace=True)

# Plot log returns
plt.figure(figsize=(12, 5))
plt.plot(df["LogReturn"], color='blue', alpha=0.7)
plt.title("AAPL Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.grid(True)
plt.show()
