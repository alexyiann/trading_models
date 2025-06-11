# Install required libraries
!pip install yfinance arch --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Ensure plots display inline in Colab
%matplotlib inline

# Define tickers and date range
tickers = ["AAPL", "BNS", "GOOGL", "JPM"]  # Added JPM (financial sector)
start_date = "2018-01-01"
end_date = "2024-01-01"

# Initialize dictionary to store volatility data
vol_data = {}

# Download data, calculate returns, and fit GARCH model for each ticker
for ticker in tickers:
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Calculate log returns (scaled by 100 for GARCH stability)
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100
    
    # Fit GARCH(1,1) model
    model = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    fit = model.fit(disp="off")
    
    # Store annualized conditional volatility
    vol_data[ticker] = fit.conditional_volatility * np.sqrt(252)

# Combine volatilities into a DataFrame
vol_df = pd.DataFrame(vol_data)

# Calculate and print mean volatility for each stock
print("Mean Annualized Volatility (%):")
for ticker in tickers:
    print(f"{ticker}: {vol_df[ticker].mean():.2f}%")

# Calculate and print volatility correlation matrix
print("\nVolatility Correlation Matrix:")
print(vol_df.corr().round(2))

# Plot volatilities
plt.figure(figsize=(14, 6))
for ticker in tickers:
    plt.plot(vol_df.index, vol_df[ticker], label=f"{ticker} GARCH Volatility")
plt.title("GARCH(1,1) Volatility Comparison: Tech vs Financial Sectors")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save plot for GitHub
plt.savefig("multi_stock_volatility_plot.png")
