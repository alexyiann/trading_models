import numpy as np
import pandas as pd
from arch import arch_model
import yfinance as yf

# Load AAPL price data
df = yf.download("AAPL", start="2010-01-01")
log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
# Fit the GARCH(1,1) model
model = arch_model(log_returns * 100, vol='GARCH', p=1, q=1)
model_fit = model.fit(disp="off")

# Print results
print(model_fit.summary())
# Forecast volatility
forecast = model_fit.forecast(horizon=20)
forecast_vol = forecast.variance[-1:].T

# Plot forecasted volatility
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(forecast_vol.values.flatten(), marker='o')
plt.title("20-Day GARCH(1,1) Volatility Forecast")
plt.xlabel("Days Ahead")
plt.ylabel("Forecasted Variance (%)")
plt.grid(True)
plt.show()
