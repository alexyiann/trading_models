!pip install yfinance arch --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Step 1: Download Apple historical price data
df = yf.download('AAPL', start='2018-01-01', end='2024-01-01')

# Step 2: Calculate log returns and scale by 100
returns = np.log(df['Close'] / df['Close'].shift(1)).dropna() * 100

# Step 3: Fit GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
garch_fit = model.fit(disp='off')

# Step 4: Calculate realized volatility (rolling std dev, annualized %)
window = 20
realized_vol = returns.rolling(window).std() * np.sqrt(252)

# Step 5: Get GARCH forecasted volatility (annualized %)
garch_fitted_vol = garch_fit.conditional_volatility * np.sqrt(252)

# Align both series for plotting
realized_vol = realized_vol[-len(garch_fitted_vol):]

# Step 6: Plot realized vs GARCH forecasted volatility
plt.figure(figsize=(14,6))
plt.plot(realized_vol.index, realized_vol, label='Realized Volatility (20-day)', color='orange')
plt.plot(garch_fitted_vol.index, garch_fitted_vol, label='GARCH Forecasted Volatility', color='blue')
plt.title('AAPL: Realized vs GARCH Forecasted Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

