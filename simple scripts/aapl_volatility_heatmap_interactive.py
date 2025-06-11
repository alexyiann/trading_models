# Install required libraries
!pip install yfinance arch ipywidgets --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from ipywidgets import interact, IntSlider
from datetime import datetime, timedelta

# Ensure plots display inline in Colab
%matplotlib inline

# Download AAPL data up to the current date
ticker = "AAPL"
start_date = "2018-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')  # Dynamic end date: June 11, 2025
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Calculate log returns (scaled by 100 for GARCH stability)
returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100

# Fit GARCH(1,1)-t model (best fit from previous analysis)
model = arch_model(returns, vol="Garch", p=1, q=1, dist="t")
fit = model.fit(disp="off")
volatility = fit.conditional_volatility * np.sqrt(252)

# Create a DataFrame for heatmap
vol_df = pd.DataFrame({
    "Date": volatility.index,
    "Volatility": volatility.values
})
vol_df["Year"] = vol_df["Date"].dt.year
vol_df["Month"] = vol_df["Date"].dt.month
heatmap_data = vol_df.pivot_table(values="Volatility", index="Month", columns="Year", aggfunc="mean")

# Heatmap visualization
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': 'Annualized Volatility (%)'})
plt.title(f"AAPL Volatility Heatmap (2018-{end_date[:4]})", fontsize=14, pad=20)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Month", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Save heatmap
plt.savefig("aapl_volatility_heatmap.png")

# Interactive volatility forecasting
def plot_forecast(horizon):
    forecast = fit.forecast(horizon=horizon)
    forecast_vol = np.sqrt(forecast.variance[-1:].values) * np.sqrt(252)
    last_date = volatility.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(volatility.index[-50:], volatility[-50:], label="Historical Volatility", color="blue", linewidth=2)
    plt.plot(forecast_dates, forecast_vol.flatten(), label="Forecasted Volatility", color="red", linestyle="--", marker="o", linewidth=2)
    plt.title(f"AAPL Volatility Forecast (Horizon: {horizon} days, as of {end_date})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Annualized Volatility (%)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create interactive slider
interact(plot_forecast, horizon=IntSlider(min=10, max=60, step=5, value=20, description="Forecast Horizon (days)"))

# Save the last plot (for GitHub, though interactive won't save directly)
plt.savefig("aapl_volatility_forecast.png")
