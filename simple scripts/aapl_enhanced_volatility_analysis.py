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
end_date = datetime.today().strftime('%Y-%m-%d')  # June 11, 2025
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Calculate log returns (scaled by 100 for GARCH stability)
returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100

# Fit multiple GARCH models
models = {
    "GARCH(1,1)-Normal": arch_model(returns, vol="Garch", p=1, q=1, dist="normal"),
    "GARCH(1,1)-t": arch_model(returns, vol="Garch", p=1, q=1, dist="t"),
    "EGARCH(1,1)-Normal": arch_model(returns, vol="Egarch", p=1, q=1, dist="normal")
}
results = {}
for name, model in models.items():
    fit = model.fit(disp="off")
    results[name] = {
        "fit": fit,
        "volatility": fit.conditional_volatility * np.sqrt(252),
        "aic": fit.aic,
        "bic": fit.bic
    }
    print(f"{name} - AIC: {fit.aic:.2f}, BIC: {fit.bic:.2f}")

# Select best model (GARCH(1,1)-t based on lowest AIC)
best_model_name = min(results, key=lambda x: results[x]["aic"])
best_fit = results[best_model_name]["fit"]
volatility = results[best_model_name]["volatility"]
std_residuals = best_fit.std_resid

# Create DataFrame for heatmap and statistics
vol_df = pd.DataFrame({
    "Date": volatility.index,
    "Volatility": volatility.values
})
vol_df["Year"] = vol_df["Date"].dt.year
vol_df["Month"] = vol_df["Date"].dt.month

# Handle any NaN values in volatility
vol_df["Volatility"] = vol_df["Volatility"].ffill()  # Replace deprecated method

# Heatmap visualization with anomalies
heatmap_data = vol_df.pivot_table(values="Volatility", index="Month", columns="Year", aggfunc="mean")
plt.figure(figsize=(14, 7))  # Increased size to accommodate annotations
sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': 'Annualized Volatility (%)'})
plt.title(f"AAPL Volatility Heatmap (2018-{end_date[:4]})", fontsize=14, pad=20)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Month", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# Highlight anomalies (volatility > mean + 2*std)
yearly_stats = vol_df.groupby("Year")["Volatility"].agg(["mean", "std"]).reset_index()
yearly_max = vol_df.groupby("Year")["Volatility"].max().reset_index(name="max")
yearly_min = vol_df.groupby("Year")["Volatility"].min().reset_index(name="min")
yearly_stats = yearly_stats.merge(yearly_max, on="Year").merge(yearly_min, on="Year")
anomalies = vol_df.merge(yearly_stats, on="Year")
anomalies = anomalies[anomalies["Volatility"] > (anomalies["mean"] + 2 * anomalies["std"])]
for idx, row in anomalies.iterrows():
    plt.text(row["Year"], row["Month"], "▲", ha="center", va="bottom", color="white", fontsize=10)
plt.subplots_adjust(left=0.2, right=0.9)  # Adjust margins to fit annotations
plt.show()

# Save heatmap
plt.savefig("aapl_volatility_heatmap_with_anomalies.png")

# Yearly volatility statistics
print("\nYearly Volatility Statistics (Annualized %):")
print(yearly_stats.round(2))

# Bar chart for yearly statistics
plt.figure(figsize=(10, 6))
bar_width = 0.8
plt.bar(yearly_stats["Year"] - bar_width/2, yearly_stats["mean"], yerr=yearly_stats["std"], capsize=5, color="skyblue", label="Mean ± Std", width=bar_width)
plt.plot(yearly_stats["Year"], yearly_stats["max"], "ro-", label="Max")
plt.plot(yearly_stats["Year"], yearly_stats["min"], "go-", label="Min")
plt.title("AAPL Yearly Volatility Statistics", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Annualized Volatility (%)", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Save bar chart
plt.savefig("aapl_yearly_volatility_stats.png")

# Interactive volatility forecasting
def plot_forecast(horizon):
    forecast = best_fit.forecast(horizon=horizon)
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

# Save the last plot
plt.savefig("aapl_volatility_forecast.png")

# Calculate VaR and ES (95% confidence level) using GARCH residuals
volatility_std = std_residuals.std() * np.sqrt(252)  # Annualized std of residuals
var_95 = -np.percentile(std_residuals, 5) * volatility_std  # 5% VaR
es_95 = -std_residuals[std_residuals <= -np.percentile(std_residuals, 5)].mean() * volatility_std  # ES

# Ensure scalar values for printing
volatility_mean = volatility.mean()
print("\nRisk Metrics (based on GARCH(1,1)-t):")
print(f"Mean Annualized Volatility: {volatility_mean:.2f}%")
print(f"VaR (95%): {var_95:.2f}% (Daily loss exceeded 5% of the time)")
print(f"ES (95%): {es_95:.2f}% (Average loss in worst 5% of cases)")
