# Install required libraries
!pip install yfinance arch --quiet

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# Ensure plots display inline in Colab
%matplotlib inline

# Download AAPL data
ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2024-01-01"
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

# Calculate log returns (scaled by 100 for GARCH stability)
returns = np.log(df["Close"] / df["Close"].shift(1)).dropna() * 100

# Define GARCH model specifications
models = {
    "GARCH(1,1)-Normal": arch_model(returns, vol="Garch", p=1, q=1, dist="normal"),
    "GARCH(1,1)-t": arch_model(returns, vol="Garch", p=1, q=1, dist="t"),
    "EGARCH(1,1)-Normal": arch_model(returns, vol="EGarch", p=1, q=1, dist="normal")
}

# Fit models and store results
results = {}
for name, model in models.items():
    fit = model.fit(disp="off")
    results[name] = {
        "fit": fit,
        "volatility": fit.conditional_volatility * np.sqrt(252),
        "aic": fit.aic,
        "bic": fit.bic,
        "residuals": fit.std_resid
    }
    print(f"{name} - AIC: {fit.aic:.2f}, BIC: {fit.bic:.2f}")

# Plot conditional volatilities with subplots to avoid overlap
plt.figure(figsize=(14, 10))
for i, (name, res) in enumerate(results.items(), 1):
    plt.subplot(3, 1, i)
    plt.plot(res["volatility"].index, res["volatility"], label=name, alpha=0.7, linewidth=1.5)
    plt.title(f"{name} Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility (%)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout(pad=2.0)

plt.suptitle("AAPL Volatility: Comparison of GARCH Model Specifications", fontsize=14, y=1.02)
plt.show()

# Save volatility plot
plt.savefig("aapl_garch_model_comparison_subplots.png")

# Plot ACF of squared standardized residuals
plt.figure(figsize=(14, 12))
for i, (name, res) in enumerate(results.items(), 1):
    plt.subplot(len(models), 1, i)
    plot_acf(res["residuals"]**2, lags=40, ax=plt.gca(), title=f"ACF of Squared Residuals: {name}")
    plt.gca().set_ylim(-0.2, 0.2)
    plt.xlabel("Lags")
plt.tight_layout(pad=2.0)
plt.show()

# Save ACF plot
plt.savefig("aapl_garch_residuals_acf.png")

# Perform Ljung-Box test
print("\nLjung-Box Test p-values for Squared Residuals (up to 20 lags):")
for name, res in results.items():
    lb_test = acorr_ljungbox(res["residuals"]**2, lags=20, return_df=True)
    p_value = lb_test["lb_pvalue"].iloc[-1]
    print(f"{name}: p-value = {p_value:.4f} (p > 0.05 indicates no autocorrelation)")

# Plot squared residuals with transparency
plt.figure(figsize=(14, 6))
for name, res in results.items():
    plt.plot(res["residuals"].index, res["residuals"]**2, label=name, alpha=0.5, linewidth=1)
plt.title("Squared Standardized Residuals Across Models")
plt.xlabel("Date")
plt.ylabel("Squared Residuals")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig("aapl_garch_residuals_variance.png")
