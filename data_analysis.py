"""
data_analysis.py
----------------
Phase 1: data acquisition and statistical parameterization.

- Downloads adjusted daily closes for four Tadawul equities from Yahoo Finance.
- Caches the download to `tadawul_prices.csv` so every later run (and every
  examiner who re-runs the project) uses exactly the same data snapshot.
- Computes daily log returns, the mean log-return vector (mu) and the
  covariance matrix (Sigma) that feed the optimizer and the simulation.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# --- Global parameters --------------------------------------------------------
yf_symbols = ['1120.SR', '2010.SR', '2222.SR', '7010.SR']   # Yahoo Finance symbols
tickers = ['Al Rajhi', 'SABIC', 'Aramco', 'STC']             # display names (same order)

start_date = '2023-09-14'
end_date = '2026-09-14'
TRADING_DAYS = 252

PRICE_CACHE = 'tadawul_prices.csv'


# --- Data loading -------------------------------------------------------------
def load_prices(refresh=False):
    """Return a DataFrame of adjusted closes with columns named by `tickers`.

    Uses the local cache when it exists (reproducibility); downloads otherwise.
    Pass refresh=True to force a new download.
    """
    if os.path.exists(PRICE_CACHE) and not refresh:
        print(f"Loading cached prices from {PRICE_CACHE} ...")
        prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        return prices[tickers]

    import yfinance as yf
    print("Fetching historical data from Yahoo Finance ...")
    raw = yf.download(yf_symbols, start=start_date, end=end_date,
                      threads=False, auto_adjust=True, progress=False)['Close']

    # Select columns explicitly by symbol so names can never be mismatched
    # (yfinance does not guarantee the column order of the request).
    prices = raw[yf_symbols].copy()
    prices.columns = tickers
    prices = prices.dropna(how='any')
    prices.to_csv(PRICE_CACHE)
    print(f"Saved {len(prices)} trading days to {PRICE_CACHE}")
    return prices


data = load_prices()

# --- Core statistics (exported to the other modules) --------------------------
log_returns = np.log(data / data.shift(1)).dropna()

mean_returns = log_returns.mean().values      # daily mean LOG return per asset
cov_matrix = log_returns.cov().values         # daily covariance of log returns


def descriptive_statistics():
    """Descriptive statistics table for the report (daily and annualized)."""
    daily_mean = log_returns.mean()
    daily_std = log_returns.std()
    stats = pd.DataFrame({
        'Daily Mean Log Return': daily_mean,
        'Daily Volatility': daily_std,
        'Annualized Return': daily_mean * TRADING_DAYS,
        'Annualized Volatility': daily_std * np.sqrt(TRADING_DAYS),
        'Skewness': log_returns.apply(skew),
        'Excess Kurtosis': log_returns.apply(kurtosis),   # Fisher: normal = 0
        'Min Daily Return': log_returns.min(),
        'Max Daily Return': log_returns.max(),
    })
    stats.index.name = 'Asset'
    return stats


def data_summary():
    """Sample information to state in the report's data section."""
    return {
        'source': 'Yahoo Finance (adjusted close, auto_adjust=True)',
        'symbols': dict(zip(tickers, yf_symbols)),
        'first_date': str(data.index.min().date()),
        'last_date': str(data.index.max().date()),
        'price_observations': len(data),
        'return_observations': len(log_returns),
    }


# --- Exploratory data analysis --------------------------------------------------
def run_eda(show=True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n--- Sample ---")
    for k, v in data_summary().items():
        print(f"{k}: {v}")

    stats = descriptive_statistics()
    print("\n--- Portfolio Descriptive Statistics ---")
    print(stats.round(5).to_string())
    stats.to_csv('descriptive_statistics.csv')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f",
                vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Correlation Matrix of Daily Log Returns')

    sns.histplot(log_returns, kde=True, bins=50, ax=axes[1])
    axes[1].set_title('Distribution of Daily Log Returns (Fat-Tails Check)')
    axes[1].set_xlabel('Daily Log Return')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('fig_eda.png', dpi=200)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    run_eda()
