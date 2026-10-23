import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# --- Step 1: Define Global Parameters ---
# Symbols used strictly for the Yahoo Finance API
yf_symbols = ['1120.SR', '2010.SR', '2222.SR', '7010.SR']

# Exported as readable names so other files and plots display names automatically
tickers = ['Al Rajhi', 'SABIC', 'Aramco', 'STC']

start_date = '2023-09-14'
end_date = '2026-09-14'

# --- Step 2: Fetch Historical Data ---
print("Fetching historical data...")
data = yf.download(yf_symbols, start=start_date, end=end_date, threads=False, auto_adjust=True)['Close']

# Rename columns from symbols to actual company names for clean EDA plots
data.columns = tickers

# --- Step 3: Core Statistical Calculations ---
# Calculate log returns for time additivity in financial modeling
log_returns = np.log(data / data.shift(1)).dropna()

# Extract mean returns and covariance matrix (Exported to other modules)
mean_returns = log_returns.mean().values
cov_matrix = log_returns.cov().values


# --- Step 4: Exploratory Data Analysis (EDA) Function ---
def run_eda():
    """Generates descriptive statistics and visualizations for the portfolio."""
    print("\n--- Portfolio Descriptive Statistics ---")
    desc_stats = pd.DataFrame({
        'Daily Mean Return': log_returns.mean(),
        'Volatility (Std Dev)': log_returns.std(),
        'Skewness': log_returns.apply(skew),
        'Kurtosis': log_returns.apply(kurtosis)
    })
    print(desc_stats)

    # Plot Correlation Matrix and Return Distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap and Histograms will now show company names automatically
    sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Correlation Matrix of Log Returns')

    sns.histplot(log_returns, kde=True, bins=50, ax=axes[1])
    axes[1].set_title('Distribution of Daily Returns (Fat Tails Check)')

    axes[1].set_xlabel('Log Returns')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# --- Step 5: Execution Control ---
# Ensures EDA only runs when this specific script is executed directly
if __name__ == '__main__':
    run_eda()