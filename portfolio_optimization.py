import numpy as np
import scipy.optimize as sco
import pandas as pd
from data_analysis import tickers, mean_returns, cov_matrix


def portfolio_performance(weights, mu, cov):
    returns = np.sum(mu * weights) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    return returns, std_dev


def negative_sharpe_ratio(weights, mu, cov, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mu, cov)
    return -(p_ret - risk_free_rate) / p_std


def optimize_portfolio(mu, cov, scenario_name, display=False):
    if display: print(f"\nRunning Optimization for: {scenario_name}...")
    risk_free_rate = 0.045
    num_assets = len(tickers)

    args = (mu, cov, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = sco.minimize(negative_sharpe_ratio, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    opt_weights = np.round(result.x, 4)  # Round to 4 decimals for clean data

    # Identify the highest weighted asset (for the Corner Solution AI prompt)
    max_asset_idx = np.argmax(opt_weights)
    top_asset = tickers[max_asset_idx]

    if display:
        opt_ret, opt_std = portfolio_performance(opt_weights, mu, cov)
        print(f"\n--- Optimal Portfolio Allocation: {scenario_name} ---")
        for ticker, weight in zip(tickers, opt_weights):
            print(f"{ticker}: {weight * 100:.2f}%")
        print(f"Sharpe Ratio: {(opt_ret - risk_free_rate) / opt_std:.3f}")

    return opt_weights, top_asset


if __name__ == '__main__':
    # Execution logic remains here for isolated testing
    opt_w_norm, _ = optimize_portfolio(mean_returns, cov_matrix, "Normal Market", display=True)