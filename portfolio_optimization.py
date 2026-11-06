"""
portfolio_optimization.py
-------------------------
Phase 2: constrained mean-variance optimization (Markowitz framework).

Primary model : maximum Sharpe ratio (tangency portfolio)
                max_w (w'mu*252 - rf) / sqrt(w'Sigma w * 252)
                s.t.  sum(w) = 1,  0 <= w_i <= 1  (fully invested, no short selling)
Check model   : global minimum variance, min_w w'Sigma w (same constraints)

Note for the report: multiplying Sigma by a constant c leaves BOTH optimal
portfolios unchanged (variance scales by c, the Sharpe ratio by 1/sqrt(c) for
every portfolio). Any change in the stressed allocation therefore comes from
the drift shock, not from the volatility multiplier. `asset_diagnostics()`
shows exactly which numbers drive the stressed solution.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco

from data_analysis import tickers, mean_returns, cov_matrix, TRADING_DAYS

RISK_FREE_RATE = 0.045     # annual; state the source in the report (e.g. SAMA / 3M SAIBOR)


def portfolio_performance(weights, mu, cov):
    """Annualized expected log return and volatility of a portfolio."""
    ret = float(np.dot(mu, weights)) * TRADING_DAYS
    vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(TRADING_DAYS)
    return ret, vol


def negative_sharpe_ratio(weights, mu, cov, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, mu, cov)
    return -(p_ret - risk_free_rate) / p_vol


def portfolio_variance(weights, cov):
    # Normalized by the average asset variance so the solver's tolerance is
    # scale-free (the optimal weights are unaffected by this constant).
    return float(weights @ cov @ weights) / float(np.mean(np.diag(cov)))


def _solve(objective, args, n):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.full(n, 1.0 / n)
    result = sco.minimize(objective, x0, args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-12, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(f"Optimizer did not converge: {result.message}")
    w = np.clip(result.x, 0, None)
    w[w < 1e-6] = 0.0            # remove numerical dust
    return w / w.sum()           # re-normalize so weights sum exactly to 1


def optimize_portfolio(mu, cov, scenario_name, display=False,
                       risk_free_rate=RISK_FREE_RATE):
    """Maximum-Sharpe weights. Returns (weights, name of the largest-weight asset)."""
    w = _solve(negative_sharpe_ratio, (mu, cov, risk_free_rate), len(tickers))
    top_asset = tickers[int(np.argmax(w))]

    ret, vol = portfolio_performance(w, mu, cov)
    sharpe = (ret - risk_free_rate) / vol
    if sharpe < 0:
        print(f"[WARNING] {scenario_name}: best attainable Sharpe ratio is negative "
              f"({sharpe:.3f}). With negative excess returns the Sharpe ratio rewards "
              f"higher volatility, so the tangency portfolio is not a meaningful "
              f"'safe' allocation in this scenario. Discuss this in the report.")

    if display:
        print(f"\n--- Max-Sharpe Allocation: {scenario_name} ---")
        for t, wt in zip(tickers, w):
            print(f"{t:>9}: {wt * 100:6.2f}%")
        print(f"Annual return {ret:.2%} | volatility {vol:.2%} | Sharpe {sharpe:.3f}")
    return w, top_asset


def optimize_min_variance(cov):
    """Global minimum-variance weights (robustness check)."""
    return _solve(portfolio_variance, (cov,), len(tickers))


def asset_diagnostics(mu, cov, scenario_name, risk_free_rate=RISK_FREE_RATE):
    """Per-asset annualized return, volatility, excess return and Sharpe ratio."""
    ann_ret = mu * TRADING_DAYS
    ann_vol = np.sqrt(np.diag(cov) * TRADING_DAYS)
    df = pd.DataFrame({
        'Scenario': scenario_name,
        'Asset': tickers,
        'Annual_Return': ann_ret,
        'Annual_Volatility': ann_vol,
        'Excess_Return': ann_ret - risk_free_rate,
        'Sharpe_Ratio': (ann_ret - risk_free_rate) / ann_vol,
    })
    return df


def portfolio_summary(weights, mu, cov, label, risk_free_rate=RISK_FREE_RATE):
    ret, vol = portfolio_performance(weights, mu, cov)
    row = {'Portfolio': label}
    row.update({f'w_{t}': w for t, w in zip(tickers, weights)})
    row.update({'Annual_Return': ret, 'Annual_Volatility': vol,
                'Sharpe_Ratio': (ret - risk_free_rate) / vol})
    return row


if __name__ == '__main__':
    optimize_portfolio(mean_returns, cov_matrix, "Normal Market", display=True)
    print("\n", asset_diagnostics(mean_returns, cov_matrix, "Baseline").round(4).to_string(index=False))
