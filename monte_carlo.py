"""
monte_carlo.py
--------------
Phase 3: stochastic simulation, stress testing and tail-risk metrics.
Phase 4 hand-off: CSV exports for QuickSight and the Bedrock executive memo.

Model
-----
Each asset follows a multivariate Geometric Brownian Motion in discrete time.
Daily log returns are simulated as

    r_t = mu + L z_t,     z_t ~ N(0, I),   L L' = Sigma   (Cholesky)

where mu is the vector of historical MEAN LOG returns. Because mu is already a
log-return mean (= arithmetic drift - sigma^2/2), no further -0.5*sigma^2
correction is applied. (The previous version subtracted it a second time,
which biased every path downward and overstated VaR/CVaR.)

The portfolio is buy-and-hold: V_T = V_0 * sum_i w_i * exp(sum_t r_{t,i}).

Stress scenario: Sigma_s = 1.5^2 * Sigma (volatility x1.5, correlations
unchanged) and mu_s = mu - 0.001 per day (about -25.2% per year), applied for
the whole one-year horizon.

All scenarios reuse the same random draws (common random numbers), so
differences between scenarios come from the parameters, not from noise.

Risk metrics are defined on the LOSS L = V_0 - V_T:
    VaR_95  = 95th percentile of L
    CVaR_95 = E[L | L >= VaR_95]   (Expected Shortfall)

Usage:  python monte_carlo.py [--no-ai] [--no-plots] [--bootstrap-shocks]
"""

import argparse

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.stats import norm, skew, kurtosis

from data_analysis import (tickers, mean_returns, cov_matrix, log_returns,
                           TRADING_DAYS, descriptive_statistics, data_summary)
from portfolio_optimization import (optimize_portfolio, optimize_min_variance,
                                    asset_diagnostics, portfolio_summary,
                                    RISK_FREE_RATE)

# --- Simulation parameters ------------------------------------------------------
NUM_SIMULATIONS = 10_000
NUM_DAYS = TRADING_DAYS            # one-year horizon
INITIAL_INVESTMENT = 1_000_000     # SAR
CONFIDENCE = 0.95
SEED = 42
N_BOOTSTRAP = 1_000                # resamples for VaR/CVaR confidence intervals

# --- Stress parameters ----------------------------------------------------------
STRESS_VOL_MULTIPLIER = 1.5
STRESS_MEAN_SHOCK = -0.001         # per trading day

stress_mean_returns = mean_returns + STRESS_MEAN_SHOCK
stress_cov_matrix = cov_matrix * STRESS_VOL_MULTIPLIER ** 2

L_baseline = la.cholesky(cov_matrix, lower=True)
L_stress = la.cholesky(stress_cov_matrix, lower=True)


# =============================================================================
# Simulation
# =============================================================================
def draw_standard_normals(seed=SEED):
    """Common random numbers shared by every scenario: shape (sims, days, assets)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((NUM_SIMULATIONS, NUM_DAYS, len(tickers)))


def simulate_cumulative_log_returns(mu, L_matrix, Z):
    """One-year cumulative log return of each asset on each path: (sims, assets)."""
    daily = mu + Z @ L_matrix.T          # (sims, days, assets)
    return daily.sum(axis=1)


def bootstrap_cumulative_log_returns(vol_multiplier=1.0, mean_shock=0.0, seed=SEED):
    """Robustness check: resample historical daily return VECTORS (keeps fat tails
    and cross-asset dependence). Stress = de-meaned returns scaled by the vol
    multiplier, plus the shifted mean."""
    hist = log_returns.values
    mu_h = hist.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hist), size=(NUM_SIMULATIONS, NUM_DAYS))
    shocks = (hist - mu_h)[idx] * vol_multiplier
    return (mu_h + mean_shock + shocks).sum(axis=1)


def terminal_values(cum_log_returns, weights):
    """Buy-and-hold terminal portfolio value for each path."""
    return INITIAL_INVESTMENT * (np.exp(cum_log_returns) @ weights)


# =============================================================================
# Risk metrics
# =============================================================================
def var_cvar(losses, alpha=CONFIDENCE):
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    return var, cvar


def bootstrap_ci(losses, alpha=CONFIDENCE, n_boot=N_BOOTSTRAP, seed=SEED, level=0.95):
    """Percentile-bootstrap confidence intervals for VaR and CVaR."""
    rng = np.random.default_rng(seed + 1)
    n = len(losses)
    vars_, cvars_ = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        vars_[b], cvars_[b] = var_cvar(losses[rng.integers(0, n, n)], alpha)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {
        'VaR_CI_Low': np.quantile(vars_, lo), 'VaR_CI_High': np.quantile(vars_, hi),
        'VaR_Std_Error': vars_.std(ddof=1),
        'CVaR_CI_Low': np.quantile(cvars_, lo), 'CVaR_CI_High': np.quantile(cvars_, hi),
        'CVaR_Std_Error': cvars_.std(ddof=1),
    }


def parametric_var_cvar(values, alpha=CONFIDENCE):
    """Variance-based (normal) benchmark fitted to the SAME mean and standard
    deviation as the simulated P&L. Any gap to the Monte Carlo figures is
    therefore caused purely by the shape of the distribution (skew/tails)."""
    pnl = values - INITIAL_INVESTMENT
    m, s = pnl.mean(), pnl.std(ddof=1)
    z = norm.ppf(alpha)
    var = z * s - m
    cvar = s * norm.pdf(z) / (1 - alpha) - m
    return var, cvar


def risk_profile(values, label, weights, mu, cov, with_ci=True):
    losses = INITIAL_INVESTMENT - values
    var, cvar = var_cvar(losses)
    p_var, p_cvar = parametric_var_cvar(values)
    row = {
        'Scenario': label,
        'Mean_Terminal_Value': values.mean(),
        'Median_Terminal_Value': np.median(values),
        'Std_Terminal_Value': values.std(ddof=1),
        'Skewness_Terminal_Value': skew(values),
        'Excess_Kurtosis_Terminal_Value': kurtosis(values),
        'Prob_Loss': (losses > 0).mean(),
        'VaR_95': var,
        'CVaR_95': cvar,
        'Parametric_VaR_95': p_var,
        'Parametric_CVaR_95': p_cvar,
    }
    if with_ci:
        row.update(bootstrap_ci(losses))
    return row


def pct_change(new, old):
    return (new - old) / old * 100


# =============================================================================
# Pipeline
# =============================================================================
def main(run_ai=True, show_plots=True, bootstrap_shocks=False):
    pd.set_option('display.width', 160)

    # ---- Phase 1 summary --------------------------------------------------------
    info = data_summary()
    print("\n[DATA]", f"{info['first_date']} to {info['last_date']},",
          f"{info['return_observations']} daily log returns per asset")
    descriptive_statistics().to_csv('descriptive_statistics.csv')

    # ---- Phase 2: optimization ----------------------------------------------------
    print("\n[STEP 1] Optimizing portfolios (max Sharpe, rf = "
          f"{RISK_FREE_RATE:.1%}) ...")
    weights_normal, _ = optimize_portfolio(mean_returns, cov_matrix, "Baseline", display=True)
    weights_stress, top_stress_asset = optimize_portfolio(
        stress_mean_returns, stress_cov_matrix, "Stress", display=True)

    diagnostics = pd.concat([
        asset_diagnostics(mean_returns, cov_matrix, 'Baseline'),
        asset_diagnostics(stress_mean_returns, stress_cov_matrix, 'Stress'),
    ])
    diagnostics.to_csv('asset_diagnostics.csv', index=False)
    print("\n--- Per-asset diagnostics (annualized) ---")
    print(diagnostics.round(4).to_string(index=False))

    mv_base = optimize_min_variance(cov_matrix)
    mv_stress = optimize_min_variance(stress_cov_matrix)
    print("\n--- Robustness: minimum-variance weights ---")
    for t, a, b in zip(tickers, mv_base, mv_stress):
        print(f"{t:>9}: baseline {a:6.2%} | stress {b:6.2%}")
    print("Identical? ", np.allclose(mv_base, mv_stress, atol=1e-4),
          "(scaling Sigma by a constant cannot change min-variance weights)")

    pd.DataFrame([
        portfolio_summary(weights_normal, mean_returns, cov_matrix, 'Max-Sharpe | Baseline'),
        portfolio_summary(weights_stress, stress_mean_returns, stress_cov_matrix, 'Max-Sharpe | Stress'),
        portfolio_summary(mv_base, mean_returns, cov_matrix, 'Min-Variance | Baseline'),
        portfolio_summary(mv_stress, stress_mean_returns, stress_cov_matrix, 'Min-Variance | Stress'),
    ]).to_csv('portfolio_summary.csv', index=False)

    # ---- Phase 3: simulation --------------------------------------------------------
    print(f"\n[STEP 2] Simulating {NUM_SIMULATIONS:,} one-year paths (common random numbers) ...")
    Z = draw_standard_normals()
    cum_base = simulate_cumulative_log_returns(mean_returns, L_baseline, Z)
    cum_stress = simulate_cumulative_log_returns(stress_mean_returns, L_stress, Z)
    del Z

    baseline_results = terminal_values(cum_base, weights_normal)
    stress_results = terminal_values(cum_stress, weights_stress)

    print("[STEP 3] Computing risk metrics and bootstrap confidence intervals ...")
    base_row = risk_profile(baseline_results, 'Baseline (Normal)', weights_normal,
                            mean_returns, cov_matrix)
    stress_row = risk_profile(stress_results, 'Stress (Shock)', weights_stress,
                              stress_mean_returns, stress_cov_matrix)
    metrics = pd.DataFrame([base_row, stress_row])
    metrics.to_csv('risk_metrics.csv', index=False)

    var_increase = pct_change(stress_row['VaR_95'], base_row['VaR_95'])
    cvar_increase = pct_change(stress_row['CVaR_95'], base_row['CVaR_95'])

    for r in (base_row, stress_row):
        print(f"\n--- {r['Scenario']} (95% confidence level) ---")
        print(f"VaR  : SAR {r['VaR_95']:>12,.2f}   95% CI [{r['VaR_CI_Low']:,.0f}, {r['VaR_CI_High']:,.0f}]")
        print(f"CVaR : SAR {r['CVaR_95']:>12,.2f}   95% CI [{r['CVaR_CI_Low']:,.0f}, {r['CVaR_CI_High']:,.0f}]")
        print(f"Normal (variance-based) VaR SAR {r['Parametric_VaR_95']:,.2f} | "
              f"CVaR SAR {r['Parametric_CVaR_95']:,.2f}")
        print(f"Terminal value: mean {r['Mean_Terminal_Value']:,.0f} | "
              f"skewness {r['Skewness_Terminal_Value']:+.3f} | "
              f"excess kurtosis {r['Excess_Kurtosis_Terminal_Value']:+.3f} | "
              f"P(loss) {r['Prob_Loss']:.1%}")
    print(f"\nVaR change : {var_increase:+.2f}%")
    print(f"CVaR change: {cvar_increase:+.2f}%")

    # ---- 2x2 decomposition: allocation effect vs. shock effect --------------------
    print("\n[STEP 4] 2x2 decomposition (weights x scenario) ...")
    grid = []
    for w_label, w in (('Baseline-optimal weights', weights_normal),
                       ('Stress-optimal weights', weights_stress)):
        for s_label, cum in (('Baseline scenario', cum_base), ('Stress scenario', cum_stress)):
            v, c = var_cvar(INITIAL_INVESTMENT - terminal_values(cum, w))
            grid.append({'Weights': w_label, 'Scenario': s_label, 'VaR_95': v, 'CVaR_95': c})
    grid = pd.DataFrame(grid)
    grid.to_csv('risk_decomposition_2x2.csv', index=False)
    print("CVaR_95 (SAR):")
    print(grid.pivot(index='Weights', columns='Scenario', values='CVaR_95')
          .to_string(float_format=lambda x: f"{x:,.0f}"))

    cvar_basew_stress = grid.loc[(grid.Weights == 'Baseline-optimal weights') &
                                 (grid.Scenario == 'Stress scenario'), 'CVaR_95'].item()
    total_increase = stress_row['CVaR_95'] - base_row['CVaR_95']
    shock_effect = cvar_basew_stress - base_row['CVaR_95']
    allocation_effect = stress_row['CVaR_95'] - cvar_basew_stress
    print(f"Shock effect      : SAR {shock_effect:,.0f} ({shock_effect / total_increase:.1%} of the CVaR increase)")
    print(f"Allocation effect : SAR {allocation_effect:,.0f} ({allocation_effect / total_increase:.1%})")
    print("(Shock-first order; extensions.py also reports the reverse order and the Shapley average.)")

    # ---- Optional robustness: historical bootstrap shocks (non-Gaussian) ---------
    robust_cvar_stress = None
    if bootstrap_shocks:
        print("\n[ROBUSTNESS] Historical-bootstrap shocks ...")
        bb = terminal_values(bootstrap_cumulative_log_returns(), weights_normal)
        bs = terminal_values(bootstrap_cumulative_log_returns(
            STRESS_VOL_MULTIPLIER, STRESS_MEAN_SHOCK), weights_stress)
        rob = pd.DataFrame([
            risk_profile(bb, 'Baseline (Bootstrap)', weights_normal, mean_returns, cov_matrix, with_ci=False),
            risk_profile(bs, 'Stress (Bootstrap)', weights_stress, stress_mean_returns, stress_cov_matrix, with_ci=False),
        ])
        rob.to_csv('risk_metrics_bootstrap_shocks.csv', index=False)
        robust_cvar_stress = rob.loc[1, 'CVaR_95']
        print(rob[['Scenario', 'VaR_95', 'CVaR_95', 'Skewness_Terminal_Value']].round(3).to_string(index=False))

    # ---- Exports for QuickSight (same schema as before) ----------------------------
    print("\n[STEP 5] Exporting CSV files for QuickSight ...")
    pd.DataFrame({
        'Scenario': ['Baseline (Normal)'] * NUM_SIMULATIONS + ['Stress (Shock)'] * NUM_SIMULATIONS,
        'Terminal_Value': np.concatenate([baseline_results, stress_results]),
    }).to_csv('monte_carlo_results.csv', index=False)
    pd.DataFrame({'Asset': tickers, 'Normal_Weight': weights_normal.round(4),
                  'Stress_Weight': weights_stress.round(4)}).to_csv('portfolio_weights.csv', index=False)

    # ---- Phase 4: AI memo -------------------------------------------------------
    if run_ai:
        from datetime import date
        from ai_report_generator import generate_executive_report
        stress_diag = diagnostics[diagnostics.Scenario == 'Stress'].set_index('Asset')
        generate_executive_report({
            'date': date.today().strftime('%d %B %Y'),
            'risk_free': RISK_FREE_RATE,
            'stress_sharpe': portfolio_summary(weights_stress, stress_mean_returns,
                                              stress_cov_matrix, '')['Sharpe_Ratio'],
            'corner_vol': float(stress_diag.loc[top_stress_asset, 'Annual_Volatility']),
            'corner_is_highest_vol': bool(stress_diag['Annual_Volatility'].idxmax() == top_stress_asset),
            'all_negative_excess': bool((stress_diag['Excess_Return'] < 0).all()),
            'minvar_weights': dict(zip(tickers, mv_stress)),
            'var_ci_normal': (base_row['VaR_CI_Low'], base_row['VaR_CI_High']),
            'cvar_ci_normal': (base_row['CVaR_CI_Low'], base_row['CVaR_CI_High']),
            'var_ci_stress': (stress_row['VaR_CI_Low'], stress_row['VaR_CI_High']),
            'cvar_ci_stress': (stress_row['CVaR_CI_Low'], stress_row['CVaR_CI_High']),
            'prob_loss_normal': base_row['Prob_Loss'], 'prob_loss_stress': stress_row['Prob_Loss'],
            'cvar_basew_stress': cvar_basew_stress,
            'shock_effect': shock_effect, 'allocation_effect': allocation_effect,
            'shock_share': shock_effect / total_increase * 100,
            'allocation_share': allocation_effect / total_increase * 100,
            'robust_cvar_stress': robust_cvar_stress,
            'robust_cvar_diff_pct': (pct_change(robust_cvar_stress, stress_row['CVaR_95'])
                                     if robust_cvar_stress is not None else None),
            'var_normal': base_row['VaR_95'], 'cvar_normal': base_row['CVaR_95'],
            'var_stress': stress_row['VaR_95'], 'cvar_stress': stress_row['CVaR_95'],
            'var_increase_pct': var_increase, 'cvar_increase_pct': cvar_increase,
            'corner_asset': top_stress_asset,
            'stress_top_weight': float(weights_stress.max()),
            'baseline_weights': dict(zip(tickers, weights_normal)),
            'stress_weights': dict(zip(tickers, weights_stress)),
            'skew_normal': base_row['Skewness_Terminal_Value'],
            'skew_stress': stress_row['Skewness_Terminal_Value'],
        })

    # ---- Figure: terminal value distributions ----------------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(baseline_results, bins=100, color='skyblue', alpha=0.6, label='Baseline', kde=True, ax=ax)
    sns.histplot(stress_results, bins=100, color='salmon', alpha=0.6, label='Stress shock', kde=True, ax=ax)
    ax.axvline(INITIAL_INVESTMENT, color='black', ls='--', lw=2, label='Initial investment (1M SAR)')
    ax.axvline(INITIAL_INVESTMENT - base_row['VaR_95'], color='steelblue', ls=':', lw=2, label='Baseline 95% VaR level')
    ax.axvline(INITIAL_INVESTMENT - stress_row['VaR_95'], color='firebrick', ls=':', lw=2, label='Stress 95% VaR level')
    ax.set_title(f'Portfolio Terminal Value Distribution: Baseline vs. Stress ({NUM_SIMULATIONS:,} simulations)')
    all_vals = np.concatenate([baseline_results, stress_results])
    ax.set_xlim(np.quantile(all_vals, 0.001), np.quantile(all_vals, 0.999))  # trim extreme 0.1% for readability
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlabel('Portfolio value after one year (SAR)')
    ax.set_ylabel('Frequency (number of simulated paths)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig_terminal_distribution.png', dpi=200)
    if show_plots:
        plt.show()
    plt.close(fig)
    print("\n[DONE] Outputs written to the project folder.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tadawul stress-testing pipeline')
    parser.add_argument('--no-ai', action='store_true', help='skip the Bedrock memo')
    parser.add_argument('--no-plots', action='store_true', help='save figures without opening windows')
    parser.add_argument('--bootstrap-shocks', action='store_true',
                        help='also run the historical-bootstrap (non-Gaussian) robustness check')
    args = parser.parse_args()
    main(run_ai=not args.no_ai, show_plots=not args.no_plots,
         bootstrap_shocks=args.bootstrap_shocks)
