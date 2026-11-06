"""
extensions.py
-------------
Additional analyses requested in the supervisor's review. Uses the same cached
data, parameters and random seed as the main pipeline, so the baseline numbers
match monte_carlo.py exactly.

    python extensions.py               # run everything
    python extensions.py --skip-2020   # skip the historical calibration downloads

Analyses
  A. Order-independent risk decomposition (both orders + Shapley average)
  B. Alternative portfolio constructions, incl. a cash option:
       max-Sharpe, minimum variance, minimum CVaR (Rockafellar-Uryasev LP),
       CVaR-budgeted portfolio with cash, 100% cash.
     Portfolios are optimized on one set of simulated scenarios and evaluated
     on an independent set (out-of-sample), so CVaR is not understated.
  C. Stress sensitivity grid (drift shock x volatility multiplier)
  D. Crisis-correlation scenario (correlations pushed halfway towards 1)
  E. Calibration against historical crisis windows: the Feb-Jun 2020 COVID / oil-price
     crash and the 2026 regional conflict (1 March 2026 to the latest available day)
  F. Block bootstrap (preserves volatility clustering), several block lengths
  G. Parameter uncertainty: block bootstrap of the historical window,
     re-estimating mu and Sigma and re-optimizing each time; compares how
     stable max-Sharpe, minimum-variance and minimum-CVaR weights are, and how
     estimation error affects their tail risk; adds a resampled (averaged)
     max-Sharpe portfolio (Michaud, 1989)
  H. Out-of-sample 1-day VaR backtest with Kupiec and Christoffersen tests

All results are written to CSV files, figures to PNG, and the full console
output to extensions_output.txt (send that file plus the PNGs).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.special import xlogy
from scipy.stats import chi2, norm

from data_analysis import tickers, yf_symbols, mean_returns, cov_matrix, log_returns, TRADING_DAYS
from portfolio_optimization import _solve, negative_sharpe_ratio, optimize_min_variance, RISK_FREE_RATE
from monte_carlo import (NUM_SIMULATIONS, NUM_DAYS, INITIAL_INVESTMENT, CONFIDENCE, SEED,
                         STRESS_VOL_MULTIPLIER, STRESS_MEAN_SHOCK, var_cvar)

V0 = INITIAL_INVESTMENT
ALPHA = CONFIDENCE
CASH_GROWTH = 1.0 + RISK_FREE_RATE          # one-year growth factor of the risk-free asset
OPT_SEED = SEED + 1000                     # independent scenarios used only for optimization
N_ASSETS = len(tickers)

GRID_DRIFT = [0.0, -0.0005, -0.001, -0.0015]     # daily drift shocks
GRID_VOL = [1.0, 1.25, 1.5, 2.0]                 # volatility multipliers
CRISIS_CORR_BLEND = 0.5                          # correlations moved 50% of the way to 1
# Historical calibration windows: (label, first return date, last date or None = latest, price cache)
# Prices are downloaded once per window and cached so every later run uses the same data.
CALIBRATION_WINDOWS = [
    ('2020 COVID-19 and oil-price crash', '2020-02-20', '2020-06-30', 'tadawul_prices_2020.csv'),
    ('2026 regional conflict (since late February 2026)', '2026-03-01', None, 'tadawul_prices_2026.csv'),
]
BLOCK_LENGTHS = [1, 5, 20, 60]                   # 1 = i.i.d. bootstrap
PARAM_BOOT_REPS = 500
PARAM_BOOT_BLOCK = 20
PARAM_BOOT_LP_SCENARIOS = 2000                   # scenarios per min-CVaR LP inside the bootstrap
BACKTEST_WINDOW = 250

BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2, GRID_C = '#0b0b0b', '#52514e', '#e4e3df'


class Tee:
    """Print to the console and to a log file at the same time."""
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.console = sys.stdout

    def write(self, s):
        self.console.write(s)
        self.file.write(s)

    def flush(self):
        self.console.flush()
        self.file.flush()


def section(title):
    print('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)


# =============================================================================
# Simulation helpers
# =============================================================================
def summed_normals(seed, n=NUM_SIMULATIONS):
    """Sum over the 252 days of the i.i.d. N(0, I) shocks.

    Uses exactly the same random draws as monte_carlo.py (same generator, seed
    and shape). Because the model is linear in the shocks,
    sum_t (mu + L z_t) = 252 mu + L sum_t z_t, so this reproduces the main
    results without storing every daily path.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, NUM_DAYS, N_ASSETS)).sum(axis=1)


def growth(mu, cov, zsum):
    """One-year gross growth factor of each asset on each path (paths x assets)."""
    L = la.cholesky(cov, lower=True)
    return np.exp(NUM_DAYS * mu + zsum @ L.T)


def with_cash(G):
    return np.column_stack([G, np.full(len(G), CASH_GROWTH)])


def metrics(values):
    losses = V0 - values
    var, cvar = var_cvar(losses, ALPHA)
    return {'Mean_Terminal_Value': values.mean(), 'VaR_95': var, 'CVaR_95': cvar,
            'Prob_Loss': (losses > 0).mean()}


def max_sharpe(mu, cov):
    return _solve(negative_sharpe_ratio, (mu, cov, RISK_FREE_RATE), len(mu))


def stress_params(mu, cov, drift=STRESS_MEAN_SHOCK, vol=STRESS_VOL_MULTIPLIER):
    return mu + drift, cov * vol ** 2


def fmt_w(w, names):
    return ', '.join(f'{n} {x:.1%}' for n, x in zip(names, w) if x > 0.0005)


# =============================================================================
# Rockafellar-Uryasev linear programs
# =============================================================================
def ru_optimize(G, objective='min_cvar', cvar_budget=None, alpha=ALPHA):
    """CVaR optimization on scenario growth factors G (S scenarios x k assets).

    Works with unit capital: loss on scenario s is 1 - G_s . w.
    Variables x = [w (k), zeta (1), u (S)], u_s >= loss_s - zeta, u_s >= 0.
      min_cvar  : minimize  zeta + sum(u) / ((1 - alpha) S)
      max_return: maximize  mean(G) . w   s.t.  zeta + sum(u) / ((1 - alpha) S) <= budget
    Long-only, fully invested (sum w = 1).
    """
    S, k = G.shape
    cvar_row = np.concatenate([np.zeros(k), [1.0], np.full(S, 1.0 / ((1 - alpha) * S))])
    # -G_s w - zeta - u_s <= -1
    A = hstack([csr_matrix(-G), csr_matrix(-np.ones((S, 1))), -eye(S, format='csr')], format='csr')
    b = np.full(S, -1.0)
    if objective == 'min_cvar':
        c = cvar_row
    elif objective == 'max_return':
        c = np.concatenate([-G.mean(axis=0), [0.0], np.zeros(S)])
        A = vstack([A, csr_matrix(cvar_row[None, :])], format='csr')
        b = np.append(b, cvar_budget)
    else:
        raise ValueError(objective)
    A_eq = csr_matrix(np.concatenate([np.ones(k), [0.0], np.zeros(S)])[None, :])
    bounds = [(0, None)] * k + [(None, None)] + [(0, None)] * S
    res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method='highs')
    if res.status != 0:
        raise RuntimeError(f'LP failed: {res.message}')
    w = np.clip(res.x[:k], 0, None)
    w[w < 1e-6] = 0.0
    return w / w.sum()


# =============================================================================
# A. Decomposition
# =============================================================================
def decomposition(c_bb, c_bs, c_sb, c_ss):
    """c_xy = CVaR with x-optimal weights in scenario y (b = baseline, s = stress)."""
    total = c_ss - c_bb
    shock_a, alloc_a = c_bs - c_bb, c_ss - c_bs           # shock first
    alloc_b, shock_b = c_sb - c_bb, c_ss - c_sb           # allocation first
    shock_sh, alloc_sh = (shock_a + shock_b) / 2, (alloc_a + alloc_b) / 2
    rows = [('Shock first', shock_a, alloc_a), ('Allocation first', shock_b, alloc_b),
            ('Shapley average', shock_sh, alloc_sh)]
    df = pd.DataFrame([{'Order': o, 'Shock_Effect': s, 'Shock_Share': s / total,
                        'Allocation_Effect': a, 'Allocation_Share': a / total,
                        'Total_Increase': total} for o, s, a in rows])
    return df


# =============================================================================
# F. Block bootstrap
# =============================================================================
def block_bootstrap_growth(block, vol_mult=1.0, mean_shock=0.0, seed=SEED, n=NUM_SIMULATIONS):
    """Circular block bootstrap of de-meaned historical daily return vectors.

    Blocks of `block` consecutive days keep volatility clustering and short-run
    dependence; block = 1 is the i.i.d. bootstrap. Returns growth factors (n x assets).
    """
    hist = log_returns.values
    mu_h = hist.mean(axis=0)
    dem = hist - mu_h
    T = len(dem)
    ext = np.vstack([dem, dem[:block]])                       # circular wrap
    cs = np.vstack([np.zeros((1, N_ASSETS)), np.cumsum(ext, axis=0)])
    n_blocks = int(np.ceil(NUM_DAYS / block))
    lengths = np.full(n_blocks, block)
    lengths[-1] = NUM_DAYS - (n_blocks - 1) * block
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T, size=(n, n_blocks))
    total = np.zeros((n, N_ASSETS))
    for j in range(n_blocks):
        s = starts[:, j]
        total += cs[s + lengths[j]] - cs[s]
    return np.exp(NUM_DAYS * (mu_h + mean_shock) + vol_mult * total)


def resample_history(rng, block=PARAM_BOOT_BLOCK):
    """One circular block-bootstrap sample of the historical return matrix."""
    hist = log_returns.values
    T = len(hist)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = ((starts[:, None] + np.arange(block)) % T).ravel()[:T]
    return hist[idx]


# =============================================================================
# H. Backtest tests
# =============================================================================
def kupiec(hits, p):
    T, x = len(hits), int(hits.sum())
    phat = x / T
    lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p) - xlogy(T - x, 1 - phat) - xlogy(x, phat))
    return lr, chi2.sf(lr, 1)


def christoffersen(hits):
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return lr, chi2.sf(lr, 1)


# =============================================================================
# Figures
# =============================================================================
def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID_C)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)


def fig_weights(w_base, w_stress, path='fig_weights.png'):
    import matplotlib.pyplot as plt
    x = np.arange(N_ASSETS)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    bw = 0.36
    for off, w, col, lab in ((-bw / 2 - 0.01, w_base, BLUE, 'Baseline'), (bw / 2 + 0.01, w_stress, ORANGE, 'Stress')):
        bars = ax.bar(x + off, w * 100, bw, color=col, label=lab, edgecolor='white', linewidth=1)
        for b_, v in zip(bars, w):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 100 + 1.5, f'{v:.1%}', ha='center', va='bottom',
                    fontsize=8.5, color=INK)
    ax.set_xticks(x, tickers)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Portfolio weight (%)', color=INK2, fontsize=9)
    ax.set_title('Maximum-Sharpe weights: baseline vs. stress', loc='left', fontsize=11, color=INK, pad=14)
    _style(ax)
    # legend above the plot area, so it cannot overlap bars or value labels
    ax.legend(frameon=False, fontsize=9, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2,
              borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_strategies(df, path='fig_strategies.png'):
    import matplotlib.pyplot as plt
    st = df[df.Scenario == 'Stress'].reset_index(drop=True)
    names = list(st.Strategy)
    cols = tickers + ['Cash']
    colors = [BLUE, ORANGE, AQUA, '#e87ba4', GREY]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=200, gridspec_kw={'width_ratios': [1.15, 1]})
    left = np.zeros(len(st))
    for c, col in zip(cols, colors):
        v = st[f'w_{c}'].values * 100
        a1.barh(names, v, left=left, color=col, label=c, edgecolor='white', linewidth=1.5)
        for i, (l_, vv) in enumerate(zip(left, v)):
            if vv >= 9:
                a1.text(l_ + vv / 2, i, f'{vv:.0f}%', ha='center', va='center', fontsize=8,
                        color='white' if col in (BLUE, GREY) else INK)
        left += v
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel('Weight under stress (%)', color=INK2, fontsize=9)
    a1.set_title('Stress-scenario allocation', loc='left', fontsize=11, color=INK)
    _style(a1); a1.yaxis.grid(False); a1.xaxis.grid(True, color=GRID_C, lw=0.6)
    a1.legend(frameon=False, fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    cv = st.CVaR_95.values / 1000
    bars = a2.barh(names, cv, color=[INK2 if n == 'Max Sharpe' else '#c3c2b7' for n in names], height=0.6)
    for b_, v in zip(bars, cv):
        x_ = v if v > 0 else 0
        a2.text(x_ + 8, b_.get_y() + b_.get_height() / 2, f'{v:,.0f}k', va='center', fontsize=8.5, color=INK)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.invert_yaxis()
    a2.set_yticks(range(len(names)), [''] * len(names))
    a2.tick_params(axis='y', length=0)
    a2.set_xlabel('95% CVaR under stress (thousand SAR, out-of-sample)', color=INK2, fontsize=9)
    a2.set_title('Stressed tail risk', loc='left', fontsize=11, color=INK)
    a2.set_xlim(min(0, cv.min()) - 60, cv.max() * 1.18)
    _style(a2); a2.yaxis.grid(False); a2.xaxis.grid(True, color=GRID_C, lw=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(grid, path='fig_sensitivity.png'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    piv_c = grid.pivot(index='Drift_Annual_pp', columns='Vol_Multiplier', values='CVaR_MaxSharpe') / 1000
    piv_a = grid.pivot(index='Drift_Annual_pp', columns='Vol_Multiplier', values='Allocation_Effect') / 1000
    piv_c = piv_c.sort_index(ascending=False); piv_a = piv_a.loc[piv_c.index]
    seq = LinearSegmentedColormap.from_list('seq', ['#fdf0ea', '#f6b799', ORANGE, '#a63f12'])
    div = LinearSegmentedColormap.from_list('div', [BLUE, '#f2f1ed', ORANGE])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    lim = np.abs(piv_a.values).max()
    for ax, piv, cmap, title, vmin, vmax in (
            (axes[0], piv_c, seq, '95% CVaR of the max-Sharpe portfolio (thousand SAR)', 0, piv_c.values.max()),
            (axes[1], piv_a, div, 'Allocation effect: re-optimized minus baseline weights', -lim, lim)):
        im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                dark = (v - vmin) / (vmax - vmin + 1e-12) > 0.62 if cmap is seq else abs(v) / (lim + 1e-12) > 0.7
                ax.text(j, i, f'{v:+,.0f}' if cmap is div else f'{v:,.0f}', ha='center', va='center',
                        fontsize=9, color='white' if dark else INK)
        ax.set_xticks(range(piv.shape[1]), [f'×{v:g}' for v in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f'{v:+.1f} pp' if v else '0 pp' for v in piv.index])
        ax.set_xlabel('Volatility multiplier', color=INK2, fontsize=9)
        ax.set_ylabel('Drift shock (annual)', color=INK2, fontsize=9)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_backtest(bt, path='fig_backtest.png'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.1), dpi=200)
    ax.plot(bt.index, bt.Realized * 100, color=GREY, lw=0.8, label='Realized daily portfolio return')
    ax.plot(bt.index, -bt.VaR_Normal_95 * 100, color=BLUE, lw=1.4, label='−95% VaR (normal, rolling 250 days)')
    ax.plot(bt.index, -bt.VaR_Normal_99 * 100, color=ORANGE, lw=1.4, label='−99% VaR (normal, rolling 250 days)')
    hit = bt[bt.Hit_Normal_99]
    ax.scatter(hit.index, hit.Realized * 100, s=22, color=ORANGE, zorder=3, edgecolor='white', linewidth=0.8,
               label='99% VaR exceedance')
    ax.set_ylabel('Daily log return (%)', color=INK2, fontsize=9)
    ax.set_title('Out-of-sample 1-day VaR backtest (weights re-optimized daily on past data only)',
                 loc='left', fontsize=11, color=INK)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_weight_stability(pb, path='fig_weight_stability.png'):
    """5th-95th percentile range and median of baseline weights across resamples."""
    import matplotlib.pyplot as plt
    specs = [('Max Sharpe', 'wb', ORANGE), ('Minimum variance', 'wmv', AQUA), ('Minimum CVaR', 'wmcb', BLUE)]
    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=200)
    ypos = np.arange(N_ASSETS)
    for k, (lab, pre, col) in enumerate(specs):
        y = ypos + (k - 1) * 0.24
        for i, t in enumerate(tickers):
            c = pb[f'{pre}_{t}'] * 100
            ax.plot([c.quantile(0.05), c.quantile(0.95)], [y[i]] * 2, color=col, lw=5, alpha=0.35,
                    solid_capstyle='round')
            ax.plot(c.median(), y[i], 'o', color=col, ms=7, mec='white', mew=1.2,
                    label=lab if i == 0 else None)
    ax.set_yticks(ypos, tickers)
    ax.invert_yaxis()
    ax.set_xlim(-3, 103)
    ax.set_xlabel('Baseline weight across 500 resamples (%): dot = median, bar = 5th–95th percentile',
                  color=INK2, fontsize=9)
    ax.set_title('How much the optimal weights move when the inputs are re-estimated', loc='left',
                 fontsize=11, color=INK)
    _style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID_C, lw=0.6)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(skip_2020=False):
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    out = {}

    zs_eval = summed_normals(SEED)          # identical draws to monte_carlo.py
    zs_opt = summed_normals(OPT_SEED)       # independent draws, optimization only

    mu_b, cov_b = mean_returns, cov_matrix
    mu_s, cov_s = stress_params(mu_b, cov_b)
    w_base = max_sharpe(mu_b, cov_b)
    w_stress = max_sharpe(mu_s, cov_s)
    G_be, G_se = growth(mu_b, cov_b, zs_eval), growth(mu_s, cov_s, zs_eval)
    G_bo, G_so = growth(mu_b, cov_b, zs_opt), growth(mu_s, cov_s, zs_opt)

    # ---------------------------------------------------------------- A
    section('A. Risk decomposition: both orders and Shapley average (95% CVaR, SAR)')
    c = {k: metrics(V0 * G @ w)['CVaR_95'] for k, G, w in
         (('bb', G_be, w_base), ('bs', G_se, w_base), ('sb', G_be, w_stress), ('ss', G_se, w_stress))}
    print(f"Baseline weights: baseline {c['bb']:,.0f} | stress {c['bs']:,.0f}")
    print(f"Stress weights  : baseline {c['sb']:,.0f} | stress {c['ss']:,.0f}")
    dec = decomposition(c['bb'], c['bs'], c['sb'], c['ss'])
    dec.to_csv('decomposition_shapley.csv', index=False)
    for _, r in dec.iterrows():
        print(f"{r.Order:17s} shock {r.Shock_Effect:>10,.0f} ({r.Shock_Share:.1%}) | "
              f"allocation {r.Allocation_Effect:>9,.0f} ({r.Allocation_Share:.1%})")

    # ---------------------------------------------------------------- B
    section('B. Alternative portfolio constructions (optimized on independent scenarios,\n'
            '   evaluated out-of-sample on the main 10,000 paths)')
    budget_unit = c['bb'] / V0                     # keep tail risk at the baseline level
    rows = []
    names = tickers + ['Cash']
    for scen, mu, cov, G_opt, G_ev in (('Baseline', mu_b, cov_b, G_bo, G_be), ('Stress', mu_s, cov_s, G_so, G_se)):
        strategies = {
            'Max Sharpe': np.append(max_sharpe(mu, cov), 0.0),
            'Minimum variance': np.append(optimize_min_variance(cov), 0.0),
            'Minimum CVaR': np.append(ru_optimize(G_opt, 'min_cvar'), 0.0),
            'CVaR budget + cash': ru_optimize(with_cash(G_opt), 'max_return', cvar_budget=budget_unit),
            '100% cash': np.array([0.0] * N_ASSETS + [1.0]),
        }
        for name, w in strategies.items():
            m = metrics(V0 * with_cash(G_ev) @ w)
            row = {'Scenario': scen, 'Strategy': name}
            row.update({f'w_{n}': x for n, x in zip(names, w)})
            row.update(m)
            rows.append(row)
    strat = pd.DataFrame(rows)
    strat.to_csv('strategy_comparison.csv', index=False)
    for scen in ('Baseline', 'Stress'):
        print(f'\n--- {scen} scenario ---')
        for _, r in strat[strat.Scenario == scen].iterrows():
            w = [r[f'w_{n}'] for n in names]
            print(f"{r.Strategy:20s} mean {r.Mean_Terminal_Value:>11,.0f} | VaR {r.VaR_95:>10,.0f} | "
                  f"CVaR {r.CVaR_95:>10,.0f} | P(loss) {r.Prob_Loss:6.1%} | {fmt_w(w, names)}")
    print(f'\nCVaR budget = baseline max-Sharpe CVaR = {c["bb"]:,.0f} SAR; cash earns {RISK_FREE_RATE:.1%}.')

    # ---------------------------------------------------------------- C
    section('C. Stress sensitivity grid (max-Sharpe re-optimized in every cell)')
    grid_rows = []
    for d in GRID_DRIFT:
        for v in GRID_VOL:
            mu_g, cov_g = stress_params(mu_b, cov_b, d, v)
            w_g = max_sharpe(mu_g, cov_g)
            G_g = growth(mu_g, cov_g, zs_eval)
            w_mc = ru_optimize(growth(mu_g, cov_g, zs_opt), 'min_cvar')
            m_opt, m_fix, m_mc = metrics(V0 * G_g @ w_g), metrics(V0 * G_g @ w_base), metrics(V0 * G_g @ w_mc)
            row = {'Drift_Daily': d, 'Drift_Annual_pp': d * TRADING_DAYS * 100, 'Vol_Multiplier': v}
            row.update({f'w_{t}': x for t, x in zip(tickers, w_g)})
            row.update({'CVaR_MaxSharpe': m_opt['CVaR_95'], 'CVaR_BaselineWeights': m_fix['CVaR_95'],
                        'Allocation_Effect': m_opt['CVaR_95'] - m_fix['CVaR_95'],
                        'CVaR_MinCVaR': m_mc['CVaR_95'], 'VaR_MaxSharpe': m_opt['VaR_95'],
                        'Prob_Loss_MaxSharpe': m_opt['Prob_Loss'], 'Top_Weight': w_g.max(),
                        'Top_Asset': tickers[int(np.argmax(w_g))]})
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv('sensitivity_grid.csv', index=False)
    print(grid[['Drift_Annual_pp', 'Vol_Multiplier', 'Top_Asset', 'Top_Weight', 'CVaR_MaxSharpe',
                'CVaR_BaselineWeights', 'Allocation_Effect', 'CVaR_MinCVaR']]
          .to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    print('\nMax-Sharpe weights by drift shock (identical across volatility multipliers):')
    for d in GRID_DRIFT:
        r = grid[(grid.Drift_Daily == d)].iloc[0]
        same = np.allclose(grid[grid.Drift_Daily == d][[f'w_{t}' for t in tickers]].values,
                           r[[f'w_{t}' for t in tickers]].values.astype(float), atol=1e-4)
        print(f"  drift {d * TRADING_DAYS * 100:+6.1f} pp: {fmt_w([r[f'w_{t}'] for t in tickers], tickers)}"
              f"   (same for all multipliers: {same})")

    # ---------------------------------------------------------------- D
    section(f'D. Crisis-correlation scenario (correlations moved {CRISIS_CORR_BLEND:.0%} towards 1,\n'
            f'   volatility x{STRESS_VOL_MULTIPLIER}, drift {STRESS_MEAN_SHOCK}/day)')
    sd = np.sqrt(np.diag(cov_b))
    C = cov_b / np.outer(sd, sd)
    C_c = (1 - CRISIS_CORR_BLEND) * C + CRISIS_CORR_BLEND * np.ones_like(C)
    cov_c = np.outer(sd * STRESS_VOL_MULTIPLIER, sd * STRESS_VOL_MULTIPLIER) * C_c
    off = ~np.eye(N_ASSETS, dtype=bool)
    print(f'Average pairwise correlation: {C[off].mean():.2f} -> {C_c[off].mean():.2f}')
    scen_rows = []

    def run_scenario(label, mu_x, cov_x):
        w_x = max_sharpe(mu_x, cov_x)
        G_x = growth(mu_x, cov_x, zs_eval)
        w_mc = ru_optimize(growth(mu_x, cov_x, zs_opt), 'min_cvar')
        r = {'Scenario': label}
        r.update({f'w_{t}': x for t, x in zip(tickers, w_x)})
        for tag, w in (('MaxSharpe', w_x), ('BaselineWeights', w_base), ('MinCVaR', w_mc)):
            m = metrics(V0 * G_x @ w)
            r.update({f'VaR_{tag}': m['VaR_95'], f'CVaR_{tag}': m['CVaR_95'], f'PLoss_{tag}': m['Prob_Loss']})
        r.update({f'wMinCVaR_{t}': x for t, x in zip(tickers, w_mc)})
        scen_rows.append(r)
        print(f"\n{label}\n  max-Sharpe weights: {fmt_w(w_x, tickers)}\n  min-CVaR weights  : {fmt_w(w_mc, tickers)}")
        print(f"  CVaR: max-Sharpe {r['CVaR_MaxSharpe']:,.0f} | baseline weights {r['CVaR_BaselineWeights']:,.0f} | "
              f"min-CVaR {r['CVaR_MinCVaR']:,.0f}   (P(loss) max-Sharpe {r['PLoss_MaxSharpe']:.1%})")

    run_scenario('Main stress scenario (correlations unchanged)', mu_s, cov_s)
    run_scenario('Crisis-correlation stress scenario', mu_s, cov_c)

    # ---------------------------------------------------------------- E
    section('E. Calibration against historical crisis windows')
    cal_rows = []
    for label, start, end, cache in CALIBRATION_WINDOWS:
        print(f'\n--- {label}: {start} to {end or "latest available"} ---')
        if skip_2020:
            print('Skipped (--skip-2020: no calibration downloads).')
            continue
        try:
            if os.path.exists(cache):
                px = pd.read_csv(cache, index_col=0, parse_dates=True)[tickers]
            else:
                import yfinance as yf
                dl_start = str((pd.Timestamp(start) - pd.Timedelta(days=20)).date())
                dl_end = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()) if end else None
                raw = yf.download(yf_symbols, start=dl_start, end=dl_end, threads=False,
                                  auto_adjust=True, progress=False)['Close']
                px = raw[yf_symbols].copy()
                px.columns = tickers
                px = px.dropna(how='any')
                px.to_csv(cache)
            r_c = np.log(px / px.shift(1)).dropna().loc[start:end]
            print(f'Window covers {r_c.index.min().date()} to {r_c.index.max().date()}.')
            if len(r_c) < 25:
                raise ValueError(f'only {len(r_c)} daily returns in the window')
            mu_c, cov_cr = r_c.mean().values, r_c.cov().values
            vol_ratio = np.sqrt(np.diag(cov_cr) / np.diag(cov_b))
            sd_c = np.sqrt(np.diag(cov_cr))
            Ccr = cov_cr / np.outer(sd_c, sd_c)
            cum = r_c.sum().values
            cal = pd.DataFrame({'Window': label, 'Asset': tickers, 'Days': len(r_c),
                                'Cumulative_Log_Return': cum,
                                'Sample_Annual_Return': mu_b * TRADING_DAYS,
                                'Window_Annual_Return': mu_c * TRADING_DAYS,
                                'Drift_Shift_Daily': mu_c - mu_b,
                                'Sample_Annual_Vol': np.sqrt(np.diag(cov_b) * TRADING_DAYS),
                                'Window_Annual_Vol': np.sqrt(np.diag(cov_cr) * TRADING_DAYS),
                                'Vol_Ratio': vol_ratio,
                                'Window_Avg_Correlation': Ccr[off].mean()})
            cal_rows.append(cal)
            print(f'{len(r_c)} daily returns in the window.')
            print(cal.drop(columns=['Window', 'Days', 'Window_Avg_Correlation'])
                  .to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
            print(f'Implied average drift shift : {np.mean(mu_c - mu_b):+.5f} per day '
                  f'({np.mean(mu_c - mu_b) * TRADING_DAYS * 100:+.1f} pp per year)   [model: {STRESS_MEAN_SHOCK}]')
            print(f'Implied average vol ratio   : x{vol_ratio.mean():.2f} (range x{vol_ratio.min():.2f} '
                  f'to x{vol_ratio.max():.2f})   [model: x{STRESS_VOL_MULTIPLIER}]')
            print(f'Average pairwise correlation: sample {C[off].mean():.2f} | window {Ccr[off].mean():.2f}')
            run_scenario(f'Historical scenario: {label} parameters sustained for one year', mu_c, cov_cr)
        except Exception as e:
            print(f'[SKIPPED] {label}: {e}')
    if cal_rows:
        pd.concat(cal_rows).to_csv('calibration_windows.csv', index=False)
    pd.DataFrame(scen_rows).to_csv('scenario_comparison.csv', index=False)

    # ---------------------------------------------------------------- F
    section('F. Block bootstrap (keeps volatility clustering); block = 1 is i.i.d.')
    m_gb, m_gs = metrics(V0 * G_be @ w_base), metrics(V0 * G_se @ w_stress)
    bb_rows = [{'Shock_Model': 'Gaussian (main model)', 'Block': np.nan,
                'Baseline_VaR': m_gb['VaR_95'], 'Baseline_CVaR': m_gb['CVaR_95'],
                'Stress_VaR': m_gs['VaR_95'], 'Stress_CVaR': m_gs['CVaR_95'],
                'Std_Annual_LogReturn_Baseline': np.std(np.log(G_be @ w_base), ddof=1)}]
    for blk in BLOCK_LENGTHS:
        Gb = block_bootstrap_growth(blk)
        Gs = block_bootstrap_growth(blk, STRESS_VOL_MULTIPLIER, STRESS_MEAN_SHOCK)
        mb, ms = metrics(V0 * Gb @ w_base), metrics(V0 * Gs @ w_stress)
        bb_rows.append({'Shock_Model': 'i.i.d. bootstrap' if blk == 1 else f'Block bootstrap ({blk} days)',
                        'Block': blk, 'Baseline_VaR': mb['VaR_95'], 'Baseline_CVaR': mb['CVaR_95'],
                        'Stress_VaR': ms['VaR_95'], 'Stress_CVaR': ms['CVaR_95'],
                        'Std_Annual_LogReturn_Baseline': np.std(np.log(Gb @ w_base), ddof=1)})
    bb = pd.DataFrame(bb_rows)
    for col in ('Baseline_CVaR', 'Stress_CVaR'):
        bb[col + '_vs_Gaussian_%'] = (bb[col] / bb[col].iloc[0] - 1) * 100
    bb.to_csv('block_bootstrap.csv', index=False)
    print(bb.drop(columns='Block').to_string(index=False, float_format=lambda x: f'{x:,.3f}'))

    # ---------------------------------------------------------------- G
    section(f'G. Parameter uncertainty: {PARAM_BOOT_REPS} block-bootstrap resamples of the historical window\n'
            f'   (block {PARAM_BOOT_BLOCK} days; mu and Sigma re-estimated, weights re-optimized each time)')
    rng = np.random.default_rng(SEED + 7)
    zs_lp = zs_opt[:PARAM_BOOT_LP_SCENARIOS]
    pb = []
    fails = 0
    for _ in range(PARAM_BOOT_REPS):
        h = resample_history(rng)
        mu_r, cov_r = h.mean(axis=0), np.cov(h, rowvar=False)
        mu_rs, cov_rs = stress_params(mu_r, cov_r)
        try:
            wb, ws = max_sharpe(mu_r, cov_r), max_sharpe(mu_rs, cov_rs)
        except RuntimeError:
            fails += 1
            continue
        try:
            wmv = optimize_min_variance(cov_r)
            wmc_b = ru_optimize(growth(mu_r, cov_r, zs_lp), 'min_cvar')
            wmc_s = ru_optimize(growth(mu_rs, cov_rs, zs_lp), 'min_cvar')
        except RuntimeError:
            fails += 1
            continue
        Gb, Gs = growth(mu_r, cov_r, zs_eval), growth(mu_rs, cov_rs, zs_eval)
        cbb = metrics(V0 * Gb @ wb)['CVaR_95']
        cbs = metrics(V0 * Gs @ wb)['CVaR_95']
        css = metrics(V0 * Gs @ ws)['CVaR_95']
        rec = {f'wb_{t}': x for t, x in zip(tickers, wb)}
        rec.update({f'ws_{t}': x for t, x in zip(tickers, ws)})
        rec.update({'CVaR_Baseline': cbb, 'CVaR_Stress': css, 'CVaR_BaseWeights_Stress': cbs,
                    'Allocation_Effect': css - cbs, 'Stress_Corner': ws.max() > 0.99,
                    'Stress_Top_Asset': tickers[int(np.argmax(ws))],
                    'All_Stress_Excess_Negative': bool(np.all(mu_rs * TRADING_DAYS < RISK_FREE_RATE))})
        rec.update({f'wmv_{t}': x for t, x in zip(tickers, wmv)})
        rec.update({f'wmcb_{t}': x for t, x in zip(tickers, wmc_b)})
        rec.update({f'wmcs_{t}': x for t, x in zip(tickers, wmc_s)})
        # Estimation-error experiment: weights estimated on the resample, evaluated
        # under the full-sample parameters (treated as the true parameters).
        for tag, w_b_, w_s_ in (('MaxSharpe', wb, ws), ('MinVar', wmv, wmv), ('MinCVaR', wmc_b, wmc_s)):
            rec[f'TrueCVaR_Baseline_{tag}'] = metrics(V0 * G_be @ w_b_)['CVaR_95']
            rec[f'TrueCVaR_Stress_{tag}'] = metrics(V0 * G_se @ w_s_)['CVaR_95']
        pb.append(rec)
    pb = pd.DataFrame(pb)
    pb.to_csv('parameter_uncertainty.csv', index=False)
    q = lambda s: f'{s.mean():6.1%}  [5%: {s.quantile(0.05):6.1%}, 95%: {s.quantile(0.95):6.1%}]'
    print(f'Successful resamples: {len(pb)} (failed: {fails})')
    print('\nBaseline max-Sharpe weight: mean [5th, 95th percentile] | share of resamples with zero weight')
    for t in tickers:
        print(f'  {t:9s} {q(pb[f"wb_{t}"])} | zero in {np.mean(pb[f"wb_{t}"] < 1e-4):.0%}')
    print(f'\nStress allocation is a 100% corner in {pb.Stress_Corner.mean():.0%} of resamples; corner asset:')
    for t, f in pb[pb.Stress_Corner].Stress_Top_Asset.value_counts(normalize=True).items():
        print(f'  {t:9s} {f:.0%}')
    print(f'All stressed excess returns negative in {pb.All_Stress_Excess_Negative.mean():.0%} of resamples')
    for col, lab in (('CVaR_Baseline', 'Baseline CVaR'), ('CVaR_Stress', 'Stress CVaR')):
        s = pb[col]
        print(f'{lab:14s}: median {s.median():>9,.0f}  90% interval [{s.quantile(0.05):,.0f} – {s.quantile(0.95):,.0f}]')
    s = pb.Allocation_Effect
    pos, neg = s > 1.0, s < -1.0
    same = ~pos & ~neg
    print('Allocation effect of re-optimizing under stress (stress CVaR, re-optimized minus baseline weights):')
    print(f'  adds risk       in {pos.mean():.0%} of resamples (median when it does: {s[pos].median():,.0f} SAR)'
          if pos.any() else '  adds risk       in 0% of resamples')
    print(f'  reduces risk    in {neg.mean():.0%} of resamples' +
          (f' (median when it does: {s[neg].median():,.0f} SAR)' if neg.any() else ''))
    print(f'  no change       in {same.mean():.0%} of resamples (baseline allocation already the same corner)')

    print('\n--- Stability and estimation error by portfolio construction ---')
    print('Weight variability = average over assets of the standard deviation of the weight across resamples.')
    print('Tail risk = CVaR of the weights estimated on each resample, evaluated under the full-sample\n'
          'parameters (estimation-error experiment).')
    w_full = {'MaxSharpe': (w_base, w_stress), 'MinVar': (optimize_min_variance(cov_b),) * 2,
              'MinCVaR': (ru_optimize(G_bo, 'min_cvar'), ru_optimize(G_so, 'min_cvar'))}
    est = []
    for tag, label, pb_, ps_ in (('MaxSharpe', 'Max Sharpe', 'wb', 'ws'), ('MinVar', 'Minimum variance', 'wmv', 'wmv'),
                                 ('MinCVaR', 'Minimum CVaR', 'wmcb', 'wmcs')):
        Wb = pb[[f'{pb_}_{t}' for t in tickers]].values
        Ws = pb[[f'{ps_}_{t}' for t in tickers]].values
        cb, cs_ = pb[f'TrueCVaR_Baseline_{tag}'], pb[f'TrueCVaR_Stress_{tag}']
        est.append({'Strategy': label,
                    'Weight_SD_Baseline_pp': Wb.std(axis=0, ddof=1).mean() * 100,
                    'Weight_SD_Stress_pp': Ws.std(axis=0, ddof=1).mean() * 100,
                    'Full_Sample_Baseline_CVaR': metrics(V0 * G_be @ w_full[tag][0])['CVaR_95'],
                    'Baseline_CVaR_Median': cb.median(), 'Baseline_CVaR_P05': cb.quantile(0.05),
                    'Baseline_CVaR_P95': cb.quantile(0.95),
                    'Full_Sample_Stress_CVaR': metrics(V0 * G_se @ w_full[tag][1])['CVaR_95'],
                    'Stress_CVaR_Median': cs_.median(), 'Stress_CVaR_P05': cs_.quantile(0.05),
                    'Stress_CVaR_P95': cs_.quantile(0.95)})
        print(f'\n{label}')
        for t in tickers:
            col = pb[f'{pb_}_{t}']
            print(f'  baseline weight {t:9s} median {col.median():6.1%}  [5%: {col.quantile(0.05):6.1%}, '
                  f'95%: {col.quantile(0.95):6.1%}]')
        e = est[-1]
        print(f"  weight variability: baseline {e['Weight_SD_Baseline_pp']:.1f} pp | stress {e['Weight_SD_Stress_pp']:.1f} pp")
        print(f"  baseline CVaR: median {e['Baseline_CVaR_Median']:,.0f} [5%: {e['Baseline_CVaR_P05']:,.0f}, "
              f"95%: {e['Baseline_CVaR_P95']:,.0f}]")
        print(f"  stress CVaR  : median {e['Stress_CVaR_Median']:,.0f} [5%: {e['Stress_CVaR_P05']:,.0f}, "
              f"95%: {e['Stress_CVaR_P95']:,.0f}]")
    beats = (pb['TrueCVaR_Stress_MinCVaR'] < pb['TrueCVaR_Stress_MaxSharpe']).mean()
    beats_b = (pb['TrueCVaR_Baseline_MinCVaR'] < pb['TrueCVaR_Baseline_MaxSharpe']).mean()
    print(f'\nMinimum CVaR has lower tail risk than max Sharpe in {beats_b:.0%} of resamples (baseline) '
          f'and {beats:.0%} (stress).')

    # Resampled max-Sharpe portfolio (Michaud, 1989): average the resampled weights
    w_rs_b = pb[[f'wb_{t}' for t in tickers]].mean().values
    w_rs_s = pb[[f'ws_{t}' for t in tickers]].mean().values
    w_rs_b, w_rs_s = w_rs_b / w_rs_b.sum(), w_rs_s / w_rs_s.sum()
    mrb, mrs = metrics(V0 * G_be @ w_rs_b), metrics(V0 * G_se @ w_rs_s)
    est.append({'Strategy': 'Resampled max Sharpe', 'Weight_SD_Baseline_pp': np.nan, 'Weight_SD_Stress_pp': np.nan,
                'Full_Sample_Baseline_CVaR': mrb['CVaR_95'], 'Full_Sample_Stress_CVaR': mrs['CVaR_95']})
    print('\nResampled max-Sharpe portfolio (average of the 500 resampled weight vectors):')
    print(f"  baseline: {fmt_w(w_rs_b, tickers)} | CVaR {mrb['CVaR_95']:,.0f} | mean {mrb['Mean_Terminal_Value']:,.0f}")
    print(f"  stress  : {fmt_w(w_rs_s, tickers)} | CVaR {mrs['CVaR_95']:,.0f} | mean {mrs['Mean_Terminal_Value']:,.0f}")
    est = pd.DataFrame(est)
    est.to_csv('estimation_error_comparison.csv', index=False)
    pd.DataFrame({'Asset': tickers, 'Resampled_Baseline_Weight': w_rs_b,
                  'Resampled_Stress_Weight': w_rs_s}).to_csv('resampled_weights.csv', index=False)
    out['pb'] = pb

    # ---------------------------------------------------------------- H
    section(f'H. Out-of-sample 1-day VaR backtest (rolling {BACKTEST_WINDOW}-day window, max-Sharpe\n'
            '   weights re-optimized every day using only past data)')
    R = log_returns.values
    dates = log_returns.index
    recs = []
    w_prev = np.full(N_ASSETS, 1 / N_ASSETS)
    for t in range(BACKTEST_WINDOW, len(R)):
        hist = R[t - BACKTEST_WINDOW:t]
        mu_w, cov_w = hist.mean(axis=0), np.cov(hist, rowvar=False)
        try:
            w = max_sharpe(mu_w, cov_w)
        except RuntimeError:
            w = w_prev
        w_prev = w
        m_, s_ = mu_w @ w, np.sqrt(w @ cov_w @ w)
        port_hist = np.log(np.exp(hist) @ w)
        rec = {'Date': dates[t], 'Realized': float(np.log(np.exp(R[t]) @ w))}
        for a in (0.95, 0.99):
            tag = int(round(a * 100))
            rec[f'VaR_Normal_{tag}'] = norm.ppf(a) * s_ - m_
            rec[f'VaR_HistSim_{tag}'] = -np.quantile(port_hist, 1 - a)
        recs.append(rec)
    bt = pd.DataFrame(recs).set_index('Date')
    tests = []
    for model in ('Normal', 'HistSim'):
        for a in (0.95, 0.99):
            tag = int(round(a * 100))
            hits = (-bt.Realized > bt[f'VaR_{model}_{tag}']).values
            bt[f'Hit_{model}_{tag}'] = hits
            lr_k, p_k = kupiec(hits, 1 - a)
            lr_i, p_i = christoffersen(hits)
            p_cc = chi2.sf(lr_k + lr_i, 2)
            tests.append({'Model': 'Normal (model assumption)' if model == 'Normal' else 'Historical simulation',
                          'Level': f'{tag}%', 'Days': len(hits), 'Expected': round(len(hits) * (1 - a), 1),
                          'Exceedances': int(hits.sum()), 'Rate': hits.mean(),
                          'Kupiec_LR': lr_k, 'Kupiec_p': p_k, 'Christoffersen_LR': lr_i,
                          'Christoffersen_p': p_i, 'Conditional_Coverage_p': p_cc,
                          'Verdict_5%': 'reject' if p_cc < 0.05 or p_k < 0.05 else 'not rejected'})
    tests = pd.DataFrame(tests)
    tests.to_csv('var_backtest_tests.csv', index=False)
    bt.to_csv('var_backtest_series.csv')
    print(f'Test period: {bt.index.min().date()} to {bt.index.max().date()} ({len(bt)} days)')
    print(tests.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))

    # ---------------------------------------------------------------- figures
    section('Figures')
    fig_weights(w_base, w_stress)
    fig_strategies(strat)
    fig_sensitivity(grid)
    fig_backtest(bt)
    fig_weight_stability(out['pb'])
    print('Saved fig_weights.png, fig_strategies.png, fig_sensitivity.png, fig_backtest.png, '
          'fig_weight_stability.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supplementary analyses for the Tadawul stress test')
    parser.add_argument('--skip-2020', action='store_true', help='skip the historical calibration downloads')
    args = parser.parse_args()
    sys.stdout = Tee('extensions_output.txt')
    try:
        main(skip_2020=args.skip_2020)
    finally:
        sys.stdout.flush()
        sys.stdout = sys.stdout.console
    print('\nFull log saved to extensions_output.txt')
