import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import from your other modular files
from data_analysis import tickers, mean_returns, cov_matrix
from portfolio_optimization import optimize_portfolio
from ai_report_generator import generate_executive_report

num_simulations = 10000
num_days = 252
initial_investment = 1000000

L_baseline = la.cholesky(cov_matrix, lower=True)
stress_volatility_multiplier = 1.5
stress_mean_shock = -0.001
stress_mean_returns = mean_returns + stress_mean_shock
stress_cov_matrix = cov_matrix * (stress_volatility_multiplier ** 2)
L_stress = la.cholesky(stress_cov_matrix, lower=True)


# Updated to accept dynamic weights
def run_simulation(mu, cov, L_matrix, weights, seed_value=42):
    np.random.seed(seed_value)
    portfolio_end_values = np.zeros(num_simulations)
    for i in range(num_simulations):
        Z = np.random.normal(0, 1, size=(num_days, len(tickers)))
        correlated_shocks = Z.dot(L_matrix.T)

        drift = mu - (0.5 * np.diag(cov))
        simulated_daily_returns = np.exp(drift + correlated_shocks)

        price_paths = np.vstack([np.ones(len(tickers)), np.cumprod(simulated_daily_returns, axis=0)])
        portfolio_path = initial_investment * (price_paths @ weights)
        portfolio_end_values[i] = portfolio_path[-1]
    return portfolio_end_values


# Updated to return VaR and CVaR for the AI prompt
def calculate_risk_metrics(results, scenario_name):
    losses = np.maximum(initial_investment - results, 0)
    var_95 = np.percentile(losses, 95)
    cvar_95 = losses[losses >= var_95].mean()

    print(f"\n--- {scenario_name} Risk Metrics (95% Confidence) ---")
    print(f"Value at Risk (VaR): SAR {var_95:,.2f}")
    print(f"Expected Shortfall (CVaR): SAR {cvar_95:,.2f}")

    return var_95, cvar_95


if __name__ == '__main__':
    print("\n[PIPELINE START] Calculating Optimal Markowitz Weights...")
    weights_normal, _ = optimize_portfolio(mean_returns, cov_matrix, "Baseline")
    weights_stress, top_stress_asset = optimize_portfolio(stress_mean_returns, stress_cov_matrix, "Stress")

    print("\n[PIPELINE STEP 2] Generating 10,000 paths for Baseline Scenario...")
    baseline_results = run_simulation(mean_returns, cov_matrix, L_baseline, weights_normal)

    print("\n[PIPELINE STEP 3] Generating 10,000 paths for Geopolitical Stress Scenario...")
    stress_results = run_simulation(stress_mean_returns, stress_cov_matrix, L_stress, weights_stress)

    var_norm, cvar_norm = calculate_risk_metrics(baseline_results, "Baseline (Normal Market)")
    var_stress, cvar_stress = calculate_risk_metrics(stress_results, "Stress (Geopolitical Shock)")

    print("\n[PIPELINE STEP 4] Exporting data to CSV for QuickSight/PowerBI...")
    df_simulations = pd.DataFrame({
        'Scenario': ['Baseline (Normal)'] * num_simulations + ['Stress (Shock)'] * num_simulations,
        'Terminal_Value': np.concatenate([baseline_results, stress_results])
    })
    df_simulations.to_csv('monte_carlo_results.csv', index=False)

    df_weights = pd.DataFrame({
        'Asset': tickers,
        'Normal_Weight': weights_normal,
        'Stress_Weight': weights_stress
    })
    df_weights.to_csv('portfolio_weights.csv', index=False)

    # Trigger AWS Bedrock Execution
    generate_executive_report(var_norm, cvar_norm, var_stress, cvar_stress, top_stress_asset)

    # Plot terminal value distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(baseline_results, bins=100, color='skyblue', alpha=0.6, label='Baseline', kde=True)
    sns.histplot(stress_results, bins=100, color='salmon', alpha=0.6, label='Stress Shock', kde=True)
    plt.axvline(initial_investment, color='black', linestyle='dashed', linewidth=2, label='Initial Investment (1M SAR)')
    plt.title('Portfolio Terminal Value Distribution: Baseline vs. Stress')
    plt.xlabel('Final Portfolio Value after 1 Year (SAR)')
    plt.ylabel('Frequency (Number of Scenarios)')
    plt.legend()
    plt.tight_layout()
    plt.show()