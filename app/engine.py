import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_parametric_var(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.99) -> float:
    """
    Calculates Parametric (Variance-Covariance) VaR for a multi-asset portfolio.
    Assumes standard normal distribution of returns.
    """
    # 1. Calculate daily mean returns and the covariance matrix between assets
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # 2. Calculate overall portfolio expected return and volatility (standard deviation)
    port_return = np.sum(mean_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # 3. Get the Z-score corresponding to our confidence level (e.g., 2.33 for 99%)
    z_score = norm.ppf(confidence_level)
    
    # 4. Compute the VaR percentage
    var_pct = z_score * port_volatility - port_return
    return float(var_pct)

def calculate_historical_var(returns: pd.DataFrame, weights: np.ndarray, confidence_level: float = 0.99) -> float:
    """
    Calculates Historical Simulation VaR for a multi-asset portfolio.
    Does NOT assume normal distribution; maps actual historic movements.
    """
    # 1. Calculate historical portfolio daily returns by multiplying asset returns by weights
    portfolio_historical_returns = returns.dot(weights)
    
    # 2. Find the lower percentile boundary (e.g., the worst 1% of days if confidence is 99%)
    quantile = 1 - confidence_level
    var_pct = -np.percentile(portfolio_historical_returns, quantile * 100)
    
    return float(var_pct)

def calculate_monte_carlo_var(returns: pd.DataFrame, weights: np.ndarray, portfolio_value: float, confidence_level: float = 0.99, num_simulations: int = 10000):
    """
    Calculates Monte Carlo VaR and returns the threshold along with a histogram 
    distribution breakdown of simulated portfolio dollar returns.
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    chol_matrix = np.linalg.cholesky(cov_matrix)
    num_assets = len(weights)
    
    random_shocks = np.random.normal(0, 1, (num_assets, num_simulations))
    correlated_shocks = np.dot(chol_matrix, random_shocks)
    
    drift = mean_returns - 0.5 * np.diag(cov_matrix)
    simulated_asset_returns = np.exp(drift[:, np.newaxis] + correlated_shocks) - 1
    simulated_portfolio_returns = np.dot(weights, simulated_asset_returns)
    
    quantile = 1 - confidence_level
    var_pct = -np.percentile(simulated_portfolio_returns, quantile * 100)
    
    # --- NEW CHART DATA PROCESSING ---
    # Convert percentages to actual simulated portfolio dollar returns
    simulated_dollar_returns = simulated_portfolio_returns * portfolio_value
    
    # Segment the 10,000 runs into 30 uniform histogram bars (bins)
    counts, bin_edges = np.histogram(simulated_dollar_returns, bins=30)
    
    chart_data = []
    for i in range(len(counts)):
        # Calculate mid-point of the bin for a clean label string
        bin_label = f"${int((bin_edges[i] + bin_edges[i+1]) / 2):,}"
        chart_data.append({
            "bin": bin_label,
            "frequency": int(counts[i])
        })
    # ---------------------------------
    
    return float(var_pct), chart_data