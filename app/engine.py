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

def calculate_monte_carlo_var(returns: pd.DataFrame, weights: np.ndarray, portfolio_value: float, confidence_level: float = 0.99, num_simulations: int = 10000) -> float:
    """
    Calculates Value at Risk using a Monte Carlo Simulation (Geometric Brownian Motion)
    across a multi-asset portfolio.
    """
    # 1. Calculate the mean daily return vector and covariance matrix
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # 2. Compute the Cholesky Decomposition
    # This correlates the random shocks across different assets (e.g., if Apple drops, Microsoft usually drops too)
    chol_matrix = np.linalg.cholesky(cov_matrix)
    
    num_assets = len(weights)
    
    # 3. Generate 10,000 independent random normal shocks for each asset
    random_shocks = np.random.normal(0, 1, (num_assets, num_simulations))
    
    # 4. Correlate the shocks using our Cholesky matrix
    correlated_shocks = np.dot(chol_matrix, random_shocks)
    
    # 5. Calculate simulated daily returns for each asset using GBM drift + shock
    # We use a 1-day horizon, so dt = 1
    drift = mean_returns - 0.5 * np.diag(cov_matrix)
    simulated_asset_returns = np.exp(drift[:, np.newaxis] + correlated_shocks) - 1
    
    # 6. Aggregate asset returns into portfolio returns based on weights
    simulated_portfolio_returns = np.dot(weights, simulated_asset_returns)
    
    # 7. Sort returns and find the cutoff point for our confidence interval
    quantile = 1 - confidence_level
    var_pct = -np.percentile(simulated_portfolio_returns, quantile * 100)
    
    return float(var_pct)