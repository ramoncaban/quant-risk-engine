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