from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import yfinance as yf
from app.engine import calculate_parametric_var, calculate_historical_var, calculate_monte_carlo_var

app = FastAPI(title="FinTech Quantitative Risk Engine")

# Define what data the API expects to receive from the user/frontend
class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    portfolio_value: float
    days_historical: int = 500
    confidence_level: float = 0.99

@app.post("/api/v1/risk/var")
async def get_portfolio_var(payload: PortfolioRequest):
    # Validation: Ensure tickers array matches weights array size
    if len(payload.tickers) != len(payload.weights):
        raise HTTPException(status_code=400, detail="Tickers and weights lengths must match.")
    
    # Validation: Ensure portfolio allocations add up exactly to 100%
    if not np.isclose(sum(payload.weights), 1.0):
        raise HTTPException(status_code=400, detail="Portfolio weights must sum to 1.0.")
        
    try:
        # Fetch actual historical closing prices via yfinance
        # yfinance returns a clean pandas DataFrame
        # --- AFTER (FIXED) ---
        data = yf.download(payload.tickers, period=f"{payload.days_historical}d", auto_adjust=False)['Adj Close']   
        
        # Calculate daily logarithmic returns and drop empty row rows
        returns = np.log(data / data.shift(1)).dropna()
        
        # Convert user weights into a numpy mathematical array
        weights_arr = np.array(payload.weights)
        
        # Run our mathematical calculations from engine.py
        # --- Run all 3 math calculations ---
        p_var_pct = calculate_parametric_var(returns, weights_arr, payload.confidence_level)
        h_var_pct = calculate_historical_var(returns, weights_arr, payload.confidence_level)
        mc_var_pct = calculate_monte_carlo_var(returns, weights_arr, payload.portfolio_value, payload.confidence_level)
        
        return {
            "status": "success",
            "metadata": {
                "tickers_analyzed": payload.tickers,
                "confidence_level": payload.confidence_level,
                "days_calculated": payload.days_historical,
                "simulations_run": 10000
            },
            "metrics": {
                "parametric_var_dollar": round(p_var_pct * payload.portfolio_value, 2),
                "parametric_var_percent": round(p_var_pct * 100, 4),
                "historical_var_dollar": round(h_var_pct * payload.portfolio_value, 2),
                "historical_var_percent": round(h_var_pct * 100, 4),
                "monte_carlo_var_dollar": round(mc_var_pct * payload.portfolio_value, 2),
                "monte_carlo_var_percent": round(mc_var_pct * 100, 4)
            }
        }
        
        # Return a beautifully structured JSON response
        return {
            "status": "success",
            "metadata": {
                "tickers_analyzed": payload.tickers,
                "confidence_level": payload.confidence_level,
                "days_calculated": payload.days_historical
            },
            "metrics": {
                "parametric_var_dollar": round(p_var_pct * payload.portfolio_value, 2),
                "parametric_var_percent": round(p_var_pct * 100, 4),
                "historical_var_dollar": round(h_var_pct * payload.portfolio_value, 2),
                "historical_var_percent": round(h_var_pct * 100, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk Engine Exception: {str(e)}")