from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from app.engine import calculate_parametric_var, calculate_historical_var, calculate_monte_carlo_var, calculate_component_var

app = FastAPI(title="FinTech Quantitative Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    portfolio_value: float
    days_historical: int = 500
    confidence_level: float = 0.99
    stress_scenario: Optional[str] = None

@app.post("/api/v1/risk/var")
async def get_portfolio_var(payload: PortfolioRequest):
    if len(payload.tickers) != len(payload.weights):
        raise HTTPException(status_code=400, detail="Tickers and weights lengths must match.")
    
    if not np.isclose(sum(payload.weights), 1.0):
        raise HTTPException(status_code=400, detail="Portfolio weights must sum to 1.0.")
        
    try:
        # 1. Establish the timeline windows based on stress test parameters
        if payload.stress_scenario == "black_monday":
            start_dt, end_dt = "1987-08-01", "1987-11-30"
        elif payload.stress_scenario == "gfc":
            start_dt, end_dt = "2008-01-01", "2009-03-31"
        elif payload.stress_scenario == "covid":
            start_dt, end_dt = "2020-02-01", "2020-05-01"
        else:
            import datetime
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=int(payload.days_historical * 1.5))
            start_dt, end_dt = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        # 2. Sequential asset parsing loop (Bypasses yfinance MultiIndex framing completely)
        df = pd.DataFrame()
        for ticker in payload.tickers:
            try:
                asset_data = yf.download(ticker, start=start_dt, end=end_dt, progress=False)
                
                # Snag Adjusted Close or standard Close safely
                if "Adj Close" in asset_data.columns:
                    series = asset_data["Adj Close"]
                elif "Close" in asset_data.columns:
                    series = asset_data["Close"]
                else:
                    continue
                
                # Clean up single asset series into a clean flat column vector
                series = series.dropna()
                if not series.empty:
                    df[ticker] = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
            except Exception:
                pass

        # 3. Guardrail tracking validation check
        missing_tickers = [t for t in payload.tickers if t not in df.columns]
        if missing_tickers:
            context_msg = "during the selected crash window" if payload.stress_scenario else "on the historical network feed"
            raise HTTPException(
                status_code=400, 
                detail=f"The following assets do not have active trading data {context_msg}: {', '.join(missing_tickers)}. Please remove them or change your stress configurations."
            )
        
        # Align rows across assets and drop structural null values
        df = df.dropna()
        if payload.stress_scenario is None:
            df = df.tail(payload.days_historical)

        # 4. Math analytics matrices calculations
        returns = df.pct_change().dropna()
        weights_arr = np.array(payload.weights)
        
        portfolio_returns_vector = np.dot(returns.values, weights_arr)
        worst_single_day_pct = float(np.min(portfolio_returns_vector)) if len(portfolio_returns_vector) > 0 else 0.0
        worst_single_day_dollar = round(worst_single_day_pct * payload.portfolio_value, 2)
        
        p_var_pct = calculate_parametric_var(returns, weights_arr, payload.confidence_level)
        h_var_pct = calculate_historical_var(returns, weights_arr, payload.confidence_level)
        mc_var_pct, mc_chart_data = calculate_monte_carlo_var(returns, weights_arr, payload.portfolio_value, payload.confidence_level)
        risk_components = calculate_component_var(returns, weights_arr, payload.portfolio_value, payload.confidence_level)
        
        return {
            "status": "success",
            "metadata": {
                "tickers_analyzed": payload.tickers,
                "confidence_level": payload.confidence_level,
                "days_calculated": len(df),
                "simulations_run": 10000
            },
            "metrics": {
                "parametric_var_dollar": round(p_var_pct * payload.portfolio_value, 2),
                "parametric_var_percent": round(p_var_pct * 100, 4),
                "historical_var_dollar": round(h_var_pct * payload.portfolio_value, 2),
                "historical_var_percent": round(h_var_pct * 100, 4),
                "monte_carlo_var_dollar": round(mc_var_pct * payload.portfolio_value, 2),
                "monte_carlo_var_percent": round(mc_var_pct * 100, 4),
                "worst_crash_day_dollar": worst_single_day_dollar,
                "worst_crash_day_percent": round(worst_single_day_pct * 100, 2)
            },
            "chart_data": mc_chart_data,
            "component_data": risk_components
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk Engine Exception: {str(e)}")