# FinTech Quantitative Risk Engine 🚀

An institutional-grade, full-stack quantitative portfolio analytics dashboard. This application calculates real-time market risk metrics—**Value at Risk (VaR)** and **Component VaR (Risk Contribution)**—across multi-asset portfolios using parametric, historical simulation, and Monte Carlo frameworks, alongside historical regime stress testing.


<img width="800" height="571" alt="New_Demo" src="https://github.com/user-attachments/assets/3849acb5-ca5c-4e22-8c35-aa77f9177499" />


## 🛠️ System Architecture & Tech Stack

- **Core Engine (Backend):** FastAPI (Python 3.11+), NumPy, Pandas, SciPy, Yahoo Finance API (`yfinance`).
- **Client Interface (Frontend):** React 18, TypeScript, Tailwind CSS, Recharts (Dynamic Canvas Layouts), Lucide Icons.
- **Environment Replication:** Fully encapsulated virtual environments tracking explicit matrix processing dependencies.

---

## 📊 Core Analytical Features & Methodologies

### 1. Value at Risk (VaR) Tri-Engine Framework
The engine processes a 99% confidence interval across three standard quantitative methodologies simultaneously:
- **Parametric (Variance-Covariance):** Utilizes historical asset covariance matrices and portfolio weight vectors to calculate volatility under a standard normal distribution assumption ($Z = 2.33$).
- **Historical Simulation:** Non-parametric approach that slices real asset return distributions over a dynamic 500-day lookback horizon to find the empirical 1st percentile.
- **Monte Carlo Simulation:** Executes **10,000 randomized asset path matrices** utilizing Geometric Brownian Motion (GBM) to map out simulated ending portfolio value distributions.

### 2. Marginal & Component VaR Risk Breakdown (Option 2)
Instead of treating portfolio risk as a flat metric, the engine performs mathematical risk decomposition:
- Computes the **Marginal Contribution to VaR (M-VaR)** to determine how a marginal dollar allocation increases or decreases overall portfolio volatility.
- Isolates **Component VaR (CVaR)** to display exactly what percentage of total portfolio risk is driven by an individual asset node, rendering it inside an interactive React Donut Chart.

### 3. Historical Crisis Regime Stress Testing (Option 1)
Bypasses standard lookback matrices to force historical date-window overrides, stress-testing current asset configurations against legendary black-swan market anomalies:
- **1987 Black Monday Hyper-Crash:** Isolates the historic October 1987 liquidity collapse (e.g., handles long-horizon assets like `SCHW`).
- **2008 Great Financial Crisis (GFC):** Captures the systemic subprime mortgage banking collapse.
- **2020 COVID Liquidity Shock:** Captures the extreme correlation-breakdown and volatility spike during March 2020.

---

## ⚡ Quick Start & Local Replication 
 
### Backend Engine Setup
1. Spin Up the Backend Engine
```bash
# Navigate to the root directory
cd quant-risk-engine

# Activate your virtual environment
# Windows PowerShell:
.venv\Scripts\activate

# Mac/Linux Terminal:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Start the FastAPI engine node
python -m uvicorn app.main:app --reload

## ⚛️ Frontend Client Setup
cd quant-risk-frontend

# Install the necessary node packages
npm install

# Run the Vite local development compiler server
npm run dev
