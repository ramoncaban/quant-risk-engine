# FinTech Quantitative Risk Engine 🚀

<img width="800" height="566" alt="demo" src="https://github.com/user-attachments/assets/6ac4dfb0-c241-4988-848c-b31e2fa09d69" />



An end-to-end quantitative portfolio analytics workstation. This application utilizes historical market data vectors via Yahoo Finance to compute Parametric, Historical, and Monte Carlo (Geometric Brownian Motion) Value at Risk (VaR) profiles across multi-asset portfolios.

### Architecture Overview
* **Backend:** Python, FastAPI, NumPy, Pandas, Uvicorn (Mathematical Core)
* **Frontend:** React, TypeScript, Vite, Tailwind CSS v4, Recharts (Visual Workspace)

---

## How to Run Locally

### 1. Spin Up the Backend Engine
```bash
# Navigate to the root directory
cd quant-risk-engine

# Activate your virtual environment
.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Start the FastAPI engine node
python -m uvicorn app.main:app --reload
