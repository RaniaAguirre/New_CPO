import yfinance as yf
import pandas as pd
import numpy as np
import random

prices = yf.download("^GSPC", start="2010-01-01", end="2025-01-02", interval="1d", progress=False)["Close"].dropna()

results = []
risk_free_rate = 0.042
rfr_daily = risk_free_rate / 252
n_simulations = 1_000

def get_random_backtest_period():
    first_valid = pd.Timestamp("2010-01-01")
    last_valid = prices.index.max() - pd.DateOffset(years=5)
    if last_valid < first_valid:
        raise ValueError("Pocos datos para un backtesting de 5 años")
    span_days = (last_valid-first_valid).days
    offset = random.randint(0, span_days)
    start_date = first_valid + pd.Timedelta(days=offset)
    end_date = start_date + pd.DateOffset(years=5)
    return start_date, end_date

for sim in range(1, n_simulations + 1):
    print(f"Simulación {sim}/{n_simulations}...")
    start_date, end_date = get_random_backtest_period()
    prices_period = prices.loc[start_date:end_date].dropna()
    start_price = float(prices_period.iloc[0])
    end_price = float(prices_period.iloc[-1])
    if len(prices_period) < 2:
        continue
    returns = prices_period.pct_change().dropna()
    if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
        returns = returns.iloc[:, 0]

    annual_return = returns.mean() * 252
    annual_std = returns.std(ddof=1) * np.sqrt(252)
    rend_efectivo = (end_price/start_price) -1
    N = len(returns)
    if N > 0:
        growth = (1 + returns).prod()
        cagr = (growth ** (252.0 / N)) - 1
    else:
        cagr = np.nan
    excess = returns - rfr_daily
    downside = excess[excess < rfr_daily]
    if not downside.empty:
        downside_std = float(downside.std(ddof=1) * np.sqrt(252))
        if downside_std > 0:
            sortino = (annual_return - risk_free_rate) / downside_std
        else:
            sortino = np.nan
    else:
        downside_std = np.nan
        sortino     = np.nan

    results.append({
        "Simulación": sim,
        "Start Date": start_date,
        "End Date": end_date,
        "Rendimiento Anual Promedio": annual_return,
        "Desviación Estándar Anual": annual_std,
        "Downside Risk": downside_std,
        "CAGR": cagr,
        "Rendimiento Efectivo": rend_efectivo,
        "Sortino Ratio": sortino
    })

df = pd.DataFrame(results)
to_num = ["Rendimiento Anual Promedio", "Desviación Estándar Anual", "Downside Risk", "CAGR", "Sortino Ratio", "Rendimiento Efectivo"]
for col in to_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.to_csv("sp500_performance.csv", index=False)

summary = df [["Rendimiento Anual Promedio", "Desviación Estándar Anual", "CAGR", "Sortino Ratio", "Downside Risk", "Rendimiento Efectivo"]].mean(numeric_only=True)
summary.to_csv("sp500_performance_summary.csv")
    


    
