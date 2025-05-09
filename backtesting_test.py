import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Backtesting import BacktestMultiStrategy, AssetClassifier
import os
import yfinance as yf
import pandas as pd


# Crear carpeta para guardar gráficas
os.makedirs("plots", exist_ok=True)

# === Cargar el dataset combinado ===
data = pd.read_csv("dbs/prices100_merged.csv", index_col=0, parse_dates=True)

# === Identificar columnas ===
indicator_cols = data.columns[-12:].tolist()
price_cols = data.columns.difference(indicator_cols).tolist()

# === Cargar modelo SVR ===
svr_models = {
    'high_cap': pickle.load(open("trained_models/Dataset_1_SVR_mc.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/Dataset_2_SVR_mc.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/Dataset_3_SVR_mc.pkl", "rb"))
}

# === Cargar modelo XGBoost ===
xgboost_models = {
    'high_cap': pickle.load(open("trained_models/Dataset_1_XGBoost_Sharpe.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/Dataset_2_XGBoost_Sharpe.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/Dataset_3_XGBoost_Sharpe.pkl", "rb"))
}

def select_valid_assets(data, price_cols, n_assets, rebalance_dates):
    for _ in range(100):
        sampled_assets = np.random.choice(price_cols, size=n_assets, replace=False).tolist()
        if all(not data.loc[data.index.asof(date), sampled_assets].isnull().any() for date in rebalance_dates):
            return sampled_assets
    raise ValueError("No se encontraron suficientes activos válidos para todas las fechas.")

def asset_ranking(price_cols):
    """
    Obtiene un ranking de capitalización de mercado para los tickers dados.

    Args:
        price_cols (list): Lista de tickers.

    Returns:
        dict: Diccionario ticker -> ranking (1 = mayor market cap).
    """
    market_caps = {}
    for ticker in price_cols:
        try:
            info = yf.Ticker(ticker).info
            cap = info.get('marketCap')
            if cap:
                market_caps[ticker] = cap
        except:
            market_caps[ticker] = None

    ranking_df = pd.DataFrame.from_dict(market_caps, orient='index', columns=['MarketCap'])
    ranking_df = ranking_df.dropna().sort_values(by='MarketCap', ascending=False).reset_index()
    ranking_df.columns = ['Ticker', 'MarketCap']
    ranking_df['Rank'] = ranking_df.index + 1

    ranking_dict = dict(zip(ranking_df['Ticker'], ranking_df['Rank']))
    return ranking_dict

# === Inicializar clasificador ===
ranking_dict = asset_ranking(price_cols)
classifier = AssetClassifier(data, ranking_dict = ranking_dict)

# === Simulaciones ===
n_simulations = 1000
results = []
all_paths = {method: [] for method in ['SVR-CPO', 'XGBoost-CPO', 'EqualWeight', 'MinVar', 'MaxSharpe']}
risk_free_rate = 0.042
rfr_monthly = risk_free_rate/12

rebalance_template = pd.date_range("2015-01-01", "2025-01-01", freq='12MS') + pd.offsets.MonthBegin(1)

for sim in range(n_simulations):
    print(f"Simulación {sim + 1}/{n_simulations}...")
    sampled_assets = select_valid_assets(data, price_cols, 15, rebalance_template)

    price_multi = pd.concat([data[sampled_assets]], axis=1, keys=["Price"])
    ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
    combined_data = pd.concat([price_multi, ind_multi], axis=1)

    # Clasificación de activos
    cap_type = classifier.get_cap_type(sampled_assets)
    print(f"Cap type seleccionado para esta simulación: {cap_type}")

    # Correr backtesting
    bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models, ranking_dict)
    bt.classifier = classifier
    bt.simulate(monthly = True)

    monthly_returns = bt.evolution()

    for strategy, path in bt.results.items():
        history = monthly_returns[strategy]
        Pt, Po = path[-1], path[0]

        rendimiento_anual = history.mean() * 12
        std_anual = history.std() * np.sqrt(12)
        excess_returns = history - rfr_monthly
        downside_returns = excess_returns[excess_returns < rfr_monthly]
        downside_deviation = downside_returns.std() * np.sqrt(12) if not downside_returns.empty else np.nan
        sortino = (rendimiento_anual - risk_free_rate) / downside_deviation if downside_deviation and downside_deviation != 0 else np.na
        
        results.append({
            "Simulación": sim + 1,
            "Metodología": strategy,
            "Rendimiento Anual Promedio": rendimiento_anual,
            "Desviación Estándar": std_anual,
            "Rendimiento Efectivo": (Pt / Po) - 1,
            "Downside Risk": downside_deviation,
            "CAGR": (Pt / Po) ** (1 / ((bt.end_date - bt.start_date).days / 365.25)) - 1,
            'Sortino ratio': sortino
        })

        all_paths[strategy].append(path)

# === Crear DataFrame con los resultados ===
results_df = pd.DataFrame(results)

# === Mostrar promedio de cada métrica por metodología ===
summary = results_df.groupby("Metodología").mean(numeric_only=True)
print("\nResumen de métricas promedio por metodología:")
print(summary)

# === Gráficas ===
summary["Rendimiento Anual Promedio"].plot(kind='bar', title='Rendimiento Anual Promedio por Metodología', ylabel='Promedio', xlabel='Metodología')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/rendimiento_anual_promedio.png")
plt.close()

sns.boxplot(data=results_df, x="Metodología", y="CAGR")
plt.title("Distribución del CAGR por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/boxplot_cagr.png")
plt.close()

strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Rendimiento Anual Promedio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Rendimiento Anual Promedio: {method}")
    axes[i].set_xlabel("Rendimiento Anual Promedio")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots/histogramas_rendimiento_anual.png")
plt.close()

summary["CAGR"].plot(kind="bar", title="CAGR Promedio por Metodología", ylabel="CAGR Promedio", xlabel="Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cagr_promedio.png")
plt.close()

# === Guardar resultados ===
results_df.to_csv("plots/resultados_simulaciones.csv", index=False)
summary.to_csv("plots/resumen_metricas.csv")

with open("plots/results_list.pkl", "wb") as f:
    pickle.dump(results, f)