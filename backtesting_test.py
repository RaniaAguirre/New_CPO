import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Backtesting import BacktestMultiStrategy
import os



# Crear carpeta para guardar gráficas
os.makedirs("plots", exist_ok=True)

data = pd.read_csv("daily_dbs/dbs_backtesting.csv", index_col=0)
data.index = pd.to_datetime(data.index)

indicator_cols = data.columns[-9:].tolist()
price_cols = data.columns.difference(indicator_cols).tolist()

# === Cargar modelo SVR ===
svr_models = {
    'mid_cap': pickle.load(open("trained_models\LowCaps_SVR_mc.pkl", "rb")),
}

# === Cargar modelo XGBoost ===
"""
xgboost_models = {
    'high_cap': pickle.load(open("trained_models/Dataset_1_XGBoost_Sharpe.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/Dataset_2_XGBoost_Sharpe.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/Dataset_3_XGBoost_Sharpe.pkl", "rb"))
}
"""

# === Inicializar clasificador ===
#classifier = pickle.load(open("portfolio_classifier.pkl", "rb"))

# === Simulaciones ===
n_simulations = 10
results = []
risk_free_rate = 0.042
rfr_daily = risk_free_rate/252
strategies = ["SVR-CPO", "XGBoost-CPO", "EqualWeight", "MinVar", "MaxSharpe"]

for sim in range(n_simulations):
    print(f"Simulación {sim + 1}/{n_simulations}...")
    sampled_assets = np.random.choice(price_cols, size=20, replace=False).tolist()
    price_multi = pd.concat([data[sampled_assets]], axis=1, keys=["Price"])
    ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
    combined_data = pd.concat([price_multi, ind_multi], axis=1)
    cap_type = "mid_cap"
    print(f"Cap type seleccionado para esta simulación: {cap_type}")

    # Correr backtesting
    bt = BacktestMultiStrategy(combined_data, svr_models)
    

    bt.simulate(cap_type)

    daily_returns = bt.evolution()

    for strategy, path in bt.results.items():
        history = daily_returns[strategy]
        Pt, Po = path[-1], path[0]

        if not history.empty:
            rendimiento_anual = history.mean() * 252
            print(f"[DEBUG] N de retornos diarios en {strategy}: {len(history)}. Numero de valores unicos en history: {history.nunique()}")
            print(f"[DEBUG] Std sin annualizar: {history.std(ddof=1)}")
            std_anual = history.std(ddof=1) * np.sqrt(252)
            excess_returns = history - rfr_daily
            downside_returns = excess_returns[excess_returns < rfr_daily]
            downside_deviation = downside_returns.std(ddof=1) * np.sqrt(252) if not downside_returns.empty else np.nan
            sortino = excess_returns.mean() / downside_deviation
        else:
            rendimiento_anual = std_anual = downside_deviation = sortino = np.nan
            
        results.append({
            "Simulación": sim + 1,
            "Start Date": bt.start_date,
            "End Date": bt.end_date,
            "Metodología": strategy,
            "Rendimiento Anual Promedio": rendimiento_anual,
            "Desviación Estándar": std_anual,
            "Rendimiento Efectivo": (Pt / Po) - 1,
            "Downside Risk": downside_deviation,
            "CAGR": (Pt / Po) ** (1 / ((bt.end_date - bt.start_date).days / 365.25)) - 1,
            'Sortino ratio': sortino
        })

# === Crear DataFrame con los resultados ===
results_df = pd.DataFrame(results)
numeric_cols = ["Rendimiento Anual Promedio",
    "Desviación Estándar",
    "Rendimiento Efectivo",
    "Downside Risk",
    "CAGR",
    "Sortino ratio"]
summary = results_df.groupby("Metodología")[numeric_cols].mean(numeric_only=True)
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