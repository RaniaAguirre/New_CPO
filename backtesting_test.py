import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Backtesting import BacktestMultiStrategy, AssetClassifier
import os

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

# === Simulaciones ===
n_simulations = 1_000
results = []
all_paths = {method: [] for method in ['SVR-CPO', 'XGBoost-CPO', 'EqualWeight', 'MinVar', 'MaxSharpe']}
risk_free_rate = 0.042

rebalance_template = pd.date_range("2015-01-01", "2025-01-01", freq='12MS') + pd.offsets.MonthBegin(1)

for sim in range(n_simulations):
    print(f"Simulación {sim + 1}/{n_simulations}...")
    sampled_assets = select_valid_assets(data, price_cols, 15, rebalance_template)

    price_multi = pd.concat([data[sampled_assets]], axis=1, keys=["Price"])
    ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
    combined_data = pd.concat([price_multi, ind_multi], axis=1)

    bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models)
    bt.classifier = AssetClassifier(data)
    bt.simulate()

    for strategy, values in bt.results.items():
        returns = np.diff(values) / values[:-1]
        portfolio_return = np.mean(returns)

        excess_return = portfolio_return - risk_free_rate
        downside_returns = returns[returns < risk_free_rate]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else np.nan
        sortino = excess_return / downside_deviation if downside_deviation and downside_deviation != 0 else np.nan

        results.append({
            "Simulación": sim + 1,
            "Metodología": strategy,
            "Rendimiento Anual Promedio": portfolio_return,
            "Desviación Estándar": np.std(returns),
            "Rendimiento Efectivo": np.prod(1 + returns) - 1,
            "Downside Risk": downside_deviation,
            "CAGR": (values[-1] / values[0]) ** (1 / ((bt.end_date - bt.start_date).days / 365.25)) - 1,
            'Sortino ratio': sortino
        })

        all_paths[strategy].append(values)

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

mean_paths = {method: np.mean(np.array(all_paths[method]), axis=0) for method in all_paths}
plt.figure(figsize=(10, 6))
for method, avg_path in mean_paths.items():
    plt.plot(avg_path, label=method)
plt.title("Evolución Promedio del Valor del Portafolio por Metodología")
plt.xlabel("Rebalanceo Anual")
plt.ylabel("Valor del Portafolio Promedio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/evolucion_promedio_portafolio.png")
plt.close()

# === Guardar resultados ===
results_df.to_csv("plots/resultados_simulaciones.csv", index=False)
summary.to_csv("plots/resumen_metricas.csv")

with open("plots/results_list.pkl", "wb") as f:
    pickle.dump(results, f)