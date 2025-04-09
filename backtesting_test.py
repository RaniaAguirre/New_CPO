import pandas as pd
import pickle
from Backtesting import BacktestMultiStrategy, AssetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Cargar el dataset combinado ===
data = pd.read_csv("Backtesting_data.csv", index_col=0, parse_dates=True)

# === Definir columnas ===
price_cols = data.columns[1:15].tolist()  # columnas 2 a 16 (15 activos)
indicator_cols = [col for col in data.columns if col not in price_cols]

# === Crear estructura MultiIndex ===
price_multi = pd.concat([data[price_cols]], axis=1, keys=["Price"])
ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
combined_data = pd.concat([price_multi, ind_multi], axis=1)

# === Cargar modelo SVR ===
svr_models = {
    'high_cap': pickle.load(open("trained_models/Dataset_1_SVR.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/Dataset_2_SVR.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/Dataset_3_SVR.pkl", "rb"))
}

# === Cargar modelo XGBoost ===
xgboost_models = {
    'high_cap': pickle.load(open("trained_models/Dataset_1_XGBoost_Sharpe.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/Dataset_2_XGBoost_Sharpe.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/Dataset_3_XGBoost_Sharpe.pkl", "rb"))
}

# === Instanciar clasificador ===
classifier = AssetClassifier(data)

# === Simulaciones ===
n_simulations = 5
results = []
all_paths = {method: [] for method in ['SVR-CPO', 'XGBoost-CPO', 'EqualWeight', 'MinVar', 'MaxSharpe']}


for sim in range(n_simulations):
    bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models)
    bt.classifier = classifier
    bt.simulate()

    for strategy, values in bt.results.items():
        returns = np.diff(values) / values[:-1]
        downside_returns = returns[returns < 0]

        results.append({
            "Simulación": sim + 1,
            "Metodología": strategy,
            "Rendimiento Anual Promedio": np.mean(returns) * 12,
            "Desviación Estándar": np.std(returns) * np.sqrt(12),
            "Rendimiento Efectivo": np.prod(1 + returns) - 1,
            "Downside Risk": (np.sqrt(np.mean(downside_returns ** 2)))*np.sqrt(12) if len(downside_returns) > 0 else 0,
            "CAGR": (values[-1] / values[0]) ** (1 / ((bt.end_date - bt.start_date).days / 365.25)) - 1
        })

# === Crear DataFrame con los resultados ===
results_df = pd.DataFrame(results)

# === Mostrar promedio de cada métrica por metodología ===
summary = results_df.groupby("Metodología").mean(numeric_only=True)
print("\nResumen de métricas promedio por metodología:")
print(summary)

# === Gráficas ===
# Gráfico de barras de rendimiento promedio
summary["Rendimiento Anual Promedio"].plot(kind='bar', title='Rendimiento Anual Promedio por Metodología', ylabel='Promedio', xlabel='Metodología')
plt.grid(True)
plt.show()

# Boxplot de CAGR
sns.boxplot(data=results_df, x="Metodología", y="CAGR")
plt.title("Distribución del CAGR por Metodología")
plt.grid(True)
plt.show()

# Histograma por metodología
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Rendimiento Anual Promedio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Rendimiento Anual Promedio: {method}")
    axes[i].set_xlabel("Rendimiento Anual Promedio")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.show()

# Evolución promedio de cada metodología
mean_paths = {method: np.mean(np.array(all_paths[method]), axis=0) for method in all_paths}

plt.figure(figsize=(10, 6))
for method, avg_path in mean_paths.items():
    plt.plot(avg_path, label=method)
plt.title("Evolución Promedio del Valor del Portafolio por Metodología")
plt.xlabel("Rebalanceo Anual")
plt.ylabel("Valor del Portafolio Promedio")
plt.legend()
plt.grid(True)
plt.show()


# Comparación de CAGR promedio
summary["CAGR"].plot(kind="bar", title="CAGR Promedio por Metodología", ylabel="CAGR Promedio", xlabel="Metodología")
plt.grid(True)
plt.show()
