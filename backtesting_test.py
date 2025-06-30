import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Backtesting import BacktestMultiStrategy
import os
import joblib


# Crear carpeta para guardar gráficas
os.makedirs("plots2", exist_ok=True)

data = pd.read_csv("daily_dbs/dbs_backtesting.csv", index_col=0)
data.index = pd.to_datetime(data.index)

indicator_cols = data.columns[-9:].tolist()
price_cols = data.columns.difference(indicator_cols).tolist()

df_caps = pd.read_csv("market_caps_kmeans.csv").set_index("Ticker")
market_caps = df_caps["Market Cap"].to_dict()

# Cargar modelo SVR
svr_models = {
    'high_cap': pickle.load(open("trained_models/HighCap_SVR_2.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/MidCap_SVR_2.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/LowCap_SVR_2.pkl", "rb"))
}

# Cargar modelo XGBoost
xgboost_models = {
    'high_cap': pickle.load(open("trained_models/xgboost_model_HighCap.pkl", "rb")),
    'mid_cap': pickle.load(open("trained_models/xgboost_model_MidCap.pkl", "rb")),
    'low_cap': pickle.load(open("trained_models/xgboost_model_LowCap.pkl", "rb"))
}

# Inicializar clasificador
kmeans, cluster_map, _, _ = joblib.load("trained_models/kmeans_model.pkl")
id_to_key = {
    -1: "low_cap",
     0: "mid_cap",
     1: "high_cap"
}

# Simulaciones 
n_simulations = 1_000

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

    def classify_portfolio(sampled_assets):
        weights = [1/len(sampled_assets)] * len(sampled_assets)
        cap = sum(market_caps.get(t, 0) * w for t, w in zip(sampled_assets, weights))
        log_cap = np.log1p(cap)
        cluster_idx = kmeans.predict([[log_cap]])[0]
        cluster_id = cluster_map[cluster_idx]
        cap_key = id_to_key[cluster_id]
        return cap_key
    
    cap_type = classify_portfolio(sampled_assets)
    print(f"Cap type seleccionado para esta simulación: {cap_type}")
    # Correr backtesting
    bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models)
    

    bt.simulate(cap_type)

    daily_returns = bt.evolution()

    for strategy, path in bt.results.items():
        history = daily_returns[strategy]
        Pt, Po = path[-1], path[0]

        if not history.empty:
            rendimiento_anual = history.mean() * 252
            std_anual = history.std(ddof=1) * np.sqrt(252)
            excess_daily = history - rfr_daily
            downside_daily = excess_daily[excess_daily < rfr_daily]
            downside_ann = downside_daily.std() * np.sqrt(252) if not downside_daily.empty else np.nan
            sortino = ((rendimiento_anual - risk_free_rate) / downside_ann)
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
            "Downside Risk": downside_ann,
            "CAGR": (Pt / Po) ** (1 / ((bt.end_date - bt.start_date).days / 365.25)) - 1,
            'Sortino ratio': sortino
        })

# Crear DataFrame con los resultados
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

# Gráficas

# Plot Rendimiento anual
summary["Rendimiento Anual Promedio"].plot(kind='bar', title='Rendimiento Anual Promedio por Metodología', ylabel='Promedio', xlabel='Metodología')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/rendimiento_anual_promedio.png")
plt.close()

# Histograma Rendimiento Efectivo
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Rendimiento Efectivo"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Rendimiento Efectivo: {method}")
    axes[i].set_xlabel("Rendimiento Efectivo")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histograma_rend_efectivo.png")
plt.close()

# Histograma Desviación Estándar
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Desviación Estándar"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Desviación Estándar: {method}")
    axes[i].set_xlabel("Desviación Estándar")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histograma_volatilidad.png")
plt.close()

# Histograma downside risk
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Downside Risk"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Downside Risk: {method}")
    axes[i].set_xlabel("Downside Risk")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histograma_downside.png")
plt.close()

# Histograma Rendimiento Anual
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Rendimiento Anual Promedio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Rendimiento Anual Promedio: {method}")
    axes[i].set_xlabel("Rendimiento Anual Promedio")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histogramas_rendimiento_anual.png")
plt.close()

# Histograma Ratio de Sortino
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Sortino ratio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del Ratio de Sortino: {method}")
    axes[i].set_xlabel("Ratio de Sortino")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histogramas_sortino.png")
plt.close()

# Hisotgrama CAGR
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["CAGR"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histograma del CAGR: {method}")
    axes[i].set_xlabel("CAGR")
    axes[i].set_ylabel("Densidad")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots2/histogramas_cagr.png")
plt.close()

# Boxplot Desviación Estándar
sns.boxplot(data=results_df, x="Metodología", y="Desviación Estándar")
plt.title("Distribución de la volatilidad por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_std.png")
plt.close()

# Boxplot downside risk
sns.boxplot(data=results_df, x="Metodología", y="Downside Risk")
plt.title("Distribución del Downside Risk por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_downside.png")
plt.close()

# Boxplot CAGR
sns.boxplot(data=results_df, x="Metodología", y="CAGR")
plt.title("Distribución del CAGR por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_cagr.png")
plt.close()

# Boxplot ratio de sortino
sns.boxplot(data=results_df, x="Metodología", y="Sortino ratio")
plt.title("Distribución del Sortino Ratio por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_sortino.png")
plt.close()

# Boxplot Rendimiento Anual
sns.boxplot(data=results_df, x="Metodología", y="Rendimiento Anual Promedio")
plt.title("Distribución del Rendimiento Anual Promedio por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_rend_anual.png")
plt.close()

# Boxplot Rend Efectivo
sns.boxplot(data=results_df, x="Metodología", y="Rendimiento Efectivo")
plt.title("Distribución del Rendimiento Efectivo por Metodología")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots2/boxplot_rend_efectivo.png")
plt.close()


results_df.to_csv("plots2/resultados_simulaciones.csv", index=False)
summary.to_csv("plots2/resumen_metricas.csv")

with open("plots2/results_list.pkl", "wb") as f:
    pickle.dump(results, f)