import pandas as pd
import pickle
from Backtesting import BacktestMultiStrategy

# === Cargar el dataset combinado ===
data = pd.read_csv("Backtesting_data.csv", index_col=0, parse_dates=True)

# === Definir columnas ===
price_cols = data.columns[1:15].tolist()  # columnas 2 a 16 (15 activos)
indicator_cols = [col for col in data.columns if col not in price_cols]

# === Crear estructura MultiIndex ===
price_multi = pd.concat([data[price_cols]], axis=1, keys=["Price"])
ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
combined_data = pd.concat([price_multi, ind_multi], axis=1)

# === Cargar modelo SVR (único modelo, lo usaremos para todos los tipos) ===
with open("trained_models/Dataset_2_SVR.pkl", "rb") as f:
    svr_model = pickle.load(f)

# === Cargar modelo XGBoost (único modelo) ===
with open("trained_models/Dataset_2_XGBoost_Sharpe.pkl", "rb") as f:
    xgboost_model = pickle.load(f)

# Diccionarios para pasar a la clase (usamos el mismo modelo para todos)
svr_models = {
    'high_cap': svr_model,
    'mid_cap': svr_model,
    'low_cap': svr_model
}

xgboost_models = {
    'high_cap': xgboost_model,
    'mid_cap': xgboost_model,
    'low_cap': xgboost_model
}

# === Instanciar backtest ===
bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models)

# === Ejecutar simulación ===
bt.simulate()

# === Mostrar resultados ===
print(bt.evaluate())
bt.plot_results()
