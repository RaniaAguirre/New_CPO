import pandas as pd
import pickle
from backtesting import BacktestMultiStrategy

# === Cargar el dataset combinado ===
data = pd.read_csv("Backtesting_data.csv", index_col=0, parse_dates=True)

# === Definir columnas ===
price_cols = data.columns[1:16].tolist()  # columnas 2 a 16 (15 activos)
indicator_cols = [col for col in data.columns if col not in price_cols]

# === Crear estructura MultiIndex ===
price_multi = pd.concat([data[price_cols]], axis=1, keys=["Price"])
ind_multi = pd.concat([data[indicator_cols]], axis=1, keys=["Indicator"])
combined_data = pd.concat([price_multi, ind_multi], axis=1)

# === Cargar modelo SVR ===
with open("trained_models/Dataset_1_SVR.pkl", "rb") as f:
    svr_model = pickle.load(f)

# === Cargar modelo XGBoost entrenado ===
with open("trained_models/Dataset_1_XGBoost_Sharpe.pkl", "rb") as f:
    xgboost_model = pickle.load(f)

# Diccionarios para pasar a la clase
svr_models = {'high_cap': svr_model, 'mid_cap': svr_model, 'low_cap': svr_model}

# === Seleccionar activos ===
selected_assets = price_cols

# === Clase personalizada ===
class CustomBacktest(BacktestMultiStrategy):
    def select_assets(self, date):
        return selected_assets
    def get_cap_type(self, date):
        return 'high_cap'

# === Ejecutar simulación ===
bt = CustomBacktest(combined_data, svr_models, xgboost_model)
bt.simulate()

# === Mostrar resultados ===
print(bt.evaluate())
bt.plot_results()
