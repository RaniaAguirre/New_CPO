import pandas as pd
import pickle
from Backtesting import BacktestMultiStrategy, AssetClassifier

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


# === Instanciar backtest ===
classifier = AssetClassifier(data)
bt = BacktestMultiStrategy(combined_data, svr_models, xgboost_models)
bt.classifier = classifier

# === Ejecutar simulación ===
bt.simulate()

# === Mostrar resultados ===
print(bt.evaluate())
bt.plot_results()
