import pandas as pd
import pickle
from backtesting import BacktestMultiStrategy  

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
indicators = pd.read_csv("Data_Base/Data_Base.csv", index_col=0, parse_dates=True)

# Alinear fechas
indicators.index = indicators.index.to_period("M").to_timestamp("M")

# Seleccionar 5 activos
selected_assets = prices.columns[:5].tolist()

# Crear estructura multi-nivel
price_multi = pd.concat([prices[selected_assets]], axis=1, keys=["Price"])
ind_multi = pd.concat([indicators], axis=1, keys=["Indicator"])

# Alinear índices comunes
common_index = price_multi.index.intersection(ind_multi.index)
combined_data = pd.concat([price_multi.loc[common_index], ind_multi.loc[common_index]], axis=1)


with open("trained_models/Dataset_1_SVR.pkl", "rb") as f:
    svr_model = pickle.load(f)

svr_models = {'high_cap': svr_model, 'mid_cap': svr_model, 'low_cap': svr_model}
xgboost_model = svr_model  

class CustomBacktest(BacktestMultiStrategy):
    def select_assets(self, date):
        return selected_assets
    def get_cap_type(self, date):
        return 'high_cap'


bt = CustomBacktest(combined_data, svr_models, xgboost_model)
bt.simulate()

# === Mostrar resultados ===
print(bt.evaluate())
bt.plot_results()
