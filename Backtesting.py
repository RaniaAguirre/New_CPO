import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


class BacktestMultiStrategy:
    def __init__(self, data, svr_models, xgboost_models, ranking_dict, initial_capital=1_000_000, cap_type = 'mid_cap'):
        """
        data: DataFrame que contiene precios históricos (mensuales) y los indicadores de mercado,
              debe tener columnas multi-indexadas: nivel 0 = tipo de dato ("Price", "Indicator"), nivel 1 = nombre
        svr_models: dict con modelos SVR {'high_cap': model, 'mid_cap': model, 'low_cap': model}
        xgboost_model: modelo CPO original
        initial_capital: capital inicial para el portafolio
        """
        self.data = data
        self.svr_models = svr_models
        self.xgboost_models = xgboost_models
        self.initial_capital = initial_capital
        self.cap_type = cap_type

        self.start_date = pd.to_datetime('2015-01-01')
        self.end_date = pd.to_datetime('2025-01-01')
        self.training_period = pd.to_datetime('2014-01-01')

        self.classifier = AssetClassifier(data['Price'], ranking_dict)

        self.results = {
            'SVR-CPO': [],
            'XGBoost-CPO': [],
            'EqualWeight': [],
            'MinVar': [],
            'MaxSharpe': []
        }

    def simulate(self, monthly=False):
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq='12MS') + pd.offsets.MonthBegin(1)
        strategies = list(self.results.keys())
        portfolio_values = {s: [self.initial_capital] for s in strategies}

        print("Fechas de rebalanceo:")
        print(rebalance_dates)

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            print(f"\nRebalanceo: {date.date()} -> {next_date.date()}")

            train_start = date - pd.DateOffset(years=1)
            train_end = date

            selected_assets = self.select_assets(date)
            #cap_type = self.classifier.get_cap_type(selected_assets)

            if not selected_assets:
                print("No se seleccionaron activos.")
                continue

            print(f"Activos seleccionados: {selected_assets}")
            #print(f"Capitalización predominante: {cap_type}")

            weights_dict = {
                'SVR-CPO': self.allocate_svr(selected_assets, date, self.cap_type),
                'XGBoost-CPO': self.allocate_xgboost(selected_assets, date, self.cap_type),
                'EqualWeight': self.equal_weight(selected_assets),
                'MinVar': self.min_var(selected_assets, date),
                'MaxSharpe': self.max_sharpe(selected_assets, date)
            }

            prices = self.data.loc[:, ('Price', selected_assets)]
            returns = prices.pct_change().loc[date:next_date].dropna()

            for strategy in strategies:
                weights = weights_dict[strategy]
                weight_vec = np.array([weights.get(a, 0) for a in selected_assets])
                print(f"{strategy} pesos: {weight_vec}, suma: {np.sum(weight_vec)}")

                if len(returns) == 0:
                    print(f"No hay retornos disponibles para {strategy} en este periodo.")
                    portfolio_values[strategy].append(portfolio_values[strategy][-1])
                    continue

                if monthly:
                    for dt in returns.index:
                        strat_return = returns.loc[dt].dot(weight_vec)
                        new_value = portfolio_values[strategy][-1] * (1 + strat_return)
                        portfolio_values[strategy].append(new_value)
                else:
                    strat_returns = returns.dot(weight_vec)
                    cumulative = np.prod(1 + strat_returns)
                    new_value = portfolio_values[strategy][-1] * cumulative
                    portfolio_values[strategy].append(new_value)

        for strategy in strategies:
            self.results[strategy] = portfolio_values[strategy]


    def evolution(self):

        monthly_returns = {}
        for strategy, values in self.results.items():
            values = np.array(values)
            returns = np.diff(values) / values[:-1]
            monthly_returns[strategy] = pd.Series(returns)
        return monthly_returns

    def select_assets(self, date):
        all_assets = self.data['Price'].columns.tolist()
        asof_date = self.data.index[self.data.index.get_indexer([date], method='pad')[0]]
        available_assets = [asset for asset in all_assets if not pd.isna(self.data.loc[asof_date, ('Price', asset)])]
        return available_assets


    def allocate_svr(self, assets, date, cap_type, n_samples=1_000):
        model = self.svr_models[cap_type]
        indicators = self.data.loc[self.data.index.asof(date), ('Indicator', slice(None))].values
        candidate_weights = self.sample_weight_combinations(len(assets), n_samples)

        best_score = -np.inf
        best_weights = None

        for w in candidate_weights:
            features = np.concatenate([w, indicators])
            score = model.predict([features])[0]
            if score > best_score:
                best_score = score
                best_weights = w

        return dict(zip(assets, best_weights))

    def allocate_xgboost(self, assets, date, cap_type, n_samples=1_000):
        model = self.xgboost_models[cap_type]
        indicators = self.data.loc[self.data.index.asof(date), ('Indicator', slice(None))].values
        candidate_weights = self.sample_weight_combinations(len(assets), n_samples)

        best_score = -np.inf
        best_weights = None

        for w in candidate_weights:
            features = np.concatenate([w, indicators])
            score = model.predict([features])[0]
            if score > best_score:
                best_score = score
                best_weights = w

        return dict(zip(assets, best_weights))


    def sample_weight_combinations(self, n_assets, n_samples):
        return np.random.dirichlet(np.ones(n_assets), size=n_samples)

    def equal_weight(self, assets):
        n = len(assets)
        return {a: 1/n for a in assets} if n > 0 else {}

    def min_var(self, assets, date):
        prices = self.data.loc[:date, ('Price', assets)].dropna()
        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov()
        weights = self.min_variance_portfolio_constrained(cov_matrix)
        return dict(zip(assets, weights))

    def max_sharpe(self, assets, date, risk_free_rate=0.05):
        prices = self.data.loc[:date, ('Price', assets)].dropna()
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 12
        cov_matrix = returns.cov() * 12
        weights = self.max_sharpe_portfolio_constrained(mean_returns.values, cov_matrix.values, risk_free_rate)
        return dict(zip(assets, weights))

    def min_variance_portfolio_constrained(self, cov_matrix):
        n = len(cov_matrix)
        x0 = np.ones(n) / n  # pesos iniciales iguales

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix.values, w))

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def max_sharpe_portfolio_constrained(self, expected_returns, cov_matrix, risk_free_rate):
        n = len(expected_returns)
        x0 = np.ones(n) / n

        def neg_sharpe(w):
            port_return = np.dot(w, expected_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -((port_return - risk_free_rate) / port_vol)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def normalize_weights(self, weight_dict):
        total = sum(weight_dict.values())
        return {k: v / total for k, v in weight_dict.items()} if total != 0 else weight_dict

    
class AssetClassifier:
    def __init__(self, data: pd.DataFrame, ranking_dict: dict, indicators: list = None):
        """
        Inicializa la clase de clasificación de activos.

        Args:
            data (pd.DataFrame): DataFrame con columnas de tickers e indicadores.
            ranking_dict (dict): Diccionario con los tickers como llave y su ranking (1 a 100) como valor.
            indicators (list, optional): Lista de nombres de columnas que son indicadores de mercado.
        """
        self.data = data
        self.ranking = ranking_dict
        self.indicators = indicators or [
            'MOM', 'Treasury Bond 3M', 'WTI index', 'Dollar index', 'TRCCRB',
            'BCI', 'CCI', 'CLI', 'GPRI', 'Unemployment rate'
        ]
        self.tickers = [col for col in data.columns if col not in self.indicators]

    def get_cap_type(self, selected_assets: list) -> str:
        """
        Clasifica una lista de activos según su ranking.

        Modelos:
            - Modelo 1 (high cap): mayoría en posiciones 1-30
            - Modelo 2 (mid cap): mayoría en posiciones 31-65
            - Modelo 3 (low cap): mayoría en posiciones 66-100
            En caso de empate o falta de mayoría, devuelve 'mid_cap'.

        Args:
            selected_assets (list): Lista de tickers seleccionados.

        Returns:
            str: Tipo de capitalización dominante ('high_cap', 'mid_cap' o 'low_cap')
        """
        count_model1 = sum(1 for t in selected_assets if self.ranking.get(t, 101) <= 30)
        count_model2 = sum(1 for t in selected_assets if 31 <= self.ranking.get(t, 101) <= 65)
        count_model3 = sum(1 for t in selected_assets if 66 <= self.ranking.get(t, 101) <= 100)

        total = len(selected_assets)
        if count_model1 > total / 2:
            return 'high_cap'
        elif count_model2 > total / 2:
            return 'mid_cap'
        elif count_model3 > total / 2:
            return 'low_cap'
        else:
            return 'mid_cap'

