import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

class BacktestMultiStrategy:
    def __init__(self, data, svr_models, xgboost_model, initial_capital=1000000):
        """
        data: DataFrame que contiene precios históricos (mensuales) y los indicadores de mercado,
              debe tener columnas multi-indexadas: nivel 0 = tipo de dato ("Price", "Indicator"), nivel 1 = nombre
        svr_models: dict con modelos SVR {'high_cap': model, 'mid_cap': model, 'low_cap': model}
        xgboost_model: modelo CPO original
        initial_capital: capital inicial para el portafolio
        """
        self.data = data
        self.svr_models = svr_models
        self.xgboost_model = xgboost_model
        self.initial_capital = initial_capital

        self.start_date = pd.to_datetime('2015-01-01')
        self.end_date = pd.to_datetime('2025-01-01')
        self.training_period = pd.to_datetime('2014-01-01')

        self.results = {
            'SVR-CPO': [],
            'XGBoost-CPO': [],
            'EqualWeight': [],
            'MinVar': [],
            'MaxSharpe': []
        }

    def simulate(self):
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq='12MS')
        strategies = list(self.results.keys())
        portfolio_values = {s: [self.initial_capital] for s in strategies}

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            train_start = date - pd.DateOffset(years=1)
            train_end = date

            selected_assets = self.select_assets(date)
            cap_type = self.get_cap_type(date)

            if not selected_assets:
                continue

            weights_dict = {
                'SVR-CPO': self.allocate_svr(selected_assets, date, cap_type),
                'XGBoost-CPO': self.allocate_xgboost(selected_assets, date),
                'EqualWeight': self.equal_weight(selected_assets),
                'MinVar': self.min_var(selected_assets, date),
                'MaxSharpe': self.max_sharpe(selected_assets, date)
            }

            prices = self.data.loc[:, ('Price', selected_assets)]
            returns = prices.pct_change().loc[date:next_date].dropna()

            for strategy in strategies:
                weights = weights_dict[strategy]
                weight_vec = np.array([weights[a] for a in selected_assets])
                strat_returns = returns.dot(weight_vec)
                cumulative = np.prod(1 + strat_returns)
                new_value = portfolio_values[strategy][-1] * cumulative
                portfolio_values[strategy].append(new_value)

        for strategy in strategies:
            self.results[strategy] = portfolio_values[strategy]

    def select_assets(self, date):
        return []

    def get_cap_type(self, date):
        return 'high_cap'

    def allocate_svr(self, assets, date, cap_type, n_samples=1000):
        model = self.svr_models[cap_type]
        indicators = self.data.loc[date, ('Indicator', slice(None))].values
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

    def allocate_xgboost(self, assets, date, n_samples=1000):
        model = self.xgboost_model
        indicators = self.data.loc[date, ('Indicator', slice(None))].values
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
        weights = self.min_variance_portfolio_analytical(cov_matrix)
        return dict(zip(assets, weights))

    def max_sharpe(self, assets, date, risk_free_rate=0.01):
        prices = self.data.loc[:date, ('Price', assets)].dropna()
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 12
        cov_matrix = returns.cov() * 12
        weights = self.max_sharpe_portfolio_analytical(mean_returns.values, cov_matrix.values, risk_free_rate)
        return dict(zip(assets, weights))

    def min_variance_portfolio_analytical(self, cov_matrix):
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        return np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))

    def max_sharpe_portfolio_analytical(self, expected_returns, cov_matrix, risk_free_rate):
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        excess_returns = expected_returns - risk_free_rate * ones
        num = np.dot(inv_cov, excess_returns)
        denom = np.dot(ones.T, num)
        return num / denom

    def normalize_weights(self, weight_dict):
        total = sum(weight_dict.values())
        return {k: v / total for k, v in weight_dict.items()} if total != 0 else weight_dict

    def plot_results(self):
        import seaborn as sns

        # Línea de evolución total
        plt.figure(figsize=(10, 5))
        for strategy, values in self.results.items():
            plt.plot(values, label=strategy)
        plt.title('Evolución del valor del portafolio (2015-2025)')
        plt.xlabel('Rebalanceos anuales')
        plt.ylabel('Valor del portafolio')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Boxplot de rendimientos anuales
        annual_returns = {}
        for strategy, values in self.results.items():
            returns = np.diff(values) / values[:-1]
            annual_returns[strategy] = returns

        df_returns = pd.DataFrame(annual_returns)
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_returns)
        plt.title('Distribución de rendimientos anuales por estrategia')
        plt.ylabel('Rendimiento anual')
        plt.grid(True)
        plt.show()

        # Histograma de rendimientos
        plt.figure(figsize=(10, 6))
        for strategy in self.results:
            sns.histplot(df_returns[strategy], kde=True, label=strategy, stat="density", bins=10, alpha=0.5)
        plt.title('Histogramas de rendimientos anuales por estrategia')
        plt.xlabel('Rendimiento anual')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self):
        cagr_results = {}
        n_years = (self.end_date - self.start_date).days / 365.25
        for strategy, values in self.results.items():
            initial_value = values[0]
            final_value = values[-1]
            cagr = (final_value / initial_value) ** (1 / n_years) - 1
            cagr_results[strategy] = cagr
        return pd.DataFrame(list(cagr_results.items()), columns=['Metodología', 'CAGR'])
