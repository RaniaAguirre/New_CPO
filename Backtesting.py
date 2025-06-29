import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import random


class BacktestMultiStrategy:
    def __init__(self, data, svr_models, xgboost_models, initial_capital=1_000_000):
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


        self.results = {
            'SVR-CPO': [],
            'XGBoost-CPO': [],
            'EqualWeight': [],
            'MinVar': [],
            'MaxSharpe': []
        }

    def get_random_backtest_period(self):
        first_valid = pd.Timestamp("2010-07-01")
        last_valid = self.data.index.max() - pd.DateOffset(years=5)
        if last_valid < first_valid:
            raise ValueError("Pocos datos para un backtesting de 5 años")
        span_days = (last_valid-first_valid).days
        offset = random.randint(0, span_days)
        start_date = first_valid + pd.Timedelta(days=offset)
        end_date = start_date + pd.DateOffset(years=5)
        return start_date, end_date


    def simulate(self, cap_type):
        self.cap_type = cap_type
        self.start_date, self.end_date = self.get_random_backtest_period()
        rebalance_dates = pd.date_range(start=self.start_date, end=self.end_date, freq=pd.DateOffset(months=6))
        if rebalance_dates[-1] < self.end_date:
            rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([self.end_date]))
        strategies = list(self.results.keys())
        portfolio_values = {s: [self.initial_capital] for s in strategies}

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            selected_assets = self.select_assets(date)

            if not selected_assets:
                for strategy in strategies:
                    portfolio_values[strategy].append(portfolio_values[strategy][-1])
                continue

            print(f"\nRebalanceo: {date.date()} -> {next_date.date()}")

            print(f"Activos seleccionados: {selected_assets}")

            if hasattr(self, "portfolio_classifier") and self.portfolio_classifier is not None:
                try:
                    classification = self.portfolio_classifier.classify(selected_assets)
                    print(f"Portfolio classified as: {classification}")
                except Exception as e:
                    print(f"[PortfolioClassifier ERROR]: {e}")

            weights_dict = {
                'SVR-CPO': self.allocate_svr(selected_assets, date, self.cap_type),
                'XGBoost-CPO': self.allocate_xgboost(selected_assets, date, self.cap_type),
                'EqualWeight': self.equal_weight(selected_assets),
                'MinVar': self.min_var(selected_assets, date),
                'MaxSharpe': self.max_sharpe(selected_assets, date)
            }

            prices = self.data.loc[:, ('Price', selected_assets)]
            date_asof = self.data.index[self.data.index.get_indexer([date], method='pad')[0]]
            next_date_asof = self.data.index[self.data.index.get_indexer([next_date], method='pad')[0]]

            price_subset = prices.loc[date_asof:next_date_asof]

            # Revisar si el índice tiene frecuencia diaria
            date_diffs = price_subset.index.to_series().diff().dropna()
            unique_diffs = date_diffs.unique()

            print(f"[DEBUG] Frecuencias únicas entre fechas en precios ({date_asof} a {next_date_asof}): {unique_diffs}")
            
            returns = prices.pct_change().loc[date_asof:next_date_asof].dropna()

            for strategy in strategies:
                weights = weights_dict[strategy]
                weight_vec = np.array([weights.get(a, 0) for a in selected_assets])
                print(f"{strategy} pesos: {weight_vec}, suma: {np.sum(weight_vec)}")
                if returns.shape[1] != len(weight_vec):
                    print(f"[SKIP] {strategy} - Shape mismatch between returns ({returns.shape[1]}) and weights ({len(weight_vec)})")
                    portfolio_values[strategy].append(portfolio_values[strategy][-1])
                    continue

                strat_returns = returns.dot(weight_vec)

                if strat_returns.empty:
                    print(f"[WARNING] No returns for {strategy} between {date} and {next_date}")
                    portfolio_values[strategy].append(portfolio_values[strategy][-1])
                    continue

                initial_value = portfolio_values[strategy][-1]

                cumulative_returns = (1 + strat_returns).cumprod()
                new_values = initial_value * cumulative_returns

                portfolio_values[strategy].extend(new_values.tolist())

        for strategy in strategies:
            self.results[strategy] = portfolio_values[strategy]
            print(f"{strategy} final portfolio length: {len(self.results[strategy])}, final value: {self.results[strategy][-1]}")


    def evolution(self):
        daily_returns = {}
        for strategy, values in self.results.items():
            values = np.array(values)

            if len(values) <= 1:
                print(f"[ERROR] Portfolio {strategy} has insufficient values: {values}")
                daily_returns[strategy] = pd.Series(dtype=float)
                continue

            values_series = pd.Series(values)
            returns = values_series.pct_change().dropna()
            daily_returns[strategy] = returns
        return daily_returns

    def select_assets(self, date):
        all_assets = self.data['Price'].columns.tolist()
        all_assets = [a for a in all_assets if a.lower() != "date"]
        asof_date = self.data.index[self.data.index.get_indexer([date], method='pad')[0]]
        available_assets = [asset for asset in all_assets if not pd.isna(self.data.loc[asof_date, ('Price', asset)])]
        return available_assets


    def allocate_svr(self, assets, date, cap_type, n_samples=100):
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

    
    def allocate_xgboost(self, assets, date, cap_type, n_samples=100):
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
        prices = (
            self.data
            .loc[:date, ('Price', assets)]
            .dropna(how='any')             # filas sin ningún NA
            .apply(pd.to_numeric, errors='coerce')
        )
        if prices.shape[0] < 2:
            return self.equal_weight(assets)
        returns = prices.pct_change().dropna()
        if returns.shape[0] < 2:
            return self.equal_weight(assets)
        cov = returns.cov().values
        n = cov.shape[0]
        inv_cov = np.linalg.pinv(cov)
        raw_w = inv_cov.dot(np.ones(n))
        raw_w = np.clip(raw_w, 0, None)
        if raw_w.sum() == 0:
            return self.equal_weight(assets)
        w = raw_w / raw_w.sum()

        return dict(zip(assets, w))

    def max_sharpe(self, assets, date, risk_free_rate=0.05):
        prices = self.data.loc[:date, ('Price', assets)].dropna()
        prices = prices.apply(pd.to_numeric, errors='coerce')
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        weights = self.max_sharpe_portfolio_constrained(mean_returns.values, cov_matrix.values, risk_free_rate)
        return dict(zip(assets, weights))


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