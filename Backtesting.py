import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


class BacktestMultiStrategy:
    def __init__(self, data, svr_models, xgboost_models, initial_capital=1000000):
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
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq='12MS') + pd.offsets.MonthBegin(1)
        strategies = list(self.results.keys())
        portfolio_values = {s: [self.initial_capital] for s in strategies}

        print("Fechas de rebalanceo:")
        print(rebalance_dates)

        print("\nPrimeras fechas en el dataset:")
        print(self.data['Price'].index[:10])

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            print(f"\nRebalanceo: {date.date()} ➡ {next_date.date()}")

            train_start = date - pd.DateOffset(years=1)
            train_end = date

            selected_assets = self.select_assets(date)
            cap_type = self.get_cap_type(date)

            if not selected_assets:
                print("No se seleccionaron activos.")
                continue

            print(f"Activos seleccionados: {selected_assets}")
            print(f"Capitalización predominante: {cap_type}")

            weights_dict = {
                'SVR-CPO': self.allocate_svr(selected_assets, date, cap_type),
                'XGBoost-CPO': self.allocate_xgboost(selected_assets, date, cap_type),
                'EqualWeight': self.equal_weight(selected_assets),
                'MinVar': self.min_var(selected_assets, date),
                'MaxSharpe': self.max_sharpe(selected_assets, date)
            }

            prices = self.data.loc[:, ('Price', selected_assets)]
            returns = prices.pct_change().loc[date:next_date].dropna()

            print(f"Primeras filas de retornos entre {date.date()} y {next_date.date()}:")
            print(returns.head())

            for strategy in strategies:
                weights = weights_dict[strategy]
                weight_vec = np.array([weights.get(a, 0) for a in selected_assets])
                print(f"→ {strategy} pesos: {weight_vec}, suma: {np.sum(weight_vec)}")

                if len(returns) == 0:
                    print(f"No hay retornos disponibles para {strategy} en este periodo.")
                    portfolio_values[strategy].append(portfolio_values[strategy][-1])
                    continue

                strat_returns = returns.dot(weight_vec)
                cumulative = np.prod(1 + strat_returns)
                new_value = portfolio_values[strategy][-1] * cumulative
                portfolio_values[strategy].append(new_value)

        for strategy in strategies:
            self.results[strategy] = portfolio_values[strategy]


    def select_assets(self, date):
        all_assets = self.data['Price'].columns.tolist()
        asof_date = self.data.index[self.data.index.get_indexer([date], method='pad')[0]]
        available_assets = [asset for asset in all_assets if not pd.isna(self.data.loc[asof_date, ('Price', asset)])]
        return available_assets


    def get_cap_type(self, date):
        return 'high_cap'

    def allocate_svr(self, assets, date, cap_type, n_samples=1000):
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

    def allocate_xgboost(self, assets, date, cap_type, n_samples=1000):
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

    def max_sharpe(self, assets, date, risk_free_rate=0.01):
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

    def plot_results(self):
        
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
        strategies = list(self.results.keys())
        fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 2.5 * len(strategies)))
        for i, strategy in enumerate(strategies):
            sns.histplot(df_returns[strategy], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Histograma de rendimientos anuales: {strategy}')
            axes[i].set_xlabel('Rendimiento anual')
            axes[i].set_ylabel('Densidad')
            axes[i].grid(True)
        plt.tight_layout()
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
    
class AssetClassifier:
    def __init__(self, data: pd.DataFrame, indicators: list = None):
        """
        Inicializa la clase de clasificación de activos a partir de precios e indicadores.

        Args:
            data (pd.DataFrame): DataFrame con columnas de tickers e indicadores.
            indicators (list, optional): Lista de nombres de columnas que son indicadores de mercado.
        """
        self.data = data
        self.indicators = indicators or [
            'MOM', 'Treasury Bond 3M', 'WTI index', 'Dollar index', 'TRCCRB',
            'BCI', 'CCI', 'CLI', 'GPRI', 'Unemployment rate'
        ]

        # Detectar tickers: columnas que no son indicadores
        self.tickers = [col for col in data.columns if col not in self.indicators]

        # Obtener y clasificar por capitalización real
        raw_caps = self.get_real_market_caps(self.tickers)
        self.capitalizations = {
            ticker: self.classify_market_cap(raw_caps[ticker]) for ticker in self.tickers
        }

    def get_real_market_caps(self, tickers):
        """
        Obtiene la capitalización bursátil de una lista de tickers usando yfinance.

        Args:
            tickers (list): Lista de tickers (str)

        Returns:
            dict: Diccionario ticker -> market cap
        """
        market_caps = {}
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker)
                cap = data.info.get('marketCap', None)
                market_caps[ticker] = cap
            except:
                market_caps[ticker] = None
        return market_caps

    def classify_market_cap(self, cap):
        """
        Clasifica un valor de market cap como small, mid o high.

        Args:
            cap (float): Valor de market cap en dólares.

        Returns:
            str: 'small_cap', 'mid_cap', o 'high_cap'
        """
        if cap is None:
            return 'mid_cap'  # Por defecto
        elif cap >= 10e9:
            return 'high_cap'
        elif cap >= 2e9:
            return 'mid_cap'
        else:
            return 'small_cap'

    def select_assets(self, date: pd.Timestamp, n_assets: int = 5):
        """
        Selecciona hasta n_assets activos con datos disponibles en la fecha dada.

        Args:
            date (pd.Timestamp): Fecha objetivo.
            n_assets (int): Número de activos a seleccionar.

        Returns:
            list: Lista de tickers seleccionados
        """
        asset_columns = self.tickers
        asof_date = self.data.index[self.data.index.get_indexer([date], method='pad')[0]]
        available_assets = [asset for asset in asset_columns if not pd.isna(self.data.loc[asof_date, asset])]
        return sorted(available_assets)[:n_assets]

    def get_cap_type(self, selected_assets: list):
        """
        Determina la capitalización predominante de una lista de activos.

        Args:
            selected_assets (list): Lista de tickers

        Returns:
            str: Tipo de capitalización dominante
        """
        caps = [self.capitalizations.get(asset, 'mid_cap') for asset in selected_assets]
        return max(set(caps), key=caps.count)

