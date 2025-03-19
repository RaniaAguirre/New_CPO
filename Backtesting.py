import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import warnings
import quantstats as qs

class DynamicBacktest:
    
    def __init__(self, results, prices, initial_capital, benchmark_data=None, benchmark_ticker='^GSPC'):
        """
        Inicializa la clase con los parámetros dados y descarga el benchmark solo si no se proporciona.
        
        Args:
        - results: DataFrame con columnas ['Date', 'Chosen Universe', 'Selected Stocks']
        - prices: Diccionario con tickers como claves y series de pandas con precios como valores.
        - initial_capital: Capital inicial para el portafolio.
        - benchmark_data: Serie de tiempo opcional con los precios del benchmark.
        - benchmark_ticker: Ticker del benchmark para descargar si no se proporciona data.
        """
        self.results = results
        self.prices = prices
        self.initial_capital = initial_capital
        self.portfolio_values_sortino = []
        self.portfolio_values_benchmark = []
        self.weights_history_sortino = []

        self.benchmark_data = benchmark_data if benchmark_data is not None else self.download_benchmark_data(benchmark_ticker)
        self.benchmark_data.index = pd.to_datetime(self.benchmark_data.index)

        self.benchmark_shares = None

        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                message="Values in x were outside bounds during a minimize step, clipping to bounds")

        self.run_backtest()

    def download_benchmark_data(self, ticker):
        """
        Descarga los datos históricos del benchmark usando yfinance.
        """
        benchmark_df = yf.download(ticker, start=self.results['Date'].min(), end=self.results['Date'].max(), progress=False)
        return benchmark_df['Adj Close']

    def calculate_sortino_weights(self, selected_stocks, end_date):
        """
        Calcula los pesos óptimos basados en el ratio Sortino.
        """
        start_date = end_date - pd.DateOffset(days=365)
        returns = pd.DataFrame({
            ticker: self.prices[ticker].loc[start_date:end_date].pct_change().dropna() 
            for ticker in selected_stocks if ticker in self.prices
        }).dropna(axis=1)

        def sortino_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            downside_std = np.sqrt(np.sum((returns[returns < 0].fillna(0).mean() * weights) ** 2) * 252)
            return -portfolio_return / downside_std if downside_std != 0 else np.inf

        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for _ in range(n))
        initial_weights = n * [1. / n,]

        optimized = sco.minimize(sortino_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return {ticker: optimized.x[i] for i, ticker in enumerate(returns.columns)}

    def calculate_semivariance_weights(self, selected_stocks, end_date):
        """
        Calcula los pesos óptimos minimizando la semivarianza del portafolio.
        """
        start_date = end_date - pd.DateOffset(days=365)
        returns = pd.DataFrame({
            ticker: self.prices[ticker].loc[start_date:end_date].pct_change().dropna()
            for ticker in selected_stocks if ticker in self.prices
        }).dropna(axis=1)
    
        def semivariance_loss(weights):
            portfolio_return = np.dot(returns, weights)
            downside_returns = portfolio_return[portfolio_return < 0]
            semivariance = np.mean(downside_returns ** 2)  # Semivarianza
            return semivariance
    
        n = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((0.05, 1) for _ in range(n))
        initial_weights = n * [1. / n,]
    
        optimized = sco.minimize(semivariance_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return {ticker: optimized.x[i] for i, ticker in enumerate(returns.columns)}
    
    def rebalance_portfolios(self, capital, selected_stocks, universe_type, date):
        """
        Realiza el rebalanceo utilizando Sortino para ofensivo y semivarianza para defensivo.
        """
        if universe_type == 'Offensive':
            weights = self.calculate_sortino_weights(selected_stocks, date)
        else:
            weights = self.calculate_semivariance_weights(selected_stocks, date)
    
        prices_at_rebalance = {ticker: self.prices[ticker].asof(date) for ticker in selected_stocks}
        shares = {ticker: (capital * weights[ticker]) / prices_at_rebalance[ticker] for ticker in selected_stocks}
        return shares, weights

    
    def run_backtest(self):
        """
        Ejecuta el backtest, manejando portafolios y benchmark.
        """
        capital = self.initial_capital
        rebalance_dates = pd.to_datetime(self.results['Date'].tolist())
        start_date = rebalance_dates[0]
        selected_stocks = self.results.iloc[0]['Selected Stocks']
        universe_type = self.results.iloc[0]['Chosen Universe']

        shares, weights = self.rebalance_portfolios(capital, selected_stocks, universe_type, start_date)
        self.weights_history_sortino.append((start_date, weights))
        benchmark_initial_price = self.benchmark_data.asof(start_date)
        self.benchmark_shares = self.initial_capital / benchmark_initial_price

        all_dates = pd.date_range(start_date, self.benchmark_data.index[-1])
        for current_date in all_dates:
            if current_date in rebalance_dates:
                idx = rebalance_dates.get_loc(current_date)
                capital = self.calculate_portfolio_value(shares, selected_stocks, current_date)
                selected_stocks = self.results.iloc[idx]['Selected Stocks']
                universe_type = self.results.iloc[idx]['Chosen Universe']
                shares, weights = self.rebalance_portfolios(capital, selected_stocks, universe_type, current_date)
                self.weights_history_sortino.append((current_date, weights))

            daily_value = self.calculate_portfolio_value(shares, selected_stocks, current_date)
            daily_value_benchmark = self.benchmark_shares * self.benchmark_data.asof(current_date)

            self.portfolio_values_sortino.append((current_date, daily_value))
            self.portfolio_values_benchmark.append((current_date, daily_value_benchmark))

    
    def calculate_portfolio_value(self, shares, selected_stocks, date):
        """
        Calcula el valor del portafolio para la fecha dada.
        """
        return sum(shares.get(ticker, 0) * self.prices[ticker].asof(date) for ticker in selected_stocks)


    def get_portfolio_values(self):
        sortino_df = pd.DataFrame(self.portfolio_values_sortino, columns=['Date', 'Sortino Portfolio Value'])
        benchmark_df = pd.DataFrame(self.portfolio_values_benchmark, columns=['Date', 'Benchmark Portfolio Value'])
        
        # Convertir 'Date' a datetime e indexar
        sortino_df['Date'] = pd.to_datetime(sortino_df['Date'])
        sortino_df.set_index('Date', inplace=True)
        benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
        benchmark_df.set_index('Date', inplace=True)
        
        # Generar un rango completo de fechas para asegurar la continuidad
        all_dates = pd.date_range(start=sortino_df.index.min(), end=sortino_df.index.max(), freq='D')
        sortino_df = sortino_df.reindex(all_dates).ffill().bfill()
        benchmark_df = benchmark_df.reindex(all_dates).ffill().bfill()
        
        # Combinar y rellenar posibles NaN resultantes de la intersección de fechas
        portfolio_values_df = sortino_df.join(benchmark_df, how='inner')
        portfolio_values_df.ffill(inplace=True)
        
        return portfolio_values_df


    def get_portfolio_series(self):
        """
        Convierte los valores del portafolio en una serie temporal.
        """
        sortino_df = pd.DataFrame(self.portfolio_values_sortino, columns=['Date', 'Sortino Portfolio Value'])
        sortino_df.set_index('Date', inplace=True)
        return sortino_df['Sortino Portfolio Value']

    def plot_strategies(self):
        """
        Grafica la evolución de los portafolios y el benchmark.
        """
        portfolio_values_df = self.get_portfolio_values()
        
        # Graficar usando el índice de fechas
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_values_df.index, portfolio_values_df['Sortino Portfolio Value'], label='Sortino Portfolio')
        plt.plot(portfolio_values_df.index, portfolio_values_df['Benchmark Portfolio Value'], label='Benchmark', color='red', linewidth=2)
        plt.title('Evolución del Portafolio y Benchmark')
        plt.xlabel('Fecha')
        plt.ylabel('Valor del Portafolio')
        plt.legend()
        plt.grid()
        plt.show()


    def evaluate_portfolios(self):
        """
        Calcula métricas clave de comparación entre el portafolio y el benchmark,
        con control adicional para valores extremos y supresión de advertencias de overflow.
        
        Returns:
            metrics_df (pd.DataFrame): DataFrame con las métricas calculadas para el portafolio y el benchmark.
        """
        # Configuración temporal para ignorar advertencias de overflow en numpy
        old_settings = np.seterr(over='ignore')
    
        try:
            # Obtener series de portafolio y benchmark sin límites para Beta y Alpha
            port_series = self.get_portfolio_series()
            benchmark_data = self.benchmark_data
    
            # Asegurar que ambos están alineados en frecuencia semanal para mayor estabilidad
            strategy_weekly = port_series.resample('W').last().pct_change().dropna()
            benchmark_weekly = benchmark_data.resample('W').last().pct_change().dropna()
    
            # Alinear fechas de ambos retornos semanales
            aligned_returns = pd.DataFrame({'Strategy': strategy_weekly, 'Benchmark': benchmark_weekly}).dropna()
    
            # Calcular métricas clave usando quantstats para el portafolio
            metrics = {
                'CAGR': qs.stats.cagr(port_series),
                'Sharpe Ratio': qs.stats.sharpe(port_series),
                'Sortino Ratio': qs.stats.sortino(port_series),
                'Max Drawdown': qs.stats.max_drawdown(port_series),
                'Volatility': qs.stats.volatility(port_series),
                'VaR (5%)': qs.stats.value_at_risk(port_series)
            }
    
            # Calcular Alpha y Beta manualmente con datos semanales alineados
            try:
                cov_matrix = aligned_returns.cov()
                metrics['Beta'] = cov_matrix.loc['Strategy', 'Benchmark'] / cov_matrix.loc['Benchmark', 'Benchmark']
                metrics['Alpha'] = (aligned_returns['Strategy'].mean() - metrics['Beta'] * aligned_returns['Benchmark'].mean()) * 52  # Anualizar Alpha
            except Exception as e:
                print(f"Error calculating Alpha and Beta manually: {e}")
                metrics['Beta'] = np.nan
                metrics['Alpha'] = np.nan
    
            # Calcular métricas clave para el benchmark, estableciendo Beta en 1 y Alpha en 0
            benchmark_metrics = {
                'CAGR': qs.stats.cagr(benchmark_data),
                'Sharpe Ratio': qs.stats.sharpe(benchmark_data),
                'Sortino Ratio': qs.stats.sortino(benchmark_data),
                'Max Drawdown': qs.stats.max_drawdown(benchmark_data),
                'Volatility': qs.stats.volatility(benchmark_data),
                'VaR (5%)': qs.stats.value_at_risk(benchmark_data),
                'Beta': 1,    # Beta fijo para el benchmark
                'Alpha': 0    # Alpha fijo para el benchmark
            }
    
            metrics_df = pd.DataFrame([metrics, benchmark_metrics], index=['Strategy', 'Benchmark']).T
            return metrics_df
    
        finally:
            # Restaurar configuración original de numpy
            np.seterr(**old_settings)