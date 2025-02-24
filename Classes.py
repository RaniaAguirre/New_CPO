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
import pickle

#ML libraries
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Data:
    def __init__(self, tickers=None):
        self.tickers = tickers or []  # Inicializa tickers con la lista proporcionada o una lista vacía
        self.fecha_inicio = None
        self.fecha_fin = None

    def dates(self, fecha_inicio, fecha_fin):
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin

    def sp500(self, url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
        try:
            table = pd.read_html(url)[0]
            self.tickers = table['Symbol'].tolist()
        except Exception as e:
            print(f"Error al obtener la lista de tickers: {e}")

    def sp100(self):
        return self.tickers[:100]

    def datadownload(self):
        if not self.tickers:
            raise ValueError("Debes proporcionar al menos un ticker.")
        if not self.fecha_inicio or not self.fecha_fin:
            raise ValueError("Debes establecer las fechas de inicio y fin.")

        datos = []
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.fecha_inicio, end=self.fecha_fin)
                # Seleccionar solo la columna 'Close' y renombrarla para evitar conflictos
                data = data[['Close']].rename(columns={'Close': ticker})
                data = data.resample('ME').last()
                datos.append(data)
            except Exception as e:
                print(f"Error al descargar datos para {ticker}: {e}")

        if not datos:
            return None

        df_combinado = pd.concat(datos, axis=1) # Concatenar a lo ancho para tener tickers como columnas
        return df_combinado


    def rend(self, df_precios):
        if df_precios is None:
            raise ValueError("Debes proporcionar un DataFrame con precios.")
        
        # Calcular rendimientos porcentuales diarios
        df_rendimientos = df_precios.pct_change(fill_method=None).dropna(how='all') 
        
        return df_rendimientos

class Sortino:
    def __init__(self, returns_df):
        """
        Inicializa la clase con un DataFrame de rendimientos.
        
        Parameters:
        returns_df (pd.DataFrame): DataFrame con rendimientos históricos de activos.
        """
        self.returns_df = returns_df
        self.selected_assets = None
        self.portfolio_data = None

    def select_random_assets(self, num_assets=15):
        """
        Selecciona un número específico de activos aleatorios del DataFrame de rendimientos.
        Estos activos serán los mismos para todas las fechas.
        
        Parameters:
        num_assets (int): Número de activos a seleccionar.
        """
        if num_assets > len(self.returns_df.columns):
            raise ValueError("El número de activos solicitado es mayor que el número de columnas disponibles.")
        
        # Seleccionar activos aleatorios (fijos para todas las fechas)
        self.selected_assets = np.random.choice(self.returns_df.columns, size=num_assets, replace=False)
        print(f"Activos seleccionados: {self.selected_assets}")

    def generate_multiple_weights(self, num_combinations=100):
        """
        Genera múltiples combinaciones de pesos aleatorios para los activos seleccionados.
        
        Parameters:
        num_combinations (int): Número de combinaciones de pesos a generar por fecha.
        """
        if self.selected_assets is None:
            raise ValueError("Primero debes seleccionar los activos usando 'select_random_assets'.")
        
        weights_list = []
        for date in self.returns_df.index:
            for _ in range(num_combinations):
                raw_weights = np.random.rand(len(self.selected_assets))
                normalized_weights = (raw_weights / raw_weights.sum()) * 100
                weights_list.append([date] + list(normalized_weights))
        
        # Crear un DataFrame con las combinaciones de pesos
        weights_columns = ["Date"] + [f"Weight_{asset}" for asset in self.selected_assets]
        self.weights_df = pd.DataFrame(weights_list, columns=weights_columns)
        print(f"Generadas {num_combinations} combinaciones de pesos para cada fecha.")

    def calculate_portfolio_returns(self):
        """
        Calcula los rendimientos del portafolio para cada combinación de pesos.
        """
        if self.selected_assets is None or self.weights_df is None:
            raise ValueError("Debes seleccionar activos y generar pesos antes de calcular rendimientos.")
        
        portfolio_returns = []
        for _, row in self.weights_df.iterrows():
            date = row["Date"]
            weights = row.values[1:]  # Obtener los pesos de esta fila
            selected_returns = self.returns_df.loc[date, self.selected_assets]
            portfolio_return = (selected_returns.values * weights / 100).sum()
            portfolio_returns.append(portfolio_return)
        
        self.weights_df["Portfolio_Returns"] = portfolio_returns
        print("Rendimientos del portafolio calculados para cada combinación de pesos.")

    def calculate_sortino_ratio(self, risk_free_rate=0.02):
        """
        Calcula el Ratio de Sortino para cada combinación de pesos.
        
        Parameters:
        risk_free_rate (float): Tasa libre de riesgo anualizada (default: 2%).
        """
        if "Portfolio_Returns" not in self.weights_df.columns:
            raise ValueError("Debes calcular los rendimientos del portafolio antes de calcular el Ratio de Sortino.")
        
        sortino_ratios = []
        for _, row in self.weights_df.iterrows():
            portfolio_return = row["Portfolio_Returns"]
            excess_return = portfolio_return - (risk_free_rate / 252)  # Ajuste diario
            
            # Filtrar los rendimientos negativos para calcular la desviación estándar de downside
            downside_returns = self.weights_df[self.weights_df["Portfolio_Returns"] < (risk_free_rate / 252)]["Portfolio_Returns"]
            downside_deviation = downside_returns.std() if not downside_returns.empty else np.nan
            
            # Calcular el Ratio de Sortino
            sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else np.nan
            sortino_ratios.append(sortino_ratio)
        
        self.weights_df["Sortino_Ratio"] = sortino_ratios
        print("Ratios de Sortino calculados para cada combinación de pesos.")

    def create_portfolio_dataset(self):
        """
        Devuelve el DataFrame completo con las combinaciones de pesos, rendimientos del portafolio
        y Ratios de Sortino.
        
        Returns:
        pd.DataFrame: DataFrame con la información del portafolio.
        """
        if "Sortino_Ratio" not in self.weights_df.columns:
            raise ValueError("Debes calcular los rendimientos y el Ratio de Sortino antes de crear el dataset.")
        
        return self.weights_df