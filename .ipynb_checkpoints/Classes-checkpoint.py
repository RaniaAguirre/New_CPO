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

import yfinance as yf
import pandas as pd

class Data:
    def __init__(self, tickers=None):
        self.tickers = tickers or []  # Inicializa tickers con la lista proporcionada o una lista vacía
        self.fecha_inicio = None
        self.fecha_fin = None

    def dates(self, fecha_inicio, fecha_fin):
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin

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
        df_rendimientos = df_precios.pct_change()

        return df_rendimientos

