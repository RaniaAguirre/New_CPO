import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import warnings
import random
import requests

#import quantstats as qs

#ML libraries
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
class Data:
    def __init__(self):
        self.tickers = []  # Lista de tickers inicialmente vacía
        self.fecha_inicio = None
        self.fecha_fin = None

    def dates(self, fecha_inicio, fecha_fin):
        """Establece las fechas de inicio y fin."""
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin

    def sp500(self, url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
        """Carga los tickers del S&P500 desde Wikipedia y los ordena por capitalización de mercado."""
        try:
            # Cargar la tabla desde Wikipedia
            table = pd.read_html(url)[0]
            
            # Filtrar solo la columna 'Symbol' para obtener los tickers
            tickers = table['Symbol'].tolist()
            #print(f"Tickers extraídos de Wikipedia (primeros 100): {tickers[:100]}")  # Verificar los primeros 10 tickers
            
            # Obtener la capitalización de mercado de cada ticker usando yfinance
            market_caps = {}
            for ticker in tickers:
                try:
                    # Descargar la información del ticker
                    stock_info = yf.Ticker(ticker).info
                    # Guardar el market cap si está disponible
                    if 'marketCap' in stock_info:
                        market_caps[ticker] = stock_info['marketCap']
                        print(f"{ticker}: {market_caps[ticker]}")  # Verificar los marketCaps obtenidos
                except Exception as e:
                    print(f"Error al obtener market cap para {ticker}: {e}")
            
            # Ordenar los tickers por capitalización de mercado en orden descendente
            sorted_tickers = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
            print(f"Primeros 100 tickers ordenados por market cap: {sorted_tickers[:100]}")  # Verificar la ordenación
            
            # Guardar los primeros 100 tickers en la lista self.tickers
            self.tickers = [ticker for ticker, _ in sorted_tickers[:100]]
            
            # Imprimir todos los 100 tickers seleccionados
            #print(f"Primeros 100 tickers por market cap:")
            #for i, ticker in enumerate(self.tickers, 1):
            #    print(f"{i}. {ticker}")
        
        except Exception as e:
            print(f"Error al obtener la lista de tickers: {e}")

    def prices(self):
        """Descarga los datos de precios de los tickers cargados."""
        if not self.tickers:
            raise ValueError("No hay tickers disponibles para descargar.")
        if not self.fecha_inicio or not self.fecha_fin:
            raise ValueError("Debes establecer las fechas de inicio y fin.")

        datos = []
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.fecha_inicio, end=self.fecha_fin)
                # Seleccionar solo la columna 'Close' y renombrarla para evitar conflictos
                data = data[['Close']].rename(columns={'Close': ticker})
                data = data.resample('MS').first()  # Resamplear a fin de mes
                datos.append(data)
            except Exception as e:
                print(f"Error al descargar datos para {ticker}: {e}")

        if not datos:
            raise ValueError("No se pudieron descargar datos para ningún ticker.")

        # Concatenar todos los datos en un DataFrame combinado
        df_combinado = pd.concat(datos, axis=1)

        # Eliminar columnas con más de 100 datos faltantes
        df_combinado = df_combinado.loc[:, df_combinado.isnull().sum() <= 100]

        return df_combinado

    def random(self, df_precios, num=15):
        """Selecciona aleatoriamente 'num' activos del DataFrame de precios."""
        if df_precios.empty:
            raise ValueError("El DataFrame de precios está vacío.")
        if num > len(df_precios.columns):
            raise ValueError(f"No hay suficientes activos disponibles. Máximo: {len(df_precios.columns)}")

        selected_tickers = np.random.choice(df_precios.columns, size=num, replace=False).tolist()
        return df_precios[selected_tickers]

    def returns(self, df_precios):
        """Calcula los rendimientos porcentuales diarios."""
        if df_precios is None or df_precios.empty:
            raise ValueError("Debes proporcionar un DataFrame con precios válido.")
        return df_precios.pct_change(fill_method=None).dropna(how='all')

    def load_prices_from_csv(self, csv_path):
        """
        Carga datos de precios desde un CSV pre-descargado.
        
        Args:
            csv_path (str): Ruta al archivo CSV con los precios históricos.
            
        Returns:
            pd.DataFrame: DataFrame con los precios procesados.
        """
        try:
            # Cargar CSV con fechas como índice
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            
            # Filtrar por fechas si ya están definidas
            if self.fecha_inicio and self.fecha_fin:
                df = df.loc[self.fecha_inicio:self.fecha_fin]
            
            # Resamplear a fin de mes (si no está hecho)
            df = df.resample('ME').last()
            
            # Eliminar columnas con más de 100 NaNs
            df = df.loc[:, df.isnull().sum() <= 100]
            
            # Actualizar tickers y fechas
            self.tickers = df.columns.tolist()
            
            # Si no se definieron fechas, usar las del CSV
            if not self.fecha_inicio:
                self.fecha_inicio = df.index.min()
            if not self.fecha_fin:
                self.fecha_fin = df.index.max()
            
            return df
        except Exception as e:
            print(f"Error al cargar el CSV: {e}")
            return pd.DataFrame()

    def classify_assets(self, selected_assets):
       """Clasifica los activos seleccionados según su posición en el S&P500."""
       #if not self.sp500_tickers:
        #   print("Advertencia: La lista de tickers del S&P500 no se ha cargado.")
         #  return None
       positions = []
       for asset_tuple in selected_assets.columns:
           asset = asset_tuple[0]
           try:
               position = self.tickers.index(asset) + 1
               positions.append(position)
           except ValueError:
               print(f"Advertencia: El activo {asset} no se encontró en la lista de los 100 principales del S&P500.")
       if not positions:
           return None
       count_1_30 = sum(1 for pos in positions if 1 <= pos <= 30)
       count_31_65 = sum(1 for pos in positions if 31 <= pos <= 65)
       count_66_100 = sum(1 for pos in positions if 66 <= pos <= 100)
       counts = {1: count_1_30, 2: count_31_65, 3: count_66_100}
       max_group = max(counts, key=counts.get)
       if counts[max_group] == 0:
           return None
   
       return max_group, counts

class Sortino:
    def __init__(self, returns_df, rfr_csv_path, selected_assets=None):
        """
        Inicializa la clase con un DataFrame de rendimientos y una lista de activos.
        
        Parameters:
            returns_df (pd.DataFrame): DataFrame con rendimientos históricos (mensuales).
            rfr_csv_path (str): Ruta al archivo CSV que contiene las tasas libres de riesgo mensuales.
            selected_assets (list, optional): Lista de tickers a utilizar. Si es None, se extraen de returns_df.
        """
        self.returns_df = returns_df.copy()
        self.returns_df['Date'] = pd.to_datetime(self.returns_df['Date'])
        self.returns_df.set_index('Date', inplace=True)
        self.returns_df.index = self.returns_df.index.normalize()
        self.returns_df = self.returns_df[self.returns_df.index < "2025-01-01"]

        self.rfr_df = pd.read_csv(rfr_csv_path)
        self.rfr_df.columns = [col.strip().lower() for col in self.rfr_df.columns]

        if "date" not in self.rfr_df.columns or "rfr" not in self.rfr_df.columns:
            raise ValueError("El archivo CSV debe contener las columnas 'Date' y 'rfr'.")
        
        self.rfr_df.rename(columns={"date": "Date"}, inplace=True)
        self.rfr_df["Date"] = pd.to_datetime(self.rfr_df["Date"])
        self.rfr_df.set_index("Date", inplace=True)
        self.rfr_df.index = self.rfr_df.index.normalize()
        
        # Si no se pasan activos seleccionados, se asume que son las columnas del DataFrame de rendimientos.
        if selected_assets is None:
            self.selected_assets = list(returns_df.columns)
        else:
            self.selected_assets = selected_assets
        
        self.portfolio_data = None

    def generate_multiple_weights(self, num_combinations=1000):
        """
        Genera múltiples combinaciones de pesos aleatorios para los activos seleccionados.
        
        Parameters:
            num_combinations (int): Número de combinaciones de pesos a generar por fecha.
        """
        weights_list = []
        for date in self.returns_df.index:
            date = pd.to_datetime(date).normalize()
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
        portfolio_returns = []
        for _, row in self.weights_df.iterrows():
            date = row["Date"]
            weights = row.values[1:]  # Obtener los pesos de esta fila
            selected_returns = self.returns_df.loc[date, self.selected_assets]
            portfolio_return = (selected_returns.values * weights / 100).sum()
            portfolio_returns.append(portfolio_return)
        
        self.weights_df["Portfolio_Returns"] = portfolio_returns
        print("Rendimientos del portafolio calculados para cada combinación de pesos.")

    def calculate_sortino_ratio(self):
        """
        Calcula el Ratio de Sortino para cada combinación de pesos utilizando la tasa libre de riesgo mensual.
        """
        if "Portfolio_Returns" not in self.weights_df.columns:
            raise ValueError("Debes calcular los rendimientos del portafolio antes de calcular el Ratio de Sortino.")
        
        sortino_ratios = []
        grouped = self.weights_df.groupby("Date")
        for _, row in self.weights_df.iterrows():
            date = pd.to_datetime(row["Date"]).normalize()
            portfolio_return = row["Portfolio_Returns"]
            
            # Obtener la tasa libre de riesgo correspondiente a la fecha
            try:
                risk_free_rate = self.rfr_df.loc[date, "rfr"] / 100  # Convertir a decimal
            except KeyError:
                print(f"No se encontró la tasa libre de riesgo para la fecha {date}. Usando NaN.")
                risk_free_rate = np.nan
            
            excess_return = portfolio_return - (risk_free_rate / 252)  # Ajuste diario
            
            # Filtrar los rendimientos negativos para calcular la desviación estándar de downside
            downside_returns = self.weights_df[self.weights_df["Portfolio_Returns"] < (risk_free_rate / 252)]["Portfolio_Returns"]
            downside_deviation = downside_returns.std() if not downside_returns.empty else np.nan
            
            sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else np.nan
            sortino_ratios.append(sortino_ratio)
        
        self.weights_df["Sortino_Ratio"] = sortino_ratios
        print("Ratios de Sortino calculados para cada combinación de pesos.")

    def calculate_sharpe_ratio(self):
        if "Portfolio_Returns" not in self.weights_df.columns:
            raise ValueError("Debes calcular los rendimientos del portafolio antes de calcular el Ratio de Sortino.")
        
        sharpe_ratios = []
        for _, row in self.weights_df.iterrows():
            date = pd.to_datetime(row["Date"]).normalize()
            portfolio_return = row["Portfolio_Returns"]
            
            # Obtener la tasa libre de riesgo correspondiente a la fecha
            try:
                risk_free_rate = self.rfr_df.loc[date, "rfr"] / 100  # Convertir a decimal
            except KeyError:
                print(f"No se encontró la tasa libre de riesgo para la fecha {date}. Usando NaN.")
                risk_free_rate = np.nan
            
            excess_return = portfolio_return - (risk_free_rate / 252)  # Ajuste diario
            
            std_dev = self.weights_df[self.weights_df["Date"] == date]["Portfolio_Returns"].std()
            
            sharpe_ratio = excess_return / std_dev if std_dev != 0 else np.nan
            sharpe_ratios.append(sharpe_ratio)

        self.weights_df["Sharpe_Ratio"] = sharpe_ratios
        print("Ratios de Sharpe calculados para cada combinación de pesos.")

    def create_portfolio_dataset(self):
        """
        Devuelve el DataFrame completo con las combinaciones de pesos, rendimientos del portafolio
        y Ratios de Sortino.
        
        Returns:
            pd.DataFrame: DataFrame con la información del portafolio.
        """
        if "Sharpe_Ratio" not in self.weights_df.columns:
            raise ValueError("Debes calcular los rendimientos y el Ratio de Sortino antes de crear el dataset.")
        
        return self.weights_df


class SharpeCalculatorUnified:
    def __init__(self, dataset_csv_path):
        """
        Inicializa la clase cargando un único dataset que ya contiene las combinaciones de pesos,
        los rendimientos del portafolio, el ratio de Sortino y la tasa libre de riesgo.

        Parameters:
            dataset_csv_path (str): Ruta del CSV que contiene el dataset.
                                    Se espera que tenga las columnas:
                                        - 'Date'
                                        - 'Portfolio_Returns'
                                        - 'rfr' (tasa libre de riesgo en porcentaje)
        """
        self.dataset = dataset_csv_path
        # Verificar que el dataset tenga las columnas requeridas
        required_columns = {"Date", "Portfolio_Returns", "rfr"}
        if not required_columns.issubset(self.dataset.columns):
            raise ValueError(f"El dataset debe contener las columnas: {required_columns}")

    def calculate_sharpe_ratio(self):
        """
        Calcula el ratio de Sharpe para cada combinación de pesos usando el dataset cargado.
        
        Para cada fila se agrupan los retornos del portafolio por fecha para obtener la desviación
        estándar (volatilidad total) de ese día. Luego se calcula:
        
            Exceso de rendimiento = Portfolio_Returns - (rfr / 100 / 252)
            Sharpe Ratio = Exceso de rendimiento / (desviación estándar del grupo del día)
        
        Returns:
            pd.DataFrame: Dataset actualizado con una nueva columna "Sharpe_Ratio".
        """
        df = self.dataset.copy()
        
        # Calcular la desviación estándar de los rendimientos del portafolio para cada fecha
        df["group_std"] = df.groupby("Date")["Portfolio_Returns"].transform("std")
        
        sharpe_ratios = []
        for idx, row in df.iterrows():
            portfolio_return = row["Portfolio_Returns"]
            # Convertir la tasa libre de riesgo de porcentaje a decimal y ajustarla a base diaria
            daily_rfr = (row["rfr"] / 100) / 252
            
            # Calcular el exceso de rendimiento
            excess_return = portfolio_return - daily_rfr
            
            group_std = row["group_std"]
            sharpe = excess_return / group_std if group_std != 0 else np.nan
            sharpe_ratios.append(sharpe)
        
        df["Sharpe_Ratio"] = sharpe_ratios
        # Eliminar la columna temporal
        df.drop(columns=["group_std"], inplace=True)
        
        self.dataset = df
        return self.dataset

    def get_dataset(self):
        """
        Devuelve el dataset actualizado con el ratio de Sharpe calculado.
        
        Returns:
            pd.DataFrame: Dataset con la columna "Sharpe_Ratio".
        """
        return self.dataset

class MarketFeaturesReplicator:
    def __init__(self, filepath, replication_factor=100):
        """
        Inicializa la clase con la ruta del archivo .xlsx y el factor de replicación.
        
        Parameters:
        filepath (str): Ruta del archivo .xlsx con las market features.
        replication_factor (int): Número de veces que se replica cada fila (por ejemplo, 100).
        """
        self.filepath = filepath
        self.replication_factor = replication_factor
        self.market_features = None

    def load_market_features(self):
        """
        Carga el archivo .xlsx y lo almacena en un DataFrame.
        Se espera que el archivo tenga una columna 'Date' que identifica cada día.
        
        Returns:
        pd.DataFrame: DataFrame con las market features.
        """
        self.market_features = pd.read_excel(self.filepath)
        if 'Date' in self.market_features.columns:
            self.market_features['Date'] = pd.to_datetime(self.market_features['Date'])
        else:
            raise ValueError("El archivo debe contener una columna 'Date'.")
        return self.market_features

    def replicate_market_features(self):
        """
        Replica cada fila del DataFrame de market features tantas veces como indique replication_factor.
        Esto se hace para que por cada día se tengan múltiples muestras (Tantos portafolios como se hayan calculado para el ratio de sortino)
          
         en este caso los 100 portafolios generados para cada día de muestra.
        
        Returns:
        pd.DataFrame: DataFrame con las market features replicadas.
        """
        if self.market_features is None:
            self.load_market_features()
        
        # Se asume que cada fila representa un día único.
        # Se replica cada fila replication_factor veces.
        replicated = self.market_features.loc[self.market_features.index.repeat(self.replication_factor)].copy()
        # Reiniciamos el índice para tener un DataFrame ordenado.
        replicated.reset_index(drop=True, inplace=True)
        return replicated

class SortinoSampler:
    def __init__(self, merged_df, sortino_col="Sortino_Ratio", date_col="Date"):
        """
        Inicializa la clase con el DataFrame fusionado que contiene las market features y los control features,
        incluyendo la columna del ratio de Sortino.
        
        Parameters:
            merged_df (pd.DataFrame): DataFrame resultante del merge.
            sortino_col (str): Nombre de la columna del ratio de Sortino (default "Sortino_Ratio").
            date_col (str): Nombre de la columna de fecha (default "Date").
        """
        self.df = merged_df.copy()
        self.sortino_col = sortino_col
        self.date_col = date_col

    def sample_best(self, n_best):
        """
        Para cada fecha, selecciona las n mejores muestras según el ratio de Sortino.
        
        Parameters:
            n_best (int): Número de muestras a conservar por cada fecha.
        
        Returns:
            pd.DataFrame: DataFrame con las n mejores combinaciones por fecha.
        """
        # Verifica que las columnas necesarias existan en el DataFrame
        if self.sortino_col not in self.df.columns:
            raise ValueError(f"La columna {self.sortino_col} no se encuentra en el DataFrame.")
        if self.date_col not in self.df.columns:
            raise ValueError(f"La columna {self.date_col} no se encuentra en el DataFrame.")
        
        # Agrupar por fecha, ordenar de forma descendente por el ratio de Sortino y tomar las n mejores filas de cada grupo
        sampled_df = self.df.groupby(self.date_col, group_keys=False).apply(
            lambda group: group.nlargest(n_best, self.sortino_col)
        )
        return sampled_df

    def sample_random(self, n_random):
        """
        Para cada fecha, selecciona aleatoriamente n filas.
        
        Parameters:
            n_random (int): Número de muestras aleatorias a seleccionar por cada fecha.
        
        Returns:
            pd.DataFrame: DataFrame con n filas aleatorias por fecha.
        """
        # Verifica que la columna de fecha exista
        if self.date_col not in self.df.columns:
            raise ValueError(f"La columna {self.date_col} no se encuentra en el DataFrame.")
        
        # Agrupar por fecha y aplicar sample a cada grupo.
        # Se usa replace=False asumiendo que cada grupo tiene al menos n_random filas.
        
        sampled_df = self.df.groupby(self.date_col, group_keys=False).apply(
        lambda group: group.sample(n=min(200, len(group)), replace=False, random_state=42)
        )
        return sampled_df
    
    def sample_in_chunks(df, chunk_size=1000, sample_size=300, random_state=42):
        """
        Divide el DataFrame en bloques de 'chunk_size' filas y, en cada bloque,
        selecciona aleatoriamente 'sample_size' filas. Si el bloque tiene menos de 
        'sample_size' filas, se seleccionan todas las disponibles.

            Parameters:
            -----------
            df : pd.DataFrame
                DataFrame de origen.
            chunk_size : int, opcional
                Número de filas que conforman cada bloque. Por defecto es 1000.
            sample_size : int, opcional
                Número de filas a muestrear en cada bloque. Por defecto es 200.
            random_state : int o None, opcional
                Semilla para el muestreo aleatorio para garantizar reproducibilidad.

            Returns:
            --------
            pd.DataFrame
                DataFrame resultante con las filas muestreadas de cada bloque.
            """
        sampled_chunks = []
        n_rows = len(df)

            # Itera en bloques de 'chunk_size'
        for start in range(0, n_rows, chunk_size):
            # Selecciona el bloque actual
            chunk = df.iloc[start:start + chunk_size]
            # Define el número de muestras a tomar: 200 o todas si no hay 200 filas
            n_sample = sample_size if len(chunk) >= sample_size else len(chunk)
            # Muestrea aleatoriamente sin reemplazo
        sampled_chunk = chunk.sample(n=n_sample, random_state=random_state)
        sampled_chunks.append(sampled_chunk)

        # Concatena todos los bloques muestreados y restablece el índice
        return pd.concat(sampled_chunks).reset_index(drop=True)


#General model class 
class BasePortfolioModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict using the trained model."""
        pass

    def evaluate(self, X_test, y_test):
        """Evaluate the model using MSE and R2."""
        preds = self.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Evaluation -- MSE: {mse:.4f}, R2: {r2:.4f}")
        return mse, r2
    

# Linear Regression Model   
class LinearRegressionModel(BasePortfolioModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Linear Regression model fitted.")

    def predict(self, X):
        return self.model.predict(X)
# NN model  
class NeuralNetworkModel(BasePortfolioModel):
    def __init__(self, input_dim, learning_rate=0.025, epochs=25, batch_size=64):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        print("Neural Network model trained.")

    def predict(self, X):
        return self.model.predict(X).flatten()
# SVR model    
class SVRModel(BasePortfolioModel):
    def __init__(self, kernel, C, epsilon):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        self.results_df = None  

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train) 
        print("SVR model fitted.")

    def predict(self, X):
        return self.model.predict(X)
#XGBoost   
class XGBoostModel(BasePortfolioModel):
    def __init__(self, 
                 n_estimators=300, 
                 max_depth=6, 
                 learning_rate=0.09437520854025253, 
                 subsample=0.760134409258957, 
                 colsample_bytree=0.7844564369302189, 
                 gamma=0.15628002532567653, 
                 reg_alpha=0.8255009142363116, 
                 reg_lambda=0.17066679740109292, 
                 objective='reg:squarederror',
                 **params):
        # Use new Optuna best params as defaults, but allow override
        default_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective=objective
        )
        default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("XGBoost model trained.")

    def predict(self, X):
        return self.model.predict(X)
    
class Market_Features:
    def __init__(self):
        pass
    
    def fred_data(self, api_key, indicator, start_date, end_date, frequency):
        """
        Descarga datos de la API de la Reserva Federal de St. Louis (FRED).
        
        Parámetros:
        api_key (str): Clave de la API FRED.
        indicator (str): Código del indicador a consultar.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        frequency (str): Frecuencia de los datos ('d' para diario, 'm' para mensual, etc.).
        
        Retorna:
        pd.DataFrame: DataFrame con los datos obtenidos.
        """
        params = {
            'api_key': api_key,
            'file_type': 'json',
            'series_id': indicator,
            'realtime_start': end_date,
            'realtime_end': end_date,
            "observation_start": start_date,
            "observation_end": end_date,
            'frequency': frequency,
        }

        url_base = 'https://api.stlouisfed.org/'
        endpoint = 'fred/series/observations'
        url = url_base + endpoint

        res = requests.get(url, params=params)
        data = res.json()
        
        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['Date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'value': indicator})
            df = df.drop(columns=['realtime_start', 'realtime_end', 'date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            raise ValueError("No se encontraron datos para el indicador y fechas proporcionadas.")
    
    
    def read_multiple_csv(self, folder_path):
        """
        Lee múltiples archivos CSV y XLSX desde una carpeta y los concatena en un solo DataFrame.
        Si los archivos tienen diferentes estructuras, se combinan con un merge basado en la columna 'date'.
    
        Parámetros:
        folder_path (str): Ruta de la carpeta donde están los archivos.
    
        Retorna:
        pd.DataFrame: DataFrame combinado con los datos de todos los archivos, eliminando filas con valores NaN.
        """
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
        if not all_files:
            raise ValueError("No se encontraron archivos CSV")
    
        df_list = []
        for file in all_files:
            file_path = os.path.join(folder_path, file)
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            df_list.append(df)
    
        # Unir todos los DataFrames en uno solo considerando la columna 'date' como referencia
        combined_df = df_list[0]
        for df in df_list[1:]:
            combined_df = pd.merge(combined_df, df, on='date', how='outer')
    
        # Asegurarse de que la columna 'date' esté en formato datetime y establecer como índice
        combined_df['Date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    
        # Eliminar filas con valores NaN en columnas diferentes a 'date'
        combined_df.dropna(subset=combined_df.columns.difference(['date']), inplace=True)
    
        # Establecer la columna 'date' como índice
        combined_df.set_index('Date', inplace=True)
        combined_df = combined_df.drop(columns=['date'])
        return combined_df

    
    def add_column_to_dataframe(self, df1, df2):
        """
        Agrega una nueva columna de un DataFrame a otro basado en la columna 'date'.
        
        Parámetros:
        df1 (pd.DataFrame): DataFrame base al que se agregará la nueva columna.
        df2 (pd.DataFrame): DataFrame que contiene la columna a agregar.
        
        Retorna:
        pd.DataFrame: DataFrame combinado con la nueva columna agregada.
        """
        combined_df = pd.merge(df1, df2, how='outer', on='Date')
        return combined_df
