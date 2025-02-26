import yfinance as yf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import quantstats as qs

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
                data = data.resample('M').last()
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

    def generate_multiple_weights(self, num_combinations=1000):
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
            lambda group: group.sample(n=n_random, random_state=42)
        )
        return sampled_df


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
    def __init__(self, input_dim, hidden_units=64, learning_rate=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(self.hidden_units, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        print("Neural Network model trained.")

    def predict(self, X):
        return self.model.predict(X).flatten()
# SVR model    
class SVRModel(BasePortfolioModel):
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("SVR model fitted.")

    def predict(self, X):
        return self.model.predict(X)
#XGBoost   
class XGBoostModel(BasePortfolioModel):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=learning_rate, objective='reg:squarederror')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("XGBoost model fitted.")

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
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'value': indicator})
            df = df.drop(columns=['realtime_start', 'realtime_end'])
            df.set_index('date', inplace=True)
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
        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
    
        # Eliminar filas con valores NaN en columnas diferentes a 'date'
        combined_df.dropna(subset=combined_df.columns.difference(['date']), inplace=True)
    
        # Establecer la columna 'date' como índice
        combined_df.set_index('date', inplace=True)
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
        combined_df = pd.merge(df1, df2, how='outer', on='date')
        return combined_df