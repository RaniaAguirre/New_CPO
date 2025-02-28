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
    def __init__(self):
        self.tickers = []  # Lista de tickers inicialmente vacía
        self.fecha_inicio = None
        self.fecha_fin = None

    def dates(self, fecha_inicio, fecha_fin):
        """Establece las fechas de inicio y fin."""
        self.fecha_inicio = fecha_inicio
        self.fecha_fin = fecha_fin

    def sp500(self, url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
        """Carga los tickers del S&P500 desde Wikipedia y selecciona los primeros 100."""
        try:
            table = pd.read_html(url)[0]
            self.tickers = table['Symbol'].tolist()[:100]  # Solo los primeros 100 tickers
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
                data = data.resample('ME').last()  # Resamplear a fin de mes
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

class Sortino:
    def __init__(self, returns_df, rfr_csv_path, selected_assets=None):
        """
        Inicializa la clase con un DataFrame de rendimientos y una lista de activos.
        
        Parameters:
            returns_df (pd.DataFrame): DataFrame con rendimientos históricos (mensuales).
            rfr_csv_path (str): Ruta al archivo CSV que contiene las tasas libres de riesgo mensuales.
            selected_assets (list, optional): Lista de tickers a utilizar. Si es None, se extraen de returns_df.
        """
        self.returns_df = returns_df
        
        # Leer el archivo CSV con las tasas libres de riesgo
        try:
            self.rfr_df = pd.read_csv(rfr_csv_path, parse_dates=["Date"])
            if "Date" not in self.rfr_df.columns or "rfr" not in self.rfr_df.columns:
                raise ValueError("El archivo CSV debe contener las columnas 'Date' y 'rfr'.")
            
            # Asegurar que esté resampleado a fin de mes
            self.rfr_df = self.rfr_df.set_index("Date").resample("ME").first()
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {rfr_csv_path}")
        except Exception as e:
            raise ValueError(f"Error al leer el archivo CSV: {e}")
        
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
        for _, row in self.weights_df.iterrows():
            date = row["Date"]
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
        lambda group: group.sample(n=min(200, len(group)), replace=False, random_state=42)
        )
        return sampled_df
    
    def sample_in_chunks(df, chunk_size=1000, sample_size=200, random_state=42):
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

# Función para cargar el dataset
def load_data(filepath):
    """
    Load data from a CSV file with the following structure:
      - Column 0: Date (ignored for training)
      - Columns 1 to 10: Asset allocation features
      - Columns 11 to 30: Market features
      - Last column: Sortino ratio (target)
    """
    df = pd.read_csv(filepath)
    # Uncomment the next line if you want to convert the first column to datetime
    # df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    # Combine asset allocation and market features into X, drop the date column.
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    return X, y

# Graficar resultados
def plot_model_results(models, X_test, y_test):
    """
    Plot actual versus predicted Sortino ratio for each model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, (name, model) in zip(axes, models.items()):
        preds = model.predict(X_test)
        ax.scatter(y_test, preds, alpha=0.6, label='Predictions')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
        ax.set_xlabel("Actual Sortino Ratio")
        ax.set_ylabel("Predicted Sortino Ratio")
        ax.set_title(name)
        ax.legend()

    fig.suptitle("Actual vs. Predicted Sortino Ratio for Each Model")
    plt.tight_layout()
    plt.show()


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