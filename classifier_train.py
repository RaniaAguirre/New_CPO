import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import os

class PortfolioClassifier:
    def __init__(self, market_caps_file, model_path="trained_models/kmeans_model.pkl"):
        self.df = pd.read_csv(market_caps_file)
        self.market_caps = self.df.set_index("Ticker")["Market Cap"].to_dict()
        self.reference_portfolios = []
        self.reference_weights = []
        self.kmeans = None
        self.cluster_map = None
        self.model_path = model_path

    def generate_random_weights(self, n):
        return [1/n] * n

    def get_random_portfolio(self, n=20, exclude=None):
        available = list(self.market_caps.keys())
        if exclude:
            available = list(set(available) - set(exclude))
        return random.sample(available, n)

    def add_reference_portfolio(self, tickers, weights=None):
        if weights is None:
            weights = self.generate_random_weights(len(tickers))
        self.reference_portfolios.append(tickers)
        self.reference_weights.append(weights)

    def generate_random_reference_portfolios(self, n_portfolios=500, size=20):
        for _ in range(n_portfolios):
            tickers = self.get_random_portfolio(size)
            weights = self.generate_random_weights(size)
            self.add_reference_portfolio(tickers, weights)

    def train(self):
        mc_totales = [
            sum([self.market_caps.get(t, 0) * w for t, w in zip(tickers, weights)])
            for tickers, weights in zip(self.reference_portfolios, self.reference_weights)
        ]

        X = np.log1p(mc_totales).reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

        centroides = self.kmeans.cluster_centers_.flatten()
        orden = np.argsort(centroides)
        self.cluster_map = {orden[0]: -1, orden[1]: 0, orden[2]: 1}

        self._last_mc_log = X.flatten().tolist()
        self._last_labels = self.kmeans.labels_
        self._last_centroids = self.kmeans.cluster_centers_.flatten()

        joblib.dump(
        (self.kmeans, self.cluster_map, self.reference_portfolios, self.reference_weights),
        self.model_path)
        print(f"Modelo y portafolios de referencia guardados en {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.kmeans, self.cluster_map = joblib.load(self.model_path)
            print(f"Modelo cargado desde {self.model_path}")
            return True
        return False

    def classify(self, tickers_portafolio, weights_portafolio=None):
        if self.kmeans is None or self.cluster_map is None:
            raise Exception("Primero necesitas entrenar o cargar el modelo con .train() o .load_model()")

        if weights_portafolio is None:
            weights_portafolio = self.generate_random_weights(len(tickers_portafolio))

        cap = sum([
            self.market_caps.get(t, 0) * w
            for t, w in zip(tickers_portafolio, weights_portafolio)
        ])
        log_cap = np.log1p(cap)
        etiqueta = self.kmeans.predict([[log_cap]])[0]

    def plot_clusters(self):
        if not hasattr(self, "_last_mc_log"):
            print("⚠ Debes entrenar el modelo con log antes de graficar.")
            return

        plt.figure(figsize=(10, 5))
        plt.scatter(self._last_mc_log, [0]*len(self._last_mc_log), c=self._last_labels, cmap='viridis', label='Portafolios')
        plt.scatter(self._last_centroids, [0]*3, color='red', label='Centroides', marker='x', s=100)
        plt.title("Clusters de capitalización ponderada (escala logarítmica)")
        plt.xlabel("log(1 + Capitalización ponderada del portafolio)")
        plt.yticks([])
        plt.legend()
        plt.grid(True)
        plt.show()

    def validate_examples(self):
        examples = {
            "Blue Chips": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            "Small Caps": random.sample([k for k, v in self.market_caps.items() if v < 1e9], 5),
            "Mixto": ['AAPL', 'META', random.choice(list(self.market_caps.keys())),
                      random.choice(list(self.market_caps.keys())),
                      random.choice(list(self.market_caps.keys()))]
        }
        for name, tickers in examples.items():
            weights = self.generate_random_weights(len(tickers))
            label = self.classify(tickers, weights)
            desc = { -1: 'Low Cap', 0: 'Mid Cap', 1: 'High Cap' }
            print(f"{name} → {desc[label]}")

if __name__ == "__main__":
    classifier = PortfolioClassifier("market_caps_kmeans.csv")

    if not classifier.load_model():
        print("Generando portafolios de referencia...")
        classifier.generate_random_reference_portfolios(n_portfolios=1000, size=20)

        print("Entrenando modelo KMeans...")
        classifier.train()

    print("Graficando clusters...")
    classifier.plot_clusters()

    print("Validando ejemplos conocidos...")
    classifier.validate_examples()
