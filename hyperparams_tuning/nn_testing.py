import itertools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
file_path = "/Users/alexsotomayor/code/PAP/New_CPO/sampled_data.csv"  
df = pd.read_csv(file_path)


df = df.drop(columns=['Unnamed: 0', 'Date', 'Portfolio_Returns'])
X = df.drop(columns=['Sortino_Ratio'])
y = df['Sortino_Ratio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split into train, test and validation
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define hyperparameter grid
learning_rates = [0.025, 0.05]
epochs_list = [25, 50]
batch_sizes = [32, 64]

results = []

# Grid search over hyperparameters
for lr, epochs, batch_size in itertools.product(learning_rates, epochs_list, batch_sizes):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=0)
    
    # Evaluate on test set
    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Store results
    results.append({'learning_rate': lr, 'epochs': epochs, 'batch_size': batch_size, 'mse': mse, 'r2': r2})


results_df = pd.DataFrame(results)

# Plots
plt.figure(figsize=(12, 6))
sns.scatterplot(data=results_df, x='epochs', y='mse', hue='learning_rate', size='batch_size', palette='viridis', sizes=(50, 200))
plt.title("Effect of Hyperparameters on MSE")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(title="Learning Rate & Batch Size")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=results_df, x='epochs', y='r2', hue='learning_rate', size='batch_size', palette='coolwarm', sizes=(50, 200))
plt.title("Effect of Hyperparameters on R² Score")
plt.xlabel("Epochs")
plt.ylabel("R² Score")
plt.legend(title="Learning Rate & Batch Size")
plt.show()

print(results_df)