import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
file_path = "sampled_data.csv"
df = pd.read_csv(file_path, low_memory=False)

df = df.drop(columns=['Unnamed: 0', 'Date', 'Portfolio_Returns'])
X = df.drop(columns=['Sortino_Ratio'])
y = df['Sortino_Ratio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train, test, and validation
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define singular hyperparameter values
C_value = 10
epsilon = 0.01
kernel = 'rbf'

# Train SVR model with specified parameters
model = SVR(kernel=kernel, C=C_value, epsilon=epsilon)
model.fit(X_train, y_train)

# Evaluate on test set
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Store results in DataFrame
results_df = pd.DataFrame([{'C': C_value, 'epsilon': epsilon, 'kernel': kernel, 'mse': mse, 'r2': r2}])

# Plots
plt.figure(figsize=(12, 6))
sns.scatterplot(data=results_df, x='C', y='mse', hue='kernel', size='epsilon', palette='viridis', sizes=(100, 200))
plt.title("Effect of Hyperparameters on MSE")
plt.xlabel("C")
plt.ylabel("MSE")
plt.legend(title="Kernel & Epsilon")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=results_df, x='C', y='r2', hue='kernel', size='epsilon', palette='coolwarm', sizes=(100, 200))
plt.title("Effect of Hyperparameters on R² Score")
plt.xlabel("C")
plt.ylabel("R² Score")
plt.legend(title="Kernel & Epsilon")
plt.show()

print(results_df)
