import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("/Users/adrianlopez/Documents/GitHub/New_CPO/sample_dbs_tunning.csv")
df = df.drop(columns=['Unnamed: 0', 'Date', 'Portfolio_Returns'])

# 2. Separate features and target
X = df.drop(columns=['Sortino_Ratio'])
y = df['Sortino_Ratio'].copy()

# 3. Target Transformation (choose one)
# Option 1: Winsorize (clip) target
y_winsor = y.clip(lower=y.quantile(0.01), upper=y.quantile(0.99))

# Option 2: Log-transform target
y_log = np.sign(y) * np.log1p(np.abs(y))

# Choose which target to use
y_t = y_log  # or y_winsor

# 4. Feature Transformation
# Use QuantileTransformer to gaussianize features
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_qt = pd.DataFrame(qt.fit_transform(X), columns=X.columns)

# 5. Rolling Features (using original, unshuffled data)
# For demonstration, assume data is sorted by time
window_sizes = [5, 10, 21]
factor_cols = ['Treasury Bond 3M', 'Yield Curve Spread (10Y - 2Y)', 'Treasury Bond 10Y', 'WTI Index', 'USD Index', 'SMB', 'HML', 'MOM', 'VVIX']
for col in factor_cols:
    for w in window_sizes:
        X_qt[f'{col}_roll{w}_mean'] = X_qt[col].rolling(window=w, min_periods=1).mean()
        X_qt[f'{col}_roll{w}_std'] = X_qt[col].rolling(window=w, min_periods=1).std().fillna(0)

# 6. Optional: PolynomialFeatures + PCA
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_qt)
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_pca = pca.fit_transform(X_poly)

# 7. Train/Test Split (time-based)
test_size = int(0.2 * len(X_pca))
X_train, X_test = X_pca[:-test_size], X_pca[-test_size:]
y_train, y_test = y_t.iloc[:-test_size], y_t.iloc[-test_size:]

# 8. SVR Hyperparameter Tuning
param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3]  # only used for 'poly'
}
tscv = TimeSeriesSplit(n_splits=5)
svr = SVR()
gscv = GridSearchCV(svr, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
gscv.fit(X_train, y_train)

print("Best SVR params:", gscv.best_params_)

# 9. Train Final Model
best_svr = gscv.best_estimator_
best_svr.fit(X_train, y_train)

# 10. Evaluation
y_pred = best_svr.predict(X_test)

# If you log-transformed y, invert the transformation for interpretability
y_test_inv = np.sign(y_test) * (np.expm1(np.abs(y_test)))
y_pred_inv = np.sign(y_pred) * (np.expm1(np.abs(y_pred)))

print("Test MSE (transformed):", mean_squared_error(y_test, y_pred))
print("Test R² (transformed):", r2_score(y_test, y_pred))
print("Test MSE (inverted):", mean_squared_error(y_test_inv, y_pred_inv))
print("Test R² (inverted):", r2_score(y_test_inv, y_pred_inv))

# Plot predicted vs. actual (inverted)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_inv, y=y_pred_inv, alpha=0.5)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
plt.xlabel("Actual Sortino Ratio")
plt.ylabel("Predicted Sortino Ratio")
plt.title("SVR: Actual vs. Predicted Sortino Ratio (Hold-out)")
plt.grid(True)
plt.tight_layout()
plt.show()