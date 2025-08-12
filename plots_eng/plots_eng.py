import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

results_df = pd.read_csv("plots2/resultados_simulaciones.csv")

# Histograma downside risk
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Downside Risk"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histogram of Downside Risk: {method}")
    axes[i].set_xlabel("Downside Risk")
    axes[i].set_ylabel("Density")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots_eng/histogram_downside.png", dpi = 300)
plt.close()

# Histograma Rendimiento Anual
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Rendimiento Anual Promedio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histogram of the Average Annual Return: {method}")
    axes[i].set_xlabel("Average Annual Return")
    axes[i].set_ylabel("Density")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots_eng/histograms_ann_ret.png" ,dpi=300)
plt.close()

# Histograma Ratio de Sortino
strategies = results_df["Metodología"].unique()
fig, axes = plt.subplots(nrows=len(strategies), ncols=1, figsize=(10, 3 * len(strategies)))
for i, method in enumerate(strategies):
    sns.histplot(results_df[results_df["Metodología"] == method]["Sortino ratio"], kde=True, stat="density", bins=10, ax=axes[i], alpha=0.7)
    axes[i].set_title(f"Histogram of the Sortino Ratio: {method}")
    axes[i].set_xlabel("Sortino Ratio")
    axes[i].set_ylabel("Density")
    axes[i].grid(True)
plt.tight_layout()
plt.savefig("plots_eng/histogram_sortino.png", dpi=300)
plt.close()

