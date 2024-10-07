# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload

# %autoreload 2

# **Prediction evaluation (VCMPR@k and centralities)**

# +
from utils.evaluation import evaluate_all
import networkx as nx
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality
eigen = partial(nx.eigenvector_centrality_numpy)
degree = partial(nx.degree_centrality)
diffusion = partial(diffusion_centrality, T=5)
names = ["eigenvector_centrality"]

data_names = ["wiki"]
result, k_dict, metrics_dict = evaluate_all(model_name="gat", list_of_data=data_names, 
                      method_list=[eigen], 
                      method_names=names)
# -
k_dict

metrics_dict

result[0]

# **Evaluate diffusion**

# +
from utils.difffusion_evaluation import evaluate_all

data_names = ["wiki"]
result = evaluate_all(model_name="gat",
                      list_of_data=data_names,
                      eval_type='s')

# -
result[0].keys()


# +
import pandas as pd

# Stack all DataFrames along a new axis, then calculate the mean across all files for each node
stacked_df = pd.concat(result[0]["info_vulnerability"], axis=0).groupby(level=0).mean()

# Calculate the mean error columns
stacked_df['vulnerability_error'] = stacked_df["true_vulnerability"] - stacked_df["pred_vulnerablity"]
stacked_df['recency_error'] = stacked_df['true_recency'] - stacked_df['pred_recency']

# Display the resulting DataFrame
stacked_df

# +
import matplotlib.pyplot as plt 

# Plotting prediction error for vulnerability vs degree
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(stacked_df['degree'], stacked_df['vulnerability_error'], alpha=0.5, c=stacked_df['degree'], cmap='viridis')
plt.colorbar(label='Degree')
plt.xlabel('Degree')
plt.ylabel('Predicted Vulnerability - True Vulnerability')
plt.title('Vulnerability Prediction Error vs. Degree')

# Plotting prediction error for recency vs degree
plt.subplot(1, 2, 2)
plt.scatter(stacked_df['degree'], stacked_df['recency_error'], alpha=0.5, c=stacked_df['degree'], cmap='viridis')
plt.colorbar(label='Degree')
plt.xlabel('Degree')
plt.ylabel('Predicted Recency - True Recency')
plt.title('Recency Prediction Error vs. Degree')

plt.tight_layout()
plt.show()
