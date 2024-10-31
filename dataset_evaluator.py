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

# ## **Evaluate the node-level centralities graph-level properties**

# +
from utils.evaluation import nodes_and_graph_properties
import networkx as nx
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality, Godfather
from diffusion.complex_diffusion import paralell_complex_path
eigen = partial(nx.eigenvector_centrality_numpy, max_iter=1000, tol=1e-6)
degree = partial(nx.degree_centrality)
diffusion = partial(diffusion_centrality, T=10)
complex_path = partial(paralell_complex_path, T=0.5) 

names = ["degree_centrality",
         "eigenvector_centrality",
         "diffusion_centrality",
         "complex_path_centrality"]

methods = [degree, eigen, diffusion, complex_path]

lista = [["AttributedGraphDataset","facebook"],
         ["AttributedGraphDataset","wiki"],
         ["Twitch", "ES"]]
         
for ttype, name in lista:
    node_level, graph_level = nodes_and_graph_properties(ttype, name, method_list=methods, method_names=names, save=False)



# +
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

# File paths for datasets
data_path = "data/node_level_centralities"
datasets = ["Cora.csv", "CiteSeer.csv", "facebook.csv", "wiki.csv", "ES.csv"]

# Load each dataset into a dictionary of DataFrames
dataframes = {dataset.split('.')[0]: pd.read_csv(os.path.join(data_path, dataset)) for dataset in datasets}

# Centrality measures to plot
centrality_measures = ["degree centrality", "eigenvector_centrality", "diffusion_centrality", "complex_path_centrality"]

# Plot histograms for each centrality measure across all datasets
def plot_histograms(dataframes, centrality_measures):
    for measure in centrality_measures:
        plt.figure(figsize=(14, 8))
        for name, df in dataframes.items():
            sns.histplot(df[measure], kde=True, label=name, alpha=0.5, bins=30)
        plt.title(f"Histogram of {measure.capitalize()} for Different Datasets")      
        plt.xlim([0, 15])
        plt.xlabel(f"{measure.replace('_', ' ').capitalize()}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

# Call the function to plot histograms
plot_histograms(dataframes, centrality_measures)

# Compute summary statistics
def compute_summary_statistics(dataframes, centrality_measures):
    summary_stats = {}
    for name, df in dataframes.items():
        summary_stats[name] = df[centrality_measures].describe().transpose()
    return summary_stats

# Display summary statistics for each dataset
summary_stats = compute_summary_statistics(dataframes, centrality_measures)
for name, stats in summary_stats.items():
    print(f"Summary Statistics for {name}:")
    print(stats)
    print("\n")

# -

# ## **Evaluate Graph-Level Properties**

# +
from utils.evaluation import graph_level_plot

# File paths for graph-level properties
data_path = "data/graph_level_properties"
datasets = ["Cora", "CiteSeer", "facebook", "wiki", "ES"]

properties = graph_level_plot(data_path, datasets, plot=True)


# -


