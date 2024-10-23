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

# + [markdown] jp-MarkdownHeadingCollapsed=true
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
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for graph-level properties
data_path = "data/graph_level_properties"
datasets = ["Cora.json", "CiteSeer.json", "facebook.json", "wiki.json", "ES.json"]

# Load each JSON file into a dictionary
graph_properties = {}
for dataset in datasets:
    with open(os.path.join(data_path, dataset)) as f:
        graph_properties[dataset.split('.')[0]] = json.load(f)

# Convert the graph properties into a DataFrame for easier plotting
data_list = []
for name, properties in graph_properties.items():
    row = {
        "Dataset": name,
        "Average_Degree": properties[0]["Average_Degree"],
        "Clustering_Coefficient": properties[0]["Clustering_Coefficient"],
        **{f"Gini_{k}": v for k, v in properties[0]["Gini_Indices"].items()}
    }
    data_list.append(row)

df = pd.DataFrame(data_list)

# Create subplots for Average Degree and Clustering Coefficient
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# Plot Average Degree
df.plot(kind='bar', x='Dataset', y='Average_Degree', ax=axes[0], color='skyblue', legend=False)
axes[0].set_title("Average Degree for Each Dataset")
axes[0].set_ylabel("Value")
axes[0].set_xticklabels(df["Dataset"], rotation=0)
axes[0].grid(axis='y')

# Plot Clustering Coefficient
df.plot(kind='bar', x='Dataset', y='Clustering_Coefficient', ax=axes[1], color='salmon', legend=False)
axes[1].set_title("Clustering Coefficient for Each Dataset")
axes[1].set_xticklabels(df["Dataset"], rotation=0)
axes[1].set_ylim([0,1])
axes[1].grid(axis='y')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Plot bar plots for Gini indices of centrality measures
df_melted = df.melt(id_vars=["Dataset"], value_vars=[f"Gini_{col}" for col in properties[0]["Gini_Indices"].keys()],
                    var_name="Centrality Measure", value_name="Gini Index")

plt.figure(figsize=(14, 8))
sns.barplot(data=df_melted, x="Dataset", y="Gini Index", hue="Centrality Measure")
plt.title("Gini Indices of Centrality Measures for Each Dataset")
plt.ylabel("Gini Index")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Scatter plot to explore relationships between Average Degree and Clustering Coefficient
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Average_Degree", y="Clustering_Coefficient", hue="Dataset", s=100)
plt.title("Relationship between Average Degree and Clustering Coefficient")
plt.xlabel("Average Degree")
plt.ylabel("Clustering Coefficient")
plt.grid(True)
plt.show()
# -

# ## **Evaluate diffusion**

# +
from utils.difffusion_evaluation import evaluate_dataset

data_names = ["facebook"]
result = evaluate_dataset(model_name="gcn",
                          data_name="facebook",
                          eval_type='c',
                          n_simulations=10)

# -
result["true_si"][0]

# +
import pandas as pd

# Stack all DataFrames along a new axis, then calculate the mean across all files for each node
stacked_df = pd.concat(result["info_vulnerability"], axis=0).groupby(level=0).mean()

# Calculate the mean error columns
stacked_df['vulnerability_error'] = stacked_df["true_vulnerability"] - stacked_df["pred_vulnerability"]
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
# -


