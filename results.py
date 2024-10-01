# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload

# %autoreload 2

# +
import os

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")
# -

# ### Paralell version

# +
from utils.evaluation import read_prediction_files, Evaluate
from diffusion.complex_diffusion import get_complex_path, paralell_complex_path
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality, Godfather
from tqdm import tqdm
import networkx as nx

def return_diffusion_centrality(graph, T):
    centrality = diffusion_centrality(G=graph, T=T)
    node_centrality_dict = {node: centrality[i] for i, node in enumerate(graph.nodes)}
    return node_centrality_dict

def return_godfather_centrality(graph):
    centrality = Godfather(graph)
    node_centrality_dict = {node: centrality[i] for i, node in enumerate(graph.nodes)}
    return node_centrality_dict
    
    
#partial_method = partial(return_diffusion_centrality, T=4)
partial_method = partial(nx.degree_centrality)

results = read_prediction_files(model_name="gcn", data_name="EN")
score_list = []
for file in tqdm(results):
    G = file["train_graph"]
    print(nx.degree_assortativity_coefficient(G))
# -

import pandas as pd
df = pd.concat(score_list)
mean_df = df.groupby('group').mean().reset_index()
std_df = df.groupby('group').std(ddof=0).reset_index()

# Add statistics for groups such as avg degree and avg centrality...

mean_df

graph_metrics = []
for file in tqdm(results):
    evaluater = Evaluate(file)
    data = evaluater.evaluate_graph(method=partial_method)
    graph_metrics.append(data)

pd.DataFrame(graph_metrics).mean()

a = results[0]["test_edges"]
for i in range(a.shape[1]):
    node1, node2 = a[:, i]
    print(node1, node2)



a.shape
