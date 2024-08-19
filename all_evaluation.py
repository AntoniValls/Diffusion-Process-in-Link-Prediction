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
from utils.evaluation import evaluate_all
import networkx as nx
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality
eigen = partial(nx.eigenvector_centrality_numpy)
degree = partial(nx.degree_centrality)
diffusion = partial(diffusion_centrality, T=5)
names = ["eigenvector_centrality", "degree_centrality", "diffusion_centrality"]

data_names = ["facebook", "wiki", "EN"]
result = evaluate_all(model_name="gcn", list_of_data=data_names, 
                      method_list=[eigen, degree, diffusion], 
                      method_names=names)
# -


