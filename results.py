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

from utils.evaluation import read_prediction_files, Evaluate
from diffusion.complex_diffusion import get_complex_path
from functools import partial
import networkx as nx
#partial_method = partial(get_complex_path, T=2)
partial_method = partial(nx.degree_centrality)
results = read_prediction_files(model_name="gcn", data_name="cora")
score_list = []
for file in results:
    evaluater = Evaluate(file)
    data = evaluater.evaluate_group(groups=4, method=partial_method)
    score_list.append(data)

score_list[1] 

# Add statistics for groups such as avg degree and avg centrality...
