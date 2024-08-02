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
import networkx as nx
from utils.evaluation import read_prediction_files, EvaluateCentrality
from diffusion.complex_diffusion import get_complex_path, paralell_complex_path
from functools import partial
from diffusion.jackson_metrics import diffusion_centrality
import os.path as osp
MODEL = "gcn"
DATA  = "Facebook"
CENTRALITY = "complex"
T = 8

out_dir = osp.join("data", "result", "centrality", MODEL, DATA)


partial_method = partial(paralell_complex_path, T=T)
results = read_prediction_files(model_name=MODEL, data_name=DATA)

for counter, file in enumerate(results):
    name = f"{DATA}_{CENTRALITY}_{T}_{counter}.pkl"
    evaluater = EvaluateCentrality(file)
    evaluater.write_centrality_to_disk(name=name, path=out_dir, method=partial_method)

# +
#how much faster would this be on the cluster? 
