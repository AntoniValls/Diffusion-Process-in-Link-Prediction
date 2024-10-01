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
import pandas as pd
result_df = pd.concat(result)

result_df.groupby("dataset")["vcmpr10"].mean()

# +
import pickle

with open("vcmpr10.pkl", "wb") as out:
    e = pickle.dump(result_df, out)

# +
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(result_df, x="eigenvector_centrality", y="diffusion_centrality", hue="dataset")

# +
import pickle

with open("mini_result.pkl", "rb") as input_file:
    e = pickle.load(input_file)
# -


e.groupby("dataset")["vcmpr5"].mean()
