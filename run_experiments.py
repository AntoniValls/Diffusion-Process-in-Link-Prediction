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

# # Experiments
#
# Notebook for running experiments. 
# - Model: GCN (probably different scripts for other models, like GAT)
# - Data: All datatypes in `torch_geometric.datasets` supported (theoretically)
#     - Planetoid: Cora
#     - AttributedGraphDataset: Facebook
#     - LastFMAsia
#     - Twitch: EN, PT
#
# - Epochs: only 150, could be more and earlystopping on validation AUC which probably is not best practice but so far no early stopping
# - Seeds: Run Model across 10 different seeds.
#
#
# Notes: 
# - Some weird behaviour in Twitch + LastFMAsia, metrics sometimes not ideal, or only go up very late..
# - Twitch: Scores go down? 

# %load_ext autoreload

# %autoreload 2

# !pwd

# +
import subprocess
import os
script_path = "scripts/run_gcn.py"
EPOCHS = 150
TGM_TYPE = "Attributed"
NAME = "PT"
project_root = os.path.abspath('')
env = os.environ.copy()
env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

for seed in range(10):
    result = subprocess.run(['python', script_path, '--tgm_type', str(TGM_TYPE), '--name', str(NAME), 
                             '--seed', str(seed), '--epochs', str(EPOCHS)], capture_output=True, text=True,
                           env=env, cwd=project_root)
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


# -

# !pwd


