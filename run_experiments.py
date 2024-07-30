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
# So far its only one model and one dataset. Ideally we find a good structure to include more datasets and more models. 
#

# %load_ext autoreload

# %autoreload 2

# !pwd

# +
import subprocess
import os
script_path = "scripts/run_gcn.py"
EPOCHS = 150
project_root = os.path.abspath('')
env = os.environ.copy()
env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

for seed in range(10):
    result = subprocess.run(['python', script_path, '--seed', str(seed), '--epochs', str(EPOCHS)], capture_output=True, text=True,
                           env=env, cwd=project_root)
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)

    
# -

# !pwd


