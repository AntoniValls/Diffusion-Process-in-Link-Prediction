#!/bin/bash
# SBATCH --job-name=eval_node
# SBATCH --partition=gpu
# SBATCH --gres=gpu:1
# SBATCH --time=10:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=20
# SBATCH --mem=16GB
# SBATCH --output=log/eval_node.out
# SBATCH --error=log/eval_node.err

# Load the conda environment
source /home/antoni_valls/beegfs/miniconda3/etc/profile.d/conda.sh
conda activate lp_env
# Run the Python script
python cluster_node_level_eval.py --model_name 'gcn' --data_names "Cora","CiteSeer" 
