#!/bin/bash
#SBATCH --job-name=lp_bias
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --output=lp_bias.out
#SBATCH --error=lp_bias.err

# Load the conda environment
conda activate pyg_env

# Run the Python script
python cluster_experiments.py --tgm_type 'Planetoid' --name 'cora' --epochs 200