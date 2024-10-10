#!/bin/bash
#SBATCH --job-name=lp_bias
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --output=log/lp_bias.out
#SBATCH --error=log/lp_bias.err

# Load the conda environment
source /home/antoni_valls/beegfs/miniconda3/etc/profile.d/conda.sh
conda activate lp_env
# Run the Python script
python cluster_experiments.py --model_name 'seal' --tgm_type 'Planetoid' --name 'citeseer' --epochs 5
