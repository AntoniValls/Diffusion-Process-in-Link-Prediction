#!/bin/bash
#SBATCH --job-name=diffusion_result
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --output=log/dif_result.out
#SBATCH --error=log/dif_result.err

# Load the conda environment
source /home/antoni_valls/beegfs/miniconda3/etc/profile.d/conda.sh
conda activate lp_env
# Run the Python script
python diffusion_evaluation_script.py --model "gat" --data "ES" --eval_type "s" --prob 0.5 --n_simulation 100 --paralell True
