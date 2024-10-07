#!/bin/bash
#SBATCH --job-name=diffusion_result
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --output=log/dif_result.out
#SBATCH --error=log/dif_result.err

# Load the conda environment
source /home/bkomander/miniconda3/etc/profile.d/conda.sh
conda activate /home/bkomander/beegfs/pyg_env
# Run the Python script
python diffusion_evaluation_script.py --model "gcn" --data "facebook" --n_simulation 1000 --paralell True