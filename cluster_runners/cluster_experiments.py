import subprocess
import os
import argparse
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tgm_type', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    if args.model_name == "seal":
         script_path = "scripts/run_seal.py"
    else:
        script_path = "scripts/run_gcn.py"
    project_root = os.path.abspath('')
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    for seed in tqdm(range(10)): # THIS HAS TO BE 10
        result = subprocess.run(['python', script_path, '--model_name', str(args.model_name), '--tgm_type', str(args.tgm_type), '--name', str(args.name),
                                 '--seed', str(seed), '--epochs', str(args.epochs)], capture_output=True, text=True,
                               env=env, cwd=project_root)
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)