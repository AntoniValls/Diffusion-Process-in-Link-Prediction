import subprocess
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_names', type=str, help="Comma-separated dataset names, e.g., 'Cora,PubMed'")
    args = parser.parse_args()

    script_path = "scripts/node_level_evaluate.py"
    project_root = os.path.abspath('')
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(['python', script_path, '--model_name', str(args.model_name), '--data_names', args.data_names], capture_output=True, text=True,
                           env=env, cwd=project_root)
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
