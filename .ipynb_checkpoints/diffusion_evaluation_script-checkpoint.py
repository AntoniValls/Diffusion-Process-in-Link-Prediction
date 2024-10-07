from utils.difffusion_evaluation import evaluate_dataset
import os
import os.path as osp
import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--n_simulations', type=int)
    parser.add_argument('--paralell', type=bool)
    args = parser.parse_args()

    project_root = os.path.abspath('')
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

    result = evaluate_dataset(model_name=args.model, data_name=args.data,
                              n_simulations=args.n_simulations, paralell=args.paralell)

    output_dir = 'data/results/diffusion'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{args.data}_si_{args.n_simulations}.pkl"
    outname = osp.join(output_dir, file_name)

    with open(outname, 'wb') as f:
        pickle.dump(result, f)
