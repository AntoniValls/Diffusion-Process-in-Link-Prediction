import os.path as osp
import os
import pickle
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import argparse
from models.train_gnn import train_and_predict
from torch_geometric import seed_everything
from utils.data_utils import data_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tgm_type', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    # check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # set seed
    seed_everything(args.seed)
    # load data
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])

    dataset = data_loader(tgm_type=args.tgm_type, name=args.name, transform=transform)

    train_data, val_data, test_data = dataset[0]
    n_features = dataset.num_features
    # train model
    result = train_and_predict(model_name=args.model_name,
                               train_data=train_data,
                               val_data=val_data,
                               test_data=test_data,
                               n_features=n_features,
                               device=device,
                               epochs=args.epochs,
                               seed=args.seed)
    
    # save outputs as pickle
    output_dir = f'data/results/{args.model_name}'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{args.name}_seed_{args.seed}.pkl"
    outname = osp.join(output_dir, file_name)

    with open(outname, 'wb') as f:
        pickle.dump(result, f)

    torch.cuda.empty_cache()
