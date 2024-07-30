import os.path as osp
import os
import pickle
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import argparse
from ..models.train_gnn import train_and_predict
from torch_geometric import seed_everything

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()

    #check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    #set seed
    seed_everything(args.seed)
    #load data
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    path = osp.join('data', 'Planetoid')
    dataset = Planetoid(path, name='Cora', transform=transform)
    # After applying the `RandomLinkSplit` transform, the data is transformed from
    # a data object to a list of tuples (train_data, val_data, test_data), with
    # each element representing the corresponding split.
    train_data, val_data, test_data = dataset[0]

    n_features = dataset.num_features

    result = train_and_predict(train_data=train_data, val_data=val_data, test_data=test_data,
                               n_features=n_features, device=device, epochs=args.epochs,
                               seed=args.seed)

    output_dir = 'data/results/gcn'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"cora_seed_{args.seed}"
    outname = osp.join(output_dir, file_name, "pkl")

    with open(outname, 'wb') as f:
        pickle.dump(result, f)

    torch.cuda.empty_cache()