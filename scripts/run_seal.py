import os.path as osp
import os
import pickle
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import argparse
import scipy.sparse as ssp
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from models.train_gnn import train_and_predict
from torch_geometric import seed_everything
from utils.seal_utils import *
from utils.data_utils import data_loader
from utils.seal_datasets import SEALDataset
from models.gcn import GCN, GCN_woBatch
from torch_geometric.utils.convert import to_networkx

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def get_graph_results(model, train_dataset, val_dataset, test_dataset, device):
    train_data = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))).to(device)
    val_data = next(iter(DataLoader(val_dataset, batch_size=len(val_dataset)))).to(device)
    test_data = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset)))).to(device)
    
    G = to_networkx(train_data, to_undirected=True)

    val_logits = model(val_data.z, val_data.edge_index, val_data.batch, val_data.x, None, val_data.node_id)
    val_predictions = val_logits.view(-1).sigmoid().detach().cpu().numpy()
    val_label = val_data.y.view(-1).cpu().numpy().astype("float32")
    pos_val_edges, neg_val_edges = get_pos_neg_edges('valid', split_edge, val_data.edge_index, val_data.num_nodes, percent=100)
    val_edges = torch.cat((pos_val_edges, neg_val_edges), dim=1).numpy()
    
    test_logits = model(test_data.z, test_data.edge_index, test_data.batch, test_data.x, None, test_data.node_id)
    test_predictions = test_logits.view(-1).sigmoid().detach().cpu().numpy()
    test_label = test_data.y.view(-1).cpu().numpy().astype("float32")
    pos_test_edges, neg_test_edges = get_pos_neg_edges('test', split_edge, test_data.edge_index, test_data.num_nodes, percent=100)
    test_edges = torch.cat((pos_test_edges, neg_test_edges), dim=1).numpy()
    return {"train_graph": G,
            "test_predictions": test_predictions,
            "test_labels": test_label,
            "val_predictions": val_predictions,
            "val_labels": val_label,
            "val_edges": val_edges,
            "test_edges": test_edges}

def train_wBatch():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)

def train():
    model.train()

    optimizer.zero_grad()
    x = train_dataset.x if args.use_feature else None
    edge_weight = train_dataset.edge_weight if args.use_edge_weight else None
    node_id = train_dataset.node_id if emb else None
    logits = model(train_dataset.z, train_dataset.data.edge_label_index, x, edge_weight, node_id)
    loss = BCEWithLogitsLoss()(logits.view(-1), train_dataset.edge_label.to(torch.float))
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test_wBatch():
    model.eval()

    y_pred, y_true = [], []
    for data in val_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for data in test_loader:
        data = data.to(device)
        x = data.x if args.use_feature else None
        edge_weight = data.edge_weight if args.use_edge_weight else None
        node_id = data.node_id if emb else None
        logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

@torch.no_grad()
def test():
    model.eval()

    x = val_dataset.x if args.use_feature else None
    edge_weight = val_dataset.edge_weight if args.use_edge_weight else None
    node_id = val_dataset.node_id if emb else None
    logits = model(val_dataset.z, val_dataset.data.edge_label_index, x, edge_weight, node_id)
    val_pred = logits.view(-1).cpu()
    val_true = val_dataset.y.view(-1).cpu().to(torch.float)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    x = test_dataset.x if args.use_feature else None
    edge_weight = test_dataset.edge_weight if args.use_edge_weight else None
    node_id = test_dataset.node_id if emb else None
    logits = model(test_dataset.z, test_dataset.data.edge_label_index, x, edge_weight, node_id)
    test_pred = logits.view(-1).cpu()
    test_true = test_dataset.y.view(-1).cpu().to(torch.float)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    results = evaluate_auc(val_pred, val_true, test_pred, test_true)

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--tgm_type', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--fast_split', action='store_true', default=True, 
                    help="for large custom datasets (not OGB), do a fast data split")
    
    # GNN settings
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--max_z', type=float, default=1000) # set a large max_z so that every z has embeddings to look up
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--use_edge_weight', type=int, default=None)

    # Training settings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    # Subgraph extraction settings
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--node_label', type=str, default='drnl', 
                        help="which specific labeling trick to use") # Useful?
    parser.add_argument('--use_feature', action='store_true', default=True, 
                        help="whether to use raw node features as GNN input")
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

    # Load dataset
    dataset = data_loader(tgm_type=args.tgm_type, name=args.name, transform=None)
    split_edge = do_edge_split(dataset, args.fast_split)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
    directed = False

    # convert the data in seal_datasets, i.e. with the subgraphs done
    path = f"data/{args.tgm_type}/{args.name}/SEAL"
    dataset_class = 'SEALDataset'
    train_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=100, 
        split='train', 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    ) 
    if False:  # visualize some graphs
        import networkx as nx
        from torch_geometric.utils import to_networkx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        for g in loader:
            f = plt.figure(figsize=(20, 20))
            limits = plt.axis('off')
            g = g.to(device)
            node_size = 100
            with_labels = True
            G = to_networkx(g, node_attrs=['z'])
            labels = {i: G.nodes[i]['z'] for i in range(len(G))}
            nx.draw(G, node_size=node_size, arrows=True, with_labels=with_labels,
                    labels=labels)
            f.savefig('tmp_vis.png')
            pdb.set_trace()
    
    val_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=100, 
        split='valid', 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )
    
    test_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=100, 
        split='test', 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=args.max_nodes_per_hop, 
        directed=directed, 
    )
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # define the training process
    emb = None
    max_z = 1000  # set a large max_z so that every z has embeddings to look up
    model = GCN(args.hidden_channels, args.num_layers, max_z, train_dataset, 
                    args.use_feature).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
    
    for epoch in range(args.epochs):
        loss = train_wBatch()
        results = test_wBatch()
        for key, result in results.items():
            valid_res, test_res = result
            to_print = (f'Epoch: {epoch:02d}, ' +
                        f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                        f'Test: {100 * test_res:.2f}%')
            print(key)
            print(to_print)

    # get the graph results
    gresults = get_graph_results(model,
                                train_dataset,
                                val_dataset,
                                test_dataset,
                                device)
    
    # save outputs as pickle
    output_dir = f'data/results/{args.model_name}'
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{args.name}_seed_{args.seed}.pkl"
    outname = osp.join(output_dir, file_name)

    with open(outname, 'wb') as f:
        pickle.dump(gresults, f)

    del model
    del optimizer

    torch.cuda.empty_cache()

    
    
    

