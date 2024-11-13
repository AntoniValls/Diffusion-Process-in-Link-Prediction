from models.gcn import Net, GATNet, SuperGATNet, GCN, GraphSAGENet
import torch
import os.path as osp

import torch
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.convert import to_networkx
from utils.utils import EarlyStopper
from torch_geometric import seed_everything


def train_and_predict(model_name, train_data, val_data, test_data, n_features, device, epochs, args=None, seed=42, printer=False):
    
    seed_everything(seed)
    if model_name == "gcn":
        model = Net(n_features, 128, 64).to(device)
    elif model_name == "gat":
        model = GATNet(n_features, 128, 64).to(device)
    elif model_name == "supergat":
        model = SuperGATNet(n_features, 128, 64).to(device)
    elif model_name == "sage":
        model = GraphSAGENet(n_features, 128, 64).to(device)
    
    else:
        raise TypeError("Invalid model: only gcn, gat, supergat, sage or seal") 
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    best_val_auc = final_test_auc = 0
    early_stopper = EarlyStopper(patience=10, min_delta=0)
            
    for epoch in range(1, epochs):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        if printer:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, 'f'Test: {test_auc:.4f}')
        if early_stopper.early_stop_score(val_auc):
            break

    print(f'Final Test: {final_test_auc:.4f}')        
    
    z = model.encode(test_data.x, test_data.edge_index)
    test_predictions = model.decode(z, test_data.edge_label_index).view(-1).sigmoid().detach().cpu().numpy()
    test_label = test_data.edge_label.cpu().numpy()
    z_val = model.encode(val_data.x, val_data.edge_index)
    val_predictions = model.decode(z_val, val_data.edge_label_index).view(-1).sigmoid().detach().cpu().numpy()
    val_label = val_data.edge_label.cpu().numpy()

    G = to_networkx(train_data, to_undirected=True)
    val_edges = val_data.edge_label_index.cpu().numpy()

    #TODO: We need the edge indices also for the negative ones, I'm stupid!
    #positive_test_edges = test_data.edge_label_index[:, test_data.edge_label == 1].cpu().numpy()
    all_test_edges = test_data.edge_label_index.cpu().numpy()
        
    del model
    del optimizer
    del criterion


    return {"train_graph": G,
            "test_predictions": test_predictions,
            "test_labels": test_label,
            "val_predictions": val_predictions,
            "val_labels": val_label,
            "val_edges": val_edges,
            "test_edges": all_test_edges}


    
