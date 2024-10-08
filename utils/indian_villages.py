# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import numpy as np
import os
import os.path as osp


class IndianVillages(Dataset):
    def __init__(self, root, adj, features, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.adj_matrix = adj
        self.features = features
        self.transform = transform
        self.pretransform = pre_transform
        super(IndianVillages, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return "IndianVillages.csv"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return "not_implemented.pt"

    def download(self):
        pass

    def process(self):
        row, col = np.where(self.adj_matrix > 0)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
        num_nodes = self.adj_matrix.shape[0]
        self.data = Data(x=self.features, edge_index=edge_index)

        self.data = self.data if self.pre_transform is None else self.pre_transform(self.data)

        torch.save(self.data, os.path.join(self.processed_dir, f'data.pt'))

    def len(self):
        return self.data.num_nodes

    def get(self, idx):
        ''' Here the idx isnt usefull as we only have 1 graph. Possible useful if I load all the villages '''
        data = torch.load(os.path.join(self.processed_dir, 
                                 f'data.pt'))
            
        return data
