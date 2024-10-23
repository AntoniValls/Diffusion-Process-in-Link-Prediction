import networkx as nx
import matplotlib.pyplot as plt
import cmocean
import torch_geometric as tg
import numpy as np
from utils.data_utils import data_loader
from diffusion.jackson_metrics import decay_centrality, diffusion_centrality, Godfather
from diffusion.complex_diffusion import get_complex_path, paralell_complex_path
import os
import json
import pandas as pd


class CentralityMeasures:
    '''
    This object computes the centrality measures of any graph from the torch_geometric library
    Old way to do it but works good!
    '''
    def __init__(self, tgm_type, name, transform=None):

        self.tgm_type = tgm_type
        self.name = name
        self.transform = transform

        # Obtaing the graph object
        self.dataset = data_loader(self.tgm_type, self.name, self.transform)
        self.G = tg.utils.to_networkx(self.dataset[0], to_undirected=True)
        
        self.degree_centrality = None
        self.betweenness_centrality = None
        self.decay_centrality = None
        self.diffusion_centrality = None
        self.godfather = None
        self.complex_path = None
        self.parallel_complex_path = None

    def calculate_degree_centrality(self):
        self.degree_centrality = nx.degree_centrality(self.G)
        return self.degree_centrality
        
    def calculate_betweenness_centrality(self):
        self.betweenness_centrality = nx.betweenness_centrality(self.G)
        return self.betweenness_centrality
        
    def calculate_decay_centrality(self, p, T):
        self.decay_centrality = decay_centrality(self.G, p=p, T=T)
        return self.decay_centrality
        
    def calculate_diffusion_centrality(self, T):
        self.diffusion_centrality = diffusion_centrality(self.G, T=T)
        return self.diffusion_centrality
        
    def calculate_godfather(self):
        self.godfather = Godfather(self.G)
        return self.godfather
        
    def calculate_complex_path(self, treshold):
        self.complex_path = get_complex_path(self.G, treshold)
        return self.complex_path
        
    def calculate_parallel_complex_path(self, treshold):
        self.parallel_complex_path = paralell_complex_path(self.G, treshold)
        return self.parallel_complex_path

    def calculate_any_centrality(self, method, *args, **kwargs):
        node_centrality = method(self.G, *args, **kwargs)
        return node_centrality

    def get_degree_dist(self):
        degseq = [v for k, v in self.G.degree()]
        dmax = max(degseq) + 1
        freq = [0 for d in range(dmax)]
        for d in degseq:
            freq[d] += 1
        return freq

    def visualize_centralities(self):
        '''
        Visualitzation of each centrality by coloring the nodes.
        Not feasible for big graphs
        '''
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        centrality_measures = [
            ('Degree Centrality', self.degree_centrality),
            ('Betweenness Centrality', self.betweenness_centrality),
            ('Decay Centrality', self.decay_centrality),
            ('Diffusion Centrality', self.diffusion_centrality),
            ('Godfather Centrality', self.godfather),
            ('Complex Path Centrality', self.complex_path),
            ('Parallel Complex Path Centrality', self.parallel_complex_path)
        ]
        
        # Define a fixed layout
        pos = nx.spring_layout(self.G, seed=42)
        
        for ax, (title, centrality) in zip(axes, centrality_measures):
            if centrality is not None:
                nx.draw(self.G, pos=pos, with_labels=True, node_color=list(centrality.values()), ax=ax, node_size=500, cmap=cmocean.cm.thermal)
                ax.set_title(title)
            else:
                ax.set_title(f"{title} (Not Calculated)")
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def compare_centralities(self):
        '''
        Greedy "comparision"
        '''
        measures = {
            'Degree Centrality': self.degree_centrality,
            'Betweenness Centrality': self.betweenness_centrality,
            'Decay Centrality': self.decay_centrality,
            'Diffusion Centrality': self.diffusion_centrality,
            'Godfather Centrality': self.godfather,
            'Complex Path Centrality': self.complex_path,
            'Parallel Complex Path Centrality': self.parallel_complex_path
        }
        
        for measure_name, centrality in measures.items():
            if centrality is not None:
                plt.figure(figsize=(10, 5))
                plt.bar(centrality.keys(), centrality.values(), color='b', alpha=0.7)
                plt.xlabel('Nodes')
                plt.ylabel('Centrality Value')
                plt.title(measure_name)
                plt.show()
                
    def save_centralities(self):
        '''
        Function that saves the mesures inside the data folder
        '''
        
        measures = {
            'degree_centrality': self.degree_centrality,
            'betweenness_centrality': self.betweenness_centrality,
            'decay_centrality': self.decay_centrality,
            'diffusion_centrality': self.diffusion_centrality,
            'godfather_centrality': self.godfather,
            'complex_path_centrality': self.complex_path,
            'parallel_complex_path_centrality': self.parallel_complex_path
        }
        
        directory = f"data/{self.tgm_type}/{self.name}/measures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for measure_name, centrality in measures.items():
            if centrality is not None:
                file_path = os.path.join(directory, f"{measure_name}.json")
                with open(file_path, 'w') as f:
                    json.dump(centrality, f)

def run_all_centralities(tgm_type, name, p, T, threshold, transform=None):
    '''
    Run all the measures
    '''
    cm = CentralityMeasures(tgm_type, name, transform=transform)
    cm.calculate_degree_centrality()
    cm.calculate_betweenness_centrality()
    cm.calculate_decay_centrality(p=p, T=T)
    cm.calculate_diffusion_centrality(T=T)
    # cm.calculate_godfather() # has to be improved
    # cm.calculate_complex_path()
    cm.calculate_parallel_complex_path(threshold=threshold)
    cm.save_centralities()
    return
