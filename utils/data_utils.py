from torch_geometric.datasets import Planetoid, AttributedGraphDataset, Twitch, LastFMAsia, GitHub, CitationFull, Coauthor
import os.path as osp


def data_loader(tgm_type, name, transform):
    path = osp.join('data', tgm_type)

    if tgm_type == "Planetoid":
        dataset = Planetoid(path, name=name, transform=transform)
    elif tgm_type == "AttributedGraphDataset":
        dataset = AttributedGraphDataset(path, name=name, transform=transform)
    elif tgm_type == "Twitch":
        dataset = Twitch(path, name=name, transform=transform)
    elif tgm_type == "CitationFull":
        dataset = CitationFull(path, name=name, transform=transform)
    elif tgm_type == "Coauthor":
        dataset = Coauthor(path, name=name, transform=transform)
    elif tgm_type == "LastFMAsia":
        dataset = LastFMAsia(path, transform=transform) # here there is no name argument
    elif tgm_type == "GitHub":
        dataset = GitHub(path, transform=transform) # here there is no name argument
    else:
        raise NotImplementedError(f"Wrong torch_geometric type or not implemented error: {tgm_type}")

    return dataset

import networkx as nx
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import torch_geometric as tg

def fit_lognormal_distribution(tgm_type, name, transform=None, plot=False):
    ''' 
    Fits an log-normal distribution to a dataset from torch_geometric.
    Returns: mu and sigma.
    '''
    
    dataset = data_loader(tgm_type, name, transform)
    G = tg.utils.to_networkx(dataset[0])

    degrees = np.array([degree for node, degree in G.degree()])
    
    shape, loc, scale = lognorm.fit(degrees) 
    mu = np.log(scale)
    sigma = shape
    
    if plot == True:
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=30, density=True, alpha=0.6, color='g', label='Degree Histogram')
    
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = lognorm.pdf(x, shape, loc, scale)
        plt.plot(x, p, 'k', linewidth=2, label='Lognormal Fit')
    
        title = f"Fit results: mu = {mu:.2f},  sigma = {sigma:.2f} of {tgm_type}/{name}"
        plt.title(title)
        plt.xlabel('Degree')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()
    
    return mu, sigma
