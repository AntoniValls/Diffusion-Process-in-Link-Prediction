from torch_geometric.datasets import Planetoid, AttributedGraphDataset, Twitch, LastFMAsia, GitHub, CitationFull, Coauthor, KarateClub
from utils.indian_villages import IndianVillages
import torch
import pandas as pd
import os
import os.path as osp
import re
import pickle
import warnings

def search_files(directory: str, pattern: str = '.') -> list:
    """
    Parameters
    ----------
    directory : str
        File directiory to return.
    pattern : str, optional
        DESCRIPTION. The default is '.'.

    Returns
    -------
    list
        sorted list of files in directory.

    """
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    # sorting files with numbers as strings does not sort them de or increasing
    return files


def get_prediction_files(model_name, data):
    # TODO: add pickle ending tbs?
    path = osp.join("data/results", model_name)
    files = search_files(path, pattern=f"{data}.")
    return files


def read_prediction_files(model_name, data_name):
    files = get_prediction_files(model_name, data_name)
    return_list = []
    for file in files:
        with open(file, 'rb') as fp:
            result_object = pickle.load(fp)
        return_list.append(result_object)

    return return_list

def data_loader(tgm_type, name=None, transform=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
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
            dataset = LastFMAsia(path, transform=transform) 
        elif tgm_type == "GitHub":
            dataset = GitHub(path, transform=transform) 
        elif tgm_type == "KarateClub":
            dataset = KarateClub(transform=transform) 
        elif tgm_type == "IndianVillages": 
            ''' This option is for using the Indian Villages HouseHolders dataset as it 
            was loaded from TGM.
            
            tgm_type == IndianVillages
            name == (number of the village)
            '''
    
            name = int(name)
            if 1 <= name <= 77 and name not in [13, 22]:
                # Load adjacency matrix
                search_dir = "data/indian_villages/Data/1. Network Data/Adjacency Matrices"
                pattern = f".allVillageRelationships_HH_vilno_{name}." 
                file = search_files(directory=search_dir, pattern=pattern)
    
                adj_matrix = np.genfromtxt(file[0], delimiter=",")
    
                # Load features
                HH_feat_path = "data/indian_villages/Data/2. Demographics and Outcomes/household_characteristics.dta"
                HH_features = pd.read_stata(HH_feat_path)
                HH_features = HH_features[HH_features["village"] == name]
    
                # Select relevant columns
                selected_columns = ["hohreligion", "castesubcaste", "rooftype1", "rooftype2", "rooftype3", "rooftype4", "rooftype5",	 "room_no", "electricity", "latrine", "hhSurveyed", "ownrent", "leader"]
                HH_features = HH_features[selected_columns]
        
                # Process create dummies for categorical columns   
                categorical_columns = ["hohreligion", "castesubcaste", "electricity", "latrine", "ownrent"]
                HH_features = pd.get_dummies(HH_features, columns=categorical_columns, drop_first=False)
                HH_features = HH_features.map(lambda x: 1 if x is True else (0 if x is False else x))
    
                # Obtain the dataset
                dataset = IndianVillages(root='data/indian_villages/Using', adj=adj_matrix, features=HH_features.values, transform=transform)
    
            else:
                 raise ValueError(f"Error: The name '{name}' is not correct. It must be between 1 and 77, and cannot be 13 or 22.")
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
