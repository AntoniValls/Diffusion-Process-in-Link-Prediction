import os.path as osp
import os
import re
import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
import torch_geometric
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from utils.metrics import vcmpr_OG, vcmpr_IB_min, vcmpr_BJ, vcmpr_IB_max
from utils.data_utils import data_loader, read_prediction_files
import cmocean.cm as cmo 

"""
Bjorn: "The classes below are not the best structure. One thing I could do however, 
is make the Evaluate class more abstract and then for each method (groups, inter-group links and whole graph) write a children 
class."
"""


def calculate_average_degree(test_edges, test_labels):
    """
    Function to calculate the average degree of nodes based on edges and labels.
    
    Parameters:
    test_edges (numpy.ndarray): 2D array where each column represents an edge between two nodes.
    test_labels (numpy.ndarray): 1D array where each value indicates if the corresponding edge exists (1) or not (0).

    Returns:
    int: The rounded average degree of the nodes.
    """
    
    # Filter edges where test_labels == 1 (edges that exist)
    actual_edges = test_edges[:, test_labels == 1]
    
    # Initialize a dictionary to store the degree of each node
    degree_count = defaultdict(int)
    
    # Count the degree for each node
    for edge in actual_edges.T:
        node1, node2 = edge
        degree_count[node1] += 1
        degree_count[node2] += 1
    
    # Compute the total degree and number of unique nodes
    total_degree = sum(degree_count.values())
    num_unique_nodes = len(degree_count)
    
    # Compute and return the average degree
    if num_unique_nodes == 0:
        return 0.0
    return round(total_degree / num_unique_nodes)

def gini_inefficient(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    if np.mean(x) == 0:
        return 0
    else:
        rmad = mad/np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g
        
def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

class Evaluate:

    def __init__(self, result):
        self.node_df = None
        self.G_true = None
        self.G_pred = None
        self.graph = result["train_graph"]
        self.test_predictions = result["test_predictions"]
        self.test_labels = result["test_labels"]
        self.val_edges = result["val_edges"]
        self.test_edges = result["test_edges"]
        self.roc_auc = roc_auc_score(self.test_labels, self.test_predictions)

    def add_real_edges(self, add_validation=False):
        # there should be some option to omit validation edges as we don't have predictions;
        true_test = self.test_edges[:, self.test_labels == 1]
        test_edges = list(map(tuple, true_test.T))
        G_new = self.graph.copy()
        G_new.add_edges_from(test_edges)
        if add_validation:
            val_edges = list(map(tuple, self.val_edges.T))
            G_new.add_edges_from(val_edges)
        self.G_true = G_new
        num_nodes = G_new.number_of_nodes()
        num_edges = G_new.number_of_edges()
        is_directed = nx.is_directed(G_new)

    def add_predicted_edges(self):
        predicted_edges_array = self.test_edges[:, self.test_predictions >= 0.5]
        predicted_edges = list(map(tuple, predicted_edges_array.T))
        G_new = self.graph.copy()
        G_new.add_edges_from(predicted_edges)
        self.G_pred = G_new
        num_nodes = G_new.number_of_nodes()
        num_edges = G_new.number_of_edges()
        is_directed = nx.is_directed(G_new)

    def get_centrality(self, method, *args, **kwargs):

        node_centrality = method(self.graph, *args, **kwargs)

        self.node_df = pd.DataFrame(list(node_centrality.items()), columns=['node_index', 'centrality'])

class EvaluateNode(Evaluate):
    def __init__(self, result):
        super().__init__(result)

    def get_roc_auc(self):
        return self.roc_auc
        
    def calculate_average_degree(self):
        """
        Function to calculate the average degree of nodes based on edges and labels.
        
        Parameters:
        test_edges (numpy.ndarray): 2D array where each column represents an edge between two nodes.
        test_labels (numpy.ndarray): 1D array where each value indicates if the corresponding edge exists (1) or not (0).
    
        Returns:
        int: The rounded average degree of the nodes.
        """
        
        # Filter edges where test_labels == 1 (edges that exist)
        actual_edges = self.test_edges[:, self.test_labels == 1]
        
        # Initialize a dictionary to store the degree of each node
        degree_count = defaultdict(int)
        
        # Count the degree for each node
        for edge in actual_edges.T:
            node1, node2 = edge
            degree_count[node1] += 1
            degree_count[node2] += 1
        
        # Compute the total degree and number of unique nodes
        total_degree = sum(degree_count.values())
        num_unique_nodes = len(degree_count)
        
        # Compute and return the average degree
        if num_unique_nodes == 0:
            return 0.0
        return round(total_degree / num_unique_nodes)
    
    def get_centrality_df(self, list_of_methods, method_names, graph):

        results = []
        if graph == "Train":
            G = self.graph
        elif graph == "True":
            G = self.G_true
        elif graph == "Pred":
            G = self.G_pred
        else:
            raise ValueError(f"Error: The variable '{graph}' is not correct.")

        for method in list_of_methods:
            method_result = method(G)
            if isinstance(method_result, dict):
                results.append(method_result)
            else:
                raise ValueError(f"Error: The centrality '{method}' is not returning a dict object.")
                
        df = pd.DataFrame(results).T

        df.columns = method_names
        # normalize with sd
        df = df / df.std()

        return df.reset_index(names="node_index")

    def score_df(self, k):
        '''
        Add VCMPR@k score, Implicit Bias paper definition and min variation
        '''
        score_IB_max = vcmpr_IB_max(G_test=self.G_true, edges=self.test_edges,
                      labels=self.test_labels, predictions=self.test_predictions,
                      k=k)
        score_IB_max_df = pd.DataFrame(list(score_IB_max.items()), columns=["node_index", f"vcmpr_IB_max"])

        # score_IB_min = vcmpr_IB_min(G_test=self.G_true, edges=self.test_edges,
        #       labels=self.test_labels, predictions=self.test_predictions,
        #       k=k)
        
        # score_IB_min_df = pd.DataFrame(list(score_IB_min.items()), columns=["node_index", f"vcmpr_IB_min"])


        # merging
        # score_df = pd.merge(score_IB_min_df, score_IB_max_df, on="node_index")

        return score_IB_max_df

    def only_vcmpr(self, k):
        self.add_real_edges()
        self.add_predicted_edges()
        score_df = self.score_df(k=k)
        return score_df

    def final_result(self, list_of_methods, method_names, k):
        self.add_real_edges()
        self.add_predicted_edges()

        true_centrality_df = self.get_centrality_df(list_of_methods=list_of_methods,
                                               method_names=method_names, graph="True") # True Graph
        pred_centrality_df = self.get_centrality_df(list_of_methods=list_of_methods,
                                               method_names=method_names, graph="Pred") # Predicted Graph
        score_df = self.score_df(k=k)

        merge1 = pd.merge(true_centrality_df, pred_centrality_df, on='node_index', suffixes=('_true', '_pred'))

        return pd.merge(score_df, merge1, how="left") # how="right" leaves NaNs 

def roc_auc(model_name, data_list):
    """
    Function that returns the roc_auc score of model
    """
    dicto = {}
    for data in data_list:
        files = read_prediction_files(model_name=model_name, data_name=data)
        scores = []
        for idx, file in enumerate(files):
            evaluater = EvaluateNode(result=file)
            scores.append(evaluater.get_roc_auc())
        dicto[data] = np.array(scores).mean()
    df = pd.DataFrame(list(dicto.items()), columns=['Dataset', 'Roc-Auc'])
    return df

def run_vcmpr(model_name, data_list, def_k=None):
    '''
    Function that returns the vcmpr@k scores of a model
    '''
    dicto = {}
    k_dict = {}
    k = def_k
    for data in data_list:
        files = read_prediction_files(model_name=model_name, data_name=data)
        scores = []
        
        for idx, file in enumerate(files):
            evaluater = EvaluateNode(result=file)
            # define k as the average degree on the test set of the first seeded version
            if def_k==None:
                if idx == 0:
                    k = evaluater.calculate_average_degree()    
            scores.append(evaluater.only_vcmpr(k=k))
            
        result = pd.concat(scores).groupby("node_index").mean().reset_index() # mean of the different seeds
        dicto[data] = result
        k_dict[data] = k
    return dicto, k_dict 
    
def evaluate_dataset(model_name, data_name, method_list, method_names):

    files = read_prediction_files(model_name=model_name, data_name=data_name)
    
    data_list = []
    for idx, file in tqdm(enumerate(files), desc=f'{data_name}', total=10):
        evaluater = EvaluateNode(result=file)
        
        # define k as the average degree on the test set of the first seeded version
        if idx == 0:
            k = evaluater.calculate_average_degree()
            
        data_list.append(evaluater.final_result(list_of_methods=method_list,
                                                method_names=method_names, k=k))

    result = pd.concat(data_list).groupby("node_index").mean().reset_index() # mean of the different seeds
    result["dataset"] = data_name

    return result


def evaluate_all(model_name, list_of_data, method_list, method_names):
    """
    Function for evaluating the predictions and real networks. It is not the best pipeline, not efficient, as one should repeat the centralities measures for every GNNs (which makes no sense).
    """

    final_list = []
    for data in list_of_data:

        # Obtain the results, the k value, and the ROC-AUC score for each dataset
        result = evaluate_dataset(model_name=model_name, data_name=data, method_list=method_list,
                                  method_names=method_names)

        final_list.append(result)

    return final_list


###########################################################################################
#                         More efficient way to do it                                     #
###########################################################################################
def get_centrality_df(G, list_of_methods, method_names):
    """
    Calculate centrality measures and node degrees for a graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph
    list_of_methods : list
        List of centrality calculation functions from networkx
    method_names : list
        Names for each centrality method
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing node indices, degrees, and normalized centrality measures
    """
    # Calculate centralities
    results = []
    for method in list_of_methods:
        method_result = method(G)
        if isinstance(method_result, dict):
            results.append(method_result)
        else:
            raise ValueError(f"Error: The centrality '{method}' is not returning a dict object.")
    
    # Create DataFrame with centralities
    df = pd.DataFrame(results).T
    df.columns = method_names
    
    # Add degree for each node
    degrees = dict(G.degree())
    df['degree'] = pd.Series(degrees)
    
    # Normalize with standard deviation
    columns_to_normalize = method_names  # Only normalize centrality measures, not degree
    df[columns_to_normalize] = df[columns_to_normalize] / df[columns_to_normalize].std()
    
    # Reset index and rename it to node_index
    return df.reset_index(names="node_index")

def graph_level_topologies(G, centrality_df):
    """
    Function to compute graph-level properties:
        - Average Degree
        - Average Clustering Coefficient
        - Gini Index for each centrality
    """
    # Calculate average degree
    degrees = dict(G.degree())
    average_degree = np.mean(list(degrees.values()))
    
    # Calculate average clustering coefficient
    clustering_coefficient = nx.average_clustering(G)
    
    # Calculate Gini index for each centrality in the DataFrame
    gini_indices = {}
    for col in tqdm(centrality_df.columns[1:], total=len(centrality_df.columns[1:]), desc=f"Gini Index"):  # Skip the 'node_index' column
        centrality_values = centrality_df[col].values
        gini_indices[col] = gini(centrality_values)
    
    # Create a dictionary of the results
    graph_properties = {
        "Average_Degree": average_degree,
        "Clustering_Coefficient": clustering_coefficient,
        "Gini_Indices": gini_indices
    }
    
    return graph_properties

      
def nodes_and_graph_properties(tgm_type, data_name, method_list, method_names, save=False):
    """
    Function that loads the dataset and computes the node-level centrality
    measures and the graph-level properties.
    """
    dataset = data_loader(tgm_type=tgm_type, name=data_name, transform=None)
    G = torch_geometric.utils.to_networkx(dataset[0])
    G = G.to_undirected() # in case it is not
    result = get_centrality_df(G, method_list, method_names)

    # Compute graph-level properties
    graph_properties = graph_level_topologies(G, result)

    if save:
        # Create directories if they do not exist
        os.makedirs('data/node_level_centralities', exist_ok=True)
        os.makedirs('data/graph_level_properties', exist_ok=True)

        result.to_csv(f'data/node_level_centralities/{data_name}.csv', index=False)
        pd.DataFrame([graph_properties]).to_json(f'data/graph_level_properties/{data_name}.json', orient='records')

    return result, graph_properties

def graph_level_plot(list_of_datasets, plot=True):
    """
    Function that loads the JSON files with the graph-level features of the datasets and plots them.
    """
    data_path = "data/graph_level_properties"
    datasets = []
    for data in list_of_datasets:
        datasets.append(data + ".json")
    
    # Load each JSON file into a dictionary
    graph_properties = {}
    for dataset in datasets:
        with open(os.path.join(data_path, dataset)) as f:
            graph_properties[dataset.split('.')[0]] = json.load(f)
    
    # Convert the graph properties into a DataFrame for easier plotting
    data_list = []
    for name, properties in graph_properties.items():
        row = {
            "Dataset": name,
            "Average_Degree": properties[0]["Average_Degree"],
            "Clustering_Coefficient": properties[0]["Clustering_Coefficient"],
            **{f"Gini_{k}": v for k, v in properties[0]["Gini_Indices"].items()}
        }
        data_list.append(row)
    
    df = pd.DataFrame(data_list)

    if plot:
        sns.color_palette("tab10")
        # Create subplots for Average Degree and Clustering Coefficient
        fig = plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(2, 2)
        ax1 = fig.add_subplot(grid[:1, :1])
        ax2 = fig.add_subplot(grid[:1, 1:])
        ax3 = fig.add_subplot(grid[1:, :])
        
        # Plot Average Degree
        df.plot(kind='bar', x='Dataset', y='Average_Degree', ax=ax1, color = cmo.deep(np.linspace(0, 1, 5)), legend=False)
        ax1.set_title("Average Degree for Each Dataset")
        ax1.set_ylabel("Value")
        ax1.set_xticklabels(df["Dataset"], rotation=0)
        ax1.grid(axis='y')
        
        # Plot Clustering Coefficient
        df.plot(kind='bar', x='Dataset', y='Clustering_Coefficient', ax=ax2, color = cmo.deep(np.linspace(0, 1, 5)), legend=False)
        ax2.set_title("Clustering Coefficient for Each Dataset")
        ax2.set_xticklabels(df["Dataset"], rotation=0)
        ax2.set_ylim([0,1])
        ax2.grid(axis='y')
        
        # Plot bar plots for Gini indices of centrality measures
        df_melted = df.melt(id_vars=["Dataset"], value_vars=[f"Gini_{col}" for col in properties[0]["Gini_Indices"].keys()],
                            var_name="Centrality Measure", value_name="Gini Index")
        
        sns.barplot(data=df_melted, x="Dataset", y="Gini Index", hue="Centrality Measure", ax=ax3)
        ax3.set_title("Gini Indices of Centrality Measures for Each Dataset")
        ax3.set_ylabel("Gini Index")
        ax3.grid(axis='y')
        
        plt.savefig('images/gini.png', dpi=300, bbox_inches='tight')
        plt.show()
    return df

def node_level_load(list_of_datasets):
    '''
    Function that loads the node-level properties of a list of
    datasets and return a dictionary with them.
    '''
    data_path = "data/node_level_centralities"
    datasets = []
    for data in list_of_datasets:
        datasets.append(data + ".csv")
    
    # Load each csv files into a dictionary
    node_centralities = {}
    for dataset in datasets:
        with open(os.path.join(data_path, dataset)) as f:
            node_centralities[dataset.split('.')[0]] = pd.read_csv(f)
    return node_centralities

    
    
                    



    

    
    

    


    
    











