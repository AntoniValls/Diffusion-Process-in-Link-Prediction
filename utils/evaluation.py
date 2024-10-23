import os.path as osp
import os
import re
import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
import torch_geometric
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from utils.metrics import vcmpr_OG, vcmpr_IB_min, vcmpr_BJ, vcmpr_IB_max
from utils.data_utils import data_loader, read_prediction_files

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

    def get_groups(self, groups=4):
        self.node_df["group"] = pd.qcut(self.node_df['centrality'], q=groups, labels=False)
        # TODO: check cutting criteria

    def get_mean_groups(self):
        mean_centrality = self.node_df["centrality"].mean()
        self.node_df["group"] = self.node_df["centrality"].apply(lambda x: 1 if x > mean_centrality else 0)

    def get_inter_intra_group_indices(self, group1, group2):
        inter_group_indices = []
        intra_group_1_indices = []
        intra_group_2_indices = []

        for i in range(self.test_edges.shape[1]):
            node1, node2 = self.test_edges[:, i]
            if node1 in group1 and node2 in group1:
                intra_group_1_indices.append(i)
            elif node1 in group2 and node2 in group2:
                intra_group_2_indices.append(i)
            elif (node1 in group1 and node2 in group2) or (node1 in group2 and node2 in group1):
                inter_group_indices.append(i)

        return intra_group_1_indices, intra_group_2_indices,  inter_group_indices

    def get_group_indices(self, list_of_nodes):

        mask = np.isin(self.test_edges, list_of_nodes)
        indices = np.where(mask.any(axis=0))[0]
        return indices

    def get_group_scores(self, indices, group):
        # TODO: add scores at k
        # TODO: support should be number of links and not number of nodes
        true = self.test_labels[indices]
        pred = self.test_predictions[indices]
        pred_classes = np.where(pred >= 0.5, 1, 0)
        if len(np.unique(true)) > 1:
            return_dict = {"group": group,
                           "roc_auc": roc_auc_score(y_true=true, y_score=pred),
                           "f1": f1_score(y_true=true, y_pred=pred_classes),
                           "recall": recall_score(y_true=true, y_pred=pred_classes),
                           "precision": precision_score(y_true=true, y_pred=pred_classes),
                           "accuracy": accuracy_score(y_true=true, y_pred=pred_classes)
                           }

            return return_dict
        else:
            print(f"Group {group} has only one class in y_true. Skipping ROC AUC calculation.")
            return None

    def evaluate_group(self, groups, method, *args, **kwargs):
        if not self.node_df:
            self.get_centrality(method, *args, **kwargs)
        self.get_groups(groups)
        group_scores = []
        # get the different groups and according nodes
        for group in range(groups):
            nodes = list(self.node_df["node_index"][self.node_df["group"] == group])
            indices = self.get_group_indices(list_of_nodes=nodes)
            scores = self.get_group_scores(indices=indices, group=group)
            if scores:
                scores["support"] = len(nodes)
                group_scores.append(scores)

        return pd.DataFrame(group_scores)

    def evaluate_group_links(self, method, *args, **kwargs):
        if self.node_df is None or self.node_df.empty:
            self.get_centrality(method, *args, **kwargs)
        self.get_mean_groups()

        group_1 = list(self.node_df["node_index"][self.node_df["group"] == 0])
        group_2 = list(self.node_df["node_index"][self.node_df["group"] == 1])

        low_intra, high_intra, inter = self.get_inter_intra_group_indices(group_1, group_2)

        groups = {
            "low_intra": low_intra,
            "high_intra": high_intra,
            "inter": inter
        }

        group_scores = []
        for group_name, group in groups.items():
            scores = self.get_group_scores(indices=group, group=group_name)
            if scores:
                scores["support"] = len(group)
                group_scores.append(scores)

        return pd.DataFrame(group_scores)

    def evaluate_graph(self, method, *args, **kwargs):
        #TODO adjust for precalculated scores
        if self.node_df is None or self.node_df.empty:
            self.get_centrality(method, *args, **kwargs)
        self.add_predicted_edges()
        self.add_real_edges()

        return_dict = {"train_gini": calculate_gini(np.array(self.node_df["centrality"]))}

        names = ["test_true_gini", "test_pred_gini"]

        for counter, graph in enumerate([self.G_pred, self.G_true]):
            node_centrality = method(graph, *args, **kwargs)
            node_df = pd.DataFrame(list(node_centrality.items()), columns=['node_index', 'centrality'])
            gini = calculate_gini(np.array(node_df["centrality"]))
            return_dict[names[counter]] = gini

        return return_dict


class EvaluateCentrality(Evaluate):
    def __init__(self, result):
        super().__init__(result)

    def make_graphs(self, validation=False):

        self.add_predicted_edges()
        self.add_real_edges(add_validation=validation)

    def flex_centrality(self, graph, method,  *args, **kwargs):

        node_centrality = method(graph, *args, **kwargs)
        node_df = pd.DataFrame(list(node_centrality.items()), columns=['node_index', 'centrality'])

        return node_df

    def write_centrality_to_disk(self, name, path, method, *args, **kwargs):

        self.make_graphs()
        names = ["train", "test_true", "test_pred"]
        data_list = []
        for counter, graph in enumerate([self.graph, self.G_true, self.G_pred]):

            centrality_df = self.flex_centrality(graph=graph, method=method, *args, **kwargs)

            data_list.append(centrality_df)
            #name = f"{names[counter]}_{name}"

            #save_path = osp.join(path, name)

        name = f"all_{name}"
        save_path = osp.join(path, name)

        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)


class EvaluateNode(Evaluate):
    def __init__(self, result):
        super().__init__(result)

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
        score_IB_max_df = pd.DataFrame(list(score_IB_max.items()), columns=["node_index", f"vcmpr_IB_max{k}"])

        score_IB_min = vcmpr_IB_min(G_test=self.G_true, edges=self.test_edges,
              labels=self.test_labels, predictions=self.test_predictions,
              k=k)
        
        score_IB_min_df = pd.DataFrame(list(score_IB_min.items()), columns=["node_index", f"vcmpr_IB_min{k}"])


        # merging
        score_df = pd.merge(score_IB_min_df, score_IB_max_df, on="node_index")

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

    results = []

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
    Function that loads the dataset and computes the node-level centrality measures
    and the graph-level properties.
    """
    dataset = data_loader(tgm_type=tgm_type, name=data_name, transform=None)
    G = torch_geometric.utils.to_networkx(dataset[0])
    G = G.to_undirected()
    result = get_centrality_df(G, method_list, method_names)

    # Compute graph-level properties
    graph_properties = graph_level_topologies(G, result)

    if save:
        result.to_csv(f'data/node_level_centralities/{data_name}.csv', index=False)
        pd.DataFrame([graph_properties]).to_json(f'data/graph_level_properties/{data_name}.json', orient='records')

    return result, graph_properties




    

    
    

    


    
    











