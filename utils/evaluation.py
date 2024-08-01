import os.path as osp
import os
import re
import pickle
import networkx as nx
import numpy as np
import pandas as pd
# from diffusion.complex_diffusion import complex_path_length
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
# from torch_geometric.metrics import LinkPredPrecision, LinkPredRecall, LinkPredF1
# no idea how above works??


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


def calculate_gini(x):
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

    def add_predicted_edges(self):
        predicted_edges_array = self.test_edges[:, self.test_predictions >= 0.5]
        predicted_edges = list(map(tuple, predicted_edges_array.T))
        G_new = self.graph.copy()
        G_new.add_edges_from(predicted_edges)
        self.G_pred = G_new

    def get_centrality(self, method, *args, **kwargs):

        node_centrality = method(self.graph, *args, **kwargs)

        self.node_df = pd.DataFrame(list(node_centrality.items()), columns=['node_index', 'centrality'])

    def get_groups(self, groups=4):
        self.node_df["group"] = pd.qcut(self.node_df['centrality'], q=groups, labels=False)
        # TODO: check cutting criteria

    def get_group_indices(self, list_of_nodes):

        mask = np.isin(self.test_edges, list_of_nodes)
        indices = np.where(mask.any(axis=0))[0]
        return indices

    def get_group_scores(self, list_of_nodes, group):
        # TODO: add scores at k
        indices = self.get_group_indices(list_of_nodes=list_of_nodes)
        true = self.test_labels[indices]
        pred = self.test_predictions[indices]
        pred_classes = np.where(pred >= 0.5, 1, 0)
        if len(np.unique(true)) > 1:
            return_dict = {"group": group,
                           "roc_auc": roc_auc_score(y_true=true, y_score=pred),
                           "f1": f1_score(y_true=true, y_pred=pred_classes),
                           "recall": recall_score(y_true=true, y_pred=pred_classes),
                           "precision": precision_score(y_true=true, y_pred=pred_classes),
                           "accuracy": accuracy_score(y_true=true, y_pred=pred_classes),
                           "support": len(list_of_nodes)}

            return return_dict
        else:
            print(f"Group {group} has only one class in y_true. Skipping ROC AUC calculation.")
            return None

    def evaluate_group(self, groups, method, *args, **kwargs):

        self.get_centrality(method, *args, **kwargs)
        self.get_groups(groups)
        group_scores = []
        # get the different groups and according nodes
        for group in range(groups):
            nodes = list(self.node_df["node_index"][self.node_df["group"] == group])
            scores = self.get_group_scores(list_of_nodes=nodes, group=group)
            if scores:
                group_scores.append(scores)

        return pd.DataFrame(group_scores)

    def evaluate_graph(self, method, *args, **kwargs):
        self.get_centrality(method, *args, **kwargs)
        #this gives you a centrality for the train graph
        # get centrality for the true graph
        # get centrality for the predicted graph;
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



# problem: centrality might take a long time;

