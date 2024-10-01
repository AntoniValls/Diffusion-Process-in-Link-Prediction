import networkx as nx
import numpy as np
import os.path as osp
import os
import re
import pickle
import pandas as pd
from diffusion.contagion_models import si_simulation, threshold_diffusion
from diffusion.jackson_metrics import diffusion_centrality
from tqdm import tqdm
from collections import defaultdict
import warnings
import random
from utils.evaluation import read_prediction_files
import multiprocessing as mp

def gini_coefficient(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    if np.mean(x) == 0:
        return 0
    else:
        rmad = mad / np.mean(x)
        # Gini coefficient
        g = 0.5 * rmad
        return g


def si_simulation_wrapper(args):
    graph, p, sim_id = args
    iterations, infection_size, infection_timeline = si_simulation(graph, prob=p)
    infection_rate = infection_size * len(graph) / iterations
    return sim_id, iterations, infection_size, infection_rate, infection_timeline


class Evaluate:
    def __init__(self, result, density_correction=True, add_validation=True):
        self.node_df = None
        self.G_true = None
        self.G_pred = None
        self.graph = result["train_graph"]
        self.test_predictions = result["test_predictions"]
        self.test_labels = result["test_labels"]
        self.val_predictions = result["val_predictions"]
        self.val_labels = result["val_labels"]
        self.val_edges = result["val_edges"]
        self.test_edges = result["test_edges"]

        # Call true_predicted_edges during initialization
        self.add_edges(density_correction=density_correction, add_validation=add_validation)

    def true_predicted_edges(self, edges, labels, predictions):
        true = edges[:, labels == 1].T
        true_edges = list(map(tuple, true))

        predicted = edges[:, predictions >= 0.5].T
        predicted_edges = list(map(tuple, predicted))

        return true_edges, predicted_edges

    def add_edges(self, add_validation=True, density_correction=True):
        G_true = self.graph.copy()
        G_pred = self.graph.copy()

        true_edges, predicted_edges = self.true_predicted_edges(self.test_edges, self.test_labels,
                                                                self.test_predictions)
        G_pred.add_edges_from(predicted_edges)
        G_true.add_edges_from(true_edges)

        if add_validation:
            val_true_edges, val_predicted_edges = self.true_predicted_edges(self.val_edges, self.val_labels,
                                                                    self.val_predictions)
            G_pred.add_edges_from(val_predicted_edges)
            G_true.add_edges_from(val_true_edges)

        if density_correction:
            edge_difference = G_pred.number_of_edges() - G_true.number_of_edges()
            if edge_difference > 0:
                edges = list(G_pred.edges)
                edges_to_remove = random.sample(edges, edge_difference)
                G_pred.remove_edges_from(edges_to_remove)




        self.G_true = G_true
        self.G_pred = G_pred

    def old_true_predicted_edges(self, edges, labels, predictions,
                             density_correction=None):
        true = edges[:, labels == 1].T
        true_edges = list(map(tuple, true))
        num_true_edges = true.shape[0]#len(true_edges)

        predicted = edges[:, predictions >= 0.5].T
        num_predicted_edges = predicted.shape[0]

        if density_correction == "prob":
            sorted_idx = predictions.argsort()[::-1]
            top_predicted = edges.T[sorted_idx]
            predicted = top_predicted[:num_true_edges]
        elif density_correction == "random":
            # Number of predicted edges
            num_edges_to_remove = num_predicted_edges - num_true_edges
            print(num_edges_to_remove)
            if num_edges_to_remove < 0:
                warnings.warn("Number of predicted edges is smaller than the number of true edges. No edges to remove.")
            if num_edges_to_remove > 0:
                indices_to_remove = random.sample(range(num_predicted_edges), num_edges_to_remove)
                predicted = np.delete(predicted, indices_to_remove, axis=0)

        predicted_edges = list(map(tuple, predicted))

        return true_edges, predicted_edges

    def old_add_edges(self, add_validation=False):
        """
        Add real and predicted edges to the true and predicted graphs.
        Optionally add validation edges as well.
        """
        G_true = self.graph.copy()
        G_pred = self.graph.copy()

        # Add real edges for test
        true_test = self.test_edges[:, self.test_labels == 1]
        test_edges = list(map(tuple, true_test.T))
        predicted_edges_array = self.test_edges[:, self.test_predictions >= 0.5]
        predicted_edges = list(map(tuple, predicted_edges_array.T))

        G_pred.add_edges_from(predicted_edges)
        G_true.add_edges_from(test_edges)

        if add_validation:
            true_val = self.val_edges[:, self.val_labels == 1]
            val_edges = list(map(tuple, true_val.T))
            predicted_val_edges_array = self.val_edges[:, self.val_predictions >= 0.5]
            predicted_val_edges = list(map(tuple, predicted_val_edges_array.T))
            G_true.add_edges_from(val_edges)
            G_pred.add_edges_from(predicted_val_edges)

        self.G_true = G_true
        self.G_pred = G_pred

    def calculate_density(self):
        """
        Compare the density of the true and predicted graphs.
        """
        density_true = nx.density(self.G_true)
        density_pred = nx.density(self.G_pred)
        return density_true, density_pred

    def calculate_clustering(self):
        """
        Compare the average clustering coefficient of the true and predicted graphs.
        """
        clustering_true = nx.average_clustering(self.G_true)
        clustering_pred = nx.average_clustering(self.G_pred)
        return clustering_true, clustering_pred

    def degree_distribution_gini(self, G):
        """
        Calculate the Gini coefficient of the degree distribution in a graph.
        """
        degrees = [d for n, d in G.degree()]
        degrees = np.array(degrees)
        if np.sum(degrees) == 0:  # Handle the case of no degrees (empty graph)
            return 0.0
        return gini_coefficient(degrees)

    def compare_metrics(self):
        """
        Compare the density, clustering, and degree distribution Gini coefficient between the true and predicted graphs.
        """
        # Calculate densities
        density_true, density_pred = self.calculate_density()

        # Calculate clustering coefficients
        clustering_true, clustering_pred = self.calculate_clustering()

        # Calculate degree distribution Gini coefficients
        gini_true = self.degree_distribution_gini(self.G_true)
        gini_pred = self.degree_distribution_gini(self.G_pred)

        # Return comparison results
        metrics = {
            "density": density_pred - density_true,
            "clustering":  clustering_pred - clustering_true,
            "degree_distribution_gini": gini_pred - gini_true
        }

        return metrics


    def si_evaluation(self, graph, p=0.1, n_simulations=100):

        results = {
            "simulation_id": [],
            "iterations": [],
            "infection_size": [],
            "infection_rate": []
        }

        infection_count = defaultdict(int)  # Tracks how often a node is infected
        infection_timesteps = defaultdict(list)

        for sim_id in tqdm(range(n_simulations), desc="Running evaluations"):
            iterations, infection_size, infection_timeline = si_simulation(graph, prob=p)
            infection_rate = infection_size * len(graph) / iterations
            # infection_rate = infection_size / iterations #relative version
            # Update infection count and timesteps
            for timestep, newly_infected in infection_timeline.items():
                for node in newly_infected:
                    infection_count[node] += 1
                    infection_timesteps[node].append(1 / (timestep + 1))



            results["simulation_id"].append(sim_id)
            results["iterations"].append(iterations)
            results["infection_size"].append(infection_size)
            results["infection_rate"].append(infection_rate)


        results_df = pd.DataFrame(results)

        vulnerability = {node: infection_count[node] / n_simulations for node in graph.nodes}
        recency = {
            node: sum(infection_timesteps[node]) / n_simulations if infection_timesteps[node] else None
            for node in graph.nodes}

        return results_df, vulnerability, recency

    def paralell_si(self, graph, p=0.1, n_simulations=100, n_cores=None):

        # Prepare multiprocessing pool
        if n_cores is None:
            n_cores = mp.cpu_count()

        pool = mp.Pool(processes=n_cores)

        results = {
            "simulation_id": [],
            "iterations": [],
            "infection_size": [],
            "infection_rate": []
        }

        infection_count = defaultdict(int)  # Tracks how often a node is infected
        infection_timesteps = defaultdict(list)

        # Use pool to run simulations in parallel
        args = [(graph, p, sim_id) for sim_id in range(n_simulations)]
        for result in tqdm(pool.imap(si_simulation_wrapper, args), total=n_simulations, desc="Running evaluations"):
            sim_id, iterations, infection_size, infection_rate, infection_timeline = result

            # Store results
            results["simulation_id"].append(sim_id)
            results["iterations"].append(iterations)
            results["infection_size"].append(infection_size)
            results["infection_rate"].append(infection_rate)

            # Update infection count and timesteps
            for timestep, newly_infected in infection_timeline.items():
                for node in newly_infected:
                    infection_count[node] += 1
                    infection_timesteps[node].append(1 / (timestep + 1))

        pool.close()
        pool.join()

        results_df = pd.DataFrame(results)

        vulnerability = {node: infection_count[node] / n_simulations for node in graph.nodes}
        recency = {
            node: sum(infection_timesteps[node]) / n_simulations if infection_timesteps[node] else None
            for node in graph.nodes}

        return results_df, vulnerability, recency

    def si_pred_true(self, p=0.1, n_simulations=100, paralell=False, n_cores=None):
        if paralell:
            true_df, true_vulnerability, true_recency = self.paralell_si(graph=self.G_true, p=p,
                                                                           n_simulations=n_simulations, n_cores=n_cores)
            test_df, pred_vulnerability, pred_recency = self.paralell_si(graph=self.G_pred, p=p,
                                                                           n_simulations=n_simulations, n_cores=n_cores)
        else:
            true_df, true_vulnerability, true_recency = self.si_evaluation(graph=self.G_true, p=p, n_simulations=n_simulations)
            test_df, pred_vulnerability, pred_recency = self.si_evaluation(graph=self.G_pred, p=p, n_simulations=n_simulations)

        combined_dict = {}

        # Get the degree of each node
        degrees = dict(self.G_true.degree())
        #dif_centrality = diffusion_centrality(self.G_true, T=5)
        # Perform the left join and add degree
        for node in list(self.G_true.nodes()):
            combined_dict[node] = {
                'true_vulnerability': true_vulnerability.get(node, None),
                'true_recency': true_recency.get(node, None),
                'pred_vulnerablity': pred_vulnerability.get(node, None),
                'pred_recency': pred_recency.get(node, None),
                'degree': degrees.get(node, 0)
                #'diffusion_centrality': dif_centrality.get(node, 0)
            }

        return true_df, test_df, combined_dict

    def full_evaluate(self, p=0.1, n_simulations=100, paralell=False, n_cores=None):

        true_si, test_si, information_vulnerability = self.si_pred_true(p=p, n_simulations=n_simulations,
                                                                        paralell=paralell, n_cores=n_cores)
        metrics = self.compare_metrics()

        vulnerability_df = pd.DataFrame(information_vulnerability).T
        vulnerability_df.fillna(0, inplace=True)


        return true_si, test_si, vulnerability_df, metrics



def evaluate_dataset(model_name, data_name, p=0.1, n_simulations=100, paralell=False, n_cores=None):

    files = read_prediction_files(model_name=model_name, data_name=data_name)

    # Initialize empty lists to store results for each file
    return_dict = {
        "true_si": [],
        "pred_si": [],
        "info_vulnerability": [],
        "metrics": []
    }

    for counter, file in enumerate(files):
        evaluater = Evaluate(result=file)
        true_si, test_si, information_vulnerability, metrics = evaluater.full_evaluate(p=p, n_simulations=n_simulations,
                                                                                               paralell=paralell, n_cores=n_cores)
        true_si['file'] = counter
        test_si['file'] = counter
        information_vulnerability['file'] = counter


        # Append the results to their respective lists
        return_dict["true_si"].append(true_si)
        return_dict["pred_si"].append(test_si)
        return_dict["info_vulnerability"].append(information_vulnerability)
        return_dict["metrics"].append(metrics)


    return return_dict


def evaluate_all(model_name, list_of_data, method_list, method_names):

    final_list = []
    for data in list_of_data:
        result = evaluate_dataset(model_name=model_name, data_name=data, method_list=method_list,
                                  method_names=method_names)
        final_list.append(result)

    return final_list
