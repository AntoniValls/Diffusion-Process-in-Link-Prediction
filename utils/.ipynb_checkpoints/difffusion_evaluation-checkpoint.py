
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import random
import multiprocessing as mp
from utils.evaluation import read_prediction_files
from diffusion.contagion_models import si_simulation, threshold_diffusion
import pickle

def gini_coefficient(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad / np.mean(x) if np.mean(x) else 0
    return 0.5 * rmad

class BaseEvaluate:
    def __init__(self, result, density_correction=True, add_validation=True):
        self.graph = result["train_graph"]
        self.test_predictions = result["test_predictions"]
        self.test_labels = result["test_labels"]
        self.test_edges = result["test_edges"]
        self.val_predictions = result["val_predictions"]
        self.val_labels = result["val_labels"]
        self.val_edges = result["val_edges"]

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

    def calculate_density(self):
        density_true = nx.density(self.G_true)
        density_pred = nx.density(self.G_pred)
        return density_true, density_pred

    def calculate_clustering(self):
        clustering_true = nx.average_clustering(self.G_true)
        clustering_pred = nx.average_clustering(self.G_pred)
        return clustering_true, clustering_pred

    def degree_distribution_gini(self, G):
        degrees = [d for n, d in G.degree()]
        degrees = np.array(degrees)
        return gini_coefficient(degrees)

    def compare_metrics(self):
        density_true, density_pred = self.calculate_density()
        clustering_true, clustering_pred = self.calculate_clustering()
        gini_true = self.degree_distribution_gini(self.G_true)
        gini_pred = self.degree_distribution_gini(self.G_pred)

        metrics = {
            "density": density_pred - density_true,
            "clustering": clustering_pred - clustering_true,
            "degree_distribution_gini": gini_pred - gini_true
        }
        return metrics

    def _run_simulation(self, graph, p, n_simulations):
        """
        General method to run diffusion simulations.
        Calls the simulation_wrapper for the specific contagion model.
        """
        results = {
            "simulation_id": [],
            "iterations": [],
            "infection_size": [],
            "infection_rate": []
        }

        infection_count = defaultdict(int)
        infection_timesteps = defaultdict(list)

        for sim_id in tqdm(range(n_simulations), desc="Running evaluations"):
            sim_id, iterations, infection_size, infection_rate, infection_timeline = self.simulation_wrapper(
                (graph, p, sim_id))

            results["simulation_id"].append(sim_id)
            results["iterations"].append(iterations)
            results["infection_size"].append(infection_size)
            results["infection_rate"].append(infection_rate)

            # Update infection count and timesteps
            for timestep, newly_infected in infection_timeline.items():
                for node in newly_infected:
                    infection_count[node] += 1
                    infection_timesteps[node].append(1 / (timestep + 1))

        results_df = pd.DataFrame(results)

        vulnerability = {node: infection_count[node] / n_simulations for node in graph.nodes}
        recency = {
            node: sum(infection_timesteps[node]) / n_simulations if infection_timesteps[node] else None
            for node in graph.nodes
        }

        return results_df, vulnerability, recency

    def diffusion_evaluation(self, graph, p, n_simulations):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def paralell_diffusion(self, graph, p=0.1, n_simulations=100, n_cores=None):
        if n_cores is None:
            n_cores = mp.cpu_count()

        pool = mp.Pool(processes=n_cores)

        args = [(graph, p, sim_id) for sim_id in range(n_simulations)]
        results = []

        for result in tqdm(pool.imap(self.simulation_wrapper, args), total=n_simulations, desc="Running evaluations"):
            results.append(result)

        pool.close()
        pool.join()

        return self._process_results(results, graph, n_simulations)

    def simulation_wrapper(self, args):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _process_results(self, results, graph, n_simulations):
        """
        Process the raw results of the simulation and calculate vulnerability and recency for each node.

        :param results: List of simulation results where each result is a tuple:
                        (sim_id, iterations, infection_size, infection_rate, infection_timeline)
        :param graph: The graph on which the diffusion took place.
        :param n_simulations: The number of simulations that were run.
        :return: DataFrame of results, vulnerability dict, recency dict
        """
        # Initialize result storage
        processed_results = {
            "simulation_id": [],
            "iterations": [],
            "infection_size": [],
            "infection_rate": []
        }

        infection_count = defaultdict(int)  # To track how often a node gets infected
        infection_timesteps = defaultdict(list)  # To track the time at which a node gets infected

        # Process each simulation result
        for result in results:
            sim_id, iterations, infection_size, infection_rate, infection_timeline = result

            # Append the basic results
            processed_results["simulation_id"].append(sim_id)
            processed_results["iterations"].append(iterations)
            processed_results["infection_size"].append(infection_size)
            processed_results["infection_rate"].append(infection_rate)

            # Update infection count and infection timesteps for each node
            for timestep, newly_infected in infection_timeline.items():
                for node in newly_infected:
                    infection_count[node] += 1
                    infection_timesteps[node].append(1 / (timestep + 1))  # Adjust the time factor

        # Create a DataFrame from the processed results
        results_df = pd.DataFrame(processed_results)

        # Calculate vulnerability and recency for each node
        vulnerability = {node: infection_count[node] / n_simulations for node in graph.nodes}
        recency = {
            node: sum(infection_timesteps[node]) / n_simulations if infection_timesteps[node] else None
            for node in graph.nodes
        }

        return results_df, vulnerability, recency

    def pred_true_diffusion(self, p=0.1, n_simulations=100, paralell=False, n_cores=None):
        if paralell:
            true_df, true_vulnerability, true_recency = self.paralell_diffusion(graph=self.G_true, p=p,
                                                                                n_simulations=n_simulations,
                                                                                n_cores=n_cores)
            pred_df, pred_vulnerability, pred_recency = self.paralell_diffusion(graph=self.G_pred, p=p,
                                                                                n_simulations=n_simulations,
                                                                                n_cores=n_cores)
        else:
            true_df, true_vulnerability, true_recency = self.diffusion_evaluation(graph=self.G_true, p=p,
                                                                                  n_simulations=n_simulations)
            pred_df, pred_vulnerability, pred_recency = self.diffusion_evaluation(graph=self.G_pred, p=p,
                                                                                  n_simulations=n_simulations)

        combined_dict = {}

        # Get the degree of each node
        degrees = dict(self.G_true.degree())
        # dif_centrality = diffusion_centrality(self.G_true, T=5)
        # Perform the left join and add degree
        for node in list(self.G_true.nodes()):
            combined_dict[node] = {
                'true_vulnerability': true_vulnerability.get(node, None),
                'true_recency': true_recency.get(node, None),
                'pred_vulnerability': pred_vulnerability.get(node, None),
                'pred_recency': pred_recency.get(node, None),
                'degree': degrees.get(node, 0)
                # 'diffusion_centrality': dif_centrality.get(node, 0)
            }

        return true_df, pred_df, combined_dict

        #return true_df, pred_df

    def full_evaluate(self, p=0.1, n_simulations=100, paralell=False, n_cores=None):
        """
        Run the full evaluation, which includes running the contagion models on the true and predicted graphs,
        comparing vulnerability and recency, and calculating graph-level metrics.
        """
        true_si, test_si, information_vulnerability = self.pred_true_diffusion(p=p, n_simulations=n_simulations,
                                                                               paralell=paralell, n_cores=n_cores)
        metrics = self.compare_metrics()

        vulnerability_df = pd.DataFrame(information_vulnerability).T
        vulnerability_df.fillna(0, inplace=True)

        return true_si, test_si, vulnerability_df, metrics

    def p_search(self, ps=np.arange(0, 1, 0.01)):

        graph = self.G_true.copy()


        return_dict = {
            "infection_probability": ps,
            "epidemic_length": [],
            "epidemic_size": []
        }

        # Loop through each probability value and run simulations
        for p in tqdm(ps, desc="Probability", position=0):
            l_list = []  # List to store epidemic lengths
            s_list = []  # List to store epidemic sizes
            for sim_id in range(100):
                _, l, s, _, _ = self.simulation_wrapper(
                (graph, p, sim_id ))  # Run the SI model
                l_list.append(l)
                s_list.append(s)
            return_dict["epidemic_length"].append(np.mean(l_list))  # Store average epidemic length
            return_dict["epidemic_size"].append(np.mean(s_list))  # Store average epidemic size

        return pd.DataFrame(return_dict)

class SimpleEvaluate(BaseEvaluate):
    def simulation_wrapper(self, args):
        graph, p, sim_id = args
        iterations, infection_size, infection_timeline = si_simulation(graph, prob=p)
        infection_rate = infection_size * len(graph) / iterations
        return sim_id, iterations, infection_size, infection_rate, infection_timeline

    def diffusion_evaluation(self, graph, p=0.1, n_simulations=100):
        return self._run_simulation(graph, p, n_simulations)


class ComplexEvaluate(BaseEvaluate):
    def simulation_wrapper(self, args):
        graph, p, sim_id = args
        iterations, infection_size, infection_timeline = threshold_diffusion(graph, lam=p)
        infection_rate = infection_size * len(graph) / iterations
        return sim_id, iterations, infection_size, infection_rate, infection_timeline

    def diffusion_evaluation(self, graph, p=0.2, n_simulations=100):
        return self._run_simulation(graph, p, n_simulations)



def evaluate_dataset(model_name, data_name, eval_type="s", p=0.1, n_simulations=100, paralell=False, n_cores=None):

    files = read_prediction_files(model_name=model_name, data_name=data_name)

    # Initialize empty lists to store results for each file
    return_dict = {
        "true_si": [],
        "pred_si": [],
        "info_vulnerability": [],
        "metrics": []
    }

    if eval_type == "s":
        evaluator_class = SimpleEvaluate
    elif eval_type == "c":
        evaluator_class = ComplexEvaluate
        #p = 0.2 #change the default;
    else:
        raise ValueError("Unknown evaluation type. Use 's' for Simple Contagion or 'c' for Complex Contagion.")

    for counter, file in enumerate(files):
        evaluater = evaluator_class(result=file)
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


def evaluate_all(model_name, list_of_data, eval_type):

    final_list = []
    for data in list_of_data:
        result = evaluate_dataset(model_name=model_name,
                                  data_name=data,
                                  eval_type=eval_type)
        final_list.append(result)

    return final_list


def process_results(file_path):
    # for now just mean results
    with open(file_path, 'rb') as fp:
        result = pickle.load(fp)

    df_list_with_index = [df.reset_index() for df in result["info_vulnerability"]]
    vulnerability_df = pd.concat(df_list_with_index)
    vul_df = vulnerability_df.groupby("index").mean()
    vul_df["dif_recency"] = vul_df.pred_recency - vul_df.true_recency
    vul_df["dif_vulnerability"] = vul_df.pred_vulnerablity - vul_df.true_vulnerability

    true_si = pd.concat(result["true_si"])
    pred_si = pd.concat(result["pred_si"])
    metric_df = pd.DataFrame(result["metrics"])

    return true_si, pred_si, vul_df, metric_df