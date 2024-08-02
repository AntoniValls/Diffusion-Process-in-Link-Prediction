import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
"""
Script to caculate complex path lengths and complex centrality as in
'Topological measures for identifying and predicting the spread of complex contagions' from Guilbeault and Centola
"""

def get_sufficient_bridges(G, T):
    """
    Measure local connectivity of bridges and identify sufficient bridges.

    Parameters:
    G (networkx.Graph): Unweighted undirected graph
    T (float): Adoption threshold, homogeneous across all nodes

    Returns:
    dict: Dictionary indicating whether each bridge in G is sufficient
    """
    def closed_neighborhood(G, node):
        """Return the closed neighborhood of a node as an induced subgraph."""
        neighbors = list(G.neighbors(node)) + [node]
        return G.subgraph(neighbors)

    def overlap(N_i, N_j):
        """Return the overlap between neighborhoods N_i and N_j."""
        return set(N_i.nodes).intersection(set(N_j.nodes))

    def disjoint(N_i, N_j):
        """Return the disjoint set of nodes in N_j that are not in N_i."""
        return set(N_j.nodes).difference(set(N_i.nodes))

    def reinforcement(N_i, N_j, G):
        """Return the reinforcement set of nodes."""
        D_ij = disjoint(N_i, N_j)
        R_ij = {v for v in D_ij if any(w in N_i for w in G.neighbors(v))} #this is wrong
        return R_ij

    sufficient_bridges = np.zeros((len(G), len(G)))

    for i in G.nodes:
        N_i = closed_neighborhood(G, i)
        for j in G.nodes:
            if i != j:
                N_j = closed_neighborhood(G, j)
                O_ij = overlap(N_i, N_j)
                R_ij = reinforcement(N_i, N_j, G)
                B_W_ij = O_ij.union(R_ij)
                W_ij = len(B_W_ij)
                sufficient_bridges[i, j] = 1 if W_ij >= T else 0
    print(B_W_ij)
    return sufficient_bridges



def complex_path_length(G, T):
    """
    Potentially a very inefficient method to calculate shortest complex path lengths in a graph.
    Maybe it makes more sense to implement the model as in the paper, we you just use a simulation rather than
    exact calculations in exchange for some small error. Calculating the paths between all nodes can be quite long,
    however, there will be much less complex paths and simple paths, which could be beneficial.
    :param G:
    :param T:
    :return:
    """
    sufficient_bridges = get_sufficient_bridges(G, T)
    test = np.multiply(nx.to_numpy_array(G), sufficient_bridges)
    G_complex = nx.from_numpy_array(test)
    all_paths = dict(nx.all_pairs_shortest_path_length(G_complex))
    #we could return this but not needed
    avg_complex_path_len = {}
    for node in all_paths:
        len_sum = sum(all_paths[node].values())
        size_closed_neighorhood = len(list(G.neighbors(node))) + 1
        avg_len = len_sum / (len(G) - size_closed_neighorhood)
        avg_complex_path_len[node] = avg_len

    return avg_complex_path_len, sufficient_bridges


#the above is not working;


def complex_contagion(graph, T, source_node):
    """
    Runs a complex contagion from a source with homogenous treshold.
    Maybe not the most efficient? Given that we only act in local neighborhoods?
    :param graph: Input Graph
    :param T: Threshold for Adoption. The same for all nodes
    :param source_node: The source node from which to start the process. Will infect all neighbors
    :return: List of infected nodes
    """
    #infect all neighbors

    seed_nodes = list(graph.neighbors(source_node)) + [source_node]
    converted_list = seed_nodes[:]

    new_converted = seed_nodes[:]
    while len(new_converted) > 0:
        # Initialize list for newly converted nodes
        new_converted = []

        for node in graph.nodes:
            if node not in converted_list:
                weight = 0
                # Calculate weight from neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor in converted_list:
                        weight += 1

                if weight > T:
                    new_converted.append(node)

        converted_list.extend(new_converted)

    if set(converted_list) == set(seed_nodes):
        converted_list = []

    return converted_list


def get_complex_path(G, T):

    complex_paths = {}
    for node in G.nodes:

        infected_nodes = complex_contagion(G, T=T, source_node=node)

        if infected_nodes:
            sub = G.subgraph(infected_nodes)
            paths = nx.single_source_shortest_path_length(sub, source=node)
            len_sum = sum(paths.values())
            size_closed_neighorhood = len(list(G.neighbors(node))) + 1
            avg_len = len_sum / (len(G) - size_closed_neighorhood)
        else:
            avg_len = 0

        complex_paths[node] = avg_len

    return complex_paths


def process_node(node, G, T):
    # TODO: check if there is a mistake with calculating the shortest path to neighborhood also;
    infected_nodes = complex_contagion(G, T=T, source_node=node)
    if infected_nodes:
        sub = G.subgraph(infected_nodes)
        paths = nx.single_source_shortest_path_length(sub, source=node)
        len_sum = sum(paths.values())
        size_closed_neighorhood = len(list(G.neighbors(node))) + 1
        avg_len = len_sum / (len(G) - size_closed_neighorhood)
    else:
        avg_len = 0

    return node, avg_len


def paralell_complex_path(G, T, num_workers=None):
    complex_paths = {}
    num_workers = num_workers or multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_node, node, G, T): node for node in G.nodes}
        total_futures = len(futures)
        for future in tqdm(as_completed(futures), total=total_futures):
            node, avg_len = future.result()
            complex_paths[node] = avg_len

    return complex_paths