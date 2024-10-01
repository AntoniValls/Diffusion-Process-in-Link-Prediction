"""
Script for contagion models. Each model needs to output the total share of infected nodes,
the time it took to spread, and a dictionary with the number of newly infected nodes at each time step.
This output can be used to calculate the time of spread and other metrics such as information vulnerability.
"""

import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import truncnorm


# Simple contagion model (SI - Susceptible-Infected)

def si_step(graph, infected_node, prob):
    """
    Performs a single step of the SI contagion model for one infected node.

    Parameters:
        graph (networkx.Graph): The social network graph.
        infected_node (int): The currently infected node.
        prob (float): The probability of infection per contact.

    Returns:
        new_infected (list): A list of newly infected nodes in this step.
    """
    # Get list of contacts (edges) for the infected node
    contact_list = graph.edges(infected_node)
    new_infected = []

    # Loop through each contact and attempt infection based on the probability
    for infecter, contact in contact_list:
        if graph.edges[(infecter, contact)]["evaluated"] is False:
            if graph.nodes[contact]["infected"] == False:
                if random.random() < prob:
                    nx.set_node_attributes(graph, {contact: {"infected": True}})  # Mark contact as infected
                    new_infected.append(contact)  # Add contact to new infected list
            nx.set_edge_attributes(graph, {(infecter, contact): {"evaluated": True}})  # Mark edge as evaluated

    return new_infected


def si_full(graph, prob):
    """
    Executes the full SI contagion model until no more infections occur.

    Parameters:
        graph (networkx.Graph): The social network graph.
        prob (float): The probability of infection per contact.

    Returns:
        it (int): The number of timesteps taken for the contagion to stabilize.
        s (float): The proportion of nodes that were infected at the end.
        new_infections_per_timestep (dict): A dictionary tracking newly infected nodes at each timestep.
    """
    # Randomly select an initial infected node (patient zero)
    patient_zero = random.choice(list(graph.nodes))
    cur_infected = 1  # Current number of infected nodes
    nx.set_node_attributes(graph, {patient_zero: {"infected": True}})  # Set patient zero as infected
    new_infections_per_timestep = {0: [patient_zero]}  # Track infections at t=0
    it = 1  # Iteration counter

    while True:
        # Get list of currently infected nodes
        infected_nodes = [n for n, v in graph.nodes.data() if v['infected'] == True]
        it_new_infected = []  # Store new infections in this iteration

        # Attempt to infect neighbors of each currently infected node
        for infected in infected_nodes:
            i_new_infected = si_step(graph, infected, prob)
            it_new_infected.extend(i_new_infected)

        # Count total number of infected nodes
        new_infected = sum(dict(graph.nodes.data("infected")).values())

        # Record new infections at this timestep
        new_infections_per_timestep[it] = it_new_infected

        # If no new infections occurred, break the loop
        if cur_infected == new_infected:
            break
        else:
            it += 1
            cur_infected = new_infected  # Update current infected count

    # Calculate the proportion of infected nodes
    s = new_infected / len(graph)
    return it, s, new_infections_per_timestep


def add_spreading_params(G):
    """
    Initializes graph attributes for spreading simulation, setting all edges and nodes as unevaluated and uninfected.

    Parameters:
        G (networkx.Graph): The social network graph.
    """
    nx.set_edge_attributes(G, False, "evaluated")  # Mark all edges as unevaluated
    nx.set_node_attributes(G, False, "infected")  # Mark all nodes as uninfected


def si_simulation(graph, prob):
    """
    Runs a full SI contagion simulation.

    Parameters:
        graph (networkx.Graph): The social network graph.
        prob (float): The probability of infection per contact.

    Returns:
        Results of the si_full function.
    """
    add_spreading_params(graph)  # Initialize graph for the simulation
    return si_full(graph, prob)  # Run the full SI model


def run_simulations(graph, runs=100):
    """
    Runs multiple simulations of the SI contagion model over a range of infection probabilities.

    Parameters:
        graph (networkx.Graph): The social network graph.
        runs (int): Number of simulation runs for each probability value.

    Returns:
        A pandas DataFrame containing the average epidemic length and size for each infection probability.
    """
    ps = np.arange(0, 1, 0.01)  # Array of infection probabilities to simulate

    return_dict = {
        "infection_probability": ps,
        "epidemic_length": [],
        "epidemic_size": []
    }

    # Loop through each probability value and run simulations
    for p in tqdm(ps, desc="Probability", position=0):
        l_list = []  # List to store epidemic lengths
        s_list = []  # List to store epidemic sizes
        for _ in range(runs):
            l, s = si_simulation(graph, p)  # Run the SI model
            l_list.append(l)
            s_list.append(s)
        return_dict["epidemic_length"].append(np.mean(l_list))  # Store average epidemic length
        return_dict["epidemic_size"].append(np.mean(s_list))  # Store average epidemic size

    return pd.DataFrame(return_dict)  # Return results as a DataFram



def threshold_diffusion(graph, lam, initial_infected=None):
    """
    Simulates the threshold contagion model on the given graph.

    Parameters:
        graph (networkx.Graph): The social network graph.
        lam (float): The mean threshold value for the nodes.
        initial_infected (list, optional): List of initially infected nodes. If None, two random nodes are chosen.

    Returns:
        conversion_rate (float): The fraction of nodes that became infected.
        new_infections_per_timestep (dict): A dictionary tracking newly infected nodes at each time step.
    """

    # Initialize the converted (infected) list with the initial infected nodes or choose two random nodes
    if initial_infected is None:
        initial_infected = random.sample(list(graph.nodes),
                                         max(2, int(np.round(len(graph) * 0.01))))

    converted_list = initial_infected[:]  # Start with the initially infected nodes

    # Initialize the dictionary to track new infections
    new_infections_per_timestep = {0: initial_infected}

    # Generate threshold values for each node (mean threshold = lam, std dev = 0.2, bounded between [0, 1])
    num_nodes = len(graph.nodes)
    lower_bound, upper_bound = 0, 1
    std_dev = 0.2  #

    # Truncated normal distribution between 0 and 1
    threshold = truncnorm.rvs((lower_bound - lam) / std_dev, (upper_bound - lam) / std_dev,
                              loc=lam, scale=std_dev, size=num_nodes)

    # Map node index to threshold value
    node_threshold = {node: threshold[i] for i, node in enumerate(graph.nodes)}

    it = 1  # Iteration counter for time steps

    while True:
        # Initialize list for newly converted nodes in this iteration
        new_converted = []

        # Iterate through each node in the graph
        for node in graph.nodes:
            if node not in converted_list:
                infected_neighbors = 0
                total_neighbors = len(list(graph.neighbors(node)))

                # Calculate the number of infected neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor in converted_list:
                        infected_neighbors += 1

                # Check if the fraction of infected neighbors exceeds the node's threshold
                if total_neighbors > 0 and infected_neighbors / total_neighbors > node_threshold[node]:
                    new_converted.append(node)

        # Extend the converted list with newly infected nodes
        converted_list.extend(new_converted)

        # If no new nodes were infected, break the loop
        if len(new_converted) == 0:
            break
        else:
            # Record the newly converted nodes for the current timestep
            new_infections_per_timestep[it] = new_converted
            it += 1

    # Calculate the final conversion rate (fraction of infected nodes)
    conversion_rate = len(converted_list) / num_nodes

    return it, conversion_rate, new_infections_per_timestep


