import numpy as np
import networkx as nx
from tqdm import tqdm

'''
Script about the centrality mesures from Jackson et. al. 2020
'''


def decay_centrality(G, p, T):

    centrality = {} 
    for i in tqdm(G.nodes, desc= "Decay centrality:"):
        centrality[i] = sum(p**l * len(nx.single_source_shortest_path_length(G, i, cutoff=l)) for l in range(1, T+1))
        
    return centrality


def communication_centrality(G, T):
    '''
    THIS IS WRONG: It should be computed modelling the diffusion process.
    '''
        
    n = len(G.nodes)
    P = np.zeros((n, n))
    nodes = list(G.nodes)
    node_index = {nodes[i]: i for i in range(n)}

    for u, v in G.edges:
        P[node_index[u], node_index[v]] = G[u][v].get('weight', 1)
    
    # Raise the matrix P to the power of T
    P_T = np.linalg.matrix_power(P, T)  
    
    # Compute centrality: sum of rows in the matrix P_T
    centrality = P_T.sum(axis=1)
    
    return centrality


def largest_eigenvalue(A):
    eigenvalues = np.linalg.eigvals(A)  # L.A returns the adjacency matrix as a NumPy array
    return max(eigenvalues.real)


def get_first_eigenval(A, max_eigen=True):
    """
    A function to return the inverse of the first eigenvalue of adjancency matrix G
    :param A: Adjacency matrix
    :return q: Inverse of first eigenvalue
    """
    eigenvalues = np.linalg.eigvals(A)

    if max_eigen:
        lambda_1 = max(eigenvalues.real)
    else:
        lambda_1 = eigenvalues[0].real


    q = 1 / lambda_1

    return q



def diffusion_centrality(G, T):
    """
    Updated version of diffusion centrality as in Banerjee 2013. We set q to be the inverse of
    the first eigenvalue of the adjacency matrix. As T approaches infinity, proportial to eigenvector centrality
    But as communication is not finite, diffusion centrality might better capture the importance of nodes
    Problem: Is it the first eigenvalue or the largest eigenvalue?
    Also in another paper they suggest that T should be equal to the diameter of the graph (longest shortest path)
    as this gurantees some desirable properties


    :param G: input Graph
    :param T: Number of Iterations
    :return:
    """
    n = len(G.nodes)
    
    # Get the adjacency matrix
    A = nx.to_numpy_array(G)

    result_sum = np.zeros_like(A)

    q = get_first_eigenval(A)

    ones = np.ones(A.shape[0])
    for t in tqdm(range(1, T + 1), desc="Diffusion centrality:"):
        result_sum += np.linalg.matrix_power(q * A, t)

    centrality = np.dot(result_sum, ones)

    centrality_dict = {node: centrality[i] for i, node in enumerate(G.nodes)}
    
    return centrality_dict


def Godfather(G):

    '''
    Godfather Index function for undirected and unweighted graphs
    '''
    
    n = len(G.nodes)
    
    # Get the adjacency matrix
    g = nx.adjacency_matrix(G).todense()
    
    # Initialize Godfather index array
    godfather_index = np.zeros(n)
    
    # # Compute the Godfather index for each node i
    # for i in tqdm(range(n), "Godfather index:"):
    #     for j in range(n):
    #         for k in range(j + 1, n):
    #             godfather_index[i] += g[i, k] * g[i, j] * (1 - g[k, j])
        
    # godfather_index_dict = {node: godfather_index[i] for i, node in enumerate(G.nodes)}

    for i in tqdm(range(n), "Godfather index:"):
        # Column vector g_i (connections of node i to others)
        g_i = g[i, :]
        
        # Compute g_i * (1 - g) for all k,j
        g_complement = np.outer(g_i, g_i) * (1 - g)
        
        # Sum over k > j (upper triangle without diagonal)
        godfather_index[i] = np.sum(np.triu(g_complement, k=1))
        
    godfather_index_dict = {node: godfather_index[i] for i, node in enumerate(G.nodes)}

    return godfather_index_dict

def bla_diffusion_centrality(g, t, q):
    '''
    g = adjacency matrix
    t = iterations
    q = probability
    '''
    arg1 = sum([(q * np.array(g)) ** i for i in range(1, t + 1)])
    return np.dot(arg1, np.ones(arg1.shape[0]))
