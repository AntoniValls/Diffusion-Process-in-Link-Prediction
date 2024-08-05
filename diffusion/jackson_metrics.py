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

def diffusion_centrality(G, T):

    n = len(G.nodes)
    
    # Get the adjacency matrix
    P = nx.adjacency_matrix(G)
    P = P.todense()

    centrality = np.zeros(n)
    
    # Compute p^l for l = 1 to T and accumulate the walks
    p_power = np.eye(n)  
    for l in tqdm(range(1, T + 1), desc="Diffusion centrality:"):
        p_power = np.matmul(p_power, P)  # Compute p^l
        centrality += np.sum(p_power, axis=1)  # Sum over j

    centrality_dict = {node: centrality[i] for i, node in enumerate(G.nodes)}
    
    return centrality_dict

def Godfather(G):
    
    n = len(G.nodes)
    
    # Get the adjacency matrix
    g = nx.adjacency_matrix(G)
    g = g.todense()
    
    # Initialize Godfather index array
    godfather_index = np.zeros(n)
    
    # Compute the Godfather index for each node i
    for i in tqdm(range(n), "Godfather index:"):
        for j in range(n):
            for k in range(j + 1, n):
                if g[k, j] == 0 and g[j, k] == 0:
                    godfather_index[i] += g[k, i] * g[j, i]                    

    godfather_index_dict = {node: godfather_index[i] for i, node in enumerate(G.nodes)}

    return godfather_index_dict

    #TODO: make it less efficient
    
# def Godfather(G):
#     n = len(G.nodes)
    
#     # Get the adjacency matrix
#     g = nx.adjacency_matrix(G).todense()
    
#     # Initialize Godfather index array
#     godfather_index = np.zeros(n)
    
#     # Compute the Godfather index for each node i
#     for i in tqdm(range(n), "Godfather index_corr:"):
#         # Use broadcasting and vectorized operations
#         godfather_index[i] = np.sum((g[:, i] * g.T[:, i]) * (1 - g) * (1 - g.T))

#     godfather_index_dict = {node: godfather_index[i] for i, node in enumerate(G.nodes)}

#     return godfather_index_dict