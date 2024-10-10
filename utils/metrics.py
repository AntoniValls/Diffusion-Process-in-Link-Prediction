import numpy as np


def get_relevant_nodes(G, edges):

    relevant_nodes = []
    predicted_nodes = np.unique(edges)
    for node in predicted_nodes:
        if G.degree(node) > 0:
            relevant_nodes.append(node)

    return relevant_nodes


def get_sorted_links(edges, labels, predictions, node):
    # print(edges, labels, predictions, node)
    # Get the indices where this node appears in the test_edges
    indices_0 = np.where(edges[0] == node)[0]
    indices_1 = np.where(edges[1] == node)[0]

    # Combine the indices from both positions
    combined_indices = np.concatenate((indices_0, indices_1))

    # Get the labels and predictions for these indices
    node_labels = labels[combined_indices]
    node_predictions = predictions[combined_indices]

    # Sort the labels by the corresponding predictions
    sorted_indices = np.argsort(node_predictions)
    sorted_labels = node_labels[sorted_indices][::-1]

    return sorted_labels


def get_real_graph(G, test_edges, test_labels):
    true_test = test_edges[:,test_labels == 1]
    test_edges = list(map(tuple, true_test.T))
    G_new = G.copy()
    G_new.add_edges_from(test_edges)
    return G_new


def vcmpr_OG(G_test, edges, labels, predictions, k):

    scores = {}
    predicted_nodes = get_relevant_nodes(G_test, edges=edges)
    for node in predicted_nodes:
        
        # list of labels sorted by predictions
        sorted_labels = get_sorted_links(edges=edges, labels=labels,
                                         predictions=predictions, node=node)

        # total number of ground truth edges
        # This gives the total number of edges of a node in the true graph, not from the non-fixed edges
        node_degree = G_test.degree(node) 

        # number of ground truth edges for the top-k scores (t_i(k)) 
        true_edges_k = sorted_labels[:k].sum() # TP
        
        denominator = min(k, node_degree)
        
        vcmpr_node = true_edges_k / denominator

        scores[node] = vcmpr_node

    return scores

def vcmpr_IB_max(G_test, edges, labels, predictions, k):

    scores = {}
    predicted_nodes = get_relevant_nodes(G_test, edges=edges)
    for node in predicted_nodes:
        
        # list of labels sorted by predictions
        sorted_labels = get_sorted_links(edges=edges, labels=labels,
                                         predictions=predictions, node=node)

        node_degree_test = sorted_labels.sum() # TP + FN
        
        # number of ground truth edges for the top-k scores (t_i(k)) 
        true_edges_k = sorted_labels[:k].sum() # TP
        
        denominator = max(k, node_degree_test) # defined by Implicit Degree Bias paper

        vcmpr_node = true_edges_k / denominator

        scores[node] = vcmpr_node

    return scores

def vcmpr_IB_min(G_test, edges, labels, predictions, k):

    scores = {}
    predicted_nodes = get_relevant_nodes(G_test, edges=edges)
    for node in predicted_nodes:
        
        # list of labels sorted by predictions
        sorted_labels = get_sorted_links(edges=edges, labels=labels,
                                         predictions=predictions, node=node)

        node_degree_test = sorted_labels.sum() # TP + FN 
        
        # number of ground truth edges for the top-k scores (t_i(k)) 
        true_edges_k = sorted_labels[:k].sum() # TP

        if node_degree_test == 0:
            denominator = k
        else:
            denominator = min(k, node_degree_test) 

        vcmpr_node = true_edges_k / denominator

        scores[node] = vcmpr_node

    return scores

def vcmpr_BJ(G_test, edges, labels, predictions, k):

    scores = {}
    predicted_nodes = get_relevant_nodes(G_test, edges=edges)
    for node in predicted_nodes:
        
        # list of labels sorted by predictions
        sorted_labels = get_sorted_links(edges=edges, labels=labels,
                                         predictions=predictions, node=node)

        node_degree = G_test.degree(node)

        # number of ground truth edges for the top-k scores (t_i(k)) 
        true_edges_k = sorted_labels[:k].sum() # TP

        # len(sorted_labels) = 
        
        denominator = min(k, node_degree, len(sorted_labels))

        vcmpr_node = true_edges_k / denominator

        scores[node] = vcmpr_node

    return scores




