import numpy as np


def get_relevant_nodes(G, edges):

    relevant_nodes = []
    predicted_nodes = np.unique(edges)
    for node in predicted_nodes:
        if G.degree(node) > 0:
            relevant_nodes.append(node)

    return relevant_nodes


def get_sorted_links(edges, labels, predictions, node):
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
    sorted_labels = node_labels[sorted_indices]

    return sorted_labels


def get_real_graph(G, test_edges, test_labels):
    true_test = test_edges[:,test_labels == 1]
    test_edges = list(map(tuple, true_test.T))
    G_new = G.copy()
    G_new.add_edges_from(test_edges)
    return G_new


def vcmpr(G_test, edges, labels, predictions, k):

    scores = {}
    predicted_nodes = get_relevant_nodes(G_test, edges=edges)

    for node in predicted_nodes:
        sorted_labels = get_sorted_links(edges=edges, labels=labels,
                                         predictions=predictions, node=node)

        node_degreee = G_test.degree(node)

        true_edges_k = sorted_labels[:k].sum()

        denominator = min(k, node_degreee, len(sorted_labels))

        vcmpr_node = true_edges_k / denominator

        scores[node] = vcmpr_node

    return scores




