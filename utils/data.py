import numpy as np
import torch


def load_dataset(dataset, p):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)

    return edge_index1, edge_index2, anchor_links, test_pairs


def get_adj_from_edge_index(edge_index, n):
    """
    Get adjacency matrix from edge index.
    :param edge_index: edge index
    :param n: number of nodes
    :return: adjacency matrix
    """

    adj = torch.zeros((n, n)).to(torch.float32)
    adj[edge_index[:, 0], edge_index[:, 1]] = 1
    adj[edge_index[:, 1], edge_index[:, 0]] = 1

    return adj
