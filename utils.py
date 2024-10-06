import numpy as np
import scipy.sparse as sp
import torch.nn as nn

import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree, to_undirected
import networkx as nx

ACTIVATIONS = {
    None: "sigmoid",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
    nn.ELU: "elu",
}


def create_optimal_inner_init(nonlinearity, **kwargs):
    """ Create initializer for inner layers of policy and value networks
    based on their activation function (nonlinearity).
    """

    nonlinearity = ACTIVATIONS.get(nonlinearity, nonlinearity)
    assert isinstance(nonlinearity, str)

    if nonlinearity in ["sigmoid", "tanh"]:
        weignt_init_fn = nn.init.xavier_uniform_
        init_args = kwargs
    elif nonlinearity in ["relu", "leaky_relu"]:
        weignt_init_fn = nn.init.kaiming_normal_
        init_args = {**{"nonlinearity": nonlinearity}, **kwargs}
    else:
        raise NotImplemented

    def inner_init(layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            weignt_init_fn(layer.weight.data, **init_args)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)

    return inner_init


def out_init(layer):
    """ Initialization for output layers of policy and value networks typically
    used in deep reinforcement learning literature.
    """
    if isinstance(layer, nn.Linear):
        # TODO check different variant of the initialization
        n = layer.weight.shape[0]
        v = 3e-3
        nn.init.uniform_(layer.weight.data, -v, v)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias.data, -v, v)


def preprocess_features(features, sparse=False):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # features = features/features.shape[1]
    if sparse:
        return sparse_to_tuple(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, sparse=False):
    """
    Preprocessing of adjacency matrix for
    simple GCN model and conversion to tuple representation.
    """
    # adj_normalized = normalize_adj(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.coo_matrix(adj)
    if sparse:
        return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def to_networkx(data, node_attrs=None, edge_attrs=None):
    r"""Converts a data object graph to a networkx graph.
    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
    """

    G = nx.Graph() if data.is_undirected() else nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = data[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def from_networkx(G):
    r""""Converts a networkx graph to a data object graph.
    Args:
        G (networkx.Graph): A networkx graph.
    """

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # print(list(list(G.nodes(data=True))[0]))
    # TODO
    #  File "/home/tkhakhulin/graph-rl/utils.py", line 151, in from_networkx
    #  keys += list(list(G.edges(data=True))[0][2].keys())
    #  IndexError: list index out of range

    keys = []
    keys += list(list(G.nodes(data=True))[0][1].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}
    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for key, item in data.items():
        data[key] = torch.tensor(item)

    if not G.is_directed():
        edge_index = to_undirected(edge_index, G.number_of_nodes())
    data['edge_index'] = edge_index
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data


def nx_to_pgeom(graph, device):
    assert type(graph) == nx.classes.graph.Graph
    pgeom_graph_format = from_networkx(graph)  # directed
    edge_index = pgeom_graph_format.edge_index.contiguous()
    n_nodes = len(nx.nodes(graph))
    norm_card = (2 * degree(edge_index[0]) / n_nodes).unsqueeze(-1)
    choices = torch.zeros(n_nodes, dtype=torch.uint8)

    # graph_features = torch.cat((choices.unsqueeze(-1).to(dtype=torch.float), norm_card), dim=-1)
    # x = torch.eye(n_nodes, dtype=torch.float)
    I = np.eye(n_nodes)
    graph_features = torch.tensor(preprocess_features(I), dtype=torch.float, requires_grad=True)
    # graph_features = preprocess_features()
    return Data(x=graph_features, edge_index=edge_index)  # .to(device)


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def to_geom_mode(g, from_nx=False, to_batch=False, device=torch.device('cpu')):
    if from_nx:
        return nx_to_pgeom(g, device)
    else:
        return adj_to_pgeom(g, to_batch, device=device)


def adj_to_pgeom(graph, to_batch=False, mask=None, use_small_features=True, device=torch.device('cpu')):
    """
    Transform adjacency matrix to the data format for pytorch geometric
    :param adj:  Adjacency matrix of graph
    :param to_batch: if true return batch data format (batch = None)
    :param mask: None for giant feature space
    :param use_small_features: if true X features has 2 dimension
    :param device: Device of data
    :return:
    """
    # graph = nx.from_numpy_array(adj)
    # pgeom_graph_format = from_networkx(graph)  # directed
    pgeom_graph_format = torch_geometric.utils.from_scipy_sparse_matrix(graph)
    # print(pgeom_graph_format[1])
    edge_index = pgeom_graph_format[0]
    # edge_index = pgeom_graph_format.edge_index.contiguous().to(device)
    # n_nodes = torch.max(edge_index[0])+ 1#.shape
    n_nodes = graph.shape[0]
    norm_card = (2 * degree(edge_index[1], n_nodes) / n_nodes).to(device)
    if use_small_features:
        if mask is not None:
            choices = 1 - mask
        else:
            choices = torch.ones(n_nodes, dtype=torch.uint8, device=device)
        graph_features = torch.cat((choices.unsqueeze(-1).to(dtype=torch.float), norm_card.unsqueeze(-1)), dim=-1)
    else:
        graph_features = torch.diag(norm_card).to(device)
    if not to_batch:
        return Data(x=graph_features, edge_index=edge_index)  # .to(device)
    else:
        return Batch(batch=None, x=graph_features, edge_index=edge_index).to(device)


def pygeom_to_nx(data_graph):
    edge_index = to_undirected(data_graph.edge_index)
    return to_networkx(Data(x=data_graph.x, edge_index=edge_index))


class SimpleLogger:
    """
    Logger with base functional like in dict
    """

    def __init__(self):
        self.info = dict()

    def add_info(self, name):
        self.info[name] = []

    def add_item(self, name, x):
        self.info[name].append(x)

    def get_log(self, name):
        return self.info[name]

    def get_current(self, name):
        return self.info[name][-1]

    def get_keys(self):
        return self.info.keys()


def plot_grad_flow_v2(named_parameters):
    #     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
