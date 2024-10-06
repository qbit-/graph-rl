import networkx as nx
import itertools
import copy


def relabel_graph_nodes(graph, label_dict=None, with_data=True):
    """
    Relabel graph nodes.The graph
    is relabelled (and returned) according to the label
    dictionary and an inverted dictionary is returned.
    If no dictionary
    will be passed then nodes will be relabeled to
    consequtive integers starting from 1.

    Parameters
    ----------
    graph : networkx.Graph
            graph to relabel
    label_dict : dict-like, default None
            dictionary for relabelling {old : new}
    Returns
    -------
    new_graph : networkx.Graph
            relabeled graph
    label_dict : dict
            {new : old} dictionary for inverse relabeling
    """
    # Ensure label dictionary contains integers or create one
    if label_dict is None:
        label_dict = {old: num for num, old in
                      enumerate(graph.nodes(data=False), 1)}
    else:
        label_dict = {key: val
                      for key, val in label_dict.items()}

    new_graph = nx.relabel_nodes(graph, label_dict, copy=True)

    # invert the dictionary
    inv_label_dict = {val: key for key, val in label_dict.items()}

    return new_graph, inv_label_dict


def eliminate_node(graph, node, self_loops=True):
    """
    Eliminates node according to the tensor contraction rules.
    A new clique is formed, which includes all neighbors of the node.

    Parameters
    ----------
    graph : networkx.Graph
            Graph containing the information about the contraction
            GETS MODIFIED IN THIS FUNCTION
    node : node to contract (such that graph can be indexed by it)
    self_loops : bool
           Whether to create selfloops on the neighbors. Default False.

    Returns
    -------
    None
    """
    # ensure that the graph is not multigraph. Otherwise
    # this function needs to be modified to avoid multiple edges
    assert(graph.is_multigraph() is False)

    # Delete node itself from the list of its neighbors.
    # This eliminates possible self loop
    neighbors_wo_node = list(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    if len(neighbors_wo_node) > 1:
        edges = itertools.combinations(neighbors_wo_node, 2)
    elif len(neighbors_wo_node) == 1 and self_loops:
        # This node had a single neighbor, add self loop to it
        edges = [[neighbors_wo_node[0], neighbors_wo_node[0]]]
    else:
        # This node had no neighbors
        edges = None

    graph.remove_node(node)
    if edges is not None:
        graph.add_edges_from(edges)


def draw_graph(graph, filename=''):
    """
    Draws graph with spectral layout
    Parameters
    ----------
    graph : networkx.Graph
            graph to draw
    filename : str, default ''
            filename for image output.
            If empty string is passed the graph is displayed
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spectral_layout(graph)

    try:
        node_color = list(map(int, graph.nodes()))
    except TypeError:
        node_color = list(range(graph.number_of_nodes()))

    nx.draw(graph, pos,
            node_color=node_color,
            node_size=100,
            cmap=plt.cm.Blues,
            with_labels=True
    )

    if len(filename) == 0:
        plt.show()
    else:
        plt.savefig(filename)


def get_simple_graph(graph, parallel_edges=False, self_loops=False):
    """
    Simplifies graph: MultiGraphs are converted to Graphs,
    selfloops are removed
    """
    if not parallel_edges and graph.is_multigraph():
        # deepcopy is critical here to copy edge dictionaries
        graph = nx.Graph(copy.deepcopy(graph), copy=False)
    if not self_loops:
        graph.remove_edges_from(graph.selfloop_edges())

    return graph
