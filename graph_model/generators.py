import numpy as np
import networkx as nx
import itertools
import copy


def generate_k_tree(treewidth, n_nodes):
    """
    Generates a k-tree with a given treewidth and the
    number of nodes
    Parameters
    ----------
    treewidth: int
             desired treewidth
    n_nodes: int
            desired number of nodes. Must be at least treewidth+1
    Returns
    -------
    graph: nx.Graph
    """
    assert(treewidth + 1 <= n_nodes)

    n_cliques = n_nodes - (treewidth + 1) + 1
    tree = nx.generators.random_tree(n_cliques)

    # find a leaf
    for node in tree.nodes():
        if (len(list(tree.neighbors(node))) <= 1):
            break

    # build graph dictionary using DFS
    clique_dict = {node: frozenset(range(treewidth + 1))}
    next_index = treewidth + 1

    def build_clique_dict(node, parent):
        nonlocal next_index
        new_clique = sorted(clique_dict[parent])[1:] + [next_index]
        clique_dict[node] = frozenset(new_clique)

        next_index = next_index + 1
        children = [neighbor for neighbor
                    in tree.neighbors(node)
                    if neighbor != parent]
        if len(children) == 0:
            return
        for child in children:
            build_clique_dict(child, node)

    if n_cliques > 1:
        build_clique_dict(next(tree.neighbors(node)), node)

    graph = nx.Graph()
    graph.add_nodes_from(range(next_index))

    for node in tree.nodes():
        graph.add_edges_from(itertools.combinations(
            clique_dict[node], 2))

    return graph


def prune_k_tree(old_ktree, probability, n_cliques=1):
    """
    Prunes a k-tree preserving its treewidth (k).
    The edges are preserved with a given probability.
    The resulting graph is a union of a clique/partial k-trees
    with an Erdos graph

    Parameters
    ----------
    old_ktree: nx.Graph
               This is a ktree to prune
    probability: float
               Probability to preserve edge
    n_cliques: int, default 1
               The number of cliques to save in the result
    Returns
    -------
    pruned_ktree: nx.Graph
    """

    # save a copy to make this function pure
    ktree = copy.deepcopy(old_ktree)

    # choose some cliques to keep. We choose clique roots at random
    preserved_roots = list(np.random.choice(
        ktree.nodes, n_cliques, replace=False))
    # now extract other nodes in cliques which have these roots
    all_neighbors = []
    for root in preserved_roots:
        all_neighbors += list(ktree.neighbors(root))
    all_neighbors = list(set(all_neighbors))

    # finally extract chosen cliques/ktrees
    preserved_subgraph = nx.subgraph(ktree, all_neighbors+preserved_roots)

    # Extract edges to keep
    preserved_edges = sorted(preserved_subgraph.edges())

    # now start pruning the graph
    remove_edges = []
    for edge in ktree.edges:
        keep_edge = bool(np.random.binomial(1, probability))
        if (not keep_edge) and (edge not in preserved_edges):
            remove_edges.append(edge)
        else:
            continue
    ktree.remove_edges_from(remove_edges)

    return ktree


def generate_pruned_k_tree(treewidth, n_nodes, probability, n_cliques=1):
    """
    Generates a k-tree with a given treewidth and the
    number of nodes
    Parameters
    ----------
    treewidth: int
             desired treewidth
    n_nodes: int
            desired number of nodes. Must be at least treewidth+1
    probability: float
            probability that an edge exist in the graph
    n_cliques: int, default 1
            number of cliques of size treewidth+1 in the final graph
    Returns
    -------
    graph: nx.Graph
    """
    ktree = generate_k_tree(treewidth, n_nodes)
    graph = prune_k_tree(ktree, probability, n_cliques)
    return graph


def k_tree_test():
    from .peo_calculation import get_peo
    gr = generate_k_tree(4, 6)
    peo, tw = get_peo(gr)
    assert(tw == 4)

    gr2 = prune_k_tree(gr, 0, n_cliques=2)
    assert(gr2.number_of_edges() == gr.number_of_edges())

    gr3 = prune_k_tree(gr, 0.5, n_cliques=1)
    peo3, tw3 = get_peo(gr3)
    assert(tw3 == 4)


if __name__ == '__main__':
    k_tree_test()
