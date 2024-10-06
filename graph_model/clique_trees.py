"""
This module implements algorithms on clique trees
Clique trees are the result of tree decomposition
(they can be built out of elimination orderings)
The clique tree graph should contain sets of the nodes of the
original
"""
import networkx as nx
import copy

from .peo_calculation import (get_treewidth_from_peo)


def prune_tree(tree, start_edge):
    """
    Prunes a tree to ensure that none of its nodes are subsets
    of each other. This is a part of the tree decomposition algorithm

    Parameters
    ----------
    tree: networkx.Graph
          a tree which contains frozensets in its nodes.
    start_edge: tuple
          edge we start the search from
    """

    edges_to_check = [start_edge]

    while len(edges_to_check) > 0:
        edge = edges_to_check.pop()
        try:
            tree.edges[edge]['visited'] = True
        except KeyError:
            continue

        left, right = edge
        neighbors_left = [neighbor for neighbor
                          in tree.neighbors(left)
                          if neighbor != right]
        if left.issubset(right):
            tree.remove_node(left)
            for neighbor in neighbors_left:
                tree.add_edge(right, neighbor)
                edges_to_check.append(
                    (right, neighbor))
        else:
            for neighbor in neighbors_left:
                if not tree.edges[(left, neighbor)].get(
                        'visited', False):
                    edges_to_check.append(
                        (left, neighbor))
        neighbors_right = [neighbor for neighbor
                           in tree.neighbors(right)
                           if neighbor != left]
        if right.issubset(left):
            tree.remove_node(right)
            for neighbor in neighbors_right:
                tree.add_edge(left, neighbor)
                edges_to_check.append(
                    (left, neighbor))
        else:
            for neighbor in neighbors_right:
                if not tree.edges[(right, neighbor)].get(
                        'visited', False):
                    edges_to_check.append(
                        (right, neighbor))

    for edge in tree.edges():
        del tree.edges[edge]['visited']


def get_fillin_graph(old_graph, peo):
    """
    Provided a graph and an order of its indices, returns a
    triangulation of that graph corresponding to the order.

    The algorithm is copied from
    "Simple Linear Time Algorithm To Test Chordality of Graph"
    by R. E. Tarjan and M. Yannakakis

    Parameters
    ----------
    old_graph : nx.Graph or nx.MultiGraph
                graph to triangulate
    peo : elimination order to use for triangulation

    Returns
    -------
    nx.Graph or nx.MultiGraph
                triangulated graph
    """
    # Ensure PEO is a list of ints
    peo = list(map(int, peo))
    peo_to_conseq = dict(zip(peo, range(len(peo))))

    number_of_nodes = len(peo)
    graph = copy.deepcopy(old_graph)

    # Safeguard check. May be removed for partial triangulation
    assert number_of_nodes == graph.number_of_nodes()

    index = [0 for ii in range(number_of_nodes)]
    f = [0 for ii in range(number_of_nodes)]

    for ii in range(number_of_nodes):
        w = peo[ii]
        idx_w = peo_to_conseq[w]
        f[idx_w] = w
        index[idx_w] = ii
        neighbors = list(graph[w])
        lower_neighbors = [v for v in neighbors
                           if peo.index(v) < ii]
        for v in lower_neighbors:
            x = v
            idx_x = peo_to_conseq[x]
            while index[idx_x] < ii:
                index[idx_x] = ii
                # Check that edge does not exist
                # Tensors added here may not correspond to cliques!
                # Their names are made incompatible with Tensorflow
                # to highlight it
                if (x, w) not in graph.edges(w):
                    tensor = {'name': 'C{}'.format(w),
                              'indices': (w, ) + tuple(neighbors),
                              'data_key': None}
                    graph.add_edge(
                        x, w,
                        tensor=tensor)
                x = f[idx_x]
                idx_x = peo_to_conseq[x]
            if f[idx_x] == x:
                f[idx_x] = w
    return graph


def get_tree_from_peo(graph_old, peo):
    """
    Returns a tree decomposition of the graph,
    which corresponds to the given peo.
    The nodes in the resulting tree are frozensets

    Parameters
    ----------
    graph_old : networkx.Graph
           graph to estimate
    peo : list
           elimination order

    Returns
    -------
    tree : nx.Graph
           Tree decomposition of the graph. Nodes are root nodes of
           the maximal cliques, and cliques of the original graph
           are stored in the frozensets

    """
    graph_chordal = get_fillin_graph(graph_old, peo)

    tree = nx.Graph()

    # Go over the graph in order of peo and collect all cliques
    cliquelist = []
    for node in peo:
        neighbors = list(graph_chordal.neighbors(node))
        # Add a clique to list
        clique = set()
        clique.add(node)
        for neighbor in neighbors:
            clique.add(neighbor)
        cliquelist.append(frozenset(clique))
        graph_chordal.remove_node(node)

    # Now build a disconnected graph with resulting cliques
    for clique in cliquelist:
        tree.add_node(clique)

    # Now connect cliques into a tree in order given by peo
    for clique_idx, clique in enumerate(
            cliquelist):
        for neighbor_candidate, neighbor_root in zip(
                cliquelist[clique_idx+1:], peo[clique_idx+1:]):
            intersection = clique.intersection(neighbor_candidate)
            if neighbor_root in intersection:
                tree.add_edge(clique, neighbor_candidate)
                break

    # Now prune the tree to leave only maximal cliques
    # Execute pruning
    start_edges = []
    for component in nx.connected_component_subgraphs(tree):
        start_edges.append(next(iter(component.edges())))
    for start_edge in start_edges:
        prune_tree(tree, start_edge)

    assert len(list(tree.selfloop_edges())) == 0
    return tree


def get_subtree_by_length_width(tree, nodelist):
    """
    For each node in nodelist this function finds its corresponding
    subtree in the tree decomposition provided in tree. (As a reminder,
    the tree in the tree decomposition is an intersection of subtrees
    of each node of the original graph).
    Returns a dictionary {node: subtree_path}, where subtree_path is
    a list containing sizes of the tree vertices the node's subtree
    passes through.

    Parameters
    ----------
    tree : networkx.Graph
           graph holding the tree decomposition
    nodelist : list
           list of nodes to order

    Returns
    -------
    node_by_subtree_path : dict
           Dictionary containing nodes as keys and their subtree_path
           (widths of all nodes their subtree contributes to in the
           full tree) as values. The ssubtree_path is stored as a list.

    """
    nodes_by_pathlength = {node: [] for node in nodelist}

    def counter(node, clique, parent):
        if node in clique:
            nodes_by_pathlength[node].append(
                len(clique))
        children = [neighbor for neighbor
                    in tree.neighbors(clique)
                    if neighbor != parent]
        if len(children) == 0:
            return
        for child in children:
            counter(node, child, clique)

    # For each connected tree run DFS
    component_roots = []
    for component in nx.connected_component_subgraphs(tree):
        component_roots.append(next(iter(component)))

    for component_root in component_roots:
        for node in nodelist:
            counter(node, component_root, None)

    return nodes_by_pathlength


def rm_element_in_tree(tree, node):
    """
    Removes a specified node of the graph from all subsets in its
    tree decomposition

    Parameters
    ----------
    tree : networkx.Graph
           graph holding the tree decomposition
    node : int
           node to remove

    Returns
    -------
    new_tree : networkx.Graph()
       new tree decomposition without
       a subtree corresponding to the node

    """
    # import pdb
    # pdb.set_trace()
    new_tree = copy.deepcopy(tree)
    for clique in tree.nodes():
        if node in clique:
            neighbors = list(new_tree.neighbors(clique))
            new_tree.remove_node(clique)
            new_clique = clique - {node}
            if new_clique in neighbors:
                # degenerate case. The clique has been reduced
                # to one of its neighbors. Enlarge the list of neighbors
                neighbors = [neighbor for neighbor
                             in neighbors if neighbor != new_clique]

            new_tree.add_edges_from((new_clique, neighbor) for
                                    neighbor in neighbors)
    assert len(list(new_tree.selfloop_edges())) == 0

    # prune tree because non-maximal cliques may have emerged
    start_edges = []
    for component in nx.connected_component_subgraphs(new_tree):
        start_edges.append(next(iter(component.edges())))
    for start_edge in start_edges:
        prune_tree(new_tree, start_edge)

    return new_tree


def find_max_clique(tree):
    """
    Returns maximum clique in the tree decomposition tree
    """
    max_width = 0
    max_clique = None
    for clique in tree.nodes():
        width = len(clique)
        if width > max_width:
            max_width = width
            max_clique = clique
    return max_clique


def get_reduced_tree(tree, reduce_by):
    """
    Given a tree decomposition in tree and a required
    size of reduction, produces a new tree decomposition
    with treewidth reduced by the requested size and a list
    of eliminated nodes.
    We use a greedy algorithm to find nodes to eliminate.
    This algorithm deletes variable subtrees from the maximal
    node. The variables corresponding to larger subtrees are
    deleted first. If the length of subtrees are equal then
    subtrees passing through more nodes are removed first

    Parameters
    ----------
    tree : networkx.Graph
           tree decomposition we need to reduce
    reduce_by : int
           reduce treewidth by this amount

    Returns
    -------
    new_tree : networkx.Graph()
               reduced tree decomposition
    eliminated_nodes : list
               list of eliminated nodes
    """

    max_clique = find_max_clique(tree)
    treewidth = len(max_clique) - 1
    current_treewidth = treewidth

    if reduce_by < 0 or reduce_by > treewidth - 1:
        raise ValueError(
            'Requested reduce_by: {}, allowed range: [0, {}]'.format(
                reduce_by, treewidth-1))

    eliminated_nodes = []
    new_tree = tree
    while current_treewidth > treewidth - reduce_by:
        nodes_by_subwidth = get_subtree_by_length_width(
            tree, list(max_clique))

        # get (node, path length, total node's subtree width)
        nodes_in_rmorder = [(node, len(nodes_by_subwidth[node]),
                             sum(nodes_by_subwidth[node]))
                            for node in nodes_by_subwidth]
        # sort by path length, then by total width of subtree
        nodes_in_rmorder = sorted(
            nodes_in_rmorder,
            key=lambda x: (x[1], x[2]))

        rmnode = nodes_in_rmorder[-1][0]
        new_tree = rm_element_in_tree(new_tree, rmnode)
        eliminated_nodes.append(rmnode)
        max_clique = find_max_clique(new_tree)
        current_treewidth = len(max_clique) - 1

    assert len(list(new_tree.selfloop_edges())) == 0
    return new_tree, eliminated_nodes


def find_path_in_tree(tree, root, target):
    """
    Find path from nodeA to nodeB in tree. We use
    BFS here, which is nothing fancy and will run in O(n)
    for each query

    Parameters
    ----------
    tree : networkx.Graph
           tree we search
    root : hashable
           Starting point of the node type
    target : hashable
           End point of the node type

    Returns
    -------
    path : list
           path from root to target including endpoints
    """
    def tree_traversal(path, node, parent):
        path.append(node)
        children = [neighbor for neighbor in tree.neighbors(node)
                    if neighbor != parent]
        if len(children) == 0:
            return
        if node == target:
            return path
        for child in children:
            path = tree_traversal(path, child, node)
            if path is not None:
                return path
        return

    path = tree_traversal([], root, None)
    return path


def get_peo_from_tree(old_tree, clique_vertices=[]):
    """
    Given a tree and clique vertices, this function returns
    a peo with clique vertices at the end of the list

    Parameters
    ----------
    old_tree : networkx.Graph
           tree decomposition to build peo
    clique_vertices : list, default []
           vertices of a clique we may want to place at the end of peo

    Returns
    -------
    peo : list
          perfect elimination order with possible restriction on the
          final vertices
    """
    tree = copy.deepcopy(old_tree)

    # First determine if clique_vertices are contained in any
    # maximal clique in the tree (clique vertices may not be
    # a maximal clique).
    test_clique = frozenset(clique_vertices)
    root_clique = None
    for root_clique in tree.nodes():
        if test_clique.issubset(root_clique):
            break
    if root_clique is None:
        raise ValueError('Clique {} not found in tree'.format(
            clique_vertices))

    # For each component find a root and add all leaves to
    # peo with breadth-first search. The requested test_clique
    # is taken as a root at the last iteration

    peo = []

    # Find the first leaf or isolate
    for node in tree.nodes():
        if (len(list(tree.neighbors(node))) <= 1
           and node != root_clique):
            break

    while node:
        # process a single leaf
        try:
            parent = next(tree.neighbors(node))
        except StopIteration:
            # tree contained a single node. We are done here
            break

        intersection = node.intersection(parent)
        for index in node - intersection:
            peo.append(index)

        tree.remove_node(node)

        # take next leaf or isolate if it exists
        for node in tree.nodes():
            if (len(list(tree.neighbors(node))) <= 1
               and node != root_clique):
                break

        if node == root_clique:  # reached root. We are done here
            break
    # We should have a root clique at this point. Otherwise something
    # is wrong
    assert node == root_clique

    # Now process the root clique
    difference = root_clique - test_clique
    for index in difference:
        peo.append(index)
    # Finally, append last indices in the given order
    for index in clique_vertices:
        peo.append(index)
    return peo


def make_test_graph():
    """
    Creates a graph from the Stack Overflow post:
    https://stackoverflow.com/questions/23737690/
    algorithm-for-generating-a-tree-decomposition/23739715#23739715
    """
    g = nx.Graph()
    g.add_edges_from([
        [0, 1], [0, 2], [0, 5], [0, 6],
        [1, 2], [1, 6], [1, 7],
        [2, 6], [2, 7], [2, 3], [2, 5],
        [4, 5], [4, 6],
        [5, 6]
    ])
    peo = [4, 3, 5, 7, 6, 2, 0, 1]
    return g, peo


def make_test_tree():
    """
    Creates a tree graph from the Stack Overflow post:
    https://stackoverflow.com/questions/23737690/
    algorithm-for-generating-a-tree-decomposition/23739715#23739715
    """
    d = {frozenset({4, 5, 6}): [frozenset({0, 2, 5, 6})],
         frozenset({2, 3}): [frozenset({0, 1, 2})],
         frozenset({0, 2, 5, 6}):
         [frozenset({4, 5, 6}), frozenset({0, 1, 2, 6})],
         frozenset({1, 2, 7}): [frozenset({0, 1, 2})],
         frozenset({0, 1, 2, 6}):
         [frozenset({0, 2, 5, 6}), frozenset({0, 1, 2})],
         frozenset({0, 1, 2}): [frozenset({2, 3}),
                                frozenset({1, 2, 7}),
                                frozenset({0, 1, 2, 6}),
                                frozenset({0, 1})],
         frozenset({0, 1}): [frozenset({0, 1, 2}), frozenset({1})],
         frozenset({1}): [frozenset({0, 1})]}
    return nx.from_dict_of_lists(d)


def is_node_subtree_connected(tree, element):
    """
    Checks if the element induces a connected subtree in the tree.
    The nodes of the tree are sets of elements. For test purposes mainly

    Parameters
    ----------
    tree: networkx.Graph
           tree to check
    element: element-type
          element which may induce a
    """
    nodes_containing_element = []
    for node in tree.nodes():
        if element in node:
            nodes_containing_element.append(node)

    subtree = nx.subgraph(tree, nodes_containing_element)
    if subtree.number_of_nodes() > 0:
        connected = nx.connected.is_connected(subtree)
    else:
        connected = True  # Take empty graph as connected
    return connected


def are_subtrees_connected(tree, elements):
    """
    Checks if every element in set induces a connected subtree in
    the tree
    """
    for element in elements:
        connected = is_node_subtree_connected(tree, element)
        if not connected:
            raise ValueError(f"Subtree of {element} not connected!")


def test_tree_reduction():
    """
    Tests the deletion of variables from tree
    """
    g, peo = make_test_graph()
    f = get_tree_from_peo(g, peo)

    ff = rm_element_in_tree(f, 2)
    peo.remove(2)
    g.remove_node(2)

    tw1 = get_treewidth_from_peo(g, peo)
    tw2 = len(find_max_clique(ff)) - 1
    assert tw1 == tw2


def test_tree_to_peo():
    """
    Test tree reduction algorithm and also
    the conversion of tree to peo
    """
    g, peo = make_test_graph()
    f = get_tree_from_peo(g, peo)

    fd, el = get_reduced_tree(f, 1)
    peo = get_peo_from_tree(fd, [5, 6])
    g.remove_nodes_from(el)
    tw1 = get_treewidth_from_peo(g, peo)
    tw2 = len(find_max_clique(fd)) - 1
    assert tw1 == tw2


if __name__ == '__main__':
    test_tree_reduction()
    test_tree_to_peo()
