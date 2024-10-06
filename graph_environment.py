"""
This module is for working with reinforcement learning agents
for computing the tree decomposition of the expression graphs
"""
import numpy as np
import networkx as nx
import graph_model as gm
import copy


def sparse_graph_adjacency(G, max_size, node_to_row, weight='weight'):
    """Return the graph adjacency matrix as a SciPy sparse matrix.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the adjacency matrix.

    max_size : int
        Matrix size. May be larger than the number of nodes. Has to
        be compatible with the node_to_idx mapping.

    node_to_row : dict
        The mapping between graph nodes and rows/columns in the
        the adjacency matrix

    Returns
    -------
    M : scipy.sparse
        Zero padded adjacency matrix
    """
    from scipy import sparse

    nodelist = list(G)
    if not set(nodelist).issubset(node_to_row):
        msg = "`nodelist` is not a subset of the `node_to_row` dictionary."
        raise nx.NetworkXError(msg)

    index = {node: node_to_row[node] for node in nodelist}
    coefficients = zip(*((index[u], index[v], d.get(weight, 1))
                         for u, v, d in G.edges(nodelist, data=True)
                         if u in index and v in index))
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    # symmetrize matrix
    d = data + data
    r = row + col
    c = col + row
    # selfloop entries get double counted when symmetrizing
    # so we subtract the data on the diagonal
    selfloops = list(nx.selfloop_edges(G, data=True))
    if selfloops:
        diag_index, diag_data = zip(*((index[u], -d.get(weight, 1))
                                      for u, v, d in selfloops
                                      if u in index and v in index))
        d += diag_data
        r += diag_index
        c += diag_index
    M = sparse.coo_matrix((d, (r, c)), shape=(max_size, max_size))
    return M


def print_int_matrix(matrix):
    """
    Prints integer matrix in a readable form
    """
    for row in matrix:
        line = ' '.join(f'{e:d}' if e != 0 else '-' for e in row)
        print(line)


def print_int_tril_matrix(matrix):
    """
    Prints a lower triangular integer matrix in a
    readable form
    """
    from math import sqrt
    size = int(-0.5 + sqrt(0.25+2*len(matrix)))

    idx = 0
    for ii in range(size):
        n_elem = ii + 1
        next_idx = idx + n_elem
        line = ' '.join(f'{e:d}' if e != 0 else '-' for e in
                        matrix[idx:next_idx])
        print(line)
        idx = next_idx


def degree_cost(graph, node):
    """
    Cost function that calculates degree
    """
    return graph.degree(node) - 1


def contraction_cost_flops(graph, node):
    """
    Cost function that uses flops contraction cost
    """
    memory, flops = gm.get_cost_by_node(graph, node)
    return flops


class Environment:
    """
    Creates an environment to train the agents
    """
    def __init__(self, filename,
                 cost_function=degree_cost,
                 simple_graph=False,
                 square_index=False, use_random=False,
                 random_graph_type='erdos',
                 use_pace=False,
                 state_size=10,
                 verbose=True,
                 p=0.1,
                 tw=8):
        """
        Creates an environment for the model from file

        Parameters
        ----------
        filename : str
               file to load
        cost_function : function, optional
               function (networkx.Graph, int)->int which
               evaluates the cost of selecting a node.
               Default `contraction_cost_flops`
        random_graph_type : type of the random graph {Erdos-Reniy, K-tree with fixed treewidth}
        """
        self.state_size = state_size
        self.use_random = use_random
        self.prob = p
        self.tw = tw
        self.random_graph_type = random_graph_type

        if self.use_random and not use_pace:
            if random_graph_type == 'erdos':
                self.prob = 5.0/state_size
                initial_graph = nx.erdos_renyi_graph(self.state_size, p=self.prob)
            else:
                self.tw = 0.1*self.state_size
                initial_graph = gm.generate_pruned_k_tree(treewidth=self.tw,
                                                          n_nodes=self.state_size,
                                                          probability=self.prob)
            self.node_map = np.arange(self.state_size)
        elif use_pace:
            if self.use_random:
                initial_graph = gm.read_gr_file(
                    filename, compressed=True, from_zero=True)
                self.use_random = False
                self.state_size = initial_graph.number_of_nodes()
                self.node_map = np.arange(self.state_size)
            else:
                initial_graph = gm.read_gr_file(
                    filename, compressed=True)
                self.state_size = initial_graph.number_of_nodes()

                # self.node_map = np.arange(self.state_size)#
                self.node_map = np.arange(1, self.state_size+1)
        else:
            initial_graph = gm.read_circuit_file(filename)
            self.state_size = initial_graph.number_of_nodes()
            self.node_map = np.arange(1, self.state_size + 1)

        if initial_graph.number_of_nodes() > self.state_size:
            raise ValueError(
                f'Graph is larger than the maximal state size:' +
                f' {self.state_size}')

        if simple_graph:
            initial_graph = nx.Graph(initial_graph)
            initial_graph.remove_edges_from(
                initial_graph.selfloop_edges(data=False))

        if verbose:
            print("Initital number of nodes:", initial_graph.number_of_nodes())
            print("Initital number of edges:", initial_graph.number_of_edges())
        self.initial_graph = initial_graph
        self.cost_function = cost_function
        self.square_index = square_index
        n_nodes = self.initial_graph.number_of_nodes()
        self.entry_position = np.arange(0, self.state_size)
        self.graph_rand_indices = np.random.permutation(range(1, n_nodes+1))
        # self.minfill_indices, _ = gm.get_upper_bound_peo(self.initial_graph)
        self.reset()

    def reset_state(self, reset_nodes=False):
        return nx.to_numpy_matrix(self.graph)

    def reset(self, reset_nodes=False, new_size=0,  graph=None):
        """
        Resets the state of the environment. The graph is
        randomly permutted and a new adjacency matrix is generated
        """
        n_nodes = self.initial_graph.number_of_nodes()
        if new_size > 0:
            n_nodes = new_size
            self.state_size = n_nodes
            self.node_map = np.arange(self.state_size)
            self.entry_position = self.node_map

        if reset_nodes:
            if self.use_random:
                if self.random_graph_type == 'erdos':
                    self.prob = 4.0 / self.state_size
                    self.initial_graph = nx.erdos_renyi_graph(n_nodes, p=self.prob)
                    # _, tw = gm.get_upper_bound_peo(self.initial_graph)
                    # print(tw)
                else:
                    self.initial_graph = gm.generate_pruned_k_tree(treewidth=self.tw,
                                                              n_nodes=n_nodes,
                                                              probability=self.prob)
            if graph is not None:
                self.initial_graph = graph
                self.node_map = np.arange(self.state_size)

                # self.initial_graph = nx.erdos_renyi_graph(n_nodes, p=self.prob)
            #TODO create a new one approach to work with not random graph
            # else:
            #     self.initial_graph = gm.generate_random_graph(int(MAX_STATE_SIZE), int(3.5*MAX_STATE_SIZE))
            #     self.node_map = np.arange(1,MAX_STATE_SIZE+1)
        entry_indices = self.entry_position
        row, col = np.tril_indices(self.state_size)
        # n*(n+1)//2 number of edges that selfloops are allowed.
        node_to_idx= {key: value for (key, value) in zip(self.node_map, entry_indices)}
        idx_to_node = {key: value for (key, value) in zip(entry_indices, self.node_map)}

        self.node_to_row = node_to_idx
        self.row_to_node = idx_to_node
        self.idx_to_node = idx_to_node
        graph = copy.deepcopy(self.initial_graph)
        self.graph = graph

        # Build adjacency matrix and pack it to lower triangular
        state = sparse_graph_adjacency(self.initial_graph, self.state_size,
                                       self.node_to_row)

        self.state = state
        self.available_actions = np.zeros(self.state_size, dtype=np.int32)
        self.available_actions[list(idx_to_node.keys())] = 1
        state = self.state
        return state

    def step(self, index):
        """
        Takes 1 step in the graph elimination environment

        Parameters
        ----------
        index : int
              index in the state matrix to eliminate.
              By default a lower triangular index is expected here
        square_index : bool
              if True the index is taken as a row/column index in
              the square adjacency matrix
        """
        if not self.square_index:
            node = self.idx_to_node[index]
        else:
            node = self.row_to_node[index]

        # Calculate cost function
        cost = self.cost_function(self.graph, node)

        # Update state
        gm.eliminate_node(self.graph, node, self_loops=True)
        complete = self.graph.number_of_nodes() == 0
        # # if there is no any edges after elimination
        # # we add one self-loop for final node
        # if len(list(self.graph.edges())) == 0 and not complete:
        #     last_node = list(self.graph.nodes())[0]
        #     self.graph.add_edge(last_node, last_node)

        graph = sparse_graph_adjacency(self.graph, self.state_size,
                                   self.node_to_row)
        # if not complete:
        #     graph = nx.to_scipy_sparse_matrix(self.graph, format='coo')

        self.state = graph
        # print(self.tril_indices)
        # self.state = adj_matrix[self.tril_indices]
        self.available_actions[index] = 0
        state = self.state
        return state, cost, complete


class MaximumVertexCoverEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_nodes = 1 - self.available_actions
        self.cover_graph = nx.Graph()
        self.cover_graph.add_nodes_from(self.initial_graph.nodes)
        self.s = []

    def reset(self, reset_nodes=False, new_size=0):
        super().reset(reset_nodes=False, new_size=0)
        self.cover_graph = nx.Graph()
        self.cover_graph.add_nodes_from(self.initial_graph.nodes)
        self.s = []
        return self.state

    def step(self, index):
        """
        :param index:
        :return: state, reward, done
        """
        if not self.square_index:
            node = self.idx_to_node[index]
        else:
            node = self.row_to_node[index]

        self.s.append(node)
        complete = False
        cost = 1
        self.available_actions[index] = 0
        state = self.state

        # num_visited_vertex = np.concatenate([visited_nodes[list(edge_index_list[0])][:,None],
        #                       visited_nodes[list(edge_index_list[1])][:,None]], axis=-1).max(-1)

        self.cover_graph.add_node(node)
        self.cover_graph.add_edges_from([(node, v) for v in self.graph.neighbors(node)])

        if set(self.cover_graph.edges) == set(self.graph.edges):
            complete = True

        return state, cost, complete



if __name__ == '__main__':
    env2_name = 'inst_5x5_10_0.txt'
    env1_name = 'inst_2x2_7_0.txt'
    env3_name = './graph_test/inst_7x7_10_0.txt'
    env_name = './graph_test/pace2017_instances/gr/heuristic/he002.gr.xz'
    environment = Environment(env_name, square_index=True,
                              cost_function=degree_cost,
                              use_random=False)
    environment.reset()

    # Print triangular adjacency matrices
    costs = []
    steps = []
    complete = False

    environment.reset()

    costs = []
    steps = []
    complete = False
    worst_cost = 0
    i = 0
    while not complete:
        # print_int_matrix(environment.state_square)
        row, col = np.nonzero(environment.state_square)
        a = row[0]
        # a = environment.minfill_indices[i]
        print(environment.node_to_row)
        i+=1
        # cost, complete = environment.step(row[0], square_index=True)
        state, cost, complete = environment.step(a)
        worst_cost = np.maximum(worst_cost, cost)
        steps.append(environment.row_to_node[row[0]])
        costs.append(cost)

    # print(' Strategy\n Node | Cost:')
    # print('-'*24)
    # print('\n'.join('{:5} | {:5}'.format(step, cost)
    #                 for step, cost in zip(steps, costs)))
    print('-'*24)
    print(worst_cost)
    print(costs)
    print(environment.state_size)
    print('Heuristic cost', gm.get_upper_bound_peo(environment.initial_graph, method='min_fill')[1])
    print('Total cost: {}'.format(sum(costs)))
