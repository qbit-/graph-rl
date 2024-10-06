"""
Conversion from other data structures to the graphs supported by this library
"""
import networkx as nx
import re
import lzma
from io import StringIO


def read_gr_file(file_or_data, as_data=False, compressed=False, from_zero=False):
    """
    Reads graph from a DGF/GR file
    The file can be specified through the filename or
    its data can be given to this function directly
    Parameters
    ----------
    file_or_data: str
             Name of the file with graph data or its contensts
    as_data: bool, default False
             If filedata should be interpreted as contents of the
             data file
    compressed : bool
           if input file or data is compressed
    """
    import sys

    ENCODING = sys.getdefaultencoding()

    graph = nx.Graph()
    if as_data is False:
        if compressed:
            datafile = lzma.open(file_or_data, 'r')
        else:
            datafile = open(file_or_data, 'r+')
    else:
        if compressed:
            datafile = StringIO(lzma.decompress(file_or_data))
        else:
            datafile = StringIO(file_or_data)

    # search for the header line
    comment_patt = '^(\s*c\s+)(?P<comment>.*)'
    header_patt = (
        '^(\s*p\s+)((?P<file_type>cnf|tw)\s+)?(?P<n_nodes>\d+)\s+(?P<n_edges>\d+)')
    if compressed:
        # if file is compressed then bytes are read. Hence we need to
        # transform patterns to byte patterns
        comment_patt = comment_patt.encode(ENCODING)
        header_patt = header_patt.encode(ENCODING)

    for n, line in enumerate(datafile):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(header_patt, line)
        if m is not None:
            n_nodes = int(m.group('n_nodes'))
            n_edges = int(m.group('n_edges'))
            break
        else:
            raise ValueError(f'File format error at line {n}:\n'
                             f' expected pattern: {header_patt}')

    # add nodes as not all of them may be connected
    if from_zero:
        graph.add_nodes_from(range(n_nodes))
    else:
        graph.add_nodes_from(range(1, n_nodes+1))

    # search for the edges
    edge_patt = '^(\s*e\s+)?(?P<u>\d+)\s+(?P<v>\d+)'
    if compressed:
        # if file is compressed then bytes are read. Hence we need to
        # transform patterns to byte patterns
        edge_patt = edge_patt.encode(ENCODING)

    for nn, line in enumerate(datafile, n):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(edge_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nn}:\n'
                             f' expected pattern: {edge_patt}')
        graph.add_edge(int(m.group('u')), int(m.group('v')))

    if (graph.number_of_edges() != n_edges):
        raise ValueError('Header states:\n'
                         f' n_nodes = {n_nodes}, n_edges = {n_edges}\n'
                         'Got graph:\n'
                         f' n_nodes = {graph.number_of_nodes()},'
                         f' n_edges = {graph.number_of_edges()}\n')
    datafile.close()
    return graph


def read_td_file(file_or_data, as_data=False, compressed=False, with_time=False):
    """
    Reads file/data in the td format of the PACE 2017 competition
    Returns a tree decomposition: a nx.Graph with frozensets as nodes

    Parameters
    ----------
    filedata: str
             Name of the file with graph data or its contensts
    as_data: bool, default False
             If filedata should be interpreted as contents of the
             data file
    compressed: bool
             Input file or data is compressed
    """
    graph = nx.Graph()
    if as_data is False:
        if compressed:
            datafile = lzma.open(file_or_data, 'r+')
        else:
            datafile = open(file_or_data, 'r+')
    else:
        if compressed:
            datafile = StringIO(lzma.decompress(file_or_data))
        else:
            datafile = StringIO(file_or_data)

    # search for the header line
    comment_patt = re.compile('^(\s*c\s+)(?P<comment>.*)')
    header_patt = re.compile(
        '^(\s*s\s+)(?P<file_type>td)\s+(?P<n_cliques>\d+)\s+(?P<max_clique>\d+)\s+(?P<n_nodes>\d+)')
    for n, line in enumerate(datafile):
        m = re.search(comment_patt, line)
        if with_time:
            if n == 2:
                m = re.search('\d+', line)

                result_time = float(line[m.start(): m.end()])/100
        if m is not None:
            continue
        m = re.search(header_patt, line)
        if m is not None:
            n_nodes = int(m.group('n_nodes'))
            n_cliques = int(m.group('n_cliques'))
            treewidth = int(m.group('max_clique')) - 1
            break
        else:
            raise ValueError(f'File format error at line {n}:\n'
                             f' expected pattern: {header_patt}')

    # add nodes as not all of them may be connected
    graph.add_nodes_from(range(1, n_cliques+1))

    # search for the cliques and collect them into a dictionary
    all_nodes = set()
    clique_dict = {}
    clique_patt = re.compile('^(\s*b\s+)(?P<clique_idx>\d+)(?P<clique>( \d+)*)')
    for nn, line in zip(range(n, n+n_cliques), datafile):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(clique_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nn}:\n'
                             f' expected pattern: {clique_patt}')
        clique = frozenset(map(int, m.group('clique').split()))
        clique_dict[int(m.group('clique_idx'))] = clique
        all_nodes = all_nodes.union(clique)

    # search for the edges between cliques
    edge_patt = re.compile('^(\s*e\s+)?(?P<u>\d+)\s+(?P<v>\d+)')
    for nnn, line in enumerate(datafile, nn):
        m = re.search(comment_patt, line)
        if m is not None:
            continue
        m = re.search(edge_patt, line)
        if m is None:
            raise ValueError(f'File format error at line {nnn}:\n'
                             f' expected pattern: {edge_patt}')
        graph.add_edge(int(m.group('u')), int(m.group('v')))

    assert(len(all_nodes) == n_nodes)

    # finally, replace clique indices by their contents
    graph = nx.relabel_nodes(graph, clique_dict)

    datafile.close()
    if with_time:
        return graph, treewidth, result_time

    return graph, treewidth


def read_circuit_file(filename, max_depth=None):
    """
    Reads circuit from filename and builds its contraction graph
    This function do returns a Graph without selfloops.
    This is not a faithful representation of a circuit!!!

    Parameters
    ----------
    filename : str
             circuit file in the format of Sergio Boixo
    max_depth : int
             maximal depth of gates to read

    Returns
    -------
    graph : networkx.MultiGraph
    """
    graph = nx.Graph()

    with open(filename, 'r') as fp:
        # read the number of qubits
        qubit_count = int(fp.readline())

        n_ignored_layers = 0
        current_layer = 0

        # initialize the variables and add nodes to graph
        for i in range(1, qubit_count+1):
            graph.add_node(i)
        layer_variables = list(range(1, qubit_count+1))
        current_var = qubit_count

        for idx, line in enumerate(fp):

            # Read circuit layer by layer. Decipher contents of the line
            m = re.search(r'(?P<layer>[0-9]+) (?P<operation>h|t|cz|x_1_2|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            layer_num = int(m.group('layer'))

            # Skip layers if max_depth is set
            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue
            if layer_num > current_layer:
                current_layer = layer_num

            op_identif = m.group('operation')
            if m.group('qubit2') is not None:
                q_idx = int(m.group('qubit1')), int(m.group('qubit2'))
            else:
                q_idx = (int(m.group('qubit1')),)

            # Now apply what we got to build the graph
            if op_identif == 'cz':
                # cZ connects two variables with an edge
                var1 = layer_variables[q_idx[0]]
                var2 = layer_variables[q_idx[1]]
                graph.add_edge(var1, var2)

            # Skip Hadamard tensors - for now
            elif op_identif == 'h':
                pass

            # Add selfloops for single variable gates
            elif op_identif == 't':
                var1 = layer_variables[q_idx[0]]
                graph.add_edge(var1, var1)

            # Process non-diagonal gates X and Y
            else:
                var1 = layer_variables[q_idx[0]]
                var2 = current_var+1
                graph.add_node(var2)
                graph.add_edge(var1, var2)

                current_var += 1
                layer_variables[q_idx[0]] = current_var

        # We are done, print stats
        if n_ignored_layers > 0:
            print("Ignored {} layers".format(n_ignored_layers))

        v = graph.number_of_nodes()
        e = graph.number_of_edges()

        print(f"Generated graph with {v} nodes and {e} edges")

    return graph


def test_read_gr_files():
    from .exporters import generate_gr_file

    data = 'p tw 9 6\n1 3\n2 5\n2 6\n4 7\n4 8\n4 9\n'
    g = read_gr_file(data, as_data=True)
    g2 = read_gr_file(generate_gr_file(g), as_data=True)
    assert(nx.isomorphism.is_isomorphic(g, g2))
