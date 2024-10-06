import pickle
import networkx as nx
import os
import graph_model as gm

if __name__ == '__main__':
    sizes = range(20, 500, 10)
    erdos_dataset = {}
    dir_path = './graph_test/erdos/'
    os.makedirs(dir_path, exist_ok=True)

    for num_nodes in sizes:
        g = nx.generators.erdos_renyi_graph(num_nodes - 1, 5.0/num_nodes)
        name = os.path.join(dir_path,'inst_' + str(num_nodes) + '.gr.xz')
        gm.generate_gr_file(g, filename=name, compressed=True)
    print('done')