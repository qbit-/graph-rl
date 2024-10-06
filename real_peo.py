import sys


from rl_environment import Environment, degree_cost
import graph_model as gm


def get_real_peo(name, minfill=False):
    env = Environment(
        name,
        square_index=True,
        cost_function=degree_cost,
        simple_graph=True)
    if minfill:
        return gm.get_upper_bound_peo(env.graph)
    return gm.get_peo(env.graph)


def get_real_peo_from_path(name):
    env = Environment(
        name,
        square_index=True,
        cost_function=degree_cost,
        simple_graph=True)
    # path = [66, 49, 63, 70, 40, 18, 53, 61, 29, 67, 9, 13, 46, 20, 33, 25, 19, 39, 56, 12, 81, 10, 21, 54, 76]
    nodes = [env.row_to_node[idx] for idx in path]
    return path, gm.get_treewidth_from_peo(env.graph, nodes)


if __name__ == '__main__':
    name = sys.argv[1]
    path, treewidth = get_real_peo(name)
    # path, treewidth = get_real_peo_from_path(name)

    print("Treewidth of the {}: {}".format(name.split('/')[-1], treewidth))
    print(path[:20])
    path, treewidth = get_real_peo(name, minfill=True)
    print("Minfill treewidth of the {}: {}".format(name.split('/')[-1], treewidth))
    print(path[:20])

