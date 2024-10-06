import collections
import pickle
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from actor_critic import ActorCritic
from arguments import init_config

from graph_environment import Environment, degree_cost, contraction_cost_flops
import graph_model as gm
import networkx as nx

import os
from datetime import datetime

from graph_policy import get_appropriate_graph_policy, GraphActor


class SolutionEntity:
    def __init__(self, graph, scores, peos, graph_name, times, num_nodes=None):
        self.graph = graph
        self.labels = ['agent', 'min_fill','min_degree', "cardinality", 'tamaki', 'quickbb']
        self.graph_name = graph_name
        self._translate_score(scores)
        self._translate_peo(peos)
        self._translate_time(times)
        self.num_nodes = num_nodes

    def _translate_score(self, scores):
        for l, s in zip(self.labels, scores):
            setattr(self, l+'_score', s)

    def _translate_peo(self, peos):
        for l, s in zip(self.labels, peos):
            setattr(self, l + '_peo', s)

    def _translate_time(self, times):
        for l, s in zip(self.labels, times):
            setattr(self, l + '_time', s)


def get_agent_peo(alg, n_restarts=5):
    """
    :param alg: algorithm with eval functional
    :param num_experiments: number of reset environment
    :param n_restarts: number restarts every experiment
    :param use_min: use minimum score of all trajectories
    :param verbose:
    :return: treewidth, peo
    """
    agent_trs = []
    path_trs = []

    agent_tr, path_tr = alg.eval(with_minfill=False, reset_nodes=False, verbose=False, with_peo=True)
    agent_trs.append(agent_tr)
    path_trs.append(path_tr)

    for _ in range(n_restarts):
        agent_tr, path_tr = alg.eval(reset_nodes=False, with_peo=True)
        agent_trs.append(agent_tr)
        path_trs.append(path_tr)

    i_agent_tr = int(np.argmin(agent_trs))

    return agent_trs[i_agent_tr], path_trs[i_agent_tr]


def evaluate_method_with_specific_type(algo, graph_type, num_nodes=30, n_restarts=5, run_heuristic=True):
    if graph_type in random_graphs:
        algo.env.reset(reset_nodes=True, new_size=num_nodes)

    graph = algo.env.initial_graph
    scores = []
    peos = []
    times_tr = []

    start = time.time()
    agent_tw, agent_peo = get_agent_peo(algo, n_restarts=n_restarts)
    time_alg = (time.time() - start) / n_restarts
    scores.append(agent_tw)
    peos.append(agent_peo)
    times_tr.append(time_alg)
    if run_heuristic:
        for ht in heuristics_type:
            start = time.time()
            peo, tw = gm.get_upper_bound_peo(graph, ht)
            time_alg = time.time() - start
            scores.append(tw)
            peos.append(peo)
            times_tr.append(time_alg)

    solution = SolutionEntity(graph, scores, peos, graph_type,
                              times_tr, num_nodes=algo.env.state_size)

    return solution


def solve_one_by_name(file_name, alg, n_restart=10, only_heur=False):
    env = Environment(
        file_name,
        square_index=True,
        cost_function=degree_cost,
        simple_graph=False,
        use_random=True,
        use_pace=True,
        verbose=False,
    )
    alg.env = env
    if not only_heur:
        start = time.time()
        agent_tw, agent_peo = get_agent_peo(alg, n_restarts=n_restart)
        time_alg = (time.time() - start) / n_restart


        return float(agent_tw)

    _, m_tw = gm.get_upper_bound_peo(alg.env.initial_graph, method='tamaki')

    return m_tw


def validate(alg):
    files_erdos = ['./graph_test/erdos/inst_{}.gr.xz'.format(i) for i in range(40, 250, 10)]
    total_solutions = []
    res_tamaki = []
    res_agents = []
    scores = []

    with open('./graph_test/res_tamaki.pkl', 'rb') as fout:
        res_tamaki = pickle.load(fout)
    i = 0
    for f_name in tqdm(files_erdos):
        agent_tw = solve_one_by_name(f_name, alg, only_heur=False)
        scores.append(float(agent_tw)/res_tamaki[i])
        i+=1
        # res_agents.append(agent_tw)
        # total_solutions.append(solved_graph)

    # with open('./graph_test/res_tamaki.pkl', 'wb') as fout:
    #     pickle.dump(res_tamaki, fout)

    # print(res_tamaki)
    return float(sum(scores))/len(scores), scores, total_solutions

if __name__ == '__main__':
    args = init_config()
    # fix seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # heuristic_type = 'min_fill'
    genearal_exp_path = os.path.join('logs', args.exp_name)
    current_exp_path = os.path.join(genearal_exp_path, time_stamp)
    os.makedirs(current_exp_path, exist_ok=True)
    os.makedirs('best_models', exist_ok=True)

    use_cuda = torch.cuda.is_available()
    global device, cpu_device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda:", use_cuda)

    cpu_device = torch.device('cpu')
    env_name = args.graph_path
    costs = {'treewidth': degree_cost, 'flops': contraction_cost_flops}
    MAX_STATE_SIZE = args.max_shape

    os.makedirs('solutions', exist_ok=True)
    EXPERIMENT_SOLUTION_PATH = os.path.join('solutions', args.exp_name)
    PATH_TO_RANDOM = EXPERIMENT_SOLUTION_PATH + '_random_graphs.pkl'
    PATH_TO_PACE = EXPERIMENT_SOLUTION_PATH + '_pace_graphs.pkl'
    PATH_TO_CIRCUITS = EXPERIMENT_SOLUTION_PATH + '_circuit_graphs.pkl'
    DUMP_PATH = EXPERIMENT_SOLUTION_PATH + 'external_dump.pkl'
    print(env_name)

    env = Environment(
        env_name,
        square_index=True,
        cost_function=costs[args.cost_function],
        simple_graph=True,
        use_random=args.use_erdos,
        state_size=MAX_STATE_SIZE,
        p=args.proba,
        random_graph_type=args.random_graph_type,
        tw=args.treewidth
    )
    if args.eval_mode:
        agent_total_score = 0
        min_fill_total_score = 0
        min_degree_total_score = 0
        quickbb_total_score = 0
        cardinality_score = 0
        # with open('test_cr_gr__circuit_graphs.pkl', 'rb') as fin:
        # with open('solutions/erdos_exps_random_graphs.pkl', 'rb') as fin:
        # with open('solutions/night_test_random_graphs.pkl', 'rb') as fin:
        # with open('solutions/night_test_circuit_graphs.pkl', 'rb') as fin:
        with open('solutions/night_test_pace_graphs.pkl', 'rb') as fin:
        # with open('solutions/test_keybexternal_dump.pkl', 'rb') as fin:
            results = pickle.load(fin)
        for r in results:
            agent_total_score += r.agent_score/r.quickbb_score
            min_fill_total_score += r.min_fill_score/r.quickbb_score
            min_degree_total_score += r.min_degree_score/r.quickbb_score
            quickbb_total_score += r.quickbb_score/r.tamaki_score
            cardinality_score += r.cardinality_score/r.tamaki_score
            print(r.quickbb_score, r.agent_score)
            # print(r.quickbb_time, r.agent_time)
            # print(str(r.graph.number_of_nodes()) + ' ' + r.graph_name)

        print(agent_total_score/len(results))
        print(min_fill_total_score/len(results))
        print(min_degree_total_score/len(results))
        print(quickbb_total_score/len(results))
        print(cardinality_score/len(results))

        print(len(results))
        exit()

    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    ndim = 2 if args.use_small_features else state_shape[0]

    graph_policy = get_appropriate_graph_policy(args, ndim, device)
    actor = GraphActor(policy=graph_policy).to(device)

    actor_total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print('Actor number of parameters: ', actor_total_params)
    checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
    actor.load_state_dict(checkpoint)
    print(args.resume_path)

    alg = ActorCritic(env,
                      actor,
                      ndim=ndim,
                      gamma=0.99,
                      num_episodes=args.batch_size,
                      episode_len=MAX_STATE_SIZE,
                      device=device,
                      use_gcn=args.gcn,
                      use_log=args.use_neighbour,
                      args=args
                      )

    global random_graphs, heuristics_type

    graph_types = ['erdos']  # TODO add two graph types
    random_graphs = ['erdos']
    heuristics_type = ['min_fill', 'min_degree', "cardinality", 'tamaki', 'quickbb']
    total_solutions = []
    try:
        if args.graph_type == 'random':
            print(PATH_TO_RANDOM)
            print('Experiments on the Random Graphs with nodes less than 500')
            # for gr_type in random_graphs:
            gr_type = 'erdos'
            print('Test on the {} graphs'.format(gr_type))
            total_num_of_nodes = range(500, 1000, 15)
            n_restart = 30
            for nodes in tqdm(total_num_of_nodes):
                solved_graph = evaluate_method_with_specific_type(alg, gr_type,
                                                                  num_nodes=nodes,
                                                                  n_restarts=n_restart,
                                                                  run_heuristic=False
                                                                  )
                total_solutions.append(solved_graph)

            print(len(total_solutions))
            with open(PATH_TO_RANDOM, 'wb') as fout:
                pickle.dump(total_solutions, fout)

        elif args.graph_type == 'pace':
            max_shape_pace = 1000
            min_shape_pace = 500
            print(f'Experiments on the PACE 2017 graph with nodes less than {max_shape_pace}')
            PACE_GRAPHS_PATH = './graph_test/pace2017_instances_index.p'
            with open(PACE_GRAPHS_PATH, 'rb') as f_in:
                filename_range = collections.OrderedDict(sorted(pickle.load(f_in).items()))

            files_pace = list(dict(filter(lambda x: x[0] <= max_shape_pace and x[0]> min_shape_pace ,
                                          filename_range.items())).values())
            # print(filename_range.items())
            print('Total number of graphs:', len(files_pace))

            gr_type = 'PACE'
            for f_pace_name in tqdm(files_pace):
                env = Environment(
                    f_pace_name[0],
                    square_index=True,
                    cost_function=degree_cost,
                    simple_graph=True,
                    use_random=False,
                    use_pace=True,
                    state_size=30,
                    p=args.proba,
                    verbose=True,
                    tw=args.treewidth
                )
                alg.env = env
                solved_graph = evaluate_method_with_specific_type(alg, gr_type,
                                                                  n_restarts=30,
                                                                  run_heuristic=False)
                total_solutions.append(solved_graph)

            print(len(total_solutions))
            with open(PATH_TO_PACE, 'wb') as fout:
                pickle.dump(total_solutions, fout)
        elif args.graph_type == 'circuit':
            max_shape_circuits = 1000
            min_shape_circuits = 500
            print(f'Experiments on the Google Circuits graphs with nodes less than {max_shape_circuits}')

            with open('./graph_test/boixo_circuits_index.p', 'rb') as f_in:
                filename_range = collections.OrderedDict(sorted(pickle.load(f_in).items()))
                files_qbit = list(dict(filter(lambda x: x[0] <= max_shape_circuits and x[0]> min_shape_circuits, filename_range.items())).values())

            import itertools

            files_qbit = list(itertools.chain(*files_qbit))

            print('Total number of graphs:', len(files_qbit))
            gr_type = 'circuit'
            index_value = random.sample(files_qbit, 100)
            # index_value = files_qbit[::10]
            print(PATH_TO_CIRCUITS)

            # print(len(index_value))
            # exit()
            for f_circuit_name in tqdm(index_value):
                env = Environment(
                        f_circuit_name,
                        square_index=True,
                        cost_function=degree_cost,
                        simple_graph=True,
                        use_random=False,
                        use_pace=True,
                        verbose=True,
                    )
                alg.env = env
                solved_graph = evaluate_method_with_specific_type(alg, gr_type, n_restarts=30, run_heuristic=False)
                total_solutions.append(solved_graph)
            print(len(total_solutions))
            with open(PATH_TO_CIRCUITS, 'wb') as fout:
                pickle.dump(total_solutions, fout)
        else:
            print('Evaluate data on the holdout Erdos')
            PATH_TO_ERDOS_HOLDOUT = EXPERIMENT_SOLUTION_PATH + '_erdos_holdout.pkl'

            files_erdos = ['./graph_test/erdos/inst_{}.gr.xz'.format(i) for i in range(40, 400, 10)]
            total_solutions = []

            # with open('./graph_test/pace2017_instances_index.p', 'rb') as f_in:
            #     filename_range = collections.OrderedDict(sorted(pickle.load(f_in).items()))

            # files_erdos = list(dict(filter(lambda x: x[0] <= 100, filename_range.items())).values())

            print('Total number of graphs:', len(files_erdos))

            gr_type = 'erdos_data'
            res_tamaki = []
            res_agents = []
            for f_name in tqdm(files_erdos):
                env = Environment(
                    f_name,
                    square_index=True,
                    cost_function=degree_cost,
                    simple_graph=False,
                    use_random=True,
                    use_pace=True,
                    p=args.proba,
                    verbose=True,
                    tw=args.treewidth
                )
                alg.env = env
                # heuristics_type = ['agent']
                start = time.time()
                agent_tw, agent_peo = get_agent_peo(alg, n_restarts=20)
                print(agent_tw)
                res_agents.append(agent_tw)
                time_alg = (time.time() - start) / 10
                _, m_tw = gm.get_upper_bound_peo(alg.env.initial_graph, method='min_fill')
                res_tamaki.append(m_tw)
                solved_graph = SolutionEntity(alg.env.initial_graph, [agent_tw], [agent_peo], 'erdos_holdout',
                                          [time_alg], num_nodes=alg.env.state_size)
                # solved_graph = evaluate_method_with_specific_type(alg, gr_type, n_restarts=1)
                total_solutions.append(solved_graph)
            print(res_tamaki)
            print(res_agents)
            print(len(total_solutions))
            with open(PATH_TO_ERDOS_HOLDOUT, 'wb') as fout:
                pickle.dump(total_solutions, fout)

    except KeyboardInterrupt:
        with open(DUMP_PATH, 'wb') as fout:
            pickle.dump(total_solutions, fout)