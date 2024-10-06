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


    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    ndim = 2 if args.use_small_features else state_shape[0]

    graph_policy = get_appropriate_graph_policy(args, ndim, device)
    actor = GraphActor(policy=graph_policy).to(device)

    shape_list = [35, 60, 100]
    dirs_res_models = [f'logs/check_shape_{i}/' for i in shape_list]
    results = collections.defaultdict(list)

    for model_path, curr_shape in zip(dirs_res_models, shape_list):
        actor_total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
        print('Actor number of parameters: ', actor_total_params)
        print(model_path)
        checkpoint = torch.load(os.path.join(model_path, 'best_model.pt'), map_location=lambda storage, loc: storage)
        actor.load_state_dict(checkpoint)

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

        PATH_TO_ERDOS_HOLDOUT = model_path + '/erdos_holdout.pkl'

        files_erdos = ['./graph_test/erdos/inst_{}.gr.xz'.format(i) for i in range(40, 400, 10)]
        total_solutions = []

        print('Total number of graphs:', len(files_erdos))

        gr_type = 'erdos_data'
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
            agent_tw, agent_peo = get_agent_peo(alg, n_restarts=50)
            time_alg = (time.time() - start) / 10
            # if curr_shape == 50:
            #     _, m_tw = gm.get_upper_bound_peo(alg.env.initial_graph, method='min_fill')
            #     results['minfill'].append(m_tw)

            results[str(curr_shape)].append(agent_tw)

        # print(len(results['50']))
        print(results)

    with open('check_small_result_erdos_max_shape_part2.pkl', 'wb') as fout:
        pickle.dump(results, fout)