import torch
import torch.nn as nn
import numpy as np

import random
import os

import time
from datetime import datetime

from arguments import init_config
from graph_policy import GraphActor, get_appropriate_graph_policy
from actor_critic import ActorCritic
from graph_environment import Environment, degree_cost, contraction_cost_flops, gm
from train_ac import evaluate_agent


if __name__ == '__main__':
    args = init_config()
    # fix seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    genearal_exp_path = os.path.join('logs',args.exp_name)
    current_exp_path = os.path.join(genearal_exp_path, time_stamp)
    os.makedirs(current_exp_path, exist_ok=True)
    os.makedirs('best_models', exist_ok=True)

    use_cuda = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda:", use_cuda)

    global cpu_device
    cpu_device = torch.device('cpu')

    # TODO move parameters to arguments
    env_name = args.graph_path
    costs = {'treewidth': degree_cost, 'flops': contraction_cost_flops}
    MAX_STATE_SIZE = args.max_shape

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

    actor_total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print('Actor number of parameters: ', actor_total_params)
    checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
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

    evaluate_agent(alg,
                   num_experiments=args.num_experiments,
                   n_restarts=args.n_restarts,
                   verbose=args.verbose,
                   heuristic_type=args.heuristic_type,
                   exact=args.use_exact)
