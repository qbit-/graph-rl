import collections

import torch
import numpy as np

import random
import os
import sys
import pickle
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

from arguments import init_config
from collect_result import validate
from graph_model import get_peo, get_upper_bound_peo
from graph_policy import GraphActor, get_appropriate_graph_policy
from actor_critic import ActorCritic, SolverTD
from graph_environment import Environment, degree_cost, contraction_cost_flops, MaximumVertexCoverEnvironment


#  TODO: add parallel execution


def evaluate_agent(alg, num_experiments=10, n_restarts=5,
                   use_min=True, verbose=True, heuristic_type='min_fill', exact=False):
    """
    :param alg: algorithm with eval functional
    :param num_experiments: number of reset environment
    :param n_restarts: number restarts every experiment
    :param use_min: use minimum score of all trajectories
    :param heuristic_type: type of used heuristic for validation
    :param exact use external solver for exact solution
    :param verbose:
    :return: Final score ration
    """
    total_score = 0
    if verbose:
        if exact:
            print('Mean |' + ' Max  | ' + str(heuristic_type) + ' | ' + 'Exact')
        else:
            print('Mean |' + ' Max  | ' + str(heuristic_type))
    for i in range(num_experiments):
        agent_trs = []
        agent_tr, minfill_score = alg.eval(with_minfill=True, reset_nodes=True,
                                           heuristic_type=heuristic_type, verbose=verbose)
        agent_trs.append(agent_tr)
        for _ in range(n_restarts):
            agent_tr = alg.eval(reset_nodes=False)
            agent_trs.append(agent_tr)
        if verbose:
            if exact:
                exact_score = get_upper_bound_peo(alg.env.initial_graph, wait_time=400)
                import pickle
                with open('peo_for_pace.pd', 'wb') as fout:
                    pickle.dump(exact_score[0],fout, protocol=2)

                print('exact', exact_score[0][:10])
                print(np.mean(agent_trs), min(agent_trs), minfill_score, exact_score[1])
            else:
                print(np.mean(agent_trs), min(agent_trs), minfill_score)

        if use_min:
            agent_tr = min(agent_trs)
        else:
            agent_tr = round(np.mean(agent_trs), 3)

        total_score += minfill_score / agent_tr

    print('Final Score ratio : ', total_score / num_experiments)

    return total_score / num_experiments


def get_new_graph_size(prev_size, step=15):
    """
    Linear increase the graph size
    :param prev_size:
    :param epoch:
    :return:
    """
    return prev_size + step


max_shape_pace = 250
min_shape_pace = 15
PACE_GRAPHS_PATH = './graph_test/pace2017_instances_index.p'


if __name__ == '__main__':
    args = init_config()
    # fix seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    genearal_exp_path = os.path.join('logs', args.exp_name)
    current_exp_path = os.path.join(genearal_exp_path, time_stamp)
    os.makedirs(current_exp_path, exist_ok=True)
    os.makedirs('best_models', exist_ok=True)

    experiment_arguments = ' '.join(sys.argv[1:])
    with open(os.path.join(current_exp_path, 'args.txt'), 'w') as f_out_log:
        f_out_log.write(experiment_arguments)

    writer = SummaryWriter(log_dir=os.path.join(os.path.join('runs', args.exp_name), time_stamp), purge_step=0)

    use_cuda = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda:", use_cuda)

    global cpu_device
    cpu_device = torch.device('cpu')

    env_name = args.graph_path
    costs = {'treewidth': degree_cost, 'flops': contraction_cost_flops}
    state_shape = MAX_STATE_SIZE = args.max_shape
    print('Inititalize graph')
    print(f'Train agent to solve {args.task}')

    if args.schedule_sizes and args.use_pace:
        with open(PACE_GRAPHS_PATH, 'rb') as f_in:
            filename_range = collections.OrderedDict(sorted(pickle.load(f_in).items(),reverse=True))
        # there is a dict with the following structure: files as a value and size as a key.
        dict_pace = dict(filter(lambda x: x[0] <= max_shape_pace and x[0] > min_shape_pace, filename_range.items()))
        _ = dict_pace.popitem()

        pace_gr_pair = dict_pace.popitem()
        state_shape = pace_gr_pair[0]
        env_name = pace_gr_pair[1][0]

    if args.task == 'mvc':
        env = MaximumVertexCoverEnvironment(
            env_name,
            square_index=True,
            cost_function=costs[args.cost_function],
            simple_graph=True,
            use_random=args.use_erdos,
            state_size=MAX_STATE_SIZE,
            p=args.proba if not args.eval_mode else 50 * args.proba / MAX_STATE_SIZE,
            random_graph_type=args.random_graph_type,
            tw=args.treewidth,
            use_pace=args.use_pace
            )
    else:
        env = Environment(
            env_name,
            square_index=True,
            cost_function=costs[args.cost_function],
            simple_graph=True,
            use_random=args.use_erdos,
            state_size=state_shape,
            p=args.proba if not args.eval_mode else 50 * args.proba / MAX_STATE_SIZE,
            random_graph_type=args.random_graph_type,
            tw=args.treewidth,
            use_pace=args.use_pace
        )
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    MAX_STATE_SIZE = env.state_size
    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    if args.resume_path is not None:
        exp_gamma = 0.99
    else:
        exp_gamma = 0.999

    lr = args.lr

    if args.verbose:
        print('Learning Rate', lr)
    ndim = 2 if args.use_small_features else state_shape[0]
    # Create RL actor for training
    graph_policy = get_appropriate_graph_policy(args, ndim, device)
    actor = GraphActor(policy=graph_policy).to(device)

    actor_total_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print('Actor number of parameters: ', actor_total_params)
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
        actor.load_state_dict(checkpoint)

    alg = ActorCritic(env,
                      actor,
                      ndim=ndim,
                      gamma=0.999,
                      gae_lambda=0.8,
                      num_episodes=args.batch_size,
                      episode_len=MAX_STATE_SIZE,
                      entropy_reg_coefficient=5e-3,
                      device=device,
                      lr=lr, use_gcn=args.gcn,
                      use_log=args.use_neighbour,
                      use_swa=args.use_swa,
                      optimizer_type=args.optimizer_type,
                      use_sil=args.use_sil,
                      args=args
                      )
    if args.validate:
        solver = SolverTD(env,
                          GraphActor(policy=graph_policy).to(device),
                          device=device,
                          use_log=args.use_neighbour,
                          )

    if args.use_intrinsic:
        icm_params = sum(p.numel() for p in alg.icm.parameters() if p.requires_grad)
        print('ICM number of parameters: ', icm_params)

    if args.verbose:
        print(alg.actor_optimizer)
        print(alg.actor)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(alg.actor_optimizer, base_lr=0.01, max_lr=0.1, step_size_up=10)
    if args.eval_mode:
        # agent_tr, minfill_score = alg.eval(with_minfill=True, reset_nodes=True)
        if args.validate:
            solver.update_actor(alg.actor)
            current_val_score, scores, total_solutions = validate(solver)
            print(current_val_score)
            exit()
        evaluate_agent(alg, num_experiments=args.num_experiments,
                       n_restarts=args.n_restarts, heuristic_type=args.heuristic_type, exact=False)
        exit()
    num_episodes = 0
    avg_reward = 0
    sum_actor_loss = 0
    sum_critic_loss = 0
    best_val_score = 10
    validation_scores = []

    results_log = {'reward': [], 'value_loss': [], 'policy_loss': []}
    log_val = {'val_score': []}
    num_epochs = args.max_epoch
    log_every = args.log_every
    print('Initial score:')
    agent_tr, minfill_score = alg.eval(with_minfill=True)
    print("Agent reward: {}, Minfill reward: {}".format(agent_tr, minfill_score))
    prev_best_reward = -state_shape[0] ** 2
    best_agent_tw = -prev_best_reward
    new_gr_size = env.state_size
    try:
        for i in tqdm(range(num_epochs)):
            logs = alg.train(args.reset_nodes)
            if args.schedule_sizes:
                # TODO: add graph from pace.
                max_step_per_size = 4 if new_gr_size < 68 else 4
                if (i+1) % max_step_per_size == 0:
                    if args.use_pace:
                        pace_gr_pair = dict_pace.popitem()
                        new_gr_size = pace_gr_pair[0]
                        gr_path = pace_gr_pair[1][0]  # we use one graph with the size. FIXME
                        print(gr_path)
                        env = Environment(
                            gr_path,
                            square_index=True,
                            cost_function=degree_cost,
                            simple_graph=True,
                            use_random=False,
                            use_pace=True,
                            state_size=new_gr_size,
                            p=args.proba,
                            verbose=True,
                            tw=args.treewidth
                        )
                        alg.env = env
                    else:
                        new_gr_size = get_new_graph_size(prev_size=new_gr_size)
                        alg.env.reset(reset_nodes=True, new_size=new_gr_size)

                    alg.upgrade_gamma_matrices(new_gr_size)
                    print(f'Number of nodes in the new graph is {new_gr_size}')
            r = logs.get_log('train_reward')
            v_loss = logs.get_log('TD_error')
            p_loss = logs.get_log('policy_loss')

            results_log['reward'].append(r[-1])
            results_log['value_loss'].append(v_loss[-1])
            results_log['policy_loss'].append(p_loss[-1])

            for u,v in results_log.items():
                writer.add_scalar('Loss/' + u, float(v[-1]), i)

            s_log = ''
            for k, it in results_log.items():
                s_log += f'|{k}  = {round(it[-1],3)}|   '
            if i % log_every == 0:
                if args.use_swa and i > args.swa_freq:
                    alg.actor_optimizer.update_swa()
                cost = logs.get_log('cost')[-1]
                s_log += "cost " + str(-cost)
                print(s_log)
                if cost > prev_best_reward:
                    prev_best_reward = cost
                    path_to_model = 'model_basic_hidden_' + str(args.hidden_gcn) + '.pt'
                    torch.save(alg.actor.state_dict(), os.path.join(current_exp_path, path_to_model))

            if args.validate and (i % args.val_every == 0):
                solver.update_actor(alg.actor)
                current_val_score, scores, total_solutions = validate(solver)
                validation_scores.append(current_val_score)

                print(current_val_score)
                with open(os.path.join(genearal_exp_path, 'val.pkl'), 'wb') as f_out_log:
                    pickle.dump(validation_scores, f_out_log)

                if current_val_score < best_val_score:
                    best_val_score = current_val_score

                torch.save(alg.actor.state_dict(), os.path.join(genearal_exp_path, 'best_model.pt'))

            elif i % args.val_every == 0 and i >= 0:
                # Correctly reset minfill score whin reset nodes
                if not args.reset_nodes:
                    agent_tr, minfill_score = alg.eval(with_minfill=True)
                    # agent_tr = alg.eval(verbose=args.verbose)
                else:
                    agent_tr, minfill_score = alg.eval(with_minfill=True)

                log_val['val_score'].append(agent_tr)
                print("Agent reward: {}, Minfill reward: {}".format(agent_tr, minfill_score))
                path_to_model = 'best_models/model_reinforce.pt'
                torch.save(alg.actor.state_dict(), os.path.join(genearal_exp_path, 'best_model.pt'))
                if agent_tr <= best_agent_tw:
                    best_agent_tw = agent_tr
                    torch.save(alg.actor.state_dict(), path_to_model)

                if agent_tr <= minfill_score and i >= 100:
                    evaluate_agent(alg, num_experiments=2, heuristic_type=args.heuristic_type)
                    agent_tr, minfill_score = alg.eval(with_minfill=True)

    except KeyboardInterrupt:
        with open(os.path.join(current_exp_path, 'log.pkl'), 'wb') as f_out_log:
            pickle.dump(results_log, f_out_log)
        with open(os.path.join(current_exp_path, 'val.pkl'), 'wb') as f_out_log:
            pickle.dump(validation_scores, f_out_log)

    with open(os.path.join(current_exp_path, 'log.pkl'), 'wb') as f_out_log:
        pickle.dump(results_log, f_out_log)

    print(validation_scores)