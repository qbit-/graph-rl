"""
Implements a DQN learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy


import torch
import torch_geometric
import torchcontrib
from torch import optim
import torch.nn.functional as F

from arguments import init_config
from graph_environment import Environment, degree_cost, contraction_cost_flops
from graph_policy import GraphActor
import graph_model as gm
import random
import networkx as nx
from copy import deepcopy as dc

import random
import os
import sys
import pickle
from datetime import datetime
from tqdm import tqdm


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_utils.utils import ReplayBuffer, Logger, TestMetric, set_global_seed

class SolutionEntity:
    def __init__(self, graph, scores, peos, graph_name, times, num_nodes=None):
        self.graph = graph
        self.labels = ['agent', 'min_fill','min_degree', "cardinality", 'tamaki', 'quickbb', 's2v']
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

class DQN:
    def __init__(
        self,
        envs,
        network,

        # Initial network parameters.
        init_network_params = None,
        init_weight_std = None,

        # DQN parameters
        double_dqn = True,
        update_target_frequency=10000,
        gamma=0.99,
        clip_Q_targets=False,

        # Replay buffer.
        replay_start_size=50000,
        replay_buffer_size=1000000,
        minibatch_size=32,
        update_frequency=1,

        # Learning rate
        update_learning_rate=True,
        initial_learning_rate=0,
        peak_learning_rate=1e-3,
        peak_learning_rate_step=10000,
        final_learning_rate=5e-5,
        final_learning_rate_step=200000,
        max_grad_norm=None,
        weight_decay=0,

        # Exploration
        update_exploration=True,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,

        # Loss function
        adam_epsilon=1e-8,
        loss="mse",

        # Saving the agent
        save_network_frequency=10000,
        network_save_path='network',

        # Testing the agent
        evaluate=True,
        test_envs=None,
        test_episodes=20,
        test_frequency=10000,
        test_save_path='test_scores',
        test_metric=TestMetric.ENERGY_ERROR,

        # Other
        logging=True,
        seed=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.double_dqn = double_dqn

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay=weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration,
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        if type(envs)!=list:
            envs = [envs]
        self.envs = envs
        self.env = envs[0]
        self.acting_in_reversible_spin_env = False
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.seed = 111#random.randint(0, 1e6) if seed is None else seed

        for env in self.envs:
            set_global_seed(self.seed, env)

        self.network = network().to(self.device)
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params != None:
            print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std != None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)
                with torch.no_grad():
                    self.network.apply(init_weights)

        self.target_network = network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay)

        self.evaluate = evaluate
        if test_envs in [None,[None]]:
            # By default, test on the same environment(s) as are trained on.
            self.test_envs = self.envs
        else:
            if type(test_envs) != list:
                test_envs = [test_envs]
            self.test_envs = test_envs
        self.test_episodes = int(test_episodes)
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        self.allowed_action_state = self.env.available_actions.copy()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path

    def get_random_env(self, envs=None, reset=False):
        if envs is None:
            env = random.sample(self.envs, k=1)[0]
        else:
            env = random.sample(envs, k=1)[0]
        env.reset()
        return env, False

    def get_replay_buffer_for_env(self):
        return self.replay_buffer

    def get_random_replay_buffer(self):
        return self.replay_buffer

    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger()

        # Initialise the state
        state = torch.as_tensor(self.env.reset(), dtype=torch.float)
        self.allowed_action_state = self.env.available_actions.copy()
        worst_cost = 0
        score = self.env.state_size
        losses_eps = []
        t1 = time.time()

        test_scores = []
        losses = []

        is_training_ready = False

        for timestep in range(timesteps):

            if not is_training_ready:
                if all([len(self.replay_buffer)>=self.replay_start_size]):
                    print('\nAll buffers have {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    is_training_ready=True

            # Choose action

            # out_degree = torch.sum(state, dim=0)
            # in_degree = torch.sum(state, dim=1)
            # identity = torch.eye(state.size()[1])
            # node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(state))
            #
            # extended_state = torch.cat([state, node_features])

            # action = self.act(extended_state.to(self.device).float(), is_training_ready=is_training_ready)
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate:
                self.update_lr(timestep)

            # Perform action in environment
            try:
                state_next, reward, done = self.env.step(action)
                worst_cost = np.maximum(worst_cost, reward)

                if done:
                    reward = -worst_cost
                else:
                    reward = -worst_cost#np.log(score + 1) if score >= 1 else 0

            except nx.exception.NetworkXError:
                print(action, state)
                print(self.allowed_action_state)
                # self.env.reset()
                state = torch.as_tensor(self.env.reset(), dtype=torch.float)
                self.allowed_action_state = self.env.available_actions.copy()
                worst_cost = 0
                continue

            # self.allowed_action_state = torch.FloatTensor(self.env.available_actions.copy())
            self.allowed_action_state = self.env.available_actions.copy()
            score += reward

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next, dtype=torch.float)

            # out_degree = torch.sum(state, dim=0)
            # in_degree = torch.sum(state, dim=1)
            # identity = torch.eye(state.size()[1])
            # node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(state))
            #
            # extended_state = torch.cat([state, node_features])

            # out_degree = torch.sum(state_next, dim=0)
            # in_degree = torch.sum(state_next, dim=1)
            # identity = torch.eye(state_next.size()[1])
            # node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(state_next))
            #
            # extended_next_state = torch.cat([state_next, node_features])

            done = torch.as_tensor([done], dtype=torch.float)

            self.replay_buffer.add(state, action, reward, state_next , done)

            if done:
                # Reinitialise the state
                worst_cost = 0
                if verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep+1),

                         np.round(score,3),
                         loss_str,
                         round(time.time() - t1, 3)))

                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)
                self.env, self.acting_in_reversible_spin_env = self.get_random_env(reset=True)
                # self.replay_buffer = self.get_replay_buffer_for_env()
                state = torch.as_tensor(self.env.reset(), dtype=torch.float)

                self.allowed_action_state = self.env.available_actions.copy()

                score = 0
                losses_eps = []
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:

                # Update the main network
                if timestep % self.update_frequency == 0:

                    # Sample a batch of transitions
                    transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep,loss])
                    losses_eps.append(loss)

                    if self.logging:
                        logger.add_scalar('Loss', loss, timestep)

                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                test_score = self.evaluate_agent()
                print('\nTest score: {}\n'.format(np.round(test_score,3)))

                best_network = test_score <= score

                if best_network:
                    print('is best_network')
                    path = self.network_save_path
                    path_main, path_ext = os.path.splitext(path)
                    path_main += "_best"
                    if path_ext == '':
                        path_ext += '.pth'
                    self.save(path_main + path_ext)

                test_scores.append([timestep+1,test_score])

            if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
                path = self.network_save_path
                path_main, path_ext = os.path.splitext(path)
                path_main += str(timestep+1)
                if path_ext == '':
                    path_ext += '.pth'
                self.save(path_main+path_ext)

        if self.logging:
            logger.save()

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'

        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('test_scores saved to {}'.format(path))

        with open(self.losses_save_path, 'wb+') as output:
            pickle.dump(np.array(losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('losses saved to {}'.format(self.losses_save_path))

    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions
        if self.acting_in_reversible_spin_env:
            # Calculate target Q
            with torch.no_grad():
                if self.double_dqn:
                    greedy_actions = self.network(states_next.float()).argmax(1, True)
                    q_value_target = self.target_network(states_next.float()).gather(1, greedy_actions)
                else:
                    q_value_target = self.target_network(states_next.float()).max(1, True)[0]

        else:
            target_preds = self.target_network(states_next.float())

            # Calculate target Q, selecting ONLY ALLOWED ACTIONS greedily.
            with torch.no_grad():
                if self.double_dqn:
                    network_preds = self.network(states_next.float())
                    # Set the Q-value of disallowed actions to a large negative number (-10000) so they are not selected.

                    network_preds_allowed = network_preds.masked_fill( \
                        1 - torch.ByteTensor(self.allowed_action_state).to(device), -10000)

                    greedy_actions = network_preds_allowed.argmax(1, True)
                    q_value_target = target_preds.gather(1, greedy_actions)
                else:
                    q_value_target = target_preds.masked_fill(\
                        1 - torch.ByteTensor(self.allowed_action_state).to(device), -10000).max(1, True)[0]

                    # target_preds_allowed =
                    # q_value_target = target_preds.masked_fill(disallowed_actions_mask,-10000).max(1, True)[0]

        if self.clip_Q_targets:
            q_value_target[q_value_target < 0] = 0

        # Calculate TD target
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value
        q_value = self.network(states.float()).gather(1, actions)

        # Calculate loss
        loss = self.loss(q_value, td_target, reduction='mean')

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximises Q function
            action = self.predict(state)
        else:
            # x = (state[0, :] == self.allowed_action_state).nonzero()
            from random import choice
            my_idx = choice(np.where(self.allowed_action_state)[0])
            # To DO samlpe
            action = choice((np.arange(len(self.allowed_action_state))[np.array(self.allowed_action_state,dtype=bool)]))#np.random.randint(0, len(self.allowed_action_state))

        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
                )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (self.final_learning_rate_step - self.peak_learning_rate_step)
                )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    @torch.no_grad()
    def predict(self, states, acting_in_reversible_spin_env=None):
        qs = self.network(states)
        # print(self.allowed_action_state)
        if qs.dim() == 1:
            qs_allowed = qs.masked_fill(1-torch.ByteTensor(self.allowed_action_state).to(device), float('-inf'))
            actions = qs_allowed.argmax().item()
        else:
            qs_allowed = qs.masked_fill(1 - torch.ByteTensor(self.allowed_action_state).to(device), float('-inf'))
            actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()[0]

        return actions

    @torch.no_grad()
    def evaluate_agent(self, batch_size=None, with_minfill=False):
        """
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        """
        if batch_size is None:
            batch_size = self.minibatch_size

        score = 0
        test_env = self.test_envs[0]
        test_env = self.env
        obs = test_env.reset()
        self.allowed_action_state = self.env.available_actions.copy()

        state = torch.as_tensor(obs, dtype=torch.float)
        out_degree = torch.sum(state, dim=0)
        in_degree = torch.sum(state, dim=1)
        identity = torch.eye(state.size()[1])
        node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(state))

        # extended_state = torch.cat([state, node_features])

        old_graph = dc(obs)

        old_graph = nx.from_numpy_array(old_graph)
        old_graph_minfill_copy = dc(old_graph)

        peo = []
        done = False
        while not done:
            # action = self.act(extended_state.to(self.device).float(),)

            # action = self.predict(extended_state.to(device))
            action = self.predict(state.to(device))
            # Perform action in environment

            try:
                state_next, reward, done = self.env.step(action)
            except nx.exception.NetworkXError:
                print(action, self.allowed_action_state)
                exit()
                return self.env.state_size
                self.env.reset()
                self.allowed_action_state = self.env.available_actions.copy()
                continue
            except KeyError:
                print(action, self.allowed_action_state)

            self.allowed_action_state = self.env.available_actions.copy()
            score += reward
            peo.append(action)
            # Store transition in replay buffer
            # action = torch.as_tensor([action], dtype=torch.long)
            # reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next, dtype=torch.float)

            # out_degree = torch.sum(state_next, dim=0)
            # in_degree = torch.sum(state_next, dim=1)
            # identity = torch.eye(state_next.size()[1])
            # node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(state_next))
            #
            # extended_next_state = torch.cat([state_next, node_features])
            done = torch.as_tensor([done], dtype=torch.float)

            if done:
                worst_cost = 0
                # Reinitialise the state
                print(" score : {}".format(np.round(score, 3)))
            else:
                state = state_next
        treewidth_cost = gm.get_treewidth_from_peo(old_graph, peo)
        if with_minfill:
            graph_indices, reward_minfill = gm.get_upper_bound_peo(old_graph_minfill_copy, method='tamaki', wait_time=20)

            print(treewidth_cost, reward_minfill)
        # print(self.network.th1.data)
        # self.env.reset()
        self.allowed_action_state = self.env.available_actions.copy()
        if with_minfill:
            return treewidth_cost, reward_minfill
        return treewidth_cost

    def save(self, path='dqn_nets/network.pth'):
        if os.path.splitext(path)[-1]=='':
            path + '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location=self.device))


def eval_scores_on(args, type, agent):
    model_path = 'dqn/results'
    PATH_TO_ERDOS_HOLDOUT = model_path + '/erdos_holdout.pkl'

    files_erdos = ['./graph_test/erdos/inst_{}.gr.xz'.format(i) for i in range(40, 400, 10)]
    total_solutions = []

    print('Total number of graphs:', len(files_erdos))

    gr_type = 'erdos_data'
    results = []
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
        agent.test_envs = [env]
        agent.env = env

        # heuristics_type = ['agent']
        start = time.time()
        # agent_tw, agent_peo = get_agent_peo(alg, n_restarts=50)
        agent_tw, reward_minfill = agent.evaluate_agent(with_minfill=True)

        # time_alg = (time.time() - start) / 10
        # if curr_shape == 50:
        #     _, m_tw = gm.get_upper_bound_peo(alg.env.initial_graph, method='min_fill')
        #     results['minfill'].append(m_tw)

        results.append(agent_tw)

        # print(len(results['50']))
    print(results)
    with open('dqn_nets/results/res_erdos_val.pkl', 'wb') as fout:
        pickle.dump(results, fout)


def evaluate_dqn_and_restore(solution, agent):
    graph = solution.graph

    env = Environment(
        '',
        square_index=True,
        cost_function=degree_cost,
        simple_graph=False,
        use_random=True,
        use_pace=False,
        p=args.proba,
        state_size=solution.num_nodes,
        verbose=True,
        tw=args.treewidth

    )
    env.reset(reset_nodes=True, graph=graph)
    agent.test_envs = [env]
    agent.env = env
    agent.allowed_action_state = agent.env.available_actions.copy()

    # heuristics_type = ['agent']
    start = time.time()
    # agent_tw, agent_peo = get_agent_peo(alg, n_restarts=50)
    tw = agent.evaluate_agent(with_minfill=False)
    print(tw, solution.tamaki_score)

    time_alg = time.time() - start

    solution.s2v_score = tw
    solution.s2v_time = time_alg

    return solution


def eval_test(path_f, agent):
    with open(path_f, 'rb') as fin:
        arr_solutions = pickle.load(fin)

    total_solutions = []
    print(len(arr_solutions))
    for sol in arr_solutions:
        total_solutions.append(evaluate_dqn_and_restore(sol, agent))

    with open(path_f, 'wb') as fout:
        pickle.dump(total_solutions, fout)


if __name__ == '__main__':
    nb_steps = 8000000
    from rl_utils.s2v import S2V

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

    use_cuda = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda:", use_cuda)

    global cpu_device
    cpu_device = torch.device('cpu')

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
        p=args.proba if not args.eval_mode else 50 * args.proba / MAX_STATE_SIZE,
        random_graph_type=args.random_graph_type,
        tw=args.treewidth,
        use_pace=args.use_pace
    )
    env.reset(reset_nodes=True)
    MAX_STATE_SIZE = env.state_size
    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    if args.resume_path is not None:
        exp_gamma = 0.99
        lr = 1e-2
    else:
        lr = args.lr
        exp_gamma = 0.999

    train_envs = [env]
    test_envs = [env]
    # network_fn = lambda: MPNN(n_obs_in=train_envs[0].state_size,
    #                           n_layers=3,
    #                           n_features=64,
    #                           n_hid_readout=[],
    #                           tied_weights=False)

    network_fn = lambda: S2V(p=64)

    if args.eval_mode:
        network_path = 'trash_net_best.pth'
    agent = DQN(train_envs,
                network_fn,
                init_network_params=None,
                init_weight_std=0.01,
                double_dqn=True,
                clip_Q_targets=True,
                replay_start_size=1000,
                replay_buffer_size=10000,
                gamma=0.9999,
                update_target_frequency=1000,
                update_learning_rate=False,
                initial_learning_rate=1e-4,
                peak_learning_rate=1e-4,
                peak_learning_rate_step=20000,
                final_learning_rate=1e-4,
                final_learning_rate_step=200000,

                update_frequency=32,  # 1
                minibatch_size=64,  # 128
                max_grad_norm=None,
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=1,
                final_exploration_rate=0.05,  # 0.05
                final_exploration_step=800000,  # 40000

                adam_epsilon=1e-8,
                logging=False,
                loss="mse",

                save_network_frequency=400000,
                network_save_path='dqn_nets/trash_net',

                evaluate=True,
                test_episodes=args.n_restarts,
                test_frequency=4000,  # 10000
                test_save_path='test_res',
                test_metric=TestMetric.MAX_CUT,
                test_envs=test_envs,
                seed=None
                )

    if args.eval_mode:
        checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)

        agent.load(args.resume_path)
        path_to = './solutions/night_test_random_graphs.pkl.new'
        # scores = agent.evaluate_agent(with_minfill=True)
        # eval_scores_on(args, 'erdos', agent)
        eval_test(path_to, agent)
    else:
        try:
            start = time.time()
            agent.learn(timesteps=nb_steps, verbose=False)
            print(time.time() - start)
        except KeyboardInterrupt:
            agent.save()
