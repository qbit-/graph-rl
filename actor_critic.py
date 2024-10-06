import torch
import torch_geometric
import torchcontrib
from torch import optim
import torch.nn.functional as F

from arguments import init_config
from graph_environment import Environment, degree_cost, contraction_cost_flops
from graph_policy import GraphActor, get_appropriate_graph_policy
import graph_model as gm
from misc.adamw import AdamW

from rl_utils import create_gamma_matrix
from rl_utils.curiosity import IntrinsicReward, ICM
from rl_utils.sil_module import SIL

from utils import adj_to_pgeom, SimpleLogger

import numpy as np
import random
import networkx as nx
from copy import deepcopy as dc

class SolverTD:
    def __init__(
            self,
            env,
            actor,
            ndim=2,
            device=torch.device('cpu'),
            use_gcn=False,
            use_log=False,
    ):
        self._device = device
        self.is_graph = use_gcn
        self.use_log = use_log
        self.env = env
        self.ndim = ndim
        self.actor = actor.to(self._device)

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def run_episode(self, use_logn=False, with_actions=False,
                    reset_nodes=False, with_proba=False):
        states, actions, masks, rewards, dones = [], [], [], [], []
        next_states = []
        env = self.env
        s = env.reset(reset_nodes)
        old_graph = dc(s)

        done = False
        worst_cost = 0
        i = 0
        entropy_episode = []

        while not done:

            mask = env.available_actions.copy()
            s = adj_to_pgeom(s, device=self._device,
                             mask=self.to_tensor(mask),
                             use_small_features=(self.ndim == 2)).to(self._device)
            states.append(s)
            if i > 0:
                next_states.append(s)
            val, a, dist = self.actor(s, mask=self.to_tensor(mask), with_dist=True,
                                      greedy=False)
            a = a.cpu().detach().item()
            s, r, done = env.step(a)
            worst_cost = np.maximum(worst_cost, r)


            actions.append(a)
            masks.append(mask)

            if with_proba:
                reward = get_score_for_task(self.task, r, worst_cost, done, use_logn=use_logn)

                reward_tensor = torch.FloatTensor([reward])

                if i == 0:
                    policy_proba = dist.probs[a].unsqueeze(0)
                    reward_episode = reward_tensor.unsqueeze(0)
                    entropy_episode.append(dist)
                else:
                    new_proba = dist.probs[a].unsqueeze(0)
                    policy_proba = torch.cat([policy_proba, new_proba], dim=0)
                    reward_episode = torch.cat([reward_episode, reward_tensor.unsqueeze(0)], dim=0)
                    entropy_episode.append(dist)
                i += 1

        if with_actions and with_proba:
            return old_graph, actions, entropy_episode
        return old_graph, actions

    def update_actor(self, model):
        par1 = dict(self.actor.named_parameters())
        par2 = dict(model.named_parameters())

        for k in par1.keys():
            par1[k].data.copy_(par2[k].data)

    @torch.no_grad()
    def eval(self, with_minfill=False, verbose=False, reset_nodes=False, heuristic_type='min_fill', with_peo=False):
        self.actor.eval()
        old_graph, actions = self.run_episode(use_logn=self.use_log, with_actions=True, reset_nodes=reset_nodes)
        self.actor.train()
        old_graph = nx.from_scipy_sparse_matrix(old_graph)
        old_graph_minfill_copy = dc(old_graph)
        treewidth_cost = gm.get_treewidth_from_peo(old_graph, actions)
        if with_minfill: # put the wait time
            graph_indices, reward_minfill = gm.get_upper_bound_peo(old_graph_minfill_copy, method=heuristic_type)
            if verbose:
                print('actor', actions[:10])
                print('heuristic', graph_indices[:10])
            return treewidth_cost, reward_minfill

        if with_peo:
            return treewidth_cost, actions
        return treewidth_cost


class ActorCritic:
    def __init__(
            self,
            env,
            actor,
            gamma,
            ndim=2,
            lr=0.0001,
            gae_lambda=0.95,
            episode_len=10,
            num_episodes=64,
            entropy_reg_coefficient=0.001,
            device=torch.device('cpu'),
            optimizer_type='adam',
            use_gcn=False,
            use_log=False,
            use_swa=False,
            use_sil=None,
            args=None
    ):
        """
        Actor critic algorithm
         with default usage of the GAE and Discount Reward
        :param env: Enviorment for the problem
        :param actor: Net with two heads
        :param gamma: discount variable
        :param ndim: number of feature space
        :param lr: Learning Rate for Adam [Best practice to use here lr > 0.01 up to convergence]
        :param episode_len:
        :param entropy_reg_coefficient:
        :param device:
        :param optimizer_type: | Adam | L-BFGS | SGD
        :param use_gcn:
        :param use_log: bool Use log of the reward per step
        :param use_swa: Use SWA from https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
        """
        self._device = device
        self.is_graph = use_gcn
        self.use_log = use_log
        self.env = env
        self.ndim = ndim
        self.actor = actor.to(self._device)
        self.use_swa = use_swa
        self.optimizer_type = optimizer_type

        self.num_episodes = num_episodes
        self.gamma = gamma
        self.episode_len = episode_len
        self.entropy_reg_coefficient = entropy_reg_coefficient
        self.gae_lambda = gae_lambda
        self.use_gae = True
        self.use_priority = args.use_priority

        self.task = args.task
        if args.task == 'treewidth':
            self.eval = self.eval_treewidth
        elif args.task == 'mvc':
            self.eval = self.eval_mvc
        else:
            raise Exception('Unexpected method')

        self.use_intrinsic = args.use_intrinsic
        if self.use_intrinsic:
            self.lamda = 0.01
            # self.intrinsic_model = IntrinsicReward(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
            self.icm = ICM(ndim, hidden_dim=24, n_act=self.env.state_size).to(device)

            # self.intrinsic_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        self.gam_matrix = torch.FloatTensor(
            create_gamma_matrix(self.gamma, episode_len)
        ).to(self._device)

        if self.use_gae:
            self.gam_lam_matrix = torch.FloatTensor(
                create_gamma_matrix(self.gamma * self.gae_lambda, episode_len)
            ).to(self._device)

        if self.use_swa:
            self.base_optimizer = optim.Adam(
                self.actor.parameters(), lr=lr)
            self.actor_optimizer = torchcontrib.optim.SWA(self.base_optimizer)
        else:
            if self.optimizer_type == 'adam':
                # self.actor_optimizer = optim.Adam(
                # self.actor.parameters(), lr=lr)
                self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=lr)
            elif self.optimizer_type == 'adam2':
                self.actor_optimizer = optim.Adam([
                    {'params': self.actor.policy.policy.parameters()},
                    {'params': self.actor.policy.conv_layers.parameters()},
                    {'params': self.actor.policy.value.parameters(), 'lr': 1e-3},
                ], lr=lr)
                if self.use_intrinsic:
                    self.actor_optimizer.add_param_group({'params': self.icm.parameters(), 'lr': 1e-3})

            elif self.optimizer_type == 'adagrad':
                self.actor_optimizer = optim.Adagrad([
                    {'params': self.actor.policy.policy.parameters()},
                    {'params': self.actor.policy.conv_layers.parameters()},
                    {'params': self.actor.policy.value.parameters(), 'lr': 1e-3}
                ], lr=lr)
            elif self.optimizer_type == 'l-bfgs':
                #  works badly with closure
                self.actor_optimizer = optim.LBFGS(
                    self.actor.parameters(),
                    max_iter=20,
                    history_size=100,
                    lr=1.0
                )
            else:
                self.actor_optimizer = optim.SGD(
                    self.actor.parameters(), lr=lr)
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10)
        self.sil_model = None
        if use_sil:
            self.sil_model = SIL(self.actor, self.actor_optimizer, env.state_size, use_priority=args.use_priority)

        self.debug = False
        self.log = SimpleLogger()
        self.log.add_info('tot_return')
        self.log.add_info('TD_error')
        self.log.add_info('entropy')
        self.log.add_info('train_reward')
        self.log.add_info('policy_loss')
        self.log.add_info('cost')
        self.log.add_info('real_cost')

    def upgrade_gamma_matrices(self, new_size):
        self.gam_matrix = torch.tensor(
            create_gamma_matrix(self.gamma, new_size), dtype=torch.float
        ).to(self._device)
        self.episode_len = new_size
        if self.use_gae:
            self.gam_lam_matrix = torch.tensor(
                create_gamma_matrix(self.gamma * self.gae_lambda, new_size), dtype=torch.float
            ).to(self._device)
        if self.sil_model is not None:
            self.sil_model = SIL(self.actor, self.actor_optimizer, new_size, use_priority=self.use_priority)

    def update_model(self, proba, reward, value, states=None, next_states=None, actions=None):
        if self.use_intrinsic:
            reward_i = self.icm.reward(actions=actions, states=states, next_states=next_states).detach()
            reward[:, :-1] += self.icm.intrinsic_reward_lmbda * reward_i  # or sum with coef

        if self.use_gae:
            deltas = reward + self.gamma * value[:, 1:] - value[:, :-1]
            advantage = deltas.squeeze() @ self.gam_lam_matrix[:self.episode_len, :self.episode_len]
        else:
            advantage = value[:, :-1] - reward
        advantage = self.rescale_advantages(advantage)
        reward = reward @ self.gam_matrix[:self.episode_len, :self.episode_len]
        #  TODO add mask for mvc problem

        policy_loss = -(torch.log(proba+1e-10) * advantage).mean()
        value_loss = F.smooth_l1_loss(value[:, :-1], reward)
        # print(value, reward)
        # policy_loss = torch.nn.CrossEntropyLoss()(proba.view(32*49,-1), actions)

        # value_loss = 0.5 * (value[:, :-1] - reward).pow(2).mean()
        entropy = -(proba * proba.log()).mean()
        lambda_value = 0.05
        loss = policy_loss + lambda_value*value_loss - self.entropy_reg_coefficient * entropy
        if self.use_intrinsic:
            loss = self.icm.loss(loss, states, next_states, actions)

        if self.debug:
            loss.register_hook(lambda grad: print(grad))
            proba.register_hook(lambda grad: print(grad))
            value.register_hook(lambda grad: print(grad.squeeze()))
            self.actor.policy[-1].weight.register_hook(lambda grad: print(grad.sum()))
            self.actor.conv_layers[0].weight.register_hook(lambda grad: print(grad.sum()))

        self.actor_update(loss)
        self.log.add_item('TD_error', value_loss.detach().item())
        self.log.add_item('entropy', entropy.cpu().detach().item())
        self.log.add_item('policy_loss', policy_loss.cpu().detach().item())
        self.log.add_item('cost', torch.mean(reward, 1)[-1].cpu().detach().item())

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def run_episode(self, use_logn=False, with_actions=False, true_act=None,
                    reset_nodes=False, use_intrinsic=False, use_clonning=False, with_proba=False):
        states, actions, masks, rewards, dones = [], [], [], [], []
        next_states = []
        env = self.env
        s = env.reset(reset_nodes)
        if with_actions:
            old_graph = dc(s)
        done = False
        worst_cost = 0
        i = 0
        entropy_episode = []

        while not done:

            mask = env.available_actions.copy()
            s = adj_to_pgeom(s, device=self._device,
                             mask=self.to_tensor(mask),
                             use_small_features=(self.ndim == 2)).to(self._device)
            states.append(s)
            if i > 0:
                next_states.append(s)
            val, a, dist = self.actor(s, mask=self.to_tensor(mask), with_dist=True,
                                      greedy=False, use_clonning=use_clonning)
            a = a.cpu().detach().item()
            if true_act is not None:
                print('Action a')
                a = true_act[i]
            s, r, done = env.step(a)
            worst_cost = np.maximum(worst_cost, r)

            reward = get_score_for_task(self.task, r, worst_cost, done, use_logn=use_logn)

            actions.append(a)
            masks.append(mask)
            reward_tensor = torch.FloatTensor([reward])
            if i == 0:
                if use_clonning:
                    policy_proba = dist.probs.unsqueeze(0)
                else:
                    policy_proba = dist.probs[a].unsqueeze(0)
                reward_episode = reward_tensor.unsqueeze(0)
                value_episode = val.unsqueeze(0)
                entropy_episode.append(dist)
            else:
                if use_clonning:
                    new_proba = dist.probs.unsqueeze(0)
                else:
                    new_proba = dist.probs[a].unsqueeze(0)
                policy_proba = torch.cat([policy_proba, new_proba], dim=0)
                reward_episode = torch.cat([reward_episode, reward_tensor.unsqueeze(0)], dim=0)
                value_episode = torch.cat([value_episode, val.unsqueeze(0)], dim=0)
                entropy_episode.append(dist)
            i += 1
        tot_return = reward_episode.sum().item()
        value_episode = torch.cat([value_episode, torch.zeros_like(val).unsqueeze(0)], dim=0)

        value_episode = value_episode.squeeze()
        reward_episode = reward_episode.squeeze()
        if self.task == 'mvc':
            policy_proba, reward_episode, value_episode = \
                self._pad_transition([policy_proba, reward_episode, value_episode])
            value_episode = F.pad(value_episode, pad=[0, 1], mode='constant', value=0)

        if self.sil_model is not None:
            deltas = reward_episode + self.gamma * value_episode.cpu()[1:] - value_episode.cpu()[:-1]
            advantage = deltas.squeeze() @ self.gam_lam_matrix[:self.episode_len, :self.episode_len].cpu()
            sample = [states,
                      actions,
                      masks,
                      (self.gam_matrix.cpu() @ reward_episode).squeeze(),
                      advantage.squeeze().detach().numpy()
                      ]
            self.sil_model.replay_buffer.push_episode(sample)

        if with_actions and with_proba:
            return old_graph, actions, entropy_episode
        elif with_actions:
            return old_graph, actions
        reward_episode = reward_episode.to(self._device)
        if use_intrinsic:
            return policy_proba, reward_episode, value_episode, tot_return, \
                   torch_geometric.data.Batch.from_data_list(states[:-1]).to(self._device), \
                   torch_geometric.data.Batch.from_data_list(next_states).to(self._device), \
                   torch.from_numpy(np.array(actions)[:-1]).long().to(self._device)
        return policy_proba, reward_episode, value_episode, tot_return

    @staticmethod
    def rescale_advantages(advantages):
        adv_centered = advantages - advantages.mean()
        advantages = adv_centered / (advantages.std() + 1e-6)
        return advantages

    def actor_update(self, loss):
        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

    def _pad_transition(self, transition):
        new_transition = []
        for el in transition:
            new_transition.append(F.pad(el.squeeze(), pad=(0, self.env.state_size - len(el)), mode='constant', value=0))
        return new_transition

    def train(self, reset_nodes=False, use_clonning=False):
        mean_return = 0
        all_states, all_next_states, all_actions = [], [], []
        if use_clonning:
            import pickle
            with open('peo_for_pace.pd', 'rb') as fout:
                true_act = np.array(pickle.load(fout)) - 1
                all_actions = torch.LongTensor(true_act).repeat(32).cuda()
        for i in range(self.num_episodes):
            if self.use_intrinsic:
                pi, r, v, tot_return, states, next_states, actions = self.run_episode(use_logn=self.use_log,
                                                                                      reset_nodes=reset_nodes,
                                                                                      use_intrinsic=True)
            else:
                pi, r, v, tot_return = self.run_episode(use_logn=self.use_log,
                                                        reset_nodes=reset_nodes,
                                                        true_act=None,
                                                        use_clonning=False)
            mean_return = mean_return + tot_return

            if (i == 0):
                policy_proba = pi.unsqueeze(0)
                reward_ = r.unsqueeze(0)
                V = v.unsqueeze(0)
                if self.use_intrinsic:
                    all_actions = actions.unsqueeze(0)
            else:
                policy_proba = torch.cat([policy_proba, pi.unsqueeze(0)], dim=0)
                reward_ = torch.cat([reward_, r.unsqueeze(0)], dim=0)
                V = torch.cat([V, v.unsqueeze(0)], dim=0)
            if self.use_intrinsic:
                all_states.append(states)
                all_next_states.append(next_states)
                if i > 0:
                    all_actions = torch.cat([all_actions, actions.unsqueeze(0)], dim=0)

        mean_return = mean_return / self.num_episodes

        self.update_model(policy_proba, reward_, V, \
                          states=all_states, next_states=all_next_states, actions=all_actions)
        if self.sil_model is not None:
            self.sil_model.train_sil_model()
        self.log.add_item('train_reward', mean_return)
        return self.log

    @torch.no_grad()
    def eval_treewidth(self, with_minfill=False, verbose=False, reset_nodes=False, heuristic_type='min_fill', with_peo=False):
        self.actor.eval()
        old_graph, actions = self.run_episode(use_logn=self.use_log, with_actions=True, reset_nodes=reset_nodes)
        self.actor.train()
        old_graph = nx.from_scipy_sparse_matrix(old_graph)
        old_graph_minfill_copy = dc(old_graph)
        treewidth_cost = gm.get_treewidth_from_peo(old_graph, actions)
        if with_minfill: # put the wait time
            graph_indices, reward_minfill = gm.get_upper_bound_peo(old_graph_minfill_copy, method=heuristic_type)
            if verbose:
                print('actor', actions[:10])
                print('heuristic', graph_indices[:10])
            return treewidth_cost, reward_minfill

        if with_peo:
            return treewidth_cost, actions
        return treewidth_cost

    @torch.no_grad()
    def eval_mvc(self, with_minfill=False, verbose=False, reset_nodes=False, heuristic_type='min_fill', with_peo=False):
        from networkx.algorithms.approximation import min_weighted_vertex_cover

        old_graph, actions = self.run_episode(use_logn=False, with_actions=True, reset_nodes=reset_nodes)
        old_graph = nx.from_numpy_array(old_graph)
        old_graph_minfill_copy = dc(old_graph)
        agent_score = len(actions)
        heuristic_score = min_weighted_vertex_cover(old_graph_minfill_copy.to_undirected())
        if verbose:
            print(agent_score, len(heuristic_score))
        if with_minfill:
            return agent_score, len(heuristic_score)
        else:
            return agent_score

    @torch.no_grad()
    def eval_proba(self, with_minfill=False):
        old_graph, actions, dists = self.run_episode(use_logn=False, with_actions=True, with_proba=True)
        entropy = [d.entropy().item() for d in dists]

        old_graph = nx.from_numpy_array(old_graph)
        old_graph_minfill_copy = dc(old_graph)

        treewidth_cost = gm.get_treewidth_from_peo(old_graph, actions)
        if with_minfill:
            tamaki_tw = gm.get_upper_bound_peo(old_graph_minfill_copy, method='tamaki', wait_time=60)
            return treewidth_cost, actions, entropy, tamaki_tw

        return treewidth_cost, actions, entropy


def get_score_for_task(task, score, prev_best_score, done, use_logn=True):
    if task == 'treewidth':
        if done:
            reward = prev_best_score
        elif use_logn:
            reward = np.log(score + 1) if score >= 1 else 0
            # reward = reward - np.log(prev_best_score + 1)
            # reward = score - prev_best_score
        else:
            reward = 0
    else:
        reward = score
    return -reward


if __name__ == '__main__':
    args = init_config()
    # fix seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
        p=args.proba if not args.eval_mode else 50 * args.proba / MAX_STATE_SIZE,
        random_graph_type=args.random_graph_type,
        tw=args.treewidth,
        use_pace=args.use_pace
    )

    MAX_STATE_SIZE = env.state_size
    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    if args.resume_path is not None:
        exp_gamma = 0.99
        lr = 1e-2
    else:
        lr = args.lr
        exp_gamma = 0.999

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
                      gamma=0.99,
                      gae_lambda=0.9,
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
    print(alg)


    if args.eval_mode:
        gr = nx.Graph()
        gr.add_nodes_from(list(range(1,8)))
        gr.add_edges_from([(1, 2), (1, 3), (2, 3),
                           (2, 4), (3, 5), (4, 5),
                           (4, 6), (4, 7), (6, 7)])

        # gr.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'c'),
        #                    ('b', 'd'), ('c', 'e'), ('d', 'e'),
        #                    ('d', 'f'), ('d', 'g'), ('f', 'g')])
        #
        # gr.add_edges_from([(1, 2), (2, 3),
        #                    (2, 4), (3, 5), (4, 5),
        #                     (4, 7), (5, 6)])

        # gr.add_edges_from([(1, 2), ('b', 'c'),
        #                    ('b', 'd'), ('c', 'e'), ('d', 'e'),
        #                    ('d', 'g'), ('e', 'f')])
        # alg.env.row_to_node = {1:'a', 2: 'b',
        #                        3: 'c', 4:'d',
        #                        5:'e', 6:'f',
        #                        7:'g'}
        alg.env.reset(reset_nodes=True, graph=gr, new_size=7)

        alg.env.state_size = gr.number_of_nodes()
        alg.env.node_map = np.arange(1, alg.env.state_size + 1)

        agent_tr, minfill_score = alg.eval(with_minfill=True, reset_nodes=False,
                                           heuristic_type='min_fill', verbose=True)

        gr = nx.Graph()
        gr.add_nodes_from(list(range(1, 8)))
        gr.add_edges_from([(1, 2), (1, 3), (2, 3),
                           (2, 4), (3, 5), (4, 5),
                           (4, 6), (4, 7), (6, 7)])
        alg.env.reset(reset_nodes=True, graph=gr, new_size=7)

        alg.env.state_size = gr.number_of_nodes()
        alg.env.node_map = np.arange(1, alg.env.state_size + 1)

        agent_tr, minfill_score = alg.eval(with_minfill=True, reset_nodes=False,
                                           heuristic_type='min_fill', verbose=True)

        gr = nx.Graph()
        gr.add_nodes_from(list(range(1, 8)))
        gr.add_edges_from([(1, 2), (1, 3), (2, 3),
                           (2, 4), (3, 5), (4, 5),
                           (4, 6), (4, 7), (6, 7)])
        alg.env.reset(reset_nodes=True, graph=gr, new_size=7)

        alg.env.state_size = gr.number_of_nodes()
        alg.env.node_map = np.arange(1, alg.env.state_size + 1)

        agent_tr, minfill_score = alg.eval(with_minfill=True, reset_nodes=False,
                                           heuristic_type='tamaki', verbose=True)

        print(agent_tr, minfill_score)
        exit()