import torch
from torch.utils.data import DataLoader

import gym
import gym.spaces
import random
import copy
import numpy as np
from tqdm import tqdm
import qtree.src.graph_model as gm

from arguments import init_config
from tensorboardX import SummaryWriter

from pointnet.pointer import PointerNet
from qtree.src.rl_environment import Environment, degree_cost, contraction_cost_flops
from qtree.src.graph_model import get_peo

from simple_models import GCN
from utils import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['CUDA_LAUNCH_BLOCKING']= '1'


def to_tensor(*args, **kwargs):
    return torch.Tensor(*args, **kwargs).to(device)


def train(args):
    logger = SummaryWriter('runs/' + args.exp_name)
    # TODO move parameters to arguments
    env_name = args.graph_path
    costs = {'treewidth':degree_cost, 'flops':contraction_cost_flops}
    env = Environment(
        env_name,
        square_index=True,
        cost_function=costs[args.cost_function],
        simple_graph=True)
    MAX_STATE_SIZE = args.max_shape
    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)
    batch_size = args.batch_size
    lr = 1e-3

    num_mini_epochs = 6
    minibatch_size = 300
    print(device, state_shape[0])
    gcn = GCN(nfeat=state_shape[0], nhid=64, nclass=30, dropout=0.2, n_layers=4).to(device)

    pointer = PointerNet(30, 50, 1, 0.2, gcn=gcn).to(device)

    s = env.reset()
    I = np.eye(s.shape[0])
    I[~np.in1d(np.arange(s.shape[0]), env.minfill_indices)] = 0
    I = preprocess_features(I)
    I = to_tensor(I).unsqueeze(0).to(device)
    s = to_tensor(preprocess_adj(s)).unsqueeze(0).to(device)
    mask = I.max(-1)[0]
    # print(pointer(s, I, mask=mask)[1])
    # print(env.row_to_node.keys(), env.node_to_row.keys())
    # print(pointer_play_episode(env, pointer, batch_size)[-2])
    # exit()
    num_episodes = 0
    avg_reward = 0
    sum_actor_loss = 0
    sum_critic_loss = 0

    actor_losses = []
    critic_losses = []
    total_rewards = []
    episode_rewards = []
    max_epoch = args.max_epoch
    optimizer = torch.optim.Adam(pointer.parameters(), lr=lr)

    for epoch in tqdm(range(1, max_epoch)):

        # buffer = BufferDataset(state_shape, batch_size)
        # buffer = PointerBuffer(state_shape, batch_size)
        optimizer.zero_grad()
        try:
            actions, rewards, log_probs = pointer_play_episode(env, pointer, batch_size)
        except KeyError:
            print('KeyError')
            actions, rewards, log_probs = pointer_play_episode(env, pointer, batch_size, debug=False, greedy=True)
        # print(rewards)
        # print(log_probs)
        # print(log_probs.size())
        loss = -torch.mean(torch.mean(log_probs,dim=-1 )*rewards)
        loss.backward()
        optimizer.step()
        if epoch % args.log_every == 0:
            print(loss.item())
            print(np.mean(episode_rewards))
            print()
        episode_rewards.append(rewards.mean().item())
        # while len(buffer) < batch_size:
        #     episode = pointer_play_episode(env, pointer, batch_size)#algo.actor, args.use_neighbour, use_gcn=args.gcn)

            # TODO add value function
            # TODO add critic
            # ret, val, adv, log_pi = algo.evaluate_episode(
            #     episode[0], episode[1], episode[2], episode[3])
            # data = [episode[0], episode[1], episode[2], ret, val, adv, log_pi]
            # reward =
            # data = [log_pi, actions, reward]
            # buffer.push_episode(episode)
            # episode_rewards.append(np.sum(episode[3]))
            # num_episodes += 1
        # buffer.rescale_advantages()

        # sampler = BufferSampler(
        #     buffer=buffer,
        #     num_mini_epochs=num_mini_epochs
        # )
        # loader = DataLoader(
        #     dataset=buffer,
        #     batch_size=minibatch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers,
        #     pin_memory=torch.cuda.is_available(),
        #     sampler=sampler
        # )

        actor_loss = []
        critic_loss = []
        # for i, batch in enumerate(loader):
        #     metrics = algo.train(batch)
        #     actor_loss.append(metrics["loss_actor"])
        #     critic_loss.append(metrics["loss_critic"])
        # sum_actor_loss += np.mean(actor_loss)
        # sum_critic_loss += np.mean(critic_loss)


def pointer_play_episode(env, pointer, batch_size, debug=False, greedy=False):
    import qtree.src.graph_model as gm

    states, actions, masks, rewards, dones = [], [], [], [], []
    worst_cost = 0
    mask = env.available_actions.copy()
    eye_s = []
    masks = []
    true_minfills = []
    row_to_node_dicts = []
    env_graphs = []
    for i in range(batch_size):
        s = env.reset(True)
        I = np.eye(s.shape[0])
        I[~np.in1d(np.arange(s.shape[0]), env.minfill_indices)] = 0
        I = preprocess_features(I)
        # I = to_tensor(I).unsqueeze(0)
        # mask = I.max(-1)[0]
        mask = I.max(-1)
        masks.append(to_tensor(mask))
        eye_s.append(I)
        states.append(to_tensor(preprocess_adj(s)))
        true_minfills.append(env.minfill_indices)
        row_to_node_dicts.append(env.row_to_node)
        env_graphs.append(env.graph)
    # s = to_tensor().unsqueeze(0)
    # print(to_tensor(masks).size())
    # print(torch.stack(states, dim=0).size())
    # print(to_tensor(eye_s).size())

    log_probs, actions = pointer(torch.stack(states, dim=0).to(device),
                                 to_tensor(eye_s).to(device),
                                 mask=torch.stack(masks).to(device),
                                 greedy=greedy,
                                 debug=debug)
    np_actions = actions.cpu().numpy()
    # print(np_actions[:, :10])
    # print(true_minfills)
    # print(np_actions)
    for i, (acts, dict_path) in enumerate(zip(np_actions, row_to_node_dicts)):
        path = acts[:len(env.minfill_indices)]
        nodes = [dict_path[idx] for idx in path]
        # except KeyError:
        #     print(dict_path, acts, masks[i])
        rewards.append(-gm.get_treewidth_from_peo(env_graphs[i], nodes) )
        # log_probs[i] = log_probs[i]*torch.ge(masks[i].sum(-1), torch.tensor(i + 1, dtype=torch.float)).float()

    # path = actions.numpy()[0][:len(env.minfill_indices)]
    # print(env.row_to_node.keys())
    # print(path)
    # rewards = np.array(gm.get_treewidth_from_peo(env.graph,  nodes))
    return actions, to_tensor(rewards), log_probs
    # print(pointer(s, I, mask=mask)[1].numpy() )
    # print(env.minfill_indices)
    # path = pointer(s, I, mask=mask)[1].numpy()[0][:len(env.minfill_indices)]



if __name__ == '__main__':
    args = init_config()
    # fix seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    global device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda:", use_cuda)

    train(args)