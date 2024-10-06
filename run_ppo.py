import multiprocessing
from joblib import Parallel, delayed

import time
import torch
import math
import gym
import random
import copy
import numpy as np
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import graph_model as gm

from arguments import init_config
from tensorboardX import SummaryWriter

from graph_policy import *
from logger import Logger
from graph_environment import Environment, degree_cost, contraction_cost_flops

# from models import
from ppo_gae import PPO_GAE
from simple_models import Actor, Critic

from itertools import tee
from utils import *
from rl_utils import BufferDataset, BufferSampler, _collate_fn
from torch.utils.data import DataLoader as tDataLoader

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['CUDA_LAUNCH_BLOCKING']= '1'


def name2nn(name):
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    else:
        return name


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def to_tensor(*args, **kwargs):
    return torch.Tensor(*args, **kwargs)#.to(device)


def train(args):
    global cpu_device
    cpu_device = torch.device('cpu')

    tb_writer = SummaryWriter('runs/' + args.exp_name)
    logger = Logger(name=args.exp_name)
    # TODO move parameters to arguments
    env_name = args.graph_path
    costs = {'treewidth': degree_cost, 'flops': degree_cost}

    global MAX_STATE_SIZE
    MAX_STATE_SIZE = args.max_shape
    state_shape = (MAX_STATE_SIZE, MAX_STATE_SIZE)

    ndim = 2 if args.use_small_features else state_shape[0]

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

    batch_size = args.batch_size
    lr = 1e-1
    num_mini_epochs = 6
    minibatch_size = args.minibatch_size
    # state_shape = (15, 15)
    if args.gcn:
        conv_hiddens = [32, 32,32]
    else:
        conv_hiddens = [32, 64, 64]

    linear_hiddens = [256, 128]
    # Create RL module for training
    if args.gcn:
        policy = GCNNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
        actor = GraphActor(policy=policy)
        critic = None
    else:
        actor = Actor(
            output_size=MAX_STATE_SIZE,
            conv_hiddens=conv_hiddens,
            out_embd=args.embd_size,
            state_shape=[MAX_STATE_SIZE, MAX_STATE_SIZE],
            linear_hiddens=linear_hiddens, gcn=args.gcn)
        critic = Critic(
            output_size=1,
            conv_hiddens=conv_hiddens,
            out_embd=args.embd_size,
            state_shape=[MAX_STATE_SIZE, MAX_STATE_SIZE],
            linear_hiddens=linear_hiddens, gcn=args.gcn).to(device)
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=lambda storage, loc: storage)
        actor.load_state_dict(checkpoint['actor'])

    print(actor)
    algo = PPO_GAE(
        actor, critic,
        gamma=0.99, gae_lambda=0.2,#0.95,
        minibatch_size=minibatch_size,
        num_mini_epochs=num_mini_epochs,
        device=device, lr=lr, use_gcn=args.gcn)


    num_episodes = 0
    avg_reward = 0
    sum_actor_loss = 0
    sum_critic_loss = 0

    actor_losses = []
    critic_losses = []
    total_rewards = []
    episode_rewards = []
    max_epoch = args.max_epoch

    minfill_graph_indices, reward_minfill = gm.get_upper_bound_peo(env.initial_graph)

    for epoch in tqdm(range(1, max_epoch)):

        buffer = BufferDataset(state_shape, batch_size, is_graph_data=args.gcn)
        if args.use_parallel:
            pool = Parallel(5)

            def parallel_episode_play(x):
                episode = play_episode(env, algo.actor, args.use_neighbour, use_gcn=args.gcn)
                ret, val, adv, log_pi = algo.evaluate_episode(
                    torch_geometric.data.Batch.from_data_list(episode[0]).to(device),
                    episode[1], episode[2], episode[3])
                data = [episode[0], episode[1], episode[2], ret, val, adv, log_pi]
                # episode_rewards.append(np.sum(episode[3]))

                return data

            F = pool(delayed(parallel_episode_play)(j) for j in range(batch_size))

            for data in F:
                buffer.push_episode(data)

            num_episodes = math.ceil(batch_size / MAX_STATE_SIZE)

        else:
            while len(buffer) < batch_size:
                # states, actions, masks, rewards, dones
                episode = play_episode(env, algo.actor, args.use_neighbour, use_gcn=args.gcn, ndim=ndim)

                if args.gcn:
                    ret, val, adv, log_pi = algo.evaluate_episode(
                        torch_geometric.data.Batch.from_data_list(episode[0]).to(device),
                        episode[1], episode[2], episode[3])
                else:

                    ret, val, adv, log_pi = algo.evaluate_episode(
                        episode[0], episode[1], episode[2], episode[3])
                data = [episode[0], episode[1], episode[2], ret, val, adv, log_pi]
                buffer.push_episode(data)
                episode_rewards.append(np.sum(episode[3]))
                num_episodes += 1

        buffer.rescale_advantages()
        actor = actor.to(device)
        sampler = BufferSampler(
            buffer=buffer,
            num_mini_epochs=num_mini_epochs)
        if args.gcn:
            collate_fn = _collate_fn
        else:
            collate_fn = default_collate

        loader = tDataLoader(
            dataset=buffer,
            batch_size=minibatch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler, collate_fn=collate_fn)

        actor_loss = []
        critic_loss = []

        s = time.time()
        for i, batch in enumerate(loader):
            metrics = algo.train(batch)
            actor_loss.append(metrics["loss_actor"])
            critic_loss.append(metrics["loss_critic"])
        sum_actor_loss += np.mean(actor_loss)
        sum_critic_loss += np.mean(critic_loss)
        time_per_epoch = time.time() - s
        if epoch % args.log_every == 0:
            avg_rew = np.mean(episode_rewards)
            var_rew = np.var(episode_rewards)
            avg_actor_loss = sum_actor_loss
            avg_critic_loss = sum_critic_loss
            if args.debug:
                print("Actor loss:         ", avg_actor_loss)
                print("Critic loss:        ", avg_critic_loss)
                print("Average reward:     ", avg_rew)
                print("Num episodes:       ", num_episodes)
                print("--------------------------------------------")
            actor_losses.append(avg_actor_loss)
            critic_losses.append(avg_critic_loss)
            total_rewards.append(avg_rew)

            tb_writer.add_scalar('data/critic_loss', avg_critic_loss, epoch)
            tb_writer.add_scalar('data/actor_loss', avg_actor_loss, epoch)
            tb_writer.add_scalar('data/reward', avg_rew, epoch)

            logger.add_scalar(epoch, 'critic_loss', avg_critic_loss)
            logger.add_scalar(epoch, 'reward', avg_rew)
            logger.add_scalar(epoch, 'variance_reward', var_rew)
            logger.add_scalar(epoch, 'actor_loss', avg_actor_loss)
            logger.add_scalar(epoch, 'time_ep', time_per_epoch)
            logger.save()

            sum_actor_loss = 0
            sum_critic_loss = 0
        episode_rewards = []

        num_episodes = 0

        if epoch % 1 == 0 or args.debug:

            _, act_test,_, rev_test, _ = play_episode(env, algo.actor, False, use_gcn=args.gcn, greedy=False, ndim=ndim)
            print("Reward validation", -rev_test[-1])
            print("Actor peo: ",act_test[:10],act_test[-10:] )

            _, act_test,_, rev_test, _ = play_episode(env, algo.actor, False, use_gcn=args.gcn, greedy=True, ndim=ndim)

            print("Greedy Reward validation", -rev_test[-1])
            print("Actor peo: ",act_test[:10],act_test[-10:])
            print("Reward MinFill", reward_minfill)
            print("Minfill:", minfill_graph_indices[:10], minfill_graph_indices[-10:])
            if args.save_path is not None:
                ckpt_path = args.save_path + args.exp_name + '.pth'
                checkpoint = {}
                checkpoint['actor'] = algo.actor.state_dict()
                if not args.gcn:
                    checkpoint['critic'] = algo.critic.state_dict()
                torch.save(checkpoint, ckpt_path)


def play_episode(env, actor, use_logn=False, use_gcn=False, greedy=False, ndim=2):
    states, actions, masks, rewards, dones = [], [], [], [], []
    s = env.reset()
    actor.eval()
    done = False
    worst_cost = 0
    i = 0
    while not done:
        mask = env.available_actions.copy()
        if use_gcn:
            s = adj_to_pgeom(s, device=device, mask=to_tensor(mask).to(device), use_small_features=(ndim==2)).to(device)
            val, a, dist = actor(s, mask=to_tensor(mask).to(device), with_dist=True)
            states.append(s.to(cpu_device))
        else:
            states.append(s.tolist())
            I = None
            a, dist = actor.forward(X=I,
                                states=to_tensor(s[None, :, :, None]),
                                mask=to_tensor(mask[None, :]),
                                with_dist=True, greedy=greedy)
        a = a.cpu().detach().item()

        s, r, done = env.step(a)
        i+=1
        worst_cost = np.maximum(worst_cost, r)
        if done:
            reward = -worst_cost
        elif use_logn:
            reward = -np.log(r) if r > 1 else 0
        else:
            reward = 0

        actions.append(a)
        masks.append(mask.tolist())
        rewards.append(reward)
        dones.append(done)
    actor.train()
    actor = actor.to(device)

    return states, actions, masks, rewards, dones


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
    train(args)
