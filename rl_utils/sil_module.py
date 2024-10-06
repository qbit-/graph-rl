import numpy as np
import torch

from rl_utils import BufferSampler
from rl_utils.buffer import BufferDataset, PriorityBufferDataset, _collate_fn_geom_weights, PrioritisedReplaySampler
from rl_utils.buffer import _collate_fn_geom
from torch.utils.data import DataLoader
import torch.nn.utils as F

from rl_utils import to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SIL:
    def __init__(self, network, optimizer, state_shape, use_priority=True):
        """
        Self-Imitation Learning
        https://arxiv.org/abs/1806.05635
        Module which use episodes from training A2C
        to improve stability and convergence
        :param network: actor which is used
        :param optimizer: optimizer of A2C for actor step
        :param state_shape: shape of
        :param use_priority: if True we use priority buffer
        """
        self.actor = network
        self.running_episodes = []
        self.entropy_reg_coefficient = 0.001
        self.optimizer = optimizer
        capacity = 512
        self.num_mini_epochs = 2
        self.K_epochs_sil = 3
        self.mini_batch_size = 128
        self.clip = 0.9
        self.max_grad_norm = 5.0

        self.total_steps = []
        self.total_rewards = []
        self.use_priority = use_priority
        if self.use_priority:
            self.replay_buffer = PriorityBufferDataset((state_shape, state_shape), capacity, alpha=0.2)
        else:
            self.replay_buffer = BufferDataset((state_shape, state_shape), capacity, is_graph_data=True)

    def train_sil_model(self):

        replay_sampler = PrioritisedReplaySampler if self.use_priority else BufferSampler
        sampler = replay_sampler(
            buffer=self.replay_buffer,
            num_mini_epochs=self.num_mini_epochs)

        collate_fn_ = _collate_fn_geom_weights if self.use_priority else _collate_fn_geom
        loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.mini_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler, collate_fn=collate_fn_)

        num_valid_samples = 0
        mean_adv = 0
        for n, batch in enumerate(loader):
            if n == self.K_epochs_sil:
                return mean_adv, num_valid_samples
            obs = batch['state']
            actions = to_tensor(batch['action']).to(device)
            masks, returns = to_tensor(batch["mask"]), to_tensor(batch["return"]).to(device)
            if self.use_priority:
                weights, idxes = to_tensor(batch['weights']).to(device), batch['idxes']
            else:
                weights = 1
            # advantages = to_tensor(batch['advantage'])
            mean_adv, num_valid_samples = 0, 0
            value, _, dist = self.actor(obs.to(device), mask=masks.to(device), with_dist=True, batch=True)
            advantages = returns - value
            advantages = self.rescale_advantages(advantages).detach()

            masks = advantages > 0

            log_pi = (weights*dist.log_prob(actions)).masked_select(masks)
            entropy = (weights*dist.entropy()).masked_select(masks)
            masked_advantages = advantages.masked_select(masks)

            action_loss = torch.mean(masked_advantages * log_pi)
            entropy_reg = torch.mean(entropy)
            policy_loss = action_loss - entropy_reg * self.entropy_reg_coefficient
            value_loss = torch.mean((returns - value).masked_select(masks)**2)
            total_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            # F.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            num_valid_samples += torch.sum(masks)
            mean_adv += torch.mean(advantages)

            if self.use_priority:
                self.replay_buffer.update_priorities(idxes, advantages.clamp_min_(0).sum(dim=-1).cpu().numpy())
        return mean_adv, num_valid_samples

    @staticmethod
    def rescale_advantages(advantages):
        adv_centered = advantages - advantages.mean()
        advantages = adv_centered / (advantages.std() + 1e-6)
        return advantages