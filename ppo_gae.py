import numpy as np
import torch
import torch.optim as optim
from rl_utils import  create_gamma_matrix
import utils


class PPO_GAE:
    def __init__(
            self,
            actor, critic,
            gamma, gae_lambda=0.95,
            lr=0.0001,
            clip_eps=0.2,
            episode_len=1000,
            minibatch_size=32,
            num_mini_epochs=10,
            use_value_clipping=False,
            entropy_reg_coefficient=0.001,
            device=torch.device('cpu'),
            use_gcn = False
            ):
        self._device = device
        self.is_graph = use_gcn

        self.actor = actor.to(self._device)
        print(self._device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=lr)

        if not self.is_graph:
            self.critic = critic.to(self._device)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=lr)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.episode_len = episode_len
        self.num_mini_epochs = num_mini_epochs
        self.minibatch_size = minibatch_size
        self.use_value_clipping = use_value_clipping
        self.entropy_reg_coefficient = entropy_reg_coefficient
        # self.gam_lam_matrix = create_gamma_matrix(
        #     self.gamma * gae_lambda, episode_len)
        self.gam_matrix = create_gamma_matrix(
            self.gamma, episode_len)

    def to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def evaluate_episode(self, states, actions, masks, rewards):
        # Default regime for gcn
        self.actor = self.actor.to(self._device)
        if self.is_graph:
            states = states.to(self._device)
        else:
            states = self.to_tensor(states)

        actions = self.to_tensor(actions)
        masks = self.to_tensor(masks)
        rewards = np.array(rewards)
        ep_len = rewards.shape[0]
        # if self.is_graph:
        #     values[:ep_len] = self.critic(X=I, states=states)
        # else:
        values = torch.zeros((ep_len + 1, 1)).to(self._device)

        if self.is_graph:
            # values = values.squeeze()
            values[:ep_len], _, dist = self.actor(states, mask=masks, with_dist=True, batch=True)
            values = values.detach().cpu().numpy().reshape(-1)
        else:
            values[:ep_len] = self.critic(states)
            _, dist = self.actor(states=states, mask=masks, with_dist=True)
            values = values.detach().cpu().numpy().reshape(-1)
        log_pis = dist.log_prob(actions)
        log_pis = log_pis.detach().cpu().numpy().reshape(-1)
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        # advantages = rewards - values[1:]
        advantages = np.dot(self.gam_lam_matrix[:ep_len, :ep_len], deltas)
        returns = np.dot(self.gam_matrix[:ep_len, :ep_len], rewards)
        return [returns, values[:ep_len], advantages, log_pis]

    def train(self, batch):
        states, actions, masks, returns, values, advantages, log_pis = \
            batch["state"], batch["action"], batch["mask"], batch["return"], \
            batch["value"], batch["advantage"], batch["log_pi"]
        if not self.is_graph:
            states = self.to_tensor(states)
        else:
            states = states.to(self._device)
        actions = self.to_tensor(actions)
        masks = self.to_tensor(masks)
        returns = self.to_tensor(returns)
        old_values = self.to_tensor(values)
        advantages = self.to_tensor(advantages)
        old_log_pi = self.to_tensor(log_pis)

        self.actor.train()

        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()

        # actor loss
        if self.is_graph:
            # masks.requires_grad_(True)
            values_t, _, dist = self.actor(states=states, mask=masks, greedy=False, with_dist=True, batch=True)
        else:
            _, dist = self.actor(states, masks, with_dist=True)
            values_t = self.critic(states).squeeze()

        if self.use_value_clipping:
            values_clip = old_values + torch.clamp(
                values_t - old_values, -self.clip_eps, self.clip_eps)
            val_loss1 = (values_t - returns).pow(2)
            val_loss2 = (values_clip - returns).pow(2)
            value_loss = 0.5 * torch.max(val_loss1, val_loss2).mean()
        else:
            value_loss = 0.5 * (values_t - returns).pow(2).mean()

        log_pi = dist.log_prob(actions)
        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(
            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = -(torch.exp(log_pi) * log_pi).mean()
        entropy_reg = self.entropy_reg_coefficient * entropy

        # log_pi.register_hook(lambda grad: print('log_pi',grad.sum()))
        # ratio.register_hook(lambda grad: print('ratio',grad.sum()))
        # self.actor.policy[0].weight.register_hook(lambda grad: print(grad.sum()))

        policy_loss = policy_loss + entropy_reg

        # actor update
        loss = policy_loss + value_loss
        self.actor_update(loss)

        if not self.is_graph:
            self.critic_update(value_loss)

        metrics = {
            "loss_actor": policy_loss.item(),
            "loss_critic": value_loss.item()
        }
        return metrics

    def actor_update(self, loss):
        loss.backward()
        self.actor_optimizer.step()

    def critic_update(self, loss):
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()