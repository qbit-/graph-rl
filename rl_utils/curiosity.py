import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool

from utils import create_optimal_inner_init, out_init


class IntrinsicReward(nn.Module):
    def __init__(self, ndim=2, hidden_dim=8, device=None, activation_fn=nn.ELU):
        super(IntrinsicReward, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        self.reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.value.apply(out_init)

        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        if batch:
            mean_val = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            mean_val = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))

        value = self.value(mean_val)
        return value


class InverseModel(nn.Module):
    """
    This model take a state feature s_t and s_{t+1} and predict action vector
    """

    def __init__(self, n_actions, state_latent_features):
        super(InverseModel, self).__init__()
        # self.feat_conv = GCNConv(2*hidden_dim, hidden_dim)
        # self.fc = nn.Linear(hidden_dim, n_actions)
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions)
        )

    def forward(self, state_latent, next_state_latent):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class ForwardModel(nn.Module):
    def __init__(self, n_actions, indim, hidden_dim):
        super(ForwardModel, self).__init__()
        self.indim = indim
        self.n_actions = n_actions
        self.fc = nn.Linear(indim + n_actions, hidden_dim)
        # self.eye = torch.eye(n_actions)

    def forward(self, features, action):
        # print(self.eye[action].size(), features.size(), action)
        x = torch.cat([F.one_hot(action, num_classes=int(self.n_actions)).float().to(features.device),
                       features.view(-1, self.indim)], dim=-1)
        features = self.fc(x)  # (1, hidden_dims)
        return features


class FeatureExtractor(nn.Module):
    def __init__(self, ndim, hidden_dims):
        super(FeatureExtractor, self).__init__()
        self.fc1 = GCNConv(ndim, hidden_dims)
        self.fc2 = GCNConv(hidden_dims, hidden_dims)

        # self.fc = nn.Linear(1,)

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index
        x = self.fc1(x, edge_index)
        x = F.elu(x)

        x = self.fc2(x, edge_index)
        batch = data.batch is not None
        if batch:
            y = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            y = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))
        return y


class ICM(nn.Module):
    def __init__(self, ndim, hidden_dim, n_act, reward_scale=1, intrinsic_reward_lmbda=0.3, policy_weight=1.2):
        """

         Implements the Intrinsic Curiosity Module described in paper: https://arxiv.org/pdf/1705.05363.pdf
        The overview of the idea is to reward the agent for exploring unseen states. It is achieved by implementing two
        models. One called forward model that given the encoded state and encoded action computes predicts the encoded next
        state. The other one called inverse model that given the encoded state and encoded next_state predicts action that
        must have been taken to move from one state to the other. The final intrinsic reward is the difference between
        encoded next state and encoded next state predicted by the forward module. Inverse model is there to make sure agent
        focuses on the states that he actually can control.

        :param ndim:
        :param hidden_dim:
        :param n_act: number of availible actions
        :param intrinsic_reward_lmbda: balances the importance between extrinsic and intrinsic reward.
             Used when incorporating intrinsic into extrinsic in the ``reward`` method
        :param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method.
              Allows to control how important optimizing policy to optimizing the curiosity module
        :param reward_scale: scales the intrinsic reward returned by this module.
                Can be used to control how big the intrinsic reward is
        """
        super(ICM, self).__init__()
        self.feature_extractor = FeatureExtractor(ndim, hidden_dim)
        self.forward_model = ForwardModel(n_actions=n_act, indim=hidden_dim, hidden_dim=hidden_dim)
        self.inverse_model = InverseModel(n_act, hidden_dim)
        self.reward_scale = reward_scale
        self.intrinsic_reward_lmbda = intrinsic_reward_lmbda

        self.action_loss = nn.CrossEntropyLoss()
        self.loss_weight = 0.5
        self.fi_weight = 0.6

    def forward(self, state, next_state, action):
        f_state = self.feature_extractor(state).squeeze(-1)
        f_next_state = self.feature_extractor(next_state).squeeze(-1)
        next_state_hat = self.forward_model(f_state, action)
        action_hat = self.inverse_model(f_state, f_next_state)
        return f_next_state, next_state_hat, action_hat

    def reward(self, actions, states, next_states):
        intrinsic_rewards = []

        for i in range(len(actions)):
            next_states_latent, next_states_hat, _ = self.forward(states[i], next_states[i], actions[i])
            intrinsic_rewards.append(
                self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2))

        return -1 * torch.stack(intrinsic_rewards)

    def loss(self, policy_loss, states, next_states, actions):
        forward_loss = 0
        inverse_loss = 0
        for i in range(len(actions)):
            next_states_latent, next_states_hat, actions_hat = self.forward(states[i], next_states[i], actions[i])

            forward_loss += 0.5 * (next_states_hat - next_states_latent.detach()).norm(2, dim=-1).pow(2).mean()
            inverse_loss += self.action_loss(actions_hat, actions[0])

        curiosity_loss = self.fi_weight * forward_loss + (1 - self.fi_weight) * inverse_loss
        # print('Curiosity loss', curiosity_loss.item())

        return policy_loss + self.loss_weight * curiosity_loss


if __name__ == '__main__':
    ndim = 10
    import networkx as nx

    n = 30
    graph = nx.erdos_renyi_graph(n, p=0.2)
    from utils import adj_to_pgeom

    state = adj_to_pgeom(nx.to_numpy_array(graph))
    graph = nx.erdos_renyi_graph(n, p=0.2)
    next_state = adj_to_pgeom(nx.to_numpy_array(graph))
    a = 1
    encoder = FeatureExtractor(2, n)
    forward = ForwardModel(n, n, 10)
    inverse = InverseModel(30, 30)
    encode_state = encoder(state).squeeze()
    encode_next_state = encoder(state).squeeze()
    print(encode_state.size())
    fin = forward(encode_state, [1])
    fout = inverse(encode_state, encode_next_state)
    print(fin.size(), fout.size())

    out_icm = ICM(2, 24, 30)
    print(out_icm(state, next_state, [1])[2].size())
