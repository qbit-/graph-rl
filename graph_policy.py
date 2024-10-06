import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, ARMAConv, GATConv, RGCNConv, SGConv, \
    global_max_pool, ChebConv, GatedGraphConv, AGNNConv, GINConv, Set2Set, global_add_pool
from torch_geometric.nn.pool import TopKPooling, SAGPooling

from misc.gcn_modified import GCNConvModified, GCNConvAdvanced
from utils import create_optimal_inner_init, out_init


#GCNConvModified
# GCNConvAdvanced
class GCNNet(nn.Module):
    def __init__(self, ndim=2, hidden_dim=8, device=None, is_prefeature=True,
                 activation_fn=nn.ELU, num_layers=3, gcn_type=GCNConv):
        super(GCNNet, self).__init__()
        self.is_prefeature = is_prefeature
        if self.is_prefeature:
            self.prefeature = nn.Sequential(
                nn.Linear(ndim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )

            ndim = 64

        self.conv_layers = nn.ModuleList([
            gcn_type(ndim, hidden_dim),
        ])

        for i in range(num_layers):
            self.conv_layers.append(gcn_type(hidden_dim, hidden_dim))

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )

        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index
        if self.is_prefeature:
            x = self.prefeature(x)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            x = F.elu(x)

        if batch:
            mean_val = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            mean_val = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))

        proba = self.policy(x)
        value = self.value(mean_val)
        return proba, value


class GINNet(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None, activation_fn=nn.ELU, num_layers=2):
        super(GINNet, self).__init__()
        nn1 = nn.Sequential(nn.Linear(ndim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv_layers = nn.ModuleList([
            GINConv(nn1),
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
        ])
        for i in range(num_layers):
            nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

            self.conv_layers.append(GINConv(nn1))
            self.norm.append(nn.LayerNorm(hidden_dim))

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )

        # inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        # self.conv_layers.apply(inner_init)
        # self.policy.apply(out_init)
        # self.value.apply(out_init)
        self.device = device

class GcnPolicy(nn.Module):
    def __init__(self, ndim=2, hidden_dim=8, device=None, activation_fn=nn.ELU, num_layers=3):
        super(GcnPolicy, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
        ])
        for i in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.value = GCNConv(hidden_dim, 1)
        self.policy = GCNConv(hidden_dim, 1)

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        if batch:
            # mean_val = global_mean_pool(x, batch=data.batch.to(x.device))
            mean_val = self.value(x, edge_index)
            value = global_mean_pool(mean_val, batch=data.batch.to(x.device))

        else:
            mean_val = self.value(x, edge_index)
            value = global_mean_pool(mean_val, torch.zeros(mean_val.size()[0], device=x.device, dtype=torch.long))

        proba = self.policy(x, edge_index)
        return proba, value


class GATNet(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(GATNet, self).__init__(ndim, hidden_dim, device)
        k = 4
        print('HERE')
        self.conv_layers = nn.ModuleList([
            GATConv(ndim, hidden_dim, heads=k),
            GATConv(k * hidden_dim, hidden_dim, heads=k, dropout=0.2),
            GATConv(k * hidden_dim, hidden_dim, heads=k, dropout=0.2)
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(k*hidden_dim),
            nn.LayerNorm(k*hidden_dim),
            nn.LayerNorm(k*hidden_dim),
        ])

        self.policy = nn.Sequential(
            nn.Linear(k*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(k*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.conv_layers):
            x = layer(x, edge_index)
            x = F.elu(x)
            x = self.norm[i](x)
        if batch:
            value = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            # mean_val = self.value(x, edge_index)
            value = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))
        proba = self.policy(x)
        value = self.value(value)
        return proba, value


class GATisoNet(nn.Module):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(GATisoNet, self).__init__()
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
        ])

        self.value = GINConv(nn1)
        self.policy = GINConv(nn2)

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.policy.apply(out_init)
        self.value.apply(out_init)
        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x=x, edge_index=edge_index)
        if batch:
            mean_val = self.value(x, edge_index)

            value = global_mean_pool(mean_val, batch=data.batch.to(x.device))
        else:
            mean_val = self.value(x, edge_index)
            value = global_mean_pool(mean_val, torch.zeros(mean_val.size()[0], device=x.device, dtype=torch.long))
        proba = self.policy(x, edge_index)
        # value = self.value(mean_val, )
        return proba, value


class RGCNNet(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None,  activation_fn=nn.ELU,):
        super(RGCNNet, self).__init__(ndim, hidden_dim, device=device)
        k = 5
        self.conv_layers = nn.ModuleList([
            SGConv(ndim, hidden_dim, K=k),
        ])

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.policy.apply(out_init)
        self.value.apply(out_init)
        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
        if batch:
            mean_val = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            mean_val = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))
        proba = self.policy(x)
        value = self.value(mean_val)
        return proba, value


class ChebNet(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(ChebNet, self).__init__(ndim, hidden_dim, device=device)
        self.conv_layers = nn.ModuleList([
            ChebConv(ndim, hidden_dim, K=3),
            ChebConv(hidden_dim, hidden_dim, K=3),
            ChebConv(hidden_dim, hidden_dim, K=3),
            ChebConv(hidden_dim, hidden_dim, K=3),
        ])

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)


class GatedNet(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(GatedNet, self).__init__(ndim, hidden_dim, device=device)
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GatedGraphConv(hidden_dim, 1),
        ])

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.policy.apply(out_init)
        self.value.apply(out_init)

        self.device = device


class SagePolicyNetwork(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(SagePolicyNetwork, self).__init__(ndim, hidden_dim, device)
        self.conv_layers = nn.ModuleList([
            SAGEConv(ndim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim)
        ])

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)


class ARMAPolicyNetwork(GCNNet):
    def __init__(self, ndim=2, hidden_dim=8, device=None):
        super(ARMAPolicyNetwork, self).__init__(ndim, hidden_dim, device)
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim,improved=True),
            GCNConv(hidden_dim, hidden_dim, improved=True),
            ARMAConv(hidden_dim, hidden_dim, num_stacks=3,
                     num_layers=2, shared_weights=True, dropout=0.1, act=None),
            ARMAConv(hidden_dim, hidden_dim, num_stacks=3,
                     num_layers=2, shared_weights=True, dropout=0.1, act=None),
        ])

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)


class GraphActor(nn.Module):
    def __init__(self, policy=None, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def forward(self, states, mask, with_dist=False, greedy=False, batch=False, use_probs=False, use_clonning=False):
        logits, value = self.policy.forward(states, batch)
        unmasked_probs = F.softmax(logits.view(mask.size()), dim=-1)
        probs = (unmasked_probs + 1e-6) * mask  # fix hint
        if use_clonning:
            probs = unmasked_probs + 1e-6
        probs = probs / probs.sum(dim=-1)[..., None]
        dist = torch.distributions.Categorical(probs=probs)
        # assert not greedy
        if greedy:
            _, action = torch.max(probs, -1)
        else:
            action = dist.sample()
        if use_probs:
            return value, action, probs[:, action]
        if with_dist:
            return value, action, dist
        return value, action


class GCPoolNet(nn.Module):
    def __init__(self, ndim=2, hidden_dim=8, device=None, activation_fn=nn.ELU):
        super(GCPoolNet, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),

            GCNConv(hidden_dim, hidden_dim)
        ])
        self.pooling = SAGPooling(hidden_dim, 0.4)

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )

        inner_init = create_optimal_inner_init(nonlinearity=nn.ReLU)
        self.conv_layers.apply(inner_init)
        self.policy.apply(out_init)
        self.value.apply(out_init)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.device = device

    def forward(self, data, batch=False):
        x, edge_index = data.x, data.edge_index

        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.elu(x)
            pooled_x, edge_index, _, _, _= self.pooling(x, edge_index)
            x += pooled_x
            print(x.size())

            x = self.norm(x)
        if batch:
            mean_val = global_mean_pool(x, batch=data.batch.to(x.device))
        else:
            mean_val = global_mean_pool(x, torch.zeros(x.size()[0], device=x.device, dtype=torch.long))

        proba = self.policy(x)
        value = self.value(mean_val)
        return proba, value


def get_appropriate_graph_policy(args, ndim, device):
    """
    Retrurn graph policy according to gcn type
    :param args:
    :param ndim:
    :return: graph policy from nodes x ndim -> nodes x hidden
    """
    if args.gcn_type == 'GCN':
        graph_policy = GCNNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device, activation_fn=nn.ELU, num_layers=2)
    elif args.gcn_type == 'GAT':
        graph_policy = GATNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'SGC':
        graph_policy = RGCNNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'SAGE':
        graph_policy = SagePolicyNetwork(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'cheb':
        graph_policy = ChebNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'GatedConv':
        graph_policy = GatedNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'Arma':
        graph_policy = ARMAPolicyNetwork(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'Pools':
        graph_policy = GCPoolNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'only_gcn':
        graph_policy = GcnPolicy(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'isomorph':
        graph_policy = GATisoNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    elif args.gcn_type == 'GIN':
        graph_policy = GINNet(ndim=ndim, hidden_dim=args.hidden_gcn, device=device)
    else:
        graph_policy = None
    return graph_policy
