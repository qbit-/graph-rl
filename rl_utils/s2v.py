import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNN(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):

        super().__init__()

        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights

        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )

        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        # obs.transpose_(-1, -2)
        #
        # Calculate features to be used in the MPNN
        # node_features = obs[:, :, 0:self.n_obs_in]
        # Get graph adj matrix.
        adj = obs[::, :self.n_obs_in]
        # out_degree = torch.sum(adj, dim=1)
        # in_degree = torch.sum(adj, dim=2)
        #
        # identity = torch.eye(adj.size()[1]).repeat((adj.size()[0], 1, 1))
        # node_features = identity * in_degree + identity * out_degree - torch.diagflat(torch.diagonal(adj))
        node_features = obs[::, self.n_obs_in:]
        # adj_conns = (adj != 0).type(torch.FloatTensor).to(adj.device)

        norm = self.get_normalisation(adj)

        init_node_embeddings = self.node_init_embedding_layer(node_features)
        # print()
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        # Initialise embeddings.
        current_node_embeddings = init_node_embeddings

        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj)
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                              edge_embeddings,
                                                                              norm,
                                                                              adj)

        out = self.readout_layer(current_node_embeddings)
        out = out.squeeze()

        return out


class EdgeAndNodeEmbeddingLayer(nn.Module):
    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features

        self.edge_embedding_NN = nn.Linear(int(n_obs_in + 1), n_features - 1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm):
        edge_features = torch.cat([adj.unsqueeze(-1),
                                   node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
                                  dim=-1)
        edge_features *= (adj.unsqueeze(-1) != 0).float()

        edge_features_unrolled = torch.reshape(edge_features, (
        edge_features.shape[0], edge_features.shape[1] * edge_features.shape[1], edge_features.shape[-1]))
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[1], self.n_features - 1))
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm

        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm.max()], dim=-1)))

        return edge_embeddings


class UpdateNodeEmbeddingLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2 * n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2 * n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, norm, adj):
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings) / norm

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):
    def __init__(self, n_features, n_hid=[], bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [2 * n_features] + n_hid + [1]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        f_local = node_embeddings

        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])
        f_pooled = h_pooled.repeat(1, 1, node_embeddings.shape[1]).view(node_embeddings.shape)

        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
            else:
                out = features

        return out


class S2V(nn.Module):
    def __init__(self, p, num_l=3):
        super().__init__()
        self.p = p
        self.th1 = nn.Parameter(torch.randn(p))
        self.th2 = nn.Parameter(torch.randn((p,p)))
        self.th3 = nn.Parameter(torch.randn((p,p)))
        self.th4 = nn.Parameter(torch.randn(p))
        self.th5 = nn.Parameter(torch.randn(2*p))
        self.th6 = nn.Parameter(torch.randn((p,p)))
        self.th7 = nn.Parameter(torch.randn((p,p)))
        self.mlp = nn.Linear(p,p)
        self.relu = torch.nn.ReLU()

    def forward(self, state, T=4):
        # Define the mus
        # Initial mu
        # mu = tf.einsum('iv,k->ivk', x, theta1)
        # mu = tf.zeros(
        # Loop over t

        if state.dim() == 2:
            state = state.unsqueeze(0)

        out_degree = torch.sum(state, dim=1)
        in_degree = torch.sum(state, dim=2)
        adj = state
        x = in_degree + out_degree - torch.diagonal(state, dim1=-2, dim2=-1)

        for t in range(T):
            # First part of mu
            mu_part1 = torch.einsum('iv,k->ivk', x, self.th1)

            # Second part of mu
            if t != 0:
                mu_part2 = torch.einsum('kl,ivk->ivl', self.th2, torch.einsum('ivu,iuk->ivk', adj, mu))
                # Add some non linear transformations of the pooled neighbors' embeddings
                mu_part2 = self.relu(self.mlp(mu_part2))

            # Third part of mu
            mu_part3_0 = torch.einsum('ikvu->ikv', self.relu(torch.einsum('k,ivu->ikvu', self.th4, adj)))
            mu_part3_1 = torch.einsum('kl,ilv->ivk', self.th3, mu_part3_0)
            if t != 0:
                mu = self.relu(torch.add(mu_part1 + mu_part2, mu_part3_1))
            else:
                mu = self.relu(torch.add(mu_part1, mu_part3_1))

        Q_part1 = torch.einsum('kl,ivk->ivl', self.th6, torch.einsum('ivu,iuk->ivk', adj, mu))
        Q_part2 = torch.einsum('kl,ivk->ivl', self.th7, mu)

        res= torch.einsum('k,ivk->iv', self.th5, self.relu(torch.cat([Q_part1, Q_part2], dim=2)))
        return res.squeeze()
