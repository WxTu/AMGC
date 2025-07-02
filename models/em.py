from utils.registry import MODEL_REGISTRY
import torch
from torch.nn import Module, Parameter
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv
)
import copy


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


@MODEL_REGISTRY.register()
class EM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.m = args.m
        self.t = args.t
        self.K = args.K
        self.imputation_branch = Encoder(args.in_features, args.encoder_channels, args.hidden_channels,
                                         num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                                         bn=args.bn, layer=args.layer, activation=args.encoder_activation)

        self.cluster_branch = copy.deepcopy(self.imputation_branch)

        for param_q, param_k in zip(self.imputation_branch.parameters(), self.cluster_branch.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.structure_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                                             num_layers=args.structure_decoder_layer,
                                             dropout=args.structure_decoder_dropout)

        self.negative_sampler = negative_sampling

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.imputation_branch.parameters(), self.cluster_branch.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    def forward(self, pyg_graph, mask, cluster_labels, clusters, o_indices, m_indices):

        X_L = pyg_graph.x
        edge_index = pyg_graph.edge_index

        remaining_edges, masked_edges = mask(edge_index)

        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=pyg_graph.num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        with torch.no_grad():
            self._momentum_update_key_encoder()

        Z = self.imputation_branch(X_L, remaining_edges)
        Z = find_neighbors_from_cluster(Z, cluster_labels, m_indices, K=self.K, m=m_indices)

        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=2 ** 16, shuffle=True):
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.structure_decoder(Z, Z, batch_masked_edges, sigmoid=False)
            neg_out = self.structure_decoder(Z, Z, batch_neg_edges, sigmoid=False)

        Z = nn.functional.normalize(Z, dim=1)

        return Z, pos_out, neg_out

    def embed(self, pyg_graph):
        X_L = pyg_graph.x
        edge_index = pyg_graph.edge_index
        Z = self.cluster_branch(X_L, edge_index)
        Z = nn.functional.normalize(Z, dim=1)
        return Z

    def random_negative_sampler(self, edge_index, num_nodes, num_neg_samples):
        neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
        return neg_edges


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2,
                 dropout=0.5,
                 bn='none',
                 layer="gcn",
                 activation="elu"):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(creat_bn_layer(bn, second_channels * heads))
        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        return output


class EdgeDecoder(nn.Module):

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, z_1, z_2, edge, sigmoid=True, reduction=False):
        x = z_1[edge[0]] * z_2[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class Projector(nn.Module):

    def __init__(
            self, in_channels, hidden_channels, out_channels=1,
            num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.mlps[-1](x)
        return x


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer


class UnitNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, vectors):
        valid_index = (vectors != 0).sum(1, keepdims=True) > 0
        vectors = torch.where(valid_index, vectors, torch.randn_like(vectors))
        return vectors / (vectors ** 2).sum(1, keepdims=True).sqrt()


def creat_bn_layer(name, channels):
    if name == "batchnorm1d":
        layer = nn.BatchNorm1d(channels)
    elif name == "layernorm":
        layer = nn.LayerNorm(channels)
    elif name == 'unitnorm':
        layer = UnitNorm()
    else:
        layer = nn.Identity()
    return layer


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation")


def find_neighbors_from_cluster(data, cluster_assignments, m_indices, K=3, m=None):
    m_n = m.size(0)
    n = data.size(0)
    neighbors_list = []
    number = m_n

    for i in range(number):

        cluster_indices = torch.nonzero(torch.from_numpy(cluster_assignments == cluster_assignments[m[i]]),
                                        as_tuple=False).squeeze(dim=1)

        distances = torch.cdist(data[m[i]].unsqueeze(0), data[cluster_indices])

        if K + 1 > cluster_indices.size(0):
            _, nearest_indices = torch.topk(distances.squeeze(), cluster_indices.size(0), largest=False)
        else:
            _, nearest_indices = torch.topk(distances.squeeze(), K + 1, largest=False)

        if cluster_indices.size(0) <= 1:
            nearest_neighbors = data[cluster_indices[nearest_indices]]
        else:
            if torch.cuda.is_available():
                cluster_indices = cluster_indices.cuda()
            else:
                cluster_indices = cluster_indices
            nearest_neighbors = torch.mean(data[cluster_indices[nearest_indices[1:]]], dim=0)

        neighbors_list.append(nearest_neighbors)

    neighbors = torch.stack(neighbors_list, dim=0)
    data[m_indices] += neighbors
    return data
