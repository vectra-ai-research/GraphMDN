from __future__ import absolute_import

import torch
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv, Attention
from models.graph_non_local import GraphNonLocal
import torch.nn.functional as F

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN_MDN_Graph(nn.Module):
    def __init__(self, adj, hid_dim, num_gaussians=5, num_layers=4, nodes_group=None, p_dropout=None, tanh_out=True,
                 multivariate=False, pose_level_pi=False, uniform_sigma=False):

        super(SemGCN_MDN_Graph, self).__init__()

        assert not (multivariate and uniform_sigma)
        _gconv_input = [_GraphConv(adj, 2, hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])

            grouped_order = nodes_group[0]
            for node in nodes_group[1:]:
                grouped_order += node
            grouped_order = list(grouped_order)

            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)

        if multivariate:
            gauss_dim = 7
        else:
            gauss_dim = 5

        self.gconv_output = SemGraphConv(hid_dim, gauss_dim * num_gaussians, adj)


        self.num_gaussians = num_gaussians
        self.tanh_out = tanh_out
        self.multivariate = multivariate
        self.pose_level_pi = pose_level_pi
        self.uniform_sigma = uniform_sigma

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)

        n_gaussians = self.num_gaussians

        ## TODO: make tanh a flag in arguments
        mu = out[:, :, :n_gaussians*3].view(out.shape[0], out.shape[1], n_gaussians, 3)
        if self.tanh_out:
            mu = torch.tanh(mu)

        if self.multivariate:
            sigma = out[:, :, n_gaussians*3:n_gaussians*6].view(out.shape[0], out.shape[1], n_gaussians, 3)
            pi = out[:, :, n_gaussians*6:n_gaussians*7]
        else:
            sigma = out[:, :, n_gaussians*3:n_gaussians*4].unsqueeze(dim=3)
            pi = out[:, :, n_gaussians*4:n_gaussians*5]

        if self.uniform_sigma:
            sigma = torch.mean(sigma, dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)
        if self.pose_level_pi:
            pi = torch.mean(pi, dim=1, keepdim=True).expand(-1, out.shape[1], -1)
            #constant pi
            #pi = torch.mean(pi, dim=2, keepdim=True).expand(-1, -1, n_gaussians)
        sigma = torch.max(F.elu(sigma) + torch.ones([1]), 1e-10 * torch.ones([1]))

        return (mu, sigma, pi)
