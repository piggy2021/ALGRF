import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

coarse_adj_list2 = [
            # 1  2  3  4
            [1, 1, 1, 1],  # 1
            [1, 1, 1, 1],  # 2
            [1, 1, 1, 1],  # 3
            [1, 1, 1, 1],  # 4
        ]

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=9, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.adj = nn.Parameter(torch.FloatTensor(adj_size, adj_size))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.bn = nn.BatchNorm2d(self.out_features)
        self.bn = nn.BatchNorm1d(out_features * adj_size)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # self.adj.data.fill_(1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # adj = torch.softmax(self.adj, dim=1)
        # adj = adj.repeat(input.size(0), 1, 1)
        output_ = torch.bmm(adj, support)
        if self.bias is not None:
            output_ = output_ + self.bias
        output = output_.view(output_.size(0), output_.size(1) * output_.size(2))
        output = self.bn(output)
        output = output.view(output_.size(0), output_.size(1), output_.size(2))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, isMeanPooling=True):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        self.nhid = nhid
        self.isMeanPooling = isMeanPooling
        self.gc1 = GraphConvolution(nfeat, nhid, adj_size)
        self.gc2 = GraphConvolution(nhid, nhid, adj_size)

    def forward(self, x, adj):
        x_ = F.dropout(x, 0.5, training=self.training)
        x_ = F.relu(self.gc1(x_, adj))
        x_ = F.dropout(x_, 0.5, training=self.training)
        x_ = F.relu(self.gc2(x_, adj))

        x_mean = torch.mean(x_, 1)  # aggregate features of nodes by mean pooling
        x_cat = x_.view(x.size()[0], -1)  # aggregate features of nodes by concatenation
        x_mean = F.dropout(x_mean, 0.5, training=self.training)
        x_cat = F.dropout(x_cat, 0.5, training=self.training)

        return x_mean, x_cat

if __name__ == '__main__':
    a = torch.ones(2, 4, 64)
    print(a.shape)
    net = GCN(4, 64, 64)
    o = net(a, None)
    print(o)
