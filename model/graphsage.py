import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.data import Graph
from cogdl.utils import spmm, get_activation, get_norm_layer


class MeanAggregator(object):
    def __call__(self, graph, x):
        graph.row_norm()
        x = spmm(graph, x)
        return x


class SumAggregator(object):
    def __call__(self, graph, x):
        x = spmm(graph, x)
        return x


class MaxAggregator(object):
    def __init__(self):
        from cogdl.operators.scatter_max import scatter_max

        self.scatter_max = scatter_max

    def __call__(self, graph, x):
        x = self.scatter_max(graph.row_indptr.int(), graph.col_indices.int(), x)
        return x


class SAGELayer(nn.Module):
    def __init__(
        self, in_feats, out_feats, normalize=False, aggr="mean", dropout=0, norm=None, activation=None, residual=False
    ):
        super(SAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.fc = nn.Linear(2 * in_feats, out_feats)
        self.normalize = normalize
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if aggr == "mean":
            self.aggr = MeanAggregator()
        elif aggr == "sum":
            self.aggr = SumAggregator()
        elif aggr == "max":
            self.aggr = MaxAggregator()
        else:
            raise NotImplementedError

        if activation is not None:
            self.act = get_activation(activation, inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = get_norm_layer(norm, out_feats)
        else:
            self.norm = None

        if residual:
            self.residual = nn.Linear(in_features=in_feats, out_features=out_feats)
        else:
            self.residual = None

    def forward(self, graph, x):
        out = self.aggr(graph, x)
        out = torch.cat([x, out], dim=-1)
        out = self.fc(out)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        if self.residual:
            out = out + self.residual(x)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


def sage_sampler(adjlist, edge_index, num_sample):
    if adjlist == {}:
        row, col = edge_index
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        for i in zip(row, col):
            if not (i[0] in adjlist):
                adjlist[i[0]] = [i[1]]
            else:
                adjlist[i[0]].append(i[1])

    sample_list = []
    for i in adjlist:
        list = [[i, j] for j in adjlist[i]]
        if len(list) > num_sample:
            list = random.sample(list, num_sample)
        sample_list.extend(list)

    edge_idx = torch.as_tensor(sample_list, dtype=torch.long).t()
    return edge_idx


class GraphSage(nn.Module):
    def sampling(self, edge_index, num_sample):
        return sage_sampler(self.adjlist, edge_index, num_sample)

    def __init__(self, in_channels, out_channels, hidden_channels=[256], num_layers=2, sample_size=[10, 10], dropout=0.6, aggr="mean"):
        super().__init__()
        assert num_layers == len(sample_size)
        assert num_layers == len(hidden_channels) + 1
        self.hsize = hidden_channels
        self.adjlist = {}
        self.num_layers = num_layers
        self.sample_size = sample_size
        self.dropout = dropout
        self.aggr = aggr
        shapes = [in_channels] + hidden_channels + [out_channels]
        self.convs = nn.ModuleList(
            [SAGELayer(shapes[layer], shapes[layer + 1], aggr=aggr) for layer in range(num_layers)]
        )

    def forward(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        x = graph.x
        for i in range(self.num_layers):
            edge_index_sp = self.sampling(graph.edge_index, self.sample_size[i]).to(x.device)
            with graph.local_graph():
                graph.edge_index = edge_index_sp
                x = self.convs[i](graph, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def inference(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        for i in range(self.num_layers):
            x = self.convs[i](graph, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x
    
    def __repr__(self):
        return "graphsage" + "  dropout: " + str(self.dropout) \
            + "  sample_size: " + str(self.sample_size) + "  aggr: " + str(self.aggr) \
            + "  num_layers: " + str(self.num_layers) + "  hsize: " + str(self.hsize)
