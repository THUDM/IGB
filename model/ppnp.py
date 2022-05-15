import torch
import torch.nn as nn
import torch.nn.functional as F

from .cogdl_base_model import BaseModel
from cogdl.utils import spmm
from cogdl.data import Graph


class MLPLayer(nn.Module):
    def __init__(self, nfeat, nclass, nhid, num_layers, dropout, activation="relu", act=False, norm=None, bias=True):
        super().__init__()
        shapes = [nfeat] + [nhid] * (num_layers - 1) + [nclass]
        self.mlp = nn.ModuleList(
            [nn.Linear(shapes[layer], shapes[layer + 1], bias=bias) for layer in range(num_layers)]
        )

        self.dropout = dropout
        self.act = act
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        else:
            raise ValueError(f"activation {activation} not supported")
            
        self.norm = norm

        if norm is not None and num_layers > 1:
            if norm == "layernorm":
                self.norm_list = nn.ModuleList(nn.LayerNorm(x) for x in shapes[1:-1])
            elif norm == "batchnorm":
                self.norm_list = nn.ModuleList(nn.BatchNorm1d(x) for x in shapes[1:-1])
            else:
                raise NotImplementedError(f"{norm} is not implemented")

    def forward(self, x):
        for i, fc in enumerate(self.mlp[:-1]):
            x = fc(x)
            if self.act:
                x = self.activation(x)
                if self.norm:
                    x = self.norm_list[i](x)

            else:
                if self.norm:
                    x = self.norm_list[i](x)
                x = self.activation(x)

            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.mlp[-1](x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size=16, num_layers=2, dropout=0.5, activation="relu", norm=None, act_first=False, bias=True):
        super().__init__()
        self.nn = MLPLayer(in_feats, out_feats, hidden_size, num_layers, dropout, activation, act_first, norm, bias)

    def forward(self, x):
        if isinstance(x, Graph):
            x = x.x
        return self.nn(x)


class PPNP(nn.Module):
    def __init__(self, nfeat, nclass, nhid=512, num_layers=2, dropout=0.001, alpha=0.5, niter=6, lmbda=0):
        super().__init__()
        self.nn = MLP(nfeat, nclass, nhid, num_layers, dropout)
        self.num_layers = num_layers
        self.nhid = nhid
        self.alpha = alpha
        self.niter = niter
        self.dropout = dropout
        self.cache = dict()
        self.lmbda = lmbda

    def forward(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        def get_ready_format(input, edge_index, edge_attr=None):
            if isinstance(edge_index, tuple):
                edge_index = torch.stack(edge_index)
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.shape[1]).float().to(input.device)
            adj = torch.sparse_coo_tensor(
                edge_index,
                edge_attr,
                (input.shape[0], input.shape[0]),
            ).to(input.device)
            return adj

        x = graph.x
        graph.sym_norm()
        # get prediction
        x = F.dropout(x, p=self.dropout, training=self.training)
        local_preds = self.nn.forward(x)

        preds = local_preds
        with graph.local_graph():
            graph.edge_weight = F.dropout(graph.edge_weight, p=self.dropout, training=self.training)
            graph.set_symmetric()
            for _ in range(self.niter):
                new_features = spmm(graph, preds)
                preds = (1 - self.alpha) * new_features + self.alpha * local_preds
            final_preds = preds
        return final_preds

    def predict(self, graph):
        return self.forward(graph)

    def __repr__(self):
        return  "model: {} dropout: {} hidden size: {} alpha: {} num_iter: {} num_layers: {} lmbda: {}".format(self.__class__.__name__, self.dropout, self.nhid, self.alpha,self.niter, self.num_layers, self.lmbda)
