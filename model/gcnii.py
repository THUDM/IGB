import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from cogdl.layers import GCNIILayer
from cogdl.data import Graph


class GCNII(nn.Module):
    """
    Implementation of GCNII in paper `"Simple and Deep Graph Convolutional Networks" <https://arxiv.org/abs/2007.02133>`_.
    Parameters
    -----------
    in_feats : int
        Size of each input sample
    hidden_size : int
        Size of each hidden unit
    out_feats : int
        Size of each out sample
    num_layers : int
    dropout : float
    alpha : float
        Parameter of initial residual connection
    lmbda : float
        Parameter of identity mapping
    """
    
    def __repr__(self):
        return "GCNII" + "  dropout: " + str(self.d) + " layers: " + str(self.n) + " hidden: " + str(self.h)+" alpha: " + str(self.a) + " lmbda: " + str(self.l)

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=64, dropout=0.5,
        alpha=0.2, lmbda=0.5, residual=False, actnn=False
    ):
        super().__init__()
        Layer = GCNIILayer
        Linear = nn.Linear
        Dropout = nn.Dropout
        ReLU = nn.ReLU(inplace=True)
        if actnn:
            try:
                from cogdl.layers.actgcnii_layer import ActGCNIILayer
                from actnn.layers import QLinear, QReLU, QDropout
            except Exception:
                print("Please install the actnn library first.")
                exit(1)
            Layer = ActGCNIILayer
            Linear = QLinear
            Dropout = QDropout
            ReLU = QReLU()

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(Linear(in_channels, hidden_channels))
        self.fc_layers.append(Linear(hidden_channels, out_channels))

        self.dropout = Dropout(dropout)
        self.d = dropout
        self.n = num_layers
        self.h = hidden_channels
        self.a = alpha
        self.l = lmbda
        self.activation = ReLU

        self.layers = nn.ModuleList(
            Layer(hidden_channels, alpha, math.log(lmbda / (i + 1) + 1), residual) for i in range(num_layers)
        )
        self.fc_parameters = list(self.fc_layers.parameters())
        self.conv_parameters = list(self.layers.parameters())

    def forward(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        graph.sym_norm()
        x = graph.x
        init_h = self.dropout(x)
        init_h = self.activation(self.fc_layers[0](init_h))

        h = init_h

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h, init_h)
            h = self.activation(h)
        h = self.dropout(h)
        out = self.fc_layers[1](h)
        return out
