import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5,improved=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,improved=improved)
        self.hidden_channels = hidden_channels
        self.d = dropout
        self.conv2 = GCNConv(hidden_channels, out_channels,improved=improved)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x,training=self.training,p=self.d)
        x = self.conv2(x, edge_index)

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def __repr__(self):
        return  "model: {} dropout: {} hidden size: {}".format(self.__class__.__name__, self.d, self.hidden_channels)