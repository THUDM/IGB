import torch.nn as nn
import torch.nn.functional as F

from cogdl.layers import MixHopLayer
from cogdl.data import Graph


class MixHop(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, layer1_pows=[100, 100, 100], layer2_pows=[60, 60, 60]):
        super().__init__()
        self.Dropout = nn.Dropout(dropout)
        self.mixhops = nn.ModuleList(
            [MixHopLayer(in_channels, [0, 1, 2], layer1_pows),
            MixHopLayer(sum(layer1_pows), [0, 1, 2], layer2_pows)]
        )
        self.fc = nn.Linear(sum(layer2_pows), out_channels)

    def forward(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        x = graph.x
        for mixhop in self.mixhops:
            x = F.relu(mixhop(graph, x))
            x = self.Dropout(x)
        x = self.fc(x)
        return x
