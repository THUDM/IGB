
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import Tensor

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6, attn_drop=0.5, output_heads=1, nhead=16):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=nhead, dropout=attn_drop)
        self.conv2 = GATConv(hidden_channels * nhead, out_channels,
                             heads=output_heads, concat=False,
                             dropout=attn_drop)
        self.d = dropout
        self.nhead = nhead
        self.attn_drop = attn_drop

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.d, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.d, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def __repr__(self):
        return super().__repr__() + "  dropout: " + str(self.d) + "  nhead: " + str(self.nhead) + "  attn_drop: " + str(self.attn_drop)
