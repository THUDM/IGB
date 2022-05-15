from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
import torch

class PPNP_PYG(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, dropout,K,alpha):
        super().__init__()
        self.lin1 = Linear(num_features, num_hidden)
        self.lin2 = Linear(num_hidden, num_classes)
        self.prop1 = APPNP(K, alpha)
        self.d = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.d, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.d, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x