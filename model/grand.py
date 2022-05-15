import torch
import torch.nn as nn
import torch.nn.functional as F

from cogdl.utils import spmm
from cogdl.data import Graph
from scipy.special import kl_div


class Grand(nn.Module):
    """
    Implementation of GRAND in paper `"Graph Random Neural Networks for Semi-Supervised Learning on Graphs"`
    <https://arxiv.org/abs/2005.11079>
    Parameters
    ----------
    nfeat : int
        Size of each input features.
    nhid : int
        Size of hidden features.
    nclass : int
        Number of output classes.
    input_droprate : float
        Dropout rate of input features.
    hidden_droprate : float
        Dropout rate of hidden features.
    use_bn : bool
        Using batch normalization.
    dropnode_rate : float
        Rate of dropping elements of input features
    tem : float
        Temperature to sharpen predictions.
    lam : float
         Proportion of consistency loss of unlabelled data
    order : int
        Order of adjacency matrix
    sample : int
        Number of augmentations for consistency loss
    alpha : float
    """

    def __init__(self, in_channels, hidden_channels, out_channels, input_droprate=0.6,
        hidden_droprate=0.8, use_bn=True, dropnode_rate=0.5, order=4, sample_num=4,
        lam = 0.1, temp = 0.5
    ):
        super().__init__()
        self.layer1 = nn.Linear(in_channels, hidden_channels)
        self.layer2 = nn.Linear(hidden_channels, out_channels)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.use_bn = use_bn
        self.order = order
        self.dropnode_rate = dropnode_rate
        self.sample_num = sample_num
        self.lam = lam
        self.temp = temp

    def rand_prop(self, graph, x):
        drop_rates = self.dropnode_rate * torch.ones(x.shape[0])
        if self.training:
            masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
            x = masks.to(x.device) * x
        else:
            x = x * (1.0 - self.dropnode_rate)
        y = x
        for i in range(self.order):
            x = spmm(graph, x).detach_()
            y.add_(x)
        return y.div_(self.order + 1.0).detach_()

    def forward(self, x, edge_index):
        graph = Graph(x=x, edge_index=edge_index)
        graph.sym_norm()
        feature_inv = x.sum(1).pow_(-1)
        feature_inv.masked_fill_(feature_inv == float("inf"), 0)
        x = x * feature_inv[:, None]
        x = self.rand_prop(graph, x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)
        return x
    
    def consis_loss(self, logps):
        temp = self.temp
        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        #p2 = torch.exp(logp2)
        
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))
            # loss += nn.KLDivLoss(size_average=False)(p, sharp_p)
            # loss += torch.mean((p-sharp_p * torch.log(p)).sum(1))
        loss = loss/len(ps)
        return self.lam * loss

    def __repr__(self):
        return super().__repr__() + "  hidden_droprate: " + str(self.hidden_droprate) + "  input_drop: " + str(self.input_droprate) \
            + "  dropnode: " + str(self.dropnode_rate) + "  order: " + str(self.order) + "  use_bn: " + str(self.use_bn) \
            + "  sample_num: " + str(self.sample_num) + "  lam: " + str(self.lam) + "  temp: " + str(self.temp)
