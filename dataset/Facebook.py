from typing import Optional, Callable
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, download_url
from .BaseDataset import BaseDataset
import icecream as ic


class Facebook(BaseDataset):
    r"""The Facebook Page-Page network dataset introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent verified pages on Facebook and edges are mutual likes.
    It contains 22,470 nodes, 342,004 edges, 128 node features and 4 classes.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://graphmining.ai/datasets/ptg/facebook.npz'

    def __init__(self, root: str="./data/Facebook",
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__("Facebook", root, transform, pre_transform)

    @property
    def raw_file_names(self) -> str:
        return 'facebook.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()
        data = Data(x=x, y=y, edge_index=edge_index)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        G = nx.Graph()
        G.add_edges_from(data["edge_index"].numpy().T)
        largest_G_nodes = max(nx.connected_components(G), key=len)
        largest_G_nodes = sorted(list(largest_G_nodes))
        x = x[largest_G_nodes]
        y = y[largest_G_nodes]
        node_trans = dict(zip(largest_G_nodes, range(len(largest_G_nodes))))
        largest_G = G.subgraph(largest_G_nodes)
        edge_index = [(node_trans[i[0]], node_trans[i[1]]) for i in largest_G.edges]
        reverse_edge_index = [(i[1],i[0]) for i in edge_index if i[0]!=i[1]]
        edge_index = torch.Tensor(edge_index + reverse_edge_index).T
        edge_index = edge_index.long()
        data = Data(x=x, y=y, edge_index=edge_index)
        torch.save(self.collate([data]), self.processed_paths[0])
