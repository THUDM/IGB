import os
import shutil
import os.path as osp

import torch
from typing import Optional, Callable
from torch_geometric.data import download_url, extract_tar
from .BaseDataset import BaseDataset
from torch_geometric.data import Data
from torch_geometric.io import read_planetoid_data
from icecream import ic
import networkx as nx   
import numpy as np


class NELL(BaseDataset):
    r"""The NELL dataset, a knowledge graph from the
    `"Toward an Architecture for Never-Ending Language Learning"
    <https://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf>`_ paper.
    The dataset is processed as in the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.

    .. note::

        Entity nodes are described by sparse feature vectors of type
        :class:`torch_sparse.SparseTensor`, which can be either used directly,
        or can be converted via :obj:`data.x.to_dense()`,
        :obj:`data.x.to_scipy()` or :obj:`data.x.to_torch_sparse_coo_tensor()`.

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

    url = 'http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz'

    def __init__(self, root: str="./data/NELL",
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__("NELL", root, transform, pre_transform)
        self.data.x = self.data.x.to_dense()

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.nell.0.001.{name}' for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.root)
        extract_tar(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, 'nell_data'), self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, 'nell.0.001')
        data = data if self.pre_transform is None else self.pre_transform(data)
        G = nx.Graph()  
        G.add_edges_from(data["edge_index"].numpy().T)
        largest_G_nodes = max(nx.connected_components(G), key=len)
        largest_G_nodes = sorted(list(largest_G_nodes))
        x = data.x[largest_G_nodes]
        y = data.y[largest_G_nodes]
        train_mask = data.train_mask[largest_G_nodes]
        val_mask = data.val_mask[largest_G_nodes]
        test_mask = data.test_mask[largest_G_nodes]
        node_trans = dict(zip(largest_G_nodes, range(len(largest_G_nodes))))
        largest_G = G.subgraph(largest_G_nodes)
        edge_index = [(node_trans[i[0]], node_trans[i[1]]) for i in largest_G.edges]
        reverse_edge_index = [(i[1],i[0]) for i in edge_index if i[0]!=i[1]]
        edge_index = torch.Tensor(edge_index + reverse_edge_index).T
        edge_index = edge_index.long()

        # count word frequencies, and reserve the most frequent 10000 words
        x = x.to_dense()
        word_cnt = x.sum(0)
        word_cnt, word_cnt_idx = torch.sort(word_cnt, 0, descending=True)
        word_cnt_idx = word_cnt_idx[:10000]
        x = x[:, word_cnt_idx]
        x = x.to_sparse()
        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        torch.save(self.collate([data]), self.processed_paths[0])
