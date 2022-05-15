from typing import Optional, Callable, List

import json
import os.path as osp

import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data
from .BaseDataset import BaseDataset


class Flickr(BaseDataset):
    r"""The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing descriptions and common properties of images.

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

    Stats:
        .. list-table::
            :widths: 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - #classes
            * - 89,250
              - 899,756
              - 500
              - 7
    """

    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__("Flickr", root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd

        path = osp.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)

        path = osp.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)

        path = osp.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)

        path = osp.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        # train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        # train_mask[torch.tensor(role['tr'])] = True

        # val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        # val_mask[torch.tensor(role['va'])] = True

        # test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        # test_mask[torch.tensor(role['te'])] = True

        # data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
        #             val_mask=val_mask, test_mask=test_mask)
        data = Data(x=x, edge_index=edge_index, y=y)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        # largest_G_nodes = max(nx.connected_components(G), key=len)
        # largest_G_nodes = sorted(list(largest_G_nodes))
        # x = x[largest_G_nodes]
        # y = y[largest_G_nodes]
        # print(len(x))
        # node_trans = dict(zip(largest_G_nodes, range(len(largest_G_nodes))))
        # largest_G = G.subgraph(largest_G_nodes)
        # edge_index = [(node_trans[i[0]], node_trans[i[1]]) for i in largest_G.edges]
        # reverse_edge_index = [(i[1],i[0]) for i in edge_index if i[0]!=i[1]]
        # edge_index = torch.Tensor(edge_index + reverse_edge_index).T
        # edge_index = edge_index.long()
        # data = Data(x=x, y=y, edge_index=edge_index)
        torch.save(self.collate([data]), self.processed_paths[0])
