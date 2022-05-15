from typing import Optional, Callable, List, Union, Tuple, Dict, Iterable

import os
import os.path as osp
import shutil

import torch
from torch_sparse import coalesce, transpose
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)
import tqdm
import itertools
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from .BaseDataset import BaseDataset


class AMiner(BaseDataset):
    r"""The heterogeneous AMiner dataset from the `"metapath2vec: Scalable
    Representation Learning for Heterogeneous Networks"
    <https://ericdongyx.github.io/papers/
    KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper, consisting of nodes from
    type :obj:`"paper"`, :obj:`"author"` and :obj:`"venue"`.
    Venue categories and author research interests are available as ground
    truth labels for a subset of nodes.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=1'
    y_url = 'https://www.dropbox.com/s/nkocx16rpl4ydde/label.zip?dl=1'


    def __init__(self,
                 root: str = "./data/AMiner",
                 split: str = "RW",
                 walk_length=1000,
                 num_train_per_class: int = 20,
                 num_val: int = 500,
                 num_test: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.name = "AMiner"
        self.root = root
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.exclud_node_list = []
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.add_node_num = -1
        super().__init__(dataset_name=self.name,
                         root=root,
                         transform=transform,
                         pre_transform=pre_transform)


    @property
    def raw_file_names(self) -> List[str]:
        return [
            'id_author.txt', 'id_conf.txt', 'paper.txt', 'paper_author.txt',
            'paper_conf.txt', 'label'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'net_aminer'), self.raw_dir)
        os.unlink(path)
        path = download_url(self.y_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        import pandas as pd

        # Get author labels.
        path = osp.join(self.raw_dir, 'id_author.txt')
        author = pd.read_csv(path, sep='\t', names=['idx', 'name'],
                             index_col=1)

        path = osp.join(self.raw_dir, 'label',
                        'googlescholar.8area.author.label.txt')
        df = pd.read_csv(path, sep=' ', names=['name', 'y'])
        df = df.join(author, on='name')

        # data['author'].y = torch.from_numpy(df['y'].values) - 1
        # data['author'].y_index = torch.from_numpy(df['idx'].values)
        author_y = torch.from_numpy(df['y'].values) - 1
        author_y_index = torch.from_numpy(df['idx'].values)
        labeled_authors_index_set = set(author_y_index.numpy())


        # Get paper<->author connectivity.
        path = osp.join(self.raw_dir, 'paper_author.txt')
        paper_author = pd.read_csv(path, sep='\t', header=None)
        paper_author = torch.from_numpy(paper_author.values)
        paper_author = paper_author.t().contiguous()
        M, N = int(paper_author[0].max() + 1), int(paper_author[1].max() + 1)
        paper_author, _ = coalesce(paper_author, None, M, N)
        author_paper, _ = transpose(paper_author, None, M, N)


        key_set = set()
        dataset_dict = {}
        dataset_edge_index = paper_author.numpy()
        for i in tqdm.tqdm(range(dataset_edge_index.shape[1])):
            if(dataset_edge_index[0,i] not in key_set):
                dataset_dict[dataset_edge_index[0,i]] = []
                key_set.add(dataset_edge_index[0,i])


            if (dataset_edge_index[1,i] in labeled_authors_index_set):
                dataset_dict[dataset_edge_index[0,i]].append(dataset_edge_index[1,i])#同一个paper的各个author
        ic(len(dataset_dict.keys()))

        edg = []
        for l in tqdm.tqdm(dataset_dict.values()):

            if(len(l)>1):
                bb = list(itertools.permutations(l, 2))
                edg += bb
        ic(len(edg))

        G = nx.Graph()

        edges = np.array(edg)
        # ic(edges.shape)
        G.add_edges_from(edges)
        largest_G_nodes_set = max(nx.connected_components(G), key=lambda item: len(item))
        largest_G = G.subgraph(largest_G_nodes_set)
        largest_G_nodes_list = sorted(list(largest_G_nodes_set))
        #现在需要构建一个x,并且需要记录其对应的原始编号，以及其feature
        from collections import defaultdict
        old_label_new_label_func = defaultdict(lambda: None) #从老标签查新标签
        new_label_old_label_func = defaultdict(lambda: None) #从新标签到老标签
        for new_label, old_label in enumerate(largest_G_nodes_list):
            old_label_new_label_func[old_label] = new_label
            new_label_old_label_func[new_label] = old_label
        path = osp.join(self.raw_dir, 'paper_conf.txt')
        paper_venue = pd.read_csv(path, sep='\t', header=None)
        paper_venue = torch.from_numpy(paper_venue.values)
        paper_venue = paper_venue.t().contiguous()
        M, N = int(paper_venue[0].max() + 1), int(paper_venue[1].max() + 1)
        paper_venue, _ = coalesce(paper_venue, None, M, N)
        venue_paper, _ = transpose(paper_venue, None, M, N)
        paper_venue = paper_venue.numpy()

        num_largest_G_nodes_list = len(largest_G_nodes_list)
        x = np.zeros([num_largest_G_nodes_list,N])
        for i in tqdm.tqdm(range(paper_venue.shape[1])):
            author_list = dataset_dict[paper_venue[0,i]]
            for author_old_label in author_list:
                new_label = old_label_new_label_func[author_old_label]
                if new_label is not None:
                    x[new_label,paper_venue[1,i]] = 1
        #             print(new_label,paper_venue[1,i])
                else:
                    #             print("?????",author_old_label)
                    pass

        y = np.arange(num_largest_G_nodes_list)
        # for num in range(num_largest_G_nodes_list):
        #     y[num]
        #     author_y_index[new_label_old_label_func[num]]
        # sum=0
        author_y_index = author_y_index.numpy()
        y = np.arange(num_largest_G_nodes_list)
        for i in tqdm.tqdm(range(author_y_index.shape[0])):
            new_label = old_label_new_label_func[author_y_index[i]]
            if new_label is not None:
                y[new_label] = author_y[i]
            else:
                pass

        ic(y[:10])
        edge_index = []
        for e in tqdm.tqdm(edg):
            e0 = old_label_new_label_func[e[0]]
            e1 = old_label_new_label_func[e[1]]
            if (e0 is not None) and (e1 is not None):
                edge_index.append([e0,e1])

        edge_index = np.array(edge_index).T
        ic(x.shape,y.shape,edge_index.shape)
        # np.savez("runoob.npz", x, y, edge_index)

        data = Data(x=torch.FloatTensor(x), edge_index=torch.from_numpy(edge_index), y=torch.from_numpy(y))
        data.train_mask = torch.zeros(num_largest_G_nodes_list)
        data.val_mask = torch.zeros(num_largest_G_nodes_list)
        data.test_mask = torch.zeros(num_largest_G_nodes_list)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
