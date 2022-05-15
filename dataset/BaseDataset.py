from os import sep
from typing import Optional, Callable, List, Union, Tuple, Dict, Iterable

import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from icecream import ic
import scipy
import scipy.stats


class BaseDataset(InMemoryDataset):
    def __init__(self, dataset_name: str = "basedataset",
                 root: str = "", transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = dataset_name
        print("loading {} dataset...".format(self.name))
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.component_dist_list = [None, None]
        self.split_init_done = False
        self.random_idx = -1
        self.random_list = None
        self.G = None

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    # accept_rate = 5 means x^5 ( 0 < x < 1), high accept_rate
    def init_split(self, split: str = "public", walk_length: int = 5000, burning_round: int = 5, connected_component: bool = False, accept_rate: int = 5):
        assert split in ['public', 'random', 'random_sample', 'MHRW', 'RW']
        if split in ['public', 'random']:
            s = split
        else:
            s = "{}, walk_length : {}, burning_roud : {}".format(
                split, walk_length, burning_round)
        print("doing init split with datasplit: {}".format(s))
        self.split = split
        self.connected_component = connected_component
        self.walk_length = walk_length
        self.accept_rate = accept_rate
        self.node_kl_div_list = [float('inf')]
        self.edge_kl_div_list = [float('inf')]

        self.split_init_done = True
        self.cur_node = None
        self.node_set = set()

        if not self.component_dist_list[self.connected_component]:
            if self.connected_component:
                if self.G == None:
                    self.G = nx.Graph()
                    self.G.add_edges_from(self.data.edge_index.numpy().T)
                largest_G_nodes = max(nx.connected_components(
                    self.G), key=lambda item: len(item))
                largest_G = self.G.subgraph(largest_G_nodes).copy()
                largest_G_nodes = sorted(list(largest_G_nodes))
                reverse_edge_index = [(i[1], i[0])
                                      for i in largest_G.edges if i[0] != i[1]]
                edges = np.array(list(largest_G.edges) +
                                 reverse_edge_index).T.tolist()
                idx = largest_G_nodes

            else:
                edges = self.data.edge_index.tolist()
                idx = list(range(self.data.x.size(0)))

            neighbors = [list() for _ in range(self.data.x.size(0))]
            neighbors_min_degree = [float('inf')
                                    for _ in range(self.data.x.size(0))]

            for i in range(len(edges[0])):
                neighbors[edges[0][i]].append(edges[1][i])
                neighbors[edges[1][i]].append(edges[0][i])

            node_dist_before_sample = np.histogram(self.data.y[idx].numpy(), bins=range(self.num_classes))[
                0]/self.data.y[idx].numpy().shape[0]
            edge_dist_before_sample = self.cal_single_edge_dist(
                self.data.y.numpy(), edges)

            self.component_dist_list[self.connected_component] = [neighbors, neighbors_min_degree, idx,
                                                                  node_dist_before_sample, edge_dist_before_sample]

        for _ in range(burning_round):
            self.data_split()

        self.node_set.clear()
        print("init split done")

    def data_split(self, mMHRW= True):
        assert self.split_init_done == True

        if self.split == 'random':
            # raise NotImplementedError
            data = self.get(0)
            exclud_node_mask = torch.ones(len(self.data.y), dtype=bool)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = torch.logical_and(exclud_node_mask, (data.y == c)).nonzero(
                    as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[
                    :self.num_train_per_class]]
                data.train_mask[idx] = True

            remaining = torch.logical_and(
                exclud_node_mask, (~data.train_mask)).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:self.num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[self.num_val:self.num_val +
                                     self.num_test]] = True


            return data, {}

        elif self.split == 'RW' or self.split == 'MHRW' or self.split == 'random_sample':

            data = Data()

            neighbors, neighbors_min_degree, idx, node_dist_before_sample, edge_dist_before_sample = self.component_dist_list[
                self.connected_component]

            node_size = len(idx)

            if self.split == 'random_sample':
                np.random.shuffle(idx)
                walk = sorted(idx[:self.walk_length])
                if self.G == None:
                    self.G = nx.Graph()
                    self.G.add_edges_from(self.data.edge_index.numpy().T)
                sub_G = self.G.subgraph(walk).copy()
                walk_edge = sub_G.edges()
                connected_components_cnt = len(
                    [i for i in nx.connected_components(sub_G)])

            else:
                connected_components_cnt = 1
                # print(len(edges[0]))

                if not self.cur_node:
                    walk = [np.random.choice(idx)]
                else:
                    walk = [self.cur_node]

                walk_edge = set()

                walk_cnt = 0

                while len(walk_edge) < self.walk_length:
                    self.cur_node = walk[-1]
                    cur = walk[-1]
                    cur_nbrs = neighbors[cur]
                    walk_cnt += 1

                    # print(cur,neighbors[cur])

                    if len(cur_nbrs) > 0:
                        if self.split == 'RW':
                            nxt = np.random.choice(cur_nbrs)
                            walk.append(nxt)
                            if cur < nxt:
                                walk_edge.add((cur, nxt))
                            else:
                                walk_edge.add((nxt, cur))

                        elif self.split == 'MHRW':
                            nxt = np.random.choice(cur_nbrs)

                            if mMHRW:
                                if neighbors_min_degree[cur] == float("inf"):
                                    for nbr in cur_nbrs:
                                        neighbors_min_degree[cur] = min(
                                            neighbors_min_degree[cur], len(neighbors[nbr]))

                                p = max(neighbors_min_degree[cur], len(
                                    neighbors[cur])) / len(neighbors[nxt])
                                if neighbors_min_degree[cur] > len(neighbors[cur]) and not p <= 1:
                                    print("something goes wrong...")
                                    raise NotImplementedError
                            else:
                                p = len(
                                    neighbors[cur]) / len(neighbors[nxt])

                            if torch.rand(1) < p:
                                walk.append(nxt)
                                if cur < nxt:
                                    walk_edge.add((cur, nxt))
                                else:
                                    walk_edge.add((nxt, cur))

                    else:
                        break

                walk = list(set(walk))
                walk_edge = list(walk_edge)

            # print(walk_edge)

            x = self.data.x.numpy()[walk]
            y = self.data.y.numpy()[walk]

            # train_prob = data.train_mask.sum() / data.x.shape[0]
            train_prob = 0.1
            val_prob = 0.1
            test_prob = 0.8
            len_walk = len(walk)
            train_num = int(len_walk * train_prob)
            val_num = int(len_walk*val_prob)
            shuffle_arr = np.arange(len_walk)
            np.random.shuffle(shuffle_arr)
            data.train_mask = torch.zeros(len_walk).bool()
            data.val_mask = torch.zeros(len_walk).bool()
            data.test_mask = torch.zeros(len_walk).bool()

            data.train_mask[shuffle_arr[0:train_num]] = True
            data.val_mask[shuffle_arr[train_num:train_num+val_num]] = True
            data.test_mask[shuffle_arr[train_num+val_num:]] = True

            # data.train_mask = torch.bernoulli(torch.ones(len(walk)) * train_prob)
            # data.train_mask = data.train_mask.bool()
            # # test_prob = data.test_mask.sum() / (data.x.shape[0]*(1-train_prob))
            # val_prob = 0.2

            # test_prob = 0.7
            # data.test_mask = torch.bernoulli(torch.ones(len(walk)) * test_prob)
            # data.test_mask = data.test_mask.bool()
            # data.test_mask[data.train_mask == True] = False
            edge_index = [[], []]

            idx = {}
            for i in range(len(walk)):
                idx[walk[i]] = i

            for i, j in walk_edge:
                edge_index[0].append(idx[i])
                edge_index[1].append(idx[j])

            data.x = torch.tensor(x)
            data.y = torch.tensor(y)
            data.edge_index = torch.tensor(edge_index)

            node_dist_after_sample = np.histogram(data.y.numpy(), bins=range(
                self.num_classes))[0]/data.y.numpy().shape[0]
            edge_dist_after_sample = self.cal_single_edge_dist(
                data.y.numpy(), data.edge_index.clone())
            node_dist_before_sample += 1e-10
            node_dist_after_sample += 1e-10
            node_kl_div = self.cal_kl_div(
                node_dist_before_sample, node_dist_after_sample)
            edge_kl_div = self.cal_kl_div(
                edge_dist_before_sample, edge_dist_after_sample)

            accept_node_kl_div = np.median(self.node_kl_div_list)
            accept_edge_kl_div = np.median(self.edge_kl_div_list)
            # print("node_kl ", node_kl_div)
            # print("edge_kl ", edge_kl_div)
            # print(accept_node_kl_div)
            # print(accept_edge_kl_div)

            if torch.rand(1) < pow(min(accept_node_kl_div / node_kl_div, accept_edge_kl_div / edge_kl_div), self.accept_rate):
                for cur in walk:
                    self.node_set.add(cur)

                self.node_kl_div_list.append(node_kl_div)
                self.edge_kl_div_list.append(edge_kl_div)
                data.edge_index =torch.cat([data.edge_index,data.edge_index.flip(0)],dim=1)
                data.edge_index=data.edge_index[:,torch.randperm(data.edge_index.size()[1])]
                return data, {
                    "node_kl_div": node_kl_div,
                    "edge_kl_div": edge_kl_div,
                    "accept_node_kl_div": accept_node_kl_div,
                    "accept_edge_kl_div": accept_edge_kl_div,
                    "cur_node_cnt": len(self.node_set),
                    "node_cnt": node_size,
                    "walk_node": walk,
                    "connected_components_cnt": connected_components_cnt,
                    "walk_cnt": walk_cnt
                }
            else:
                ic("reject")
                return self.data_split()

        elif self.split != "public":
            raise NotImplementedError

    def cal_kl_div(self, dist_before_sample, dist_after_sample):
        # self.node_dist_after_sample += 1e-5
        return scipy.stats.entropy(pk=dist_before_sample, qk=dist_after_sample)

    def cal_single_edge_dist(self, label, edges):
        edges = np.array(edges)
        num_classes = self.num_classes
        connect_inter_label = np.zeros([num_classes, num_classes])

        for i in range(edges.shape[1]):
            label0 = label[edges[0][i]]
            label1 = label[edges[1][i]]
            if(label0 == label1):
                connect_inter_label[label0][label1] += 1
            else:
                connect_inter_label[label0][label1] += 1
                connect_inter_label[label1][label0] += 1
        connect_inter_label += 1e-5
        flat_trui = np.triu(connect_inter_label).flatten()
        flat_trui = flat_trui[flat_trui != 0]
        return flat_trui/sum(flat_trui)