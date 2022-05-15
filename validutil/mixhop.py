from torch_geometric.datasets import Planetoid
import tqdm
import torch
import random
import torch.nn.functional as F
import copy
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath("../model")))
from model.mixhop import MixHop
import argparse
from datasets import get_planetoid_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--max_patience', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--layer1_pows', nargs='+', type=int, default=[200, 200, 200])
parser.add_argument('--layer2_pows', nargs='+', type=int, default=[100, 100, 100])
parser.add_argument('--dataset_num', type=str, default='1')
args = parser.parse_args()


# load dataset
dataset = get_planetoid_dataset(args.dataset, False)
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data.to(device)
real_val_data = data.y[data.val_mask].clone()
num_train = data.train_mask.sum()
num_val = data.val_mask.sum()
if args.dataset=="Cora":
    train_val_data = torch.load("../pseudo_label_list/MixHop_{}_label_list_{}.pt".format(args.dataset,args.dataset_num))
else:
    train_val_data = torch.load("../pseudo_label_list/MixHop_{}_label_list.pt".format(args.dataset))
train_val_data = torch.LongTensor(train_val_data).to(device)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True


def run(extra_hyper_num):
    model = MixHop(dataset.num_features, dataset.num_classes, dropout=args.dropout, 
                layer1_pows=args.layer1_pows, layer2_pows=args.layer2_pows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    patience = 0
    epoch = 0
    val_mask_used = data.val_mask.clone()
    val_mask_used[num_train + extra_hyper_num:] = False
    for epoch in range(args.epochs):
        # train
        model.train()
        if extra_hyper_num > 0:
            data.y[val_mask_used] = train_val_data[:extra_hyper_num]
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if extra_hyper_num > 0:
            loss = F.cross_entropy(out[data.train_mask + val_mask_used], data.y[data.train_mask + val_mask_used])
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # evaluate
        model.eval()
        data.y[data.val_mask] = real_val_data
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            val_acc = int(correct) / int(data.val_mask.sum())
            if val_acc > best_val_acc:
                best_model = copy.deepcopy(model)
                best_val_acc = val_acc
            else:
                if patience == args.max_patience:
                    break
                else:
                    patience = patience + 1
    
    model = best_model
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(correct) / int(data.test_mask.sum())
    return best_val_acc, test_acc


if __name__ == '__main__':
    assert len(args.layer1_pows) == 3, 'layer1_pows must have 3 numbers'
    assert len(args.layer2_pows) == 3, 'layer2_pows must have 3 numbers'
    setup_seed(42)
    val_acc_list = []
    test_acc_list = []
    for i in range(10):
        val_acc, test_acc = run(500)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    print('validation accuracy: {} +- {}'.format(np.mean(val_acc_list), np.std(val_acc_list)))
    print('test accuracy: {} +- {}'.format(np.mean(test_acc_list), np.std(test_acc_list)))
