import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run
import random
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath("../model")))
from model.gcn import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
args = parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True
setup_seed(0)

dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
train_val_data = torch.load("../pseudo_label_list/GCN_{}_label_list.pt".format(args.dataset))
improved = (args.dataset=="CiteSeer")
print("using improved gcn: {}".format(improved))
model = GCN(
    in_channels = dataset.num_features,
    hidden_channels = args.hidden,
    out_channels = dataset.num_classes,
    dropout = args.dropout,
    improved = improved
)

permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, permute_masks,train_val_data=train_val_data)

