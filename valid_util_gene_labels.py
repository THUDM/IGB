from xmlrpc.client import boolean
from torch_geometric.datasets import Planetoid

from model.ppnp import PPNP
from model.ppnp_pyg import PPNP_PYG
from model.gcn import GCN
from model.gat_pyg import GAT
from model.mixhop import MixHop
import torch
import torch.nn.functional as F
from icecream import install, ic
import numpy as np
from tqdm import tqdm
import random
import argparse
import os
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="GCN",
                    help='Model name. Default: GCN')
parser.add_argument('--dataset', type=str, default="Cora",
                    help='Dataset name. Default: Cora')
args = parser.parse_args()

def train(model, mask, train_data,lr=0.01, weight_decay=5e-4,iters = 200):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(iters):
        optimizer.zero_grad()
        out = torch.log_softmax(model(train_data.x,train_data.edge_index), dim=-1)
        loss = F.nll_loss(out[mask], train_data.y[mask])
        loss.backward()
        optimizer.step()


def eval(model, mask, eval_data):
    model.eval()
    pred = model(eval_data.x,eval_data.edge_index).argmax(dim=1)
    correct = (pred[mask] == eval_data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return pred, acc

def valid_util(model, data, num_classes, num_train=140, num_val=500):
    train_data = data.clone()
    eval_data = data

    train(model, train_data.train_mask, train_data)
    pred, acc = eval(model, eval_data.val_mask, eval_data)
    print("Validation Accuracy: {:.4f}".format(acc))

    init_hyper = pred[num_train:num_train+num_val].clone()
    correct_label = eval_data.y[num_train:num_train+num_val].cpu().numpy()

    train_data.y[num_train:num_train+num_val] = init_hyper

    p_label_list = []

    for cntind, ind in enumerate(range(num_train, num_train+num_val)):
        acc_list = np.zeros(num_classes)
        for p_label in range(num_classes):
            
            if args.model == "PPNPPPP":
                pass

            elif args.model == "MixHop":
                model = MixHop(in_channels=dataset.num_features, out_channels=dataset.num_classes,
                dropout=0, layer1_pows=[160,160,160], layer2_pows=[80,80,80]).to(device)
            else:
                model.reset_parameters()

            train_data.y[ind] = p_label
            train(model, train_data.train_mask+train_data.val_mask, train_data, 
              lr=0.02, weight_decay=0,iters=400)
            pred, acc = eval(model, eval_data.val_mask, eval_data)
            ic(p_label,acc)
            ic(pred[ind])
            acc_list[p_label] = acc
        

        if np.where(acc_list == np.max(acc_list))[0].shape[0] > 1:
            best_p_label = init_hyper[ind-num_train].cpu().item()
            
        else:
            best_p_label = np.argmax(acc_list)
        ic(best_p_label)
        p_label_list.append(best_p_label)
        train_data.y[ind] = best_p_label
        # print(p_label_list, correct_label[:len(p_label_list)])
        sum = (np.array(p_label_list) == correct_label[:len(p_label_list)]).astype(np.int32).sum()
        ic(p_label_list, correct_label[:len(p_label_list)])
        ic(sum)
        ic(len(p_label_list))
    os.makedirs("label_lists", exist_ok=True)
    torch.save(np.array(p_label_list), 'label_lists/{}_{}_label_list.pt'.format(args.model, args.dataset))
    print(p_label_list)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='./data/', name=args.dataset)
    setup_seed(3)
    data = dataset[0].to(device)
    if args.model == "GCN":
        improved = (args.dataset=="CiteSeer")
        model = GCN(in_channels=dataset.num_node_features, hidden_channels=16,
        out_channels=dataset.num_classes,dropout=0,improved = improved).to(device)
    elif args.model == "PPNP":
        model = PPNP_PYG(num_features = dataset.num_features,
        num_hidden=16, num_classes=dataset.num_classes, dropout=0,K=10,alpha=0.5).to(device)

    elif args.model == "GAT":
        model = GAT(in_channels=dataset.num_node_features, hidden_channels=16, 
        out_channels=dataset.num_classes, dropout=0, attn_drop=0., output_heads=1, nhead=8).to(device)
    elif args.model == "MixHop":
        model = MixHop(in_channels=dataset.num_features, out_channels=dataset.num_classes,
        dropout=0, layer1_pows=[160,160,160], layer2_pows=[80,80,80]).to(device)
    valid_util(model, data, dataset.num_classes,num_train=dataset[0].train_mask.sum(),num_val=500)
