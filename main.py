from numpy.core.fromnumeric import argmax
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import torch.nn.functional as F
from model.gcn import GCN
from icecream import ic
from collections import defaultdict
import numpy as np
import argparse
from dataset import AMiner
import os
import itertools
from icecream import install
from dataset.AMiner import AMiner
from dataset.Facebook import Facebook
from dataset.NELL import NELL
from dataset.Flickr import Flickr
from model.gcn import GCN
from model.ppnp import PPNP
from model.gcnii import GCNII
from model.grand import Grand
from model.mixhop import MixHop
from model.graphsage import GraphSage
from model.gat_pyg import GAT
from icecream import ic
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import tqdm
import copy
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import random
from arguments import check_args, default_args
install()



def get_model(args, num_features, num_classes):
    if args.model == "GCN":
        iter_list = [args.drop_rate_list, args.hidden_size_list]
    elif args.model == "PPNP":
        iter_list = [args.drop_rate_list, args.hidden_size_list, args.num_layers_list, args.alpha_list, args.num_iterations_list, args.lmbda_list]
    elif args.model == "GCNII":
        iter_list = [args.drop_rate_list, args.hidden_size_list, args.alpha_list, args.num_layers_list, args.lmbda_list]
    elif args.model == "Grand":
        iter_list = [args.drop_rate_list, args.hidden_size_list, args.order_list, args.use_bn_list, args.sample_num_list, \
            args.dropnode_list, args.input_drop_list, args.lmbda_list, args.temp_list]
    elif args.model == "MixHop":
        pows_list = []
        for i in range(int(len(args.layer_pows_list)/6)):
            pows_list.append(args.layer_pows_list[6 * i : 6 * i + 6])
        iter_list = [args.drop_rate_list, pows_list]
    elif args.model == "GAT":
        iter_list = [args.drop_rate_list, args.hidden_size_list, args.nhead_list, args.attn_drop_list]
    elif args.model == "GraphSage":
        iter_list = [args.drop_rate_list, args.layer_attr_list, args.aggr_list]
    elif args.model == "DropEdge":
        iter_list = [args.drop_rate_list, args.hidden_size_list, args.num_layers_list, args.use_bn_list, args.use_loop_list]
    hyper_par_list = itertools.product(*iter_list)

    for counter, hyper_par in enumerate(hyper_par_list):
        if args.model == "GCN":
            model = GCN(in_channels=num_features,
                        hidden_channels=hyper_par[1], out_channels=num_classes, dropout=hyper_par[0])
        elif args.model == "PPNP":
            model = PPNP(nfeat=num_features, nclass=num_classes,
            nhid = hyper_par[1],num_layers = hyper_par[2],
            dropout = hyper_par[0],alpha=hyper_par[3],
            niter=hyper_par[4], lmbda=hyper_par[5])
        elif args.model == "GCNII":
            model = GCNII(in_channels=num_features, hidden_channels=hyper_par[1], 
            out_channels=num_classes, dropout=hyper_par[0], lmbda=hyper_par[4], alpha=hyper_par[2],
            num_layers=hyper_par[3])
        elif args.model == "Grand":
            model = Grand(in_channels=num_features, hidden_channels=hyper_par[1], 
            out_channels=num_classes, hidden_droprate=hyper_par[0], order=hyper_par[2],
            use_bn=hyper_par[3], sample_num=hyper_par[4], dropnode_rate=hyper_par[5], input_droprate=hyper_par[6], 
            lam=hyper_par[7], temp=hyper_par[8])
        elif args.model == "MixHop":
            model = MixHop(in_channels=num_features, out_channels=num_classes, 
            dropout=hyper_par[0], layer1_pows=hyper_par[1][0:3], layer2_pows=hyper_par[1][3:6])
        elif args.model == "GAT":
            model = GAT(in_channels=num_features, hidden_channels=hyper_par[1],
                out_channels=num_classes, dropout=hyper_par[0], nhead=hyper_par[2], attn_drop=hyper_par[3])
        elif args.model == "GraphSage":
            model = GraphSage(in_channels=num_features, hidden_channels=hyper_par[1]['hidden_size'],
            out_channels=num_classes, dropout=hyper_par[0], num_layers=hyper_par[1]['layer'], sample_size=hyper_par[1]['sample_size'], aggr=hyper_par[2])
        elif args.model == "DropEdge":
            model = DropEdge_GCN(in_channels=num_features, hidden_channels=hyper_par[1],
            out_channels=num_classes, dropout=hyper_par[0], nhidlayer=hyper_par[2], withbn=hyper_par[3], withloop=hyper_par[4])
        yield model

def train_and_eval(model, data, epochs,device,args,learning_rate, eval=False):
    best_acc = 0
    best_model = None
    patience = 0
    model.to(device)
    data.to(device)
    use_grand = (args.model == "Grand")

    if args.model == 'GCNII':
        optimizer = torch.optim.Adam(
            [
                {"params": model.fc_parameters, "weight_decay": args.wd1},
                {"params": model.conv_parameters, "weight_decay": args.wd2},
            ],
            lr=learning_rate,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    
    model.train()
    best_epoch = epochs
    for epoch in range(epochs):

        if(eval==True):
            X = data.train_mask
        else:
            X = data.train_mask+data.val_mask
        optimizer.zero_grad()
        if use_grand:
            output_list = []
            loss_train = 0.
            K = int(model.sample_num)
            for k in range(K):
                output = torch.log_softmax(model(data.x, data.edge_index), dim=-1)
                loss_train += F.nll_loss(output[X], data.y[X])
                output_list.append(output)
            loss_train = loss_train/K
            loss_consis = model.consis_loss(output_list)
            loss = loss_train + loss_consis
        elif args.model == "PPNP":
            pred = model(data.x, data.edge_index)
            loss = F.cross_entropy(pred[X], data.y[X])
            loss = loss + model.lmbda * (torch.sum(model.nn.nn.mlp[0].weight ** 2))
        else:
            pred = model(data.x, data.edge_index)
            loss = F.cross_entropy(pred[X], data.y[X])

        loss.backward()
        optimizer.step()

        if(eval==True and epoch%args.eval_step == 0):
            model.eval()
            with torch.no_grad():
                if args.model == "GraphSage":
                    pred = model.inference(data.x, data.edge_index).argmax(dim=1)
                else:
                    pred = model(data.x, data.edge_index).argmax(dim=1)
                correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
                acc = int(correct) / int(data.val_mask.sum())
                if acc > best_acc:
                    best_model = copy.deepcopy(model)
                    best_acc = acc
                    best_epoch = epoch
                else:
                    if patience >= args.max_patience and epoch >= epochs/2:
                        break
                    else:
                        patience = patience + 1
                
            model.train()
    
    if best_model==None:
        best_model = model 
    best_model.eval()
    if args.model == "GraphSage":
        pred = best_model.inference(data.x, data.edge_index).argmax(dim=1)
    else:
        pred = best_model(data.x, data.edge_index).argmax(dim=1)
    if(eval==True):
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())
    else:
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    if args.best_epoch == 0:
        return round(acc,4),int(epoch)
    else:
        return round(acc,4),int(best_epoch)



def setup_seed(seed):
    print("using seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_list  = []

    datapath = os.path.join(args.data_root,  '{}.pt'.format(args.dataset))
    dataset = torch.load(datapath)
    new_dataset = []
    for data in dataset[2:]:
        data = data[0]
        if args.model == "GAT" and args.GAT_normal == 1:
            data.transform = T.NormalizeFeatures()
            data = data.transform(data)
        new_dataset.append(data)
    dataset = dataset[0:2] + new_dataset
    dataset_list.append(dataset)
    
    plot_result_list = {}

    for dataset, dataset_name in zip(dataset_list, args.dataset):
    
        result_list = defaultdict(lambda: [])
        num_features = dataset[0]
        num_classes = dataset[1]
        for eval_num in tqdm.tqdm(range(args.eval_times)):
            data = dataset[eval_num + 2]
            epochs = args.num_epochs
            for learning_rate in args.learning_rate_list:
                count = 0
                model_list = []
                for model in get_model(args, num_features, num_classes):
                    count +=1
                    model_list.append(model.__repr__() + " learning_rate: "+str(learning_rate))
                    result_list[model.__repr__()+" learning_rate: "+str(learning_rate)].append(train_and_eval(model, data, epochs,device,args,learning_rate,eval=True))
                    # ic(result_list[model.__repr__()+" learning_rate: "+str(learning_rate)],model.__repr__()+" learning_rate: "+str(learning_rate))
        for k in result_list.keys():
            results = []
            epochs = []
            for (i,j) in result_list[k]:
                results.append(i)
                epochs.append(j)
            result_list[k] = (np.mean(results),np.std(results),epochs)


        best_par = np.argmax([result_list[k][0] for k in model_list])

        epochs = result_list[list(result_list.keys())[int(best_par)]][2]


        lr = args.learning_rate_list[int(best_par/count)]
        model_num = best_par%count

        acc_list = []
        for eval_num,e in zip(range(args.eval_times),epochs):
            count = 0
            for model in get_model(args, num_features, num_classes):
                if count == model_num:
                    model_name = model.__repr__()

                    data = dataset[eval_num + 2]
                    acc_list.append(train_and_eval(model, data, e,device,args,lr,eval=False)[0])
                    break
                else:
                    count += 1

        # ic(lr,model_name,np.mean(acc_list),np.std(acc_list))
        print("model_name: {} lr: {}".format(model_name,lr))
        print("Accuracy: {} +- {}".format(np.mean(acc_list),np.std(acc_list)))
        plot_result_list[dataset_name] = dict(result_list)
    
    return plot_result_list, np.mean(acc_list), np.std(acc_list)


if __name__ == "__main__":
    args = default_args()
    args = check_args(args)
    if args.use_seed == 1:
        total_result_list = []
        total_acc_list = []
        total_std_list = []
        
        for seed in args.seeds:
            setup_seed(seed)
            result, acc, std = main(args)
            total_acc_list.append(acc)
            total_std_list.append(std)
        # ic(args.seeds)
        # ic(total_acc_list)
        # ic(total_std_list)
    else:
        result, acc, std = main(args)
