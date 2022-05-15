import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from tqdm import tqdm

import os, sys; sys.path.append(os.path.dirname(os.path.realpath("../model")))
from model.gcn import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in tqdm(range(num_classes)):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
   
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None,train_val_data=[]):

    val_losses, val_accs, accs, durations = [], [], [], []
    train_val_data = torch.LongTensor(train_val_data).to(device)
    for _ in range(runs):
        data = dataset[0]
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        val_data = data.y[data.val_mask].clone()
        # GCN on Cora: 
        
        # train_val_data = torch.LongTensor([3, 1, 2, 2, 2, 2, 0, 2, 0, 0, 4, 3, 1, 4, 3, 3, 3, 2, 1, 5, 1, 2, 4, 2, 2, 1, 2, 2, 3, 1, 1, 1, 2, 2, 2, 3, 5, 2, 1, 4, 0, 2, 2, 3, 3, 2, 3, 5, 5, 2, 3, 5, 3, 4, 3, 5, 4, 3, 3, 3, 2, 2, 4, 3, 2, 5, 3, 3, 1, 3, 5, 5, 4, 2, 4, 5, 3, 2, 0, 1, 2, 4, 3, 4, 0, 3, 0, 1, 1, 1, 1, 1, 0, 3, 5, 4, 5, 3, 1, 3, 0, 1, 5, 3, 2, 5, 4, 5, 4, 2, 5, 2, 5, 2, 4, 2, 5, 0, 3, 1, 1, 3, 2, 2, 5, 3, 2, 0, 0, 5, 3, 0, 5, 5, 5, 2, 2, 0, 2, 4, 3, 3, 0, 1, 3, 4, 0, 3, 3, 0, 1, 0, 5, 2, 5, 4, 0, 3, 3, 3, 3, 1, 1, 4, 4, 4, 0, 1, 4, 2, 3, 5, 1, 4, 4, 2, 2, 0, 2, 2, 4, 4, 2, 2, 4, 5, 3, 2, 5, 3, 3, 2, 1, 3, 2, 1, 4, 3, 1, 4, 1, 1, 2, 5, 5, 5, 0, 3, 1, 1, 5, 5, 4, 4, 4, 1, 3, 3, 4, 4, 2, 0, 4, 3, 5, 3, 2, 1, 1, 5, 2, 2, 5, 3, 5, 0, 5, 3, 1, 4, 5, 0, 0, 2, 1, 2, 1, 1, 2, 1, 0, 4, 0, 2, 5, 2, 4, 2, 5, 3, 5, 1, 1, 3, 5, 1, 3, 2, 2, 0, 3, 5, 4, 4, 2, 2, 2, 3, 2, 2, 3, 4, 3, 3, 1, 4, 3, 3, 4, 1, 5, 4, 4, 3, 0, 1, 2, 4, 1, 4, 3, 4, 3, 2, 0, 5, 3, 2, 3, 2, 4, 3, 5, 2, 2, 2, 4, 4, 1, 2, 4, 4, 4, 5, 2, 1, 3, 5, 5, 5, 2, 2, 4, 2, 2, 4, 1, 3, 3, 3, 2, 1, 4, 3, 2, 4, 3, 0, 2, 2, 1, 0, 0, 5, 5, 4, 4, 1, 0, 0, 2, 4, 1, 3, 5, 4, 1, 1, 0, 0, 4, 1, 0, 4, 4, 2, 3, 5, 0, 2, 5, 1, 1, 1, 4, 5, 2, 4, 5, 1, 3, 2, 3, 1, 2, 4, 4, 2, 5, 4, 1, 2, 3, 4, 2, 3, 4, 2, 2, 2, 3, 4, 2, 1, 1, 1, 3, 1, 1, 2, 2, 4, 4, 0, 3, 4, 4, 5, 3, 2, 4, 2, 2, 0, 2, 4, 5, 2, 2, 2, 4, 3, 4, 4, 4, 3, 4, 4, 3, 5, 3, 2, 3, 4, 1, 3, 3, 5, 5, 1, 1, 1, 4, 1, 4, 2, 2, 4, 1, 1, 2, 1, 3, 3, 2, 2, 3, 5, 3, 3, 2, 1, 5, 5, 1, 4, 1, 3, 0, 4, 4, 3, 3, 5, 4, 5, 2, 5, 5, 5] ).to(device) 

        
        for epoch in range(1, epochs + 1):
            data.y[data.val_mask] = train_val_data
            train(model, optimizer, data)
            data[data.val_mask] = val_data
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                val_acc = eval_info['val_acc']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():

                    break
        # print(epoch)                
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        val_accs.append(val_acc)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, val_accs, acc, duration = tensor(val_losses), tensor(val_accs), tensor(accs), tensor(durations)

    print(f'Val Loss: {float(loss.mean()):.4f}, '
          f'Valid Accuracy: {float(val_accs.mean()):.3f} ± {float(val_accs.std()):.3f},'
          f'Test Accuracy: {float(acc.mean()):.3f} ± {float(acc.std()):.3f}, '
          f'Duration: {float(duration.mean()):.3f}')
    return acc.mean()


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = F.log_softmax(out, dim=1)
    # out = model(data)
    loss = F.nll_loss(out[data.train_mask+data.val_mask], data.y[data.train_mask+data.val_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, dim=1)


    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data[f'{key}_mask']
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs[f'{key}_loss'] = loss
        outs[f'{key}_acc'] = acc

    return outs
