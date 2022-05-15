import argparse
import os
import json

def default_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epoch. Default: 200')
    parser.add_argument('--eval_times', type=int, default=100,
                        help='Eval times of training. Default: 100')
    parser.add_argument('--model', type=str, default="GCN",
                        help='Model for benchmarking. Default: GCN')
    parser.add_argument('--data_root', type=str, default="./data/",
                        help='Root of local data. Default: ./data/')
    parser.add_argument('--dataset', type=str, default="AMiner",
                        help='Dataset for benchmarking. Default: ["AMiner"]')
    parser.add_argument('--learning_rate_list', nargs='+', type=float, default=[1e-2],
                        help='List of learning rate during optimization. Default: [1e-2]')
    parser.add_argument('--drop_rate_list', nargs='+', type=float, default=[0.5],
                        help='List of drop rate of the Dropout Layer. Default: [0.5]')
    parser.add_argument('--num_layers_list', nargs='+', type=int, default=[2],
                        help='List of number of layers. Default: [2]')
    parser.add_argument('--order_list', nargs='+', type=int, default=[8],
                        help='List of order of Grand. Default: [8]')
    parser.add_argument('--hidden_size_list', nargs='+', type=int, default=[16,32, 64],
                        help='hidden size lsit eg: [16,32, 64]. Default: [16,32, 64]')
    parser.add_argument('--test_datasets', nargs='+', type=str, default=["Cora",],
                        help='The test dataset for benchmarking results. Default: Cora')
    parser.add_argument('--walk_length', type=int, default=5000, 
                        help='Walk length for random walk. Default: 5000')
    parser.add_argument('--burning_round', type=int, default=5, 
                        help='Burning round for random walk initialization. Default: 5')
    parser.add_argument('--max_patience', type=int, default=100,
                        help='Maximum patience for early stopping. Default: 100')
    parser.add_argument('--weight_decay', type =float, default=1e-4, 
                        help='Weight decay for optimizer. Default: 1e-4')
    parser.add_argument('--eval_step', type=int, default=1,
                        help='Evaluation step. Default: 1') 
    parser.add_argument('--alpha_list', nargs='+', type=float, default=[0.1],
                        help='Alpha list. Default: [0.1]')
    parser.add_argument('--num_iterations_list',nargs='+', type =int, default=[8],
                        help='Number of iterations for PPNP. Default: [8]')
    parser.add_argument('--wd1', type =float, default=0,
                        help='wd1 for GCNII. Default: 0')
    parser.add_argument('--wd2', type =float, default=0,
                        help='wd2 for GCNII. Default: 0')
    parser.add_argument('--lmbda_list', nargs='+', type=float, default=[0.5],
                        help='lmbda_list. Default: [0.5]')
    parser.add_argument('--layer_pows_list', nargs='+', type=int, default=[20, 20, 20, 20, 20, 20],
                        help='layer_pows list for MixHop. Default: [20, 20, 20, 20, 20, 20]')
    parser.add_argument('--use_bn_list', nargs='+', type=int, default=[0],
                        help='use_bn list for Grand. Default: [0]')
    parser.add_argument('--use_residual_list', nargs='+', type=int, default=[0],
                        help='use_residual list for Grand. Default: [0]')
    parser.add_argument('--use_loop_list', nargs='+', type=int, default=[0],
                        help='use_loop list for DropEdge. Default: [0]')
    parser.add_argument('--aggr_list', nargs='+', type=str, default=["mean"],
                        help='aggr list for GraphSage. Default: ["mean"]')
    parser.add_argument('--nhead_list', nargs='+', type=int, default=[8],
                        help='nhead list for GAT. Default: [8]')
    parser.add_argument('--sample_size_list', nargs='+', type=int, default=[10, 10],
                        help='sample_size list for GraphSage. Default: [10, 10]')
    parser.add_argument('--k_list', nargs='+', type=int, default=[4])
    parser.add_argument('--input_drop_list', nargs='+', type=float, default=[0.4])
    parser.add_argument('--dropnode_list', nargs='+', type=float, default=[0.5])
    parser.add_argument('--temp_list', nargs='+', type=float, default=[1])
    parser.add_argument('--sample_num_list', nargs='+', type=float, default=[2])
    parser.add_argument('--attn_drop_list', nargs='+', type=float, default=[0.5])
    parser.add_argument('--GAT_normal', type=int, default=1)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--best_epoch', type=int, default=0)
    parser.add_argument('--use_seed', type=int, default=1)

    return parser.parse_args()



def check_args(args):
    assert args.model in ["GCN","PPNP","GCNII","Grand","MixHop","GAT","GraphSage","DropEdge"], "Model must be GCN, PPNP, GCNII, Grand, MixHop, GAT, GraphSage, or DropEdge"
    assert args.dataset in ["AMiner","Facebook","NELL","Flickr","Reddit","Cora","Pubmed","Citeseer"], " Dataset must be AMiner, Facebook, NELL, Flickr, Reddit, Cora, or Pubmed"
    assert len(args.learning_rate_list) > 0, " learning rate list is empty"
    assert len(args.num_layers_list)>0, " num_layer list is empty"
    assert len(args.drop_rate_list) > 0, "drop_rate list is empty"
    assert len(
        args.hidden_size_list)>0, " hidden_size_list is empty"
    assert len(args.layer_pows_list) >= 6, "layer_pows_list must have at least 6 integers"
    assert len(args.layer_pows_list) % 6 == 0, "layer_pows_list's length must be a positive multiple of 6"
    assert len(args.use_bn_list) > 0, "use_bn list is empty"
    use_bn_list = []
    for b in args.use_bn_list:
        assert b in [0, 1], "Use_bn_list only supports 0 and 1"
        if b == 1:
            use_bn_list.append(True)
        else:
            use_bn_list.append(False)
    args.use_bn_list = use_bn_list
    assert len(args.use_residual_list) > 0, "use_residual list is empty"
    use_res_list = []
    for b in args.use_residual_list:
        assert b in [0, 1], "Use_bn_list only supports 0 and 1"
        if b == 1:
            use_res_list.append(True)
        else:
            use_res_list.append(False)
    use_loop_list = []
    for b in args.use_loop_list:
        assert b in [0, 1], "Use_loop_list only supports 0 and 1"
        if b == 1:
            use_loop_list.append(True)
        else:
            use_loop_list.append(False)
    args.use_loop_list = use_loop_list
    for aggr in args.aggr_list:
        assert aggr in ["mean", "sum", "max"], "aggr only supports mean, sum, or max"
    if args.model == 'GraphSage':
        layer_attr_list = []
        s_cnt = 0
        h_cnt = 0
        for i in args.num_layers_list:
            s_rec = []
            h_rec = []
            for j in range(i):
                s_rec.append(args.sample_size_list[s_cnt])
                s_cnt += 1
            for j in range(i - 1):
                h_rec.append(args.hidden_size_list[h_cnt])
                h_cnt += 1
            layer_attr_list.append({'layer': i, 'sample_size': s_rec, 'hidden_size': h_rec})
        args.layer_attr_list = layer_attr_list


    return args

