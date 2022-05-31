# IGB

[Rethinking the Setting of Semi-supervised Learning on Graphs.](https://arxiv.org/abs/2205.14403) (Li et al., IJCAI 2022)

## Overview
Here we present the PyTorch implementation of ValidUtil, IGB evaluation pipeline and the script to reproduce our results. \
The repository is organised as follows:
* `dataset/` contains the implementation of the 4 datasets in **IGB**.
* `model/` contains the implementation of the GNN models in **IGB**.
* `scripts/` contains the `shell` scripts to reproduce **IGB** evaluation results.
* `validutil/` and `valid_util_gene_labels.py` contains the implementation of **ValidUtil**.
* `main.py` is the implementation of the evaluation pipeline we present in our paper.



## Download Data

We provide both `Tsinghua Cloud` and `Google Drive` links to download **IGB** datasets.

```shell
# Tsinghua Cloud
wget https://cloud.tsinghua.edu.cn/d/834bd180cabf488cb553/files/\?p\=%2Fdata.tar.gz\&dl\=1 -O data.tar.gz
# Google Drive Link
https://drive.google.com/file/d/1th423dYfzOImXN_9CQFSqRD2AUi-3VmC/view?usp=sharing
tar -zxvf data.tar.gz
```

## Python Environment

* python 3.9.7

```shell
pip install -r requirements.txt
```

## ValidUtil

We can first generate the pseudo-labels for the GNN models. In this step, we will search and fix the best pseudo-label one by one.

```shell
python valid_util_gene_labels.py --model <GNN_MODEL> --dataset <DATASET>
```
You may use `&>` to redirect the output into a file. <GNN_MODEL> can be chosen from: `GCN, MixHop, PPNP`.

For `GCN on CiteSeer`, we add an additional self-loop for each node for the GCN model.

\<DATASET> can be chosen from `Cora, CiteSeer, PubMed`.

The result list will be saved at `label_lists/<GNN_MODEL>_<DATASET>_label_list.pt`. And we also provide our raw results in `pseudo_label_list/`, they can be directly used as an input for the next step. Even though we have already set up a manual seed, there is still a fluctuation in the results of `MixHop on Cora`. We get 10 different results in this step (see in `pseudo_label_list/MixHop_Cora_label_list_<0-9>.pt`) , and use them as the inputs of the next step and report the average performance. 

For the next step, we train the GNN model with the additional pseudo-labels and compare the result with that of the original model. 

To reproduce the ValidUtil results:

```shell
cd validutil/
./final_scripts.sh
```

|        | Cora     | CiteSeer | PubMed   |
| ------ | -------- | -------- | -------- |
| GCN    | **85.8** | 76.0     | 83.8     |
| MixHop | 84.9     | 75.5     | 84.2     |
| PPNP   | **85.8** | **77.3** | **84.7** |

Here we still need to emphasize that **ValidUtil** is ***not*** really a method to improve GNNs, but more like a “reduction to absurdity”.

## Evaluate on IGB

To reproduce the evaluation results of the 7 models on IGB:

```shell
./scripts/run_<GNN_MODEL>_<DATASET>.sh
```

<GNN_MODEL> can be chosen from: `GCN, GAT, GCNII, Grand, Graphsage, MixHop, PPNP`.

\<DATASET> can be chosen from `AMiner, Facebook, NELL, Flickr`.

|           | AMiner       | Facebook     | NELL         | Flickr       |
| --------- | ------------ | ------------ | ------------ | ------------ |
| Grand     | 82.5±0.8 | 88.9±1.0 | 84.4±1.1 | 44.3±0.8 |
| GCN       | 76.5±1.1 | 87.9±1.0 | 93.9±0.7 | 41.9±1.3 |
| GAT       | 78.8±1.0 | 88.3±1.2 | 91.1±1.2 | 43.1±1.3 |
| GraphSAGE | 81.6±0.8 | 87.2±1.1 | **94.9±0.6** | 43.4±0.9 |
| APPNP     | 87.0±1.0 | 88.0±1.3 | 93.0±0.8 | 44.6±0.9 |
| MixHop    | 86.1±1.1 | 89.1±0.9 | 94.7±0.7 | 43.5±1.2 |
| GCNII     | **88.4±0.6** | **89.5±0.9** | 91.5±1.0 | **44.7±0.8** |

