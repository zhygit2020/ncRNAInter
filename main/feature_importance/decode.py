import os
import psutil
import pickle
import sys
sys.path.append("..")
from pathlib import Path
import joblib
import time
import numpy as np
import pandas as pd
import torch
import dgl
from dgl.contrib.sampling import EdgeSampler
from dgl.contrib.sampling import NeighborSampler
from utils import loaddata
from utils import graph
import torch.nn as nn
from models.gnn import GNN
import torch.nn.functional as F
import tqdm
from sklearn import metrics
import copy
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
import argparse
from dgl.data.utils import save_graphs
prj_path = Path(__file__).parent.resolve().parent.resolve()

def load_kth_model(model, k):
    model_path = prj_path / 'pretrained' / f'KFold_val_based_on_pair' / f'5_Fold_trainval' / 'models' / f'epochsize_512' / f'hiddensize_192' / f'learningrate_0.0005' / 'RNA_intrinsic' / 'model' / f'RNA_intrinsic_{k}th_Fold.pt'
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])

def run_kth_on_bit(graph, edgeidxs, model, num_edges_single):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) # all nodes information
    run_dataloader = dgl.dataloading.EdgeDataLoader(graph, edgeidxs, sampler,
                                                    exclude = 'reverse_id',
                                                    reverse_eids = torch.cat([torch.arange(num_edges_single) + num_edges_single, torch.arange(num_edges_single)]),
                                                    batch_size = 32,
                                                    shuffle = False,
                                                    drop_last = False,
                                                    num_workers = 1,
                                                    device = torch.device('cpu'))
    # Sets the module in evaluating mode
    model.eval()
    batch_num = 0
    total_loss = 0
    edge_labels = torch.tensor([],device=torch.device('cpu'))
    edge_logits = torch.tensor([],device=torch.device('cpu'))
    edge_ids = torch.tensor([],device=torch.device('cpu'))
    with tqdm.tqdm(run_dataloader) as tq:
        for step, (input_nodes, pair_graph, blocks_MFGS) in enumerate(tq):
            blocks = [b.to(torch.device('cpu')) for b in blocks_MFGS]
            edge_subgraph = pair_graph.to(torch.device('cpu'))
            batch_ids = edge_subgraph.edata['pair_id'].squeeze(-1)
            batch_labels = edge_subgraph.edata['Labels'].squeeze(-1)
            input_features = blocks[0].srcdata['nfea']
            with torch.no_grad():
                logits = model(edge_subgraph, blocks, input_features, tasktype = 'val')
            run_loss = F.cross_entropy(logits, batch_labels.long(), reduction = 'sum')
            # run_loss = F.binary_cross_entropy(F.sigmoid(logits[:,1]), batch_labels, reduction = 'sum')
            # run_loss = F.binary_cross_entropy(logits, batch_labels, reduction = 'sum')
            # save info of batches in one epoch
            batch_num += 1
            total_loss += run_loss.item()
            edge_labels = torch.cat([edge_labels, batch_labels.detach()],dim=0)
            edge_logits = torch.cat([edge_logits, logits.detach()],dim=0)
            edge_ids = torch.cat([edge_ids, batch_ids.detach()],dim=0)

            tq.set_postfix({'Batch:':batch_num, 'edgeprediction[0]':logits.detach()[0], 'edgelabel[0]': '%.03f' % batch_labels.detach()[0], 'batch_loss': '%.03f' % run_loss.item()}, refresh=False)

    mean_loss = total_loss/len(edgeidxs)
    return mean_loss, edge_labels.cpu(), edge_logits.cpu(), edge_ids.cpu()


graph_path = prj_path / 'pretrained' / f'KFold_val_based_on_pair' / f'5_Fold_trainval' / 'graphs' / f'epochsize_512' / f'hiddensize_192' / f'learningrate_0.0005' / 'RNA_intrinsic'
k=0
kth_graph = load_graphs(str(graph_path / f'RNA_intrinsic.bin'))[0][0]
dense_dim = len(kth_graph.ndata['nfea'][0])
id2nodeidx_trainval = np.load(graph_path / 'id2nodeidx_trainval.npy', allow_pickle=True).item()
# k'th model
kth_model = GNN(in_feats = dense_dim,
            n_hidden = 192,
            n_classes = 2,
            n_layers = 2,
            node_num = kth_graph.num_nodes(),
            id2nodeidx = id2nodeidx_trainval,
            activation = F.relu,
            dropout = 0.1).to(torch.device('cpu'))
load_kth_model(model=kth_model, k=k)
Label_data = pd.read_csv(prj_path / 'data' / 'trainval_data' / 'RNA_intrinsic' / 'RNA-RNA-Interacting.csv', index_col=0)
trainval = Label_data.index.values
num_edges_single = len(Label_data['Label'])
# run original loss value
_loss, _edge_labels, _edge_logits, _edge_ids = run_kth_on_bit(graph=kth_graph, edgeidxs=trainval, model=kth_model, num_edges_single=num_edges_single)


pickle_file = open(f'./importance_{k}th_model.pkl', 'rb')
# labels,logits,eids=pickle.load(pickle_file)
__loss,__results,__bits,labels,logits,eids=pickle.load(pickle_file)
pickle_file.close()
print(labels[0],logits[0],eids)
mirnas = pd.read_csv(prj_path/'data'/'trainval_data'/'RNA_intrinsic'/'csv_mi.csv', index_col=0)
bits = mirnas.columns[3:]
indexs=[]
for rnatype in ['mirna', 'lncrna']:
    if rnatype=='mirna':
        for idx, bit in enumerate(bits):
            indexs.append(f'{rnatype}_{bit}')
    elif rnatype=='lncrna':
        for idx, bit in enumerate(bits):
            indexs.append(f'{rnatype}_{bit}')
print(len(indexs))

results=[]
__results=[]
ids=[]
for idx,id in enumerate(indexs):
    loss = F.cross_entropy(logits[idx], labels[idx].long(), reduction = 'mean')
    sum_loss = F.cross_entropy(logits[idx], labels[idx].long(), reduction = 'sum')
    print('loss,sum_loss/len(labels[idx])', loss,sum_loss/len(labels[idx]))
    res = (loss-_loss).item()
    __res = (loss-__loss).item()
    results.append(res)
    __results.append(__res)
    ids.append(id)

pd.DataFrame(data={0:results,1:np.concatenate((__bits,__bits),axis=0),2:__results}, index=ids).to_csv('11.csv')