import pickle
import sys
sys.path.append("..")
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import dgl
from models.gnn import GNN
import torch.nn.functional as F
import tqdm
import copy
from dgl.data.utils import load_graphs
import argparse

time_start=time.time()

class Importancer:
    def __init__(self, params):
        self.params = params
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prj_path = Path(__file__).parent.resolve().parent.resolve()
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{self.params.gpu}')
        self.graph_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'graphs' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        self.model_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'models' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type / 'model'
        self.graph_trainval = load_graphs(str(self.graph_path / f'{self.params.feature_type}.bin'))[0][0] 
        self.dense_dim = len(self.graph_trainval.ndata['nfea'][0])
        print('original graph loaded')
        # load trainval info
        mirnas = pd.read_csv(self.prj_path/'data'/'trainval_data'/self.params.feature_type/'csv_mi.csv', index_col=0)
        lncrnas = pd.read_csv(self.prj_path/'data'/'trainval_data'/self.params.feature_type/'csv_lnc.csv', index_col=0)
        self.Label_data = pd.read_csv(self.prj_path / 'data' / 'trainval_data' / self.params.feature_type / 'RNA-RNA-Interacting.csv', index_col=0)
        self.num_micR = len(mirnas)
        self.num_LncR = len(lncrnas)
        bits_mi = mirnas.columns[3:]
        bits_lnc = lncrnas.columns[3:]
        self.num_edges_single = len(self.Label_data['Label'])
        if (bits_mi==bits_lnc).all() and self.num_micR+self.num_LncR==self.graph_trainval.num_nodes():
            print('data matched')
            self.bits = bits_mi.values
        else: 
            print('data are not matched') 
            sys.exit()
        # id2idx dic
        self.id2nodeidx_trainval = np.load(self.graph_path / 'id2nodeidx_trainval.npy', allow_pickle=True).item()                
        self.num_label_kinds = 2       
        # sampler define
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.params.n_layers) # all nodes information

    def run_kth_on_bit(self, graph, edgeidxs, model):
        run_dataloader = dgl.dataloading.EdgeDataLoader(graph, edgeidxs, self.sampler,
                                                        exclude = 'reverse_id',
                                                        reverse_eids = torch.cat([torch.arange(self.num_edges_single) + self.num_edges_single, torch.arange(self.num_edges_single)]),
                                                        batch_size = self.params.batch_size,
                                                        shuffle = False,
                                                        drop_last = False,
                                                        num_workers = 1,
                                                        device = self.device)
        # Sets the module in evaluating mode
        model.eval()
        batch_num = 0
        total_loss = 0
        edge_labels = torch.tensor([],device=self.device)
        edge_logits = torch.tensor([],device=self.device)
        edge_ids = torch.tensor([],device=self.device)
        with tqdm.tqdm(run_dataloader) as tq:
            for step, (input_nodes, pair_graph, blocks_MFGS) in enumerate(tq):
                blocks = [b.to(self.device) for b in blocks_MFGS]
                edge_subgraph = pair_graph.to(self.device)
                batch_ids = edge_subgraph.edata['pair_id'].squeeze(-1)
                batch_labels = edge_subgraph.edata['Labels'].squeeze(-1)
                input_features = blocks[0].srcdata['nfea']
                with torch.no_grad():
                    logits = model(edge_subgraph, blocks, input_features, tasktype = 'val')
                run_loss = F.cross_entropy(logits, batch_labels.long(), reduction = 'sum')
                # run_loss = F.binary_cross_entropy(F.sigmoid(logits[:,1]), batch_labels, reduction = 'sum')
                # save info of batches in one epoch
                batch_num += 1
                total_loss += run_loss.item()
                edge_labels = torch.cat([edge_labels, batch_labels.detach()],dim=0)
                edge_logits = torch.cat([edge_logits, logits.detach()],dim=0)
                edge_ids = torch.cat([edge_ids, batch_ids.detach()],dim=0)

                tq.set_postfix({'Batch:':batch_num, 'edgeprediction[0]':logits.detach()[0], 'edgelabel[0]': '%.03f' % batch_labels.detach()[0], 'batch_loss': '%.03f' % run_loss.item()}, refresh=False)

        mean_loss = total_loss/len(edgeidxs)
        return mean_loss, edge_labels.cpu(), edge_logits.cpu(), edge_ids.cpu()

    def run_KFold(self, k):
        kth_graph = copy.deepcopy(self.graph_trainval)
        # k'th model
        kth_model = GNN(in_feats = self.dense_dim,
                    n_hidden = self.params.hidden_dim,
                    n_classes = self.num_label_kinds,
                    n_layers = self.params.n_layers,
                    node_num = self.num_micR + self.num_LncR,
                    id2nodeidx = self.id2nodeidx_trainval,
                    activation = F.relu,
                    dropout = self.params.dropout).to(self.device)
        self.load_kth_model(model=kth_model, k=k)
        trainval = self.Label_data.index.values
        # run original loss value
        _loss, _edge_labels, _edge_logits, _edge_ids = self.run_kth_on_bit(graph=kth_graph, edgeidxs=trainval, model=kth_model)
        # initialize
        results = []
        indexs = []
        labels, logits, eids = [], [], []
        for rnatype in ['mirna', 'lncrna']:
            print(f'working on {rnatype}')
            if rnatype=='mirna':
                for idx, bit in enumerate(self.bits):
                    print(f'>>>running bit of {rnatype}_{bit}<<<')
                    time_start = time.time()
                    bit_graph = copy.deepcopy(self.graph_trainval)
                    bit_graph.ndata['nfea'][:self.num_micR,idx]=torch.tensor(0, dtype=torch.float32)
                    loss, edge_labels, edge_logits, edge_ids = self.run_kth_on_bit(graph=bit_graph, edgeidxs=trainval, model=kth_model)
                    res = loss-_loss # >0 important
                    print(f'<<<importance of {rnatype}_{bit}: {res:.4f}>>>')
                    print('time cost: %.4f min' % ((time.time()-time_start)/60))
                    indexs.append(f'{rnatype}_{bit}')
                    results.append(res)
                    labels.append(edge_labels)
                    logits.append(edge_logits)
                    eids.append(edge_ids)

            elif rnatype=='lncrna':
                for idx, bit in enumerate(self.bits):
                    print(f'>>>running bit of {rnatype}_{bit}<<<')
                    time_start = time.time()
                    bit_graph = copy.deepcopy(self.graph_trainval)
                    bit_graph.ndata['nfea'][self.num_micR:,idx]=torch.tensor(0, dtype=torch.float32)
                    loss, edge_labels, edge_logits, edge_ids = self.run_kth_on_bit(graph=bit_graph, edgeidxs=trainval, model=kth_model)
                    res = loss-_loss # >0 important
                    print(f'<<<importance of {rnatype}_{bit}: {res:.4f}>>>')
                    print('time cost: %.4f min' % ((time.time()-time_start)/60))
                    indexs.append(f'{rnatype}_{bit}')
                    results.append(res)
                    labels.append(edge_labels)
                    logits.append(edge_logits)
                    eids.append(edge_ids)

        kth_importance = pd.DataFrame(data=results, index=indexs, columns=[f'importance'])
        with open(f'./importance.pkl', 'wb') as pickle_file:
            pickle.dump((_loss,results,self.bits,labels,logits,eids), pickle_file)
        return kth_importance

    def run(self):
        k = 2
        kth_importance = self.run_KFold(k)
        print("......leave-one-out importance finished......")
        kth_importance.to_csv('./importance.csv')

    def load_kth_model(self, model, k):
        model_path = self.model_path / f'{self.params.feature_type}_{k}th_Fold.pt'
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state['model'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss") 
    parser.add_argument("--n_epochs", type=int, default=512, help="number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="number of hidden GraphSAGE units")
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden NodeSAGE layers")
    parser.add_argument("--feature_type", type=str, default='RNA_intrinsic', help="the type of RNA feature representation which would be applied")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--KFold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--KFold_val_type", type=str, default='pair', help="K-Fold Cross-Validation dataset splitting on pair")      
    params = parser.parse_args()
    print(vars(params))
    runner = Importancer(params)
    runner.run() 

