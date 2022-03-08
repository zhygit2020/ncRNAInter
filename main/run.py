import os
import psutil
import time
import numpy as np
import pandas as pd
import torch
import dgl
from dgl.contrib.sampling import EdgeSampler
from dgl.contrib.sampling import NeighborSampler
from utils import loaddata
from utils import graph
from pathlib import Path
import torch.nn as nn
from models.gnn import GNN
import torch.nn.functional as F
import tqdm
from sklearn import metrics
import copy
from sklearn.model_selection import train_test_split
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
from dgl.data.utils import save_graphs

time_start=time.time()

class Runner:
    def __init__(self, params):
        self.params = params
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prj_path = Path(__file__).parent.resolve()
        self.prerna_save_path = self.prj_path / 'run_result' / 'prerna' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        if not self.prerna_save_path.exists():
            self.prerna_save_path.mkdir(parents=True)
        self.prepair_save_path = self.prj_path / 'run_result' / 'prepair' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        if not self.prepair_save_path.exists():
            self.prepair_save_path.mkdir(parents=True)
        self.test_save_path = self.prj_path / 'run_result' / 'test' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        if not self.test_save_path.exists():
            self.test_save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{self.params.gpu}')
        self.graph_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'graphs' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        self.model_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'models' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        self.graph_trainval = load_graphs(str(self.graph_path / f'{self.params.feature_type}.bin'))[0][0] 
        self.dense_dim = len(self.graph_trainval.ndata['nfea'][0])
        print('original graph loaded')
        # graphbuilder
        self.graphbuilder = graph.graphbuilder(self.params)
        if params.task_type == 'trainval':
            print('Error >>> you give the wrong task type')
            sys.exit()
        elif params.task_type == 'run':
            pass
        # id2idx dic
        self.id2nodeidx_trainval = np.load(self.graph_path / 'id2nodeidx_trainval.npy', allow_pickle=True).item()                
        self.num_label_kinds = 2       
        # sampler define
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.params.n_layers) # all nodes information

    def test_KFold(self, k, graph, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, assess):
        print('----------test for %s th Fold model----------' %k)
        task_type = 'test'
        kth_model = GNN(in_feats = self.dense_dim,
                    n_hidden = self.params.hidden_dim,
                    n_classes = self.num_label_kinds,
                    n_layers = self.params.n_layers,
                    node_num = graph.number_of_nodes(),
                    id2nodeidx = self.id2nodeidx_trainval,
                    activation = F.relu,
                    dropout = self.params.dropout).to(self.device)
        self.load_kth_model(model=kth_model, k=k)

        kfold_test_data = {}
        # initialize
        kth_row_labels, kth_row_scores, kth_row_idxs, kth_row_ids = torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([])
        kth_row_losss = 0
        
        with tqdm.tqdm(range(Label_data_pair.shape[0])) as tq:            
            for step, row in enumerate(tq): 
                kth_row_graph, u_name, v_name, Labels = self.graphbuilder.buildgraph(graph, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, row, self.id2nodeidx_trainval)
                test_edgeidx = torch.tensor(kth_row_graph.number_of_edges() - 1, dtype=torch.int64).unsqueeze(0)
                test_dataloader = dgl.dataloading.EdgeDataLoader(kth_row_graph, test_edgeidx, self.sampler,
                                                                batch_size = 1,
                                                                shuffle = False,
                                                                drop_last = False,
                                                                num_workers = 1,
                                                                device = self.device) 
                # Sets the module in evaluating mode
                kth_model.eval()
                for input_nodes, pair_graph, blocks_MFGS in test_dataloader:
                    # feature copy to(self.device)
                    blocks = [b.to(self.device) for b in blocks_MFGS]
                    edge_subgraph = pair_graph.to(self.device)
                    input_features = blocks[0].srcdata['nfea']
                    batch_ids = edge_subgraph.edata['pair_id']
                    with torch.no_grad():
                        logits = kth_model(edge_subgraph, blocks, input_features, tasktype = task_type).cpu()
                kth_row_labels = torch.cat([kth_row_labels, torch.tensor(Labels, dtype=torch.float32)],dim=0)
                kth_row_scores = torch.cat([kth_row_scores, logits],dim=0)
                kth_row_ids = torch.cat([kth_row_ids, batch_ids],dim=0)
                kth_row_idxs = torch.cat([kth_row_idxs, torch.tensor([[row]], dtype=torch.float32)],dim=0)
                kth_row_losss += F.binary_cross_entropy(logits, torch.tensor(Labels, dtype=torch.float32), reduction = 'sum')
                tq.set_postfix({'row': '%s' % row, 'edgeprediction': '%.03f' % logits, 'Label': '%s' % Labels[0][0]}, refresh=False)

        print(f"evaluate test for {k}th Fold")
        fprs, tprs, thresholds, auc, tn, fp, fn, tp, acc, mcc, precision, specificity, recall, f1 = self.evaluate(kth_row_labels.cpu(), kth_row_scores.cpu())
        loss = kth_row_losss.numpy()/Label_data_pair.shape[0]
        # save kth Fold ROC data
        ROC_savepath = self.test_save_path / 'ROC_data' / f'{k}th_Fold'
        if not ROC_savepath.exists():
            ROC_savepath.mkdir(parents=True)
        pd.DataFrame({'fprs':fprs, 'tprs':tprs, 'thresholds':thresholds}).to_csv(ROC_savepath / f'train_ROC_for_{k}th_Fold.csv')

        # save kth Fold score data 
        score_savepath = self.test_save_path / 'score_data' / f'{k}th_Fold'
        if not score_savepath.exists():
            score_savepath.mkdir(parents=True)
        pd.DataFrame({'eidxs':kth_row_idxs.cpu().squeeze(-1), 'eids':kth_row_ids.cpu().squeeze(-1), 'labels':kth_row_labels.squeeze(-1), 'scores':kth_row_scores.squeeze(-1)}).to_csv(score_savepath / f'train_score_for_{k}th_Fold.csv')
        
        # save kth Fold statistic data
        for ass in assess:
            exec(f"kfold_test_data['{ass}'] = {ass}")

        print(f"{k}th Fold >>>> Test ACC {acc:.4f}, Test AUC {auc:.4f}, Test_MCC {mcc:.4f}, Test_loss {loss:.4f}, Test_precision {precision:.4f}, Test_recall {recall:.4f}, Test_specificity {specificity:.4f}, Test_f1 {f1:.4f}", 
            f"Test TN {tn:d}, Test FP {fp:d}, Test_FN {fn:d}, Test_TP {tp:d}")

        return kfold_test_data

    def prepair_KFold(self, k, graph, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, k_average_logits):
        print('----------predict on pairs for %s th Fold model----------' %k)
        task_type = 'predict'
        kth_model = GNN(in_feats = self.dense_dim,
                    n_hidden = self.params.hidden_dim,
                    n_classes = self.num_label_kinds,
                    n_layers = self.params.n_layers,
                    node_num = graph.number_of_nodes(),
                    id2nodeidx = self.id2nodeidx_trainval,
                    activation = F.relu,
                    dropout = self.params.dropout).to(self.device)
        self.load_kth_model(model=kth_model, k=k)

        # initialize
        kth_logits = pd.DataFrame(columns = ['pair_idx','pair_id',f'logits_{k}th_model'])
        with tqdm.tqdm(range(Label_data_pair.shape[0])) as tq:            
            for step, row in enumerate(tq): 
                kth_row_graph, u_name, v_name, Labels = self.graphbuilder.buildgraph(graph, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, row, self.id2nodeidx_trainval)
                test_edgeidx = torch.tensor(kth_row_graph.number_of_edges() - 1, dtype=torch.int64).unsqueeze(0)
                test_dataloader = dgl.dataloading.EdgeDataLoader(kth_row_graph, test_edgeidx, self.sampler,
                                                                batch_size = 1,
                                                                shuffle = False,
                                                                drop_last = False,
                                                                num_workers = 1,
                                                                device = self.device) 
                # Sets the module in evaluating mode
                kth_model.eval()
                for input_nodes, pair_graph, blocks_MFGS in test_dataloader:
                    # feature copy to(self.device)
                    blocks = [b.to(self.device) for b in blocks_MFGS]
                    edge_subgraph = pair_graph.to(self.device)
                    input_features = blocks[0].srcdata['nfea']
                    batch_ids = edge_subgraph.edata['pair_id']
                    with torch.no_grad():
                        logits = kth_model(edge_subgraph, blocks, input_features, tasktype = task_type).cpu()
                    kth_logits = kth_logits.append({'pair_idx':row, 'pair_id':batch_ids.detach().squeeze(-1).squeeze(-1).squeeze(-1).numpy(), f'logits_{k}th_model':logits.detach().squeeze(-1).squeeze(-1).numpy()}, ignore_index=True)
                tq.set_postfix({'row': '%s' % row, 'edgeprediction': '%.03f' % logits}, refresh=False)
        
        # save kth Fold predict result
        kth_logits = kth_logits.astype({'pair_idx': 'int64', 'pair_id': 'int64', f'logits_{k}th_model': 'float32'})               
        k_average_logits = pd.merge(k_average_logits,kth_logits,on=['pair_idx','pair_id'],how='left',sort=False)
        return k_average_logits

    def prerna_KFold(self, k, graph, target, for_target, mi2lnc, k_average_logits):
        print('----------predict on rnas for %s th Fold model----------' %k)
        task_type = 'predict'
        kth_model = GNN(in_feats = self.dense_dim,
                    n_hidden = self.params.hidden_dim,
                    n_classes = self.num_label_kinds,
                    n_layers = self.params.n_layers,
                    node_num = graph.number_of_nodes(),
                    id2nodeidx = self.id2nodeidx_trainval,
                    activation = F.relu,
                    dropout = self.params.dropout).to(self.device)
        self.load_kth_model(model=kth_model, k=k)
        # initialize
        kth_logits = pd.DataFrame(columns = ['rna_id',f'logits_{k}th_model'])
        
        with tqdm.tqdm(for_target) as tq:            
            for step, idx_r in enumerate(tq): 
                # kth_graph add target-pair
                if mi2lnc:
                    kth_graph = copy.deepcopy(graph)
                    kth_graph.add_edges(target, idx_r, {'edgeweight': torch.zeros(1, dtype=torch.float32).unsqueeze(-1)})
                else:
                    kth_graph = copy.deepcopy(graph)
                    kth_graph.add_edges(idx_r, target, {'edgeweight': torch.zeros(1, dtype=torch.float32).unsqueeze(-1)})
                # test
                test_edgeidx = torch.tensor(kth_graph.number_of_edges() - 1, dtype=torch.int64).unsqueeze(-1)
                test_dataloader = dgl.dataloading.EdgeDataLoader(kth_graph, test_edgeidx, self.sampler,
                                                                batch_size = 1,
                                                                shuffle = False,
                                                                drop_last = False,
                                                                num_workers = 1,
                                                                device = self.device)
                # Sets the module in evaluating mode
                kth_model.eval()
                for input_nodes, pair_graph, blocks_MFGS in test_dataloader:
                    # feature copy to(self.device)
                    blocks = [b.to(self.device) for b in blocks_MFGS]
                    edge_subgraph = pair_graph.to(self.device)
                    input_features = blocks[0].srcdata['nfea']
                    with torch.no_grad():
                        logits = kth_model(edge_subgraph, blocks, input_features, tasktype = task_type).cpu()
                    
                    kth_logits = kth_logits.append({'rna_id':kth_graph.ndata['id'][idx_r].squeeze(-1).numpy(), f'logits_{k}th_model':logits.detach().squeeze(-1).squeeze(-1).numpy()}, ignore_index=True)
                
                tq.set_postfix({'idx_for_target': '%s' % idx_r, 'edge_prediction': '%.03f' % logits}, refresh=False)

        # save kth Fold predict result
        kth_logits = kth_logits.astype({'rna_id': 'int64', f'logits_{k}th_model': 'float32'})
        k_average_logits = pd.merge(k_average_logits,kth_logits,on='rna_id',how='left',sort=False)

        return k_average_logits

    def test(self):
        assess = ['fprs', 'tprs', 'thresholds', 'tn', 'fp', 'fn', 'tp', 'loss', 'acc', 'auc', 'mcc', 'precision', 'recall', 'specificity', 'f1']
        allfold_test_data = {}
        # load test info
        Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info = self.graphbuilder.dealer.deal_test_RNA_data()
        print(f'test number = {Label_data_pair.shape[0]}')

        for k in range(self.params.KFold_num):
            allfold_test_data[k] = self.test_KFold(k, self.graph_trainval, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, assess)
        # save result data
        result = pd.DataFrame([])
        for ass in assess[3:]:
            for k in range(self.params.KFold_num):
                result.at[k,'Foldid'] = k
                result.at[k,ass] = allfold_test_data[k][ass]
                
        resultdata_savepath = self.test_save_path / 'result_data'
        if not resultdata_savepath.exists():
            resultdata_savepath.mkdir(parents=True)
        result.to_csv(resultdata_savepath / 'test_result.csv')
        Ave_res = result.mean(axis=0)
        print(
            f"Average Result >>>> Test ACC {Ave_res['acc']:.4f}, Test AUC {Ave_res['auc']:.4f}, Test MCC {Ave_res['mcc']:.4f}, Test loss {Ave_res['loss']:.4f}, Test precision {Ave_res['precision']:.4f}, Test recall {Ave_res['recall']:.4f}, Test specificity {Ave_res['specificity']:.4f}, Test f1 {Ave_res['f1']:.4f}",
            f"Test TN {Ave_res['tn']:.4f}, Test FP {Ave_res['fp']:.4f}, Test_FN {Ave_res['fn']:.4f}, Test_TP {Ave_res['tp']:.4f}")

    def predict_on_pair(self):
        # load pair info
        Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info = self.graphbuilder.dealer.deal_predict_RNA_data()
        print(f'test number = {Label_data_pair.shape[0]}')
        # result storage
        k_average_logits = pd.DataFrame({'pair_idx':Label_data_pair.index, 'pair_id':Label_data_pair['pair_id']})
        for k in range(self.params.KFold_num):
            k_average_logits = self.prepair_KFold(k, self.graph_trainval, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, k_average_logits)
        k_average_logits['logits'] = k_average_logits[[f'logits_{k}th_model' for k in range(self.params.KFold_num)]].mean(axis=1)
        logits_pair = pd.merge(k_average_logits, Label_data_pair[['mirna','lncrna']], left_on='pair_idx', right_index=True, how='left',sort=False)
        logits_pair.to_csv(self.prepair_save_path / f'logits_of_prepair.csv')
        print('predict on pair finished')

    def predict_on_rna(self):
        predataloader = loaddata.dataloader(self.params)
        target_rna, rna_info_o = predataloader.load_rna_predict()
        target_ids = list(target_rna['rna_id'].values)
        for rnaid in target_ids:
            rna_name = target_rna['rna'].values[target_rna['rna_id'].values == rnaid][0]
            print(f'predict on the RNA, id of {rnaid}, name of {rna_name}')
            graph_nodeids = np.squeeze(self.graph_trainval.ndata['id'].numpy())
            if rnaid > 0:
                ids_for_target = graph_nodeids[graph_nodeids < 0]
                mi2lnc=True
            elif rnaid < 0:
                ids_for_target = graph_nodeids[graph_nodeids > 0]
                mi2lnc=False     
            idxs_for_target = np.array([self.id2nodeidx_trainval[i] for i in ids_for_target])
            idx_target = self.id2nodeidx_trainval[rnaid]
            # result storage
            k_average_logits = pd.DataFrame({'rna_id':ids_for_target})

            for k in range(self.params.KFold_num):
                k_average_logits = self.prerna_KFold(k=k, graph=self.graph_trainval, target=idx_target, for_target=idxs_for_target, mi2lnc=mi2lnc, k_average_logits=k_average_logits)    
            k_average_logits['logits'] = k_average_logits[[f'logits_{k}th_model' for k in range(self.params.KFold_num)]].mean(axis=1)
            k_average_logits.sort_values(by='logits', ascending=False, inplace=True)
            logits_rna = pd.merge(k_average_logits, rna_info_o, on='rna_id',how='left',sort=False)

            trainval_pairs = pd.read_csv(self.prj_path / 'data' / 'processed_data' / 'pair_trainval' / 'run_info' / 'trainval_pairs_run.csv')[['mirna','lncrna','Label','mirna_id','lncrna_id','pair_id']]
            if rnaid > 0:
                trainval = trainval_pairs[trainval_pairs['mirna_id'].isin(pd.Series(rnaid))]
                logits_rna['confirmed'] = logits_rna['rna_id'].isin(trainval['lncrna_id'])
            elif rnaid < 0:
                trainval = trainval_pairs[trainval_pairs['lncrna_id'].isin(pd.Series(rnaid))]
                logits_rna['confirmed'] = logits_rna['rna_id'].isin(trainval['mirna_id'])
 
            logits_rna.to_csv(self.prerna_save_path / f'logits_of_{rna_name}.csv')
        print('predict on rna finished')

    def evaluate(self, y_label, y_score):
        # ROC, AUC
        fprs, tprs, thresholds = metrics.roc_curve(y_label, y_score, pos_label=1)
        auc = metrics.auc(fprs, tprs)
        # scores' label prediction by threshold
        threshold = 0.5
        label_pred = copy.deepcopy(y_score)
        label_pred = torch.where(y_score >= threshold, torch.ones_like(label_pred), label_pred)
        label_pred = torch.where(y_score < threshold, torch.zeros_like(label_pred), label_pred)
        # TN, FP, FN, TP
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_label, y_pred=label_pred, labels=[0,1]).ravel()
        # Model Evaluation
        acc = metrics.accuracy_score(y_label, label_pred)
        mcc = metrics.matthews_corrcoef(y_label, label_pred)
        precision = metrics.precision_score(y_label, label_pred)
        specificity = tn/(tn+fp)
        recall = metrics.recall_score(y_label, label_pred)
        f1 = metrics.f1_score(y_label, label_pred)
        return fprs, tprs, thresholds, auc, tn, fp, fn, tp, acc, mcc, precision, specificity, recall, f1

    def load_kth_model(self, model, k):
        model_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'models' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type / 'model' / f'{self.params.feature_type}_{k}th_Fold.pt'
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
    parser.add_argument("--hidden_dim", type=int, default=192, help="number of hidden GraphSAGE units")
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden NodeSAGE layers")
    parser.add_argument("--test_rate", type=float, default=0.1, help="rate for test dataset spliting, it will not split out test data when set as 0")
    parser.add_argument("--task_type", type=str, default='run', choices=['trainval', 'run'], help="task type, trainval mode or run mode")
    parser.add_argument("--run_mode", type=str, default='predict_on_pair', choices=['test', 'predict_on_rna', 'predict_on_pair'], help="run mode, test, predict_on_rna or predict_on_pair")
    parser.add_argument("--feature_type", type=str, default='RNA_intrinsic', help="the type of RNA feature representation which would be applied")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--KFold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--KFold_val_type", type=str, default='pair', help="K-Fold Cross-Validation dataset splitting on pair")      
    params = parser.parse_args()
    print(vars(params))
    runner = Runner(params)
    if params.run_mode == 'test':
        runner.test()
    elif params.run_mode == 'predict_on_rna':
        runner.predict_on_rna() 
    elif params.run_mode == 'predict_on_pair':
        runner.predict_on_pair() 

