import os
import copy
import argparse
import sys
import psutil
import time
import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from utils.AutoFlush import AutoFlushThread as AF
from utils.graph import graphbuilder
from models.gnn import GNN
AF.flush(1)
time_start=time.time()

class Trainer:
    def __init__(self, params):
        self.params = params
        print('torch.cuda.is_available():', torch.cuda.is_available())
        self.prj_path = Path(__file__).parent.resolve()
        self.save_path = self.prj_path / 'pretrained' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'models' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type 
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{self.params.gpu}')
        # graphbuilder and build base_graph
        builder = graphbuilder(self.params)
        if params.task_type == 'trainval':
            graph, id2nodeidx_trainval, u_n, v_n, Label_data_pair, mirna_node_info, lncrna_node_info = builder.buildgraph_internal()
        elif params.task_type == 'test':
            print('you give the wrong task type')
            sys.exit()
        self.base_graph = graph
        self.dense_dim = len(self.base_graph.ndata['nfea'][0])
        # graph information
            # node information
        self.mirna_node_info = mirna_node_info  # unique micRNA data
        self.lncrna_node_info = lncrna_node_info  # unique LncRNA data
        self.num_micR = len(mirna_node_info['mirna'])
        self.num_LncR = len(lncrna_node_info['lncrna'])
        self.u_notest = u_n
        self.v_notest = v_n
        self.id2nodeidx_trainval = id2nodeidx_trainval
            # edge information
        self.Label_data = Label_data_pair
        self.num_edges_single = len(Label_data_pair['Label'])
        print(f"Train-Val Number: {self.num_edges_single}")
        self.num_label_kinds = 2
        # sampler defining
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.params.n_layers) # all nodes information
        
    def train(self, graph_t, trainidxs, model, optimizer, kth_log_file):
        train_dataloader = dgl.dataloading.EdgeDataLoader(graph_t, trainidxs, self.sampler,
                                                            exclude = 'reverse_id', 
                                                            reverse_eids = torch.cat([torch.arange(self.num_edges_single) + self.num_edges_single, torch.arange(self.num_edges_single)]), 
                                                            batch_size = self.params.batch_size,
                                                            shuffle = True,
                                                            drop_last = False,
                                                            num_workers = 1,
                                                            device = self.device)
        # Sets the module in training mode
        print("start to train", file=kth_log_file)
        model.train()
        batch_num_t = 0
        total_loss_t = 0
        edge_labels_t = torch.tensor([],device=self.device)
        edge_logits_t = torch.tensor([],device=self.device)
        edge_ids_t = torch.tensor([],device=self.device)
        with tqdm.tqdm(train_dataloader, file=kth_log_file) as tq: 
            for step, (input_nodes, pair_graph, blocks_MFGS) in enumerate(tq):           
                blocks = [b.to(self.device) for b in blocks_MFGS]
                edge_subgraph = pair_graph.to(self.device)
                batch_ids = edge_subgraph.edata['pair_id'].squeeze(-1)
                batch_labels = edge_subgraph.edata['Labels'].squeeze(-1)
                input_features = blocks[0].srcdata['nfea']
                logits_t = model(edge_subgraph, blocks, input_features, tasktype = 'train') 
                loss = F.cross_entropy(logits_t, batch_labels.long(), reduction = 'sum')
                # loss = F.binary_cross_entropy(F.sigmoid(logits_t[:,1]), batch_labels, reduction = 'sum')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # save info of batches in one epoch
                batch_num_t += 1
                total_loss_t += loss.item()
                edge_labels_t = torch.cat([edge_labels_t, batch_labels.detach()],dim=0)
                edge_logits_t = torch.cat([edge_logits_t, logits_t.detach()],dim=0)
                edge_ids_t = torch.cat([edge_ids_t, batch_ids.detach()],dim=0)
                tq.set_postfix({'Batch:':batch_num_t, 'edgeprediction[0]': logits_t.detach()[0], 'edgelabel[0]': '%.03f' % batch_labels.detach()[0], 'batch_loss': '%.03f' % loss.item()}, refresh=False)

        mean_loss_t = total_loss_t/len(trainidxs)
        return mean_loss_t, edge_labels_t, edge_logits_t, edge_ids_t

    def validate(self, graph_v, edgeidxs, model, kth_log_file):
        val_dataloader = dgl.dataloading.EdgeDataLoader(graph_v, edgeidxs, self.sampler,
                                                        exclude = 'reverse_id',
                                                        reverse_eids = torch.cat([torch.arange(self.num_edges_single) + self.num_edges_single, torch.arange(self.num_edges_single)]),
                                                        batch_size = self.params.batch_size,
                                                        shuffle = True,
                                                        drop_last = False,
                                                        num_workers = 1,
                                                        device = self.device)
        # Sets the module in evaluating mode
        print("start to validation", file=kth_log_file)
        model.eval()
        batch_num_v = 0
        total_loss_v = 0
        edge_labels_v = torch.tensor([],device=self.device)
        edge_logits_v = torch.tensor([],device=self.device)
        edge_ids_v = torch.tensor([],device=self.device)
        with tqdm.tqdm(val_dataloader, file=kth_log_file) as tq:
            for step, (input_nodes, pair_graph, blocks_MFGS) in enumerate(tq):
                blocks = [b.to(self.device) for b in blocks_MFGS]
                edge_subgraph = pair_graph.to(self.device)
                batch_ids = edge_subgraph.edata['pair_id'].squeeze(-1)
                batch_labels = edge_subgraph.edata['Labels'].squeeze(-1)
                input_features = blocks[0].srcdata['nfea']
                with torch.no_grad():
                    logits_v = model(edge_subgraph, blocks, input_features, tasktype = 'val')
                val_loss = F.cross_entropy(logits_v, batch_labels.long(), reduction = 'sum')
                # val_loss = F.binary_cross_entropy(F.sigmoid(logits_v[:,1]), batch_labels, reduction = 'sum')
                # save info of batches in one epoch
                batch_num_v += 1
                total_loss_v += val_loss.item()
                edge_labels_v = torch.cat([edge_labels_v, batch_labels.detach()],dim=0)
                edge_logits_v = torch.cat([edge_logits_v, logits_v.detach()],dim=0)
                edge_ids_v = torch.cat([edge_ids_v, batch_ids.detach()],dim=0)

                tq.set_postfix({'Batch:':batch_num_v, 'edgeprediction[0]':logits_v.detach()[0], 'edgelabel[0]': '%.03f' % batch_labels.detach()[0], 'val_batch_loss': '%.03f' % val_loss.item()}, refresh=False)

        mean_loss_v = total_loss_v/len(edgeidxs)
        return mean_loss_v, edge_labels_v, edge_logits_v, edge_ids_v

    def evaluate(self, y_label, y_score):
        # ROC, AUC
        fprs, tprs, thresholds = metrics.roc_curve(y_label, y_score, pos_label=1)
        auc = metrics.auc(fprs, tprs)
        # prauc
        pres, recs, thresholds_prc = metrics.precision_recall_curve(y_label, y_score, pos_label=1)
        prauc = metrics.auc(recs, pres)
        av_prc = metrics.average_precision_score(y_label, y_score)
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
        sensitivity = tp/(tp+fn)
        recall = metrics.recall_score(y_label, label_pred)
        f1 = metrics.f1_score(y_label, label_pred)
        return fprs, tprs, thresholds, pres, recs, np.append(thresholds_prc, [1], axis=0), auc, tn, fp, fn, tp, acc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc

    def split_KFold(self, split_type):
        if split_type == 'pair':
            print(f"Apply Stratified {self.params.KFold_num}-Fold cross-validation based on pair")
            skf = StratifiedKFold(n_splits=self.params.KFold_num)
            list = []
            X = self.Label_data.index.values
            y = self.Label_data['Label'].values
            for (t,v) in skf.split(X, y):
                list.append((X[t],X[v]))
        
        return list
    
    def run_KFold(self, k, train, val, assess):
        log_path_k = self.prj_path / 'logs' / f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type / f'fold_{k}' 
        if not log_path_k.exists():
            log_path_k.mkdir(parents=True)
        kth_log_file = open(log_path_k / f"fold_{k}_trainval.log", "w")

        print(f"----------processing Fold {k}----------", file=kth_log_file)
        print(f"{len(train)} to train, {len(val)} to validation", file=kth_log_file)
        count_train = self.Label_data.iloc[train,:]['Label'].value_counts()
        count_val = self.Label_data.iloc[val,:]['Label'].value_counts()
        print(f"{count_train[1.0]} positive and {count_train[0]} negative in train, {count_val[1.0]} positive and {count_val[0]} negative in val", file=kth_log_file)
        kfold_train_data, kfold_val_data = {}, {}
        # initialize
        _epoch = 0
        _loss_t, _loss_v, _acc_t, _acc_v = 0,0,0,0
        _fprs_t, _tprs_t, _thresholds_t = 0,0,0
        _pres_t, _recs_t, _thresholds_prc_t = 0,0,0
        _tn_t, _fp_t, _fn_t, _tp_t = 0,0,0,0
        _auc_t, _mcc_t, _precision_t, _recall_t, _specificity_t, _sensitivity_t, _f1_t, _prauc_t, _av_prc_t = 0,0,0,0,0,0,0,0,0
        _fprs_v, _tprs_v, _thresholds_v = 0,0,0
        _pres_v, _recs_v, _thresholds_prc_v = 0,0,0
        _tn_v, _fp_v, _fn_v, _tp_v = 0,0,0,0
        _auc_v, _mcc_v, _precision_v, _recall_v, _specificity_v, _sensitivity_v, _f1_v, _prauc_v, _av_prc_v = 0,0,0,0,0,0,0,0,0
        # _model = None
        # _optimizer = None
        _loss_epochs_t, _acc_epochs_t, _loss_epochs_v, _acc_epochs_v = {},{},{},{}
        _epoch_edge_labels_t, _epoch_edge_logits_t, _epoch_edge_scores_t, _epoch_edge_ids_t, _epoch_edge_labels_v, _epoch_edge_logits_v, _epoch_edge_scores_v, _epoch_edge_ids_v = torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([]),torch.tensor([])

        # k'th graph remove validation information
        kth_graph = copy.deepcopy(self.base_graph)
        kth_graph.edata['edgeweight_keepval'] = kth_graph.edata['edgeweight'].clone()
        kth_graph.edata['edgeweight'][torch.cat([torch.tensor(val),torch.tensor(val+self.num_edges_single)])]=torch.tensor(0, dtype=torch.float32)

        # k'th model
        kth_model = GNN(in_feats = self.dense_dim,
                    n_hidden = self.params.hidden_dim,
                    n_classes = self.num_label_kinds,
                    n_layers = self.params.n_layers,
                    node_num = self.num_micR + self.num_LncR,
                    id2nodeidx = self.id2nodeidx_trainval,
                    activation = F.relu,
                    dropout = self.params.dropout).to(self.device)
        # k'th optimizer
        kth_optimizer = torch.optim.Adam(kth_model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        
        ### epochs start ###
        print("epoch trainval start", file=kth_log_file)
        for epoch in range(self.params.n_epochs):
            # train
            print(f"prepare to train for epoch {epoch}", file=kth_log_file)
            epoch_loss_t, epoch_edge_labels_t, epoch_edge_logits_t, epoch_edge_ids_t = self.train(graph_t = kth_graph, trainidxs = train, model = kth_model, optimizer=kth_optimizer, kth_log_file=kth_log_file)
            # move to cpu if in gpu
            epoch_edge_labels_t = epoch_edge_labels_t.cpu() 
            epoch_edge_logits_t = epoch_edge_logits_t.cpu() 
            epoch_edge_ids_t = epoch_edge_ids_t.cpu() 
            # evaluate train
            epoch_edge_scores_t = F.softmax(epoch_edge_logits_t, dim=1)[:,1]
            
            fprs_t, tprs_t, thresholds_t, pres_t, recs_t, thresholds_prc_t, auc_t, tn_t, fp_t, fn_t, tp_t, acc_t, mcc_t, precision_t, recall_t, specificity_t, sensitivity_t, f1_t, prauc_t, av_prc_t = self.evaluate(y_label=epoch_edge_labels_t, y_score=epoch_edge_scores_t)
            loss_t = epoch_loss_t
            print('time cost: %.4f min' % ((time.time()-time_start)/60), file=kth_log_file)     
            # validation
            print(f"prepare to validate for epoch {epoch}", file=kth_log_file)
            epoch_loss_v, epoch_edge_labels_v, epoch_edge_logits_v, epoch_edge_ids_v = self.validate(graph_v=kth_graph, edgeidxs= val, model = kth_model, kth_log_file=kth_log_file)
            # move to cpu if in gpu
            epoch_edge_labels_v = epoch_edge_labels_v.cpu() 
            epoch_edge_logits_v = epoch_edge_logits_v.cpu() 
            epoch_edge_ids_v = epoch_edge_ids_v.cpu() 
            # evaluate val
            epoch_edge_scores_v = F.softmax(epoch_edge_logits_v, dim=1)[:,1]
            fprs_v, tprs_v, thresholds_v, pres_v, recs_v, thresholds_prc_v, auc_v, tn_v, fp_v, fn_v, tp_v, acc_v, mcc_v, precision_v, recall_v, specificity_v, sensitivity_v, f1_v, prauc_v, av_prc_v = self.evaluate(y_label=epoch_edge_labels_v, y_score=epoch_edge_scores_v)
            loss_v = epoch_loss_v
            # store process data
            _loss_epochs_t[epoch] = loss_t
            _acc_epochs_t[epoch] = acc_t
            _loss_epochs_v[epoch] = loss_v
            _acc_epochs_v[epoch] = acc_v
            # store best validation epoch data
            if _acc_v <= acc_v:
                _epoch = epoch

                _epoch_edge_labels_t, _epoch_edge_logits_t, _epoch_edge_scores_t, _epoch_edge_ids_t, _epoch_edge_labels_v, _epoch_edge_logits_v, _epoch_edge_scores_v, _epoch_edge_ids_v = epoch_edge_labels_t, epoch_edge_logits_t, epoch_edge_scores_t, epoch_edge_ids_t, epoch_edge_labels_v, epoch_edge_logits_v, epoch_edge_scores_v, epoch_edge_ids_v 
                
                _loss_t = loss_t
                _loss_v = loss_v
                _acc_t = acc_t
                _acc_v = acc_v

                _fprs_t, _tprs_t, _thresholds_t = fprs_t, tprs_t, thresholds_t
                _pres_t, _recs_t, _thresholds_prc_t = pres_t, recs_t, thresholds_prc_t
                _tn_t, _fp_t, _fn_t, _tp_t = tn_t, fp_t, fn_t, tp_t
                _auc_t = auc_t
                _mcc_t = mcc_t
                _precision_t = precision_t
                _recall_t = recall_t
                _specificity_t = specificity_t
                _sensitivity_t = sensitivity_t
                _f1_t = f1_t
                _prauc_t, _av_prc_t = prauc_t, av_prc_t

                _fprs_v, _tprs_v, _thresholds_v = fprs_v, tprs_v, thresholds_v
                _pres_v, _recs_v, _thresholds_prc_v = pres_v, recs_v, thresholds_prc_v
                _tn_v, _fp_v, _fn_v, _tp_v = tn_v, fp_v, fn_v, tp_v
                _auc_v = auc_v
                _mcc_v = mcc_v
                _precision_v = precision_v
                _recall_v = recall_v
                _specificity_v = specificity_v
                _sensitivity_v = sensitivity_v
                _f1_v = f1_v
                _prauc_v, _av_prc_v = prauc_v, av_prc_v

                # _model = kth_model
                # _optimizer = kth_optimizer

                # save best kth model
                self.save_model(k, kth_model, kth_optimizer)

            if acc_t == 1:
                break
            print(
                f">>>>Epoch {epoch:04d}<<<<"
                f"Train acc {acc_t:.4f}, Train Loss {loss_t:.6f}, train_MCC {mcc_t:.4f}, train_precision {precision_t:.4f}, train_recall {recall_t:.4f}, train_specificity {specificity_t:.4f}, train_sensitivity {sensitivity_t:.4f}, train_f1 {f1_t:.4f}, "
                f"Val acc {acc_v:.4f}, Val Loss {loss_v:.6f}, val_MCC {mcc_v:.4f}, val_precision {precision_v:.4f}, val_recall {recall_v:.4f}, val_specificity {specificity_v:.4f}, val_sensitivity {sensitivity_v:.4f}, val_f1 {f1_v:.4f}, "
                f"Train TN {tn_t}, Train FP {fp_t}, Train FN {fn_t}, Train TP {tp_t}, Val TN {tn_v}, Val FP {fp_v}, Val FN {fn_v}, Val TP {tp_v}, "
                f"Train AUC {auc_t:.4f}, Train_prauc {prauc_t:.4f}, Train_av_prc {av_prc_t:.4f}, Val AUC {auc_v:.4f}, Val_prauc {prauc_v:.4f}, Val_av_prc {av_prc_v:.4f}", file=kth_log_file)
        print("epoch trainval finished", file=kth_log_file)
        ### epochs finish ### 

        # save best kth Fold ROC data
        ROC_savepath = self.save_path / 'ROC_data' / f'{k}th_Fold'
        if not ROC_savepath.exists():
            ROC_savepath.mkdir(parents=True)
        pd.DataFrame({'fprs':_fprs_t, 'tprs':_tprs_t, 'thresholds':_thresholds_t}).to_csv(ROC_savepath / f'train_ROC_for_{k}th_Fold.csv')
        pd.DataFrame({'fprs':_fprs_v, 'tprs':_tprs_v, 'thresholds':_thresholds_v}).to_csv(ROC_savepath / f'val_ROC_for_{k}th_Fold.csv')

        # save best kth Fold PRC data
        PRC_savepath = self.save_path / 'PRC_data' / f'{k}th_Fold'
        if not PRC_savepath.exists():
            PRC_savepath.mkdir(parents=True)      
        pd.DataFrame({'pres':_pres_t, 'recs':_recs_t, 'thresholds_prc':_thresholds_prc_t}).to_csv(PRC_savepath / f'train_PRC_for_{k}th_Fold.csv')
        pd.DataFrame({'pres':_pres_v, 'recs':_recs_v, 'thresholds_prc':_thresholds_prc_v}).to_csv(PRC_savepath / f'val_PRC_for_{k}th_Fold.csv')

        # save best kth Fold score data 
        score_savepath = self.save_path / 'score_data' / f'{k}th_Fold'
        if not score_savepath.exists():
            score_savepath.mkdir(parents=True)
        pd.DataFrame({'eids':_epoch_edge_ids_t, 'labels':_epoch_edge_labels_t, 'logits_0':_epoch_edge_logits_t[:,0], 'logits_1':_epoch_edge_logits_t[:,1], 'scores':_epoch_edge_scores_t}).to_csv(score_savepath / f'train_score_for_{k}th_Fold.csv')
        pd.DataFrame({'eids':_epoch_edge_ids_v, 'labels':_epoch_edge_labels_v, 'logits_0':_epoch_edge_logits_v[:,0], 'logits_1':_epoch_edge_logits_v[:,1], 'scores':_epoch_edge_scores_v}).to_csv(score_savepath / f'val_score_for_{k}th_Fold.csv')

        # store best kth train-val data
        for ass in assess:
            exec(f"kfold_train_data['{ass}'] = _{ass}_t")
        for ass in assess:
            exec(f"kfold_val_data['{ass}'] = _{ass}_v")

        print(f"--- Best val result for {k} th Fold:---", file=kth_log_file)
        print(
            f">Epoch {_epoch:04d}<"
            f"Train acc {_acc_t:.4f}, Train Loss {_loss_t:.6f}, train_MCC {_mcc_t:.4f}, train_precision {_precision_t:.4f}, train_recall {_recall_t:.4f}, train_specificity {_specificity_t:.4f}, train_sensitivity {_sensitivity_t:.4f}, train_f1 {_f1_t:.4f}, "
            f"Val acc {_acc_v:.4f}, Val Loss {_loss_v:.6f}, val_MCC {_mcc_v:.4f}, val_precision {_precision_v:.4f}, val_recall {_recall_v:.4f}, val_specificity {_specificity_v:.4f}, val_sensitivity {_sensitivity_v:.4f}, val_f1 {_f1_v:.4f}, "
            f"Train TN {_tn_t}, Train FP {_fp_t}, Train FN {_fn_t}, Train TP {_tp_t}, Val TN {_tn_v}, Val FP {_fp_v}, Val FN {_fn_v}, Val TP {_tp_v}, "
            f"Train AUC {_auc_t:.4f}, Train_prauc {_prauc_t:.4f}, Train_av_prc {_av_prc_t:.4f}, Val AUC {_auc_v:.4f}, Val_prauc {_prauc_v:.4f}, Val_av_prc {_av_prc_v:.4f}", file=kth_log_file)
        
        kth_log_file.close()

        return kfold_train_data, kfold_val_data

    def save_model(self, k, model, optimizer):
        state = {'model': model.state_dict(),'optimizer': optimizer.state_dict()}
        model_savepath = self.save_path / 'model'
        if not model_savepath.exists():
            model_savepath.mkdir(parents=True)
        torch.save(state, model_savepath / f'{self.params.feature_type}_{k}th_Fold.pt') 

    def fit(self): 
        print("Trainer __init__() finished")
        assess = ['loss_epochs', 'acc_epochs', 'fprs', 'tprs', 'thresholds', 'pres', 'recs', 'thresholds_prc', 'tn', 'fp', 'fn', 'tp', 'loss', 'acc', 'auc', 'mcc', 'precision', 'recall', 'specificity', 'sensitivity', 'f1', 'prauc', 'av_prc']
        allfold_train_data, allfold_val_data = {},{}
        # run KFold
        trainval_list = self.split_KFold(split_type = self.params.KFold_val_type)
        print("......K-Fold Trainval is running, please wait......")
        print(f"Foldlogs are printed in ./logs/KFold_val_based_on_{self.params.KFold_val_type}/{self.params.KFold_num}_Fold_trainval/epochsize_{self.params.n_epochs}/hiddensize_{self.params.hidden_dim}/learningrate_{self.params.lr}/{self.params.feature_type}/fold_k")
        for k in range(self.params.KFold_num):
            train, val = trainval_list[k]
            allfold_train_data[k], allfold_val_data[k] = self.run_KFold(k, train, val, assess)
        print("......K-Fold Trainval finished......")
        return assess, allfold_train_data, allfold_val_data

    def superIO(self, assess, allfold_train_data, allfold_val_data):
        # save process loss and acc 
        process_loss_t = pd.DataFrame({f'{k}th Fold' : allfold_train_data[k]['loss_epochs'] for k in range(self.params.KFold_num)}).T
        process_acc_t = pd.DataFrame({f'{k}th Fold' : allfold_train_data[k]['acc_epochs'] for k in range(self.params.KFold_num)}).T
        process_loss_v = pd.DataFrame({f'{k}th Fold' : allfold_val_data[k]['loss_epochs'] for k in range(self.params.KFold_num)}).T
        process_acc_v = pd.DataFrame({f'{k}th Fold' : allfold_val_data[k]['acc_epochs'] for k in range(self.params.KFold_num)}).T
        processdata_savepath = self.save_path / 'process_data'
        if not processdata_savepath.exists():
            processdata_savepath.mkdir(parents=True)
        process_loss_t.to_csv(processdata_savepath / 'train_process_loss.csv')
        process_acc_t.to_csv(processdata_savepath / 'train_process_acc.csv')
        process_loss_v.to_csv(processdata_savepath / 'val_process_loss.csv')
        process_acc_v.to_csv(processdata_savepath / 'val_process_acc.csv')

        # save result data
        result_t = pd.DataFrame([])
        result_v = pd.DataFrame([])
        for ass in assess[8:]:
            for k in range(self.params.KFold_num):
                result_t.at[k,'Foldid'] = k
                result_v.at[k,'Foldid'] = k
                result_t.at[k,ass] = allfold_train_data[k][ass]
                result_v.at[k,ass] = allfold_val_data[k][ass]
                
        resultdata_savepath = self.save_path / 'result_data'
        if not resultdata_savepath.exists():
            resultdata_savepath.mkdir(parents=True)
        result_t.to_csv(resultdata_savepath / 'train_result.csv')
        result_v.to_csv(resultdata_savepath / 'val_result.csv')





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
    parser.add_argument("--test_rate", type=float, default=0.1, help="rate for test dataset spliting, it will not split out test data when set as 0")
    parser.add_argument("--task_type", type=str, default='trainval', choices=['trainval', 'run'], help="task type, trainval mode or run mode")
    parser.add_argument("--feature_type", type=str, default='RNA_intrinsic', help="the type of RNA feature representation which would be applied")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--KFold_num", type=int, default=5, help="number of folds for K-Fold Cross-Validation, default 5")
    parser.add_argument("--KFold_val_type", type=str, default='pair', help="K-Fold Cross-Validation dataset splitting on pair")  
    params = parser.parse_args()
    print(vars(params))

    trainer = Trainer(params)
    assess, allfold_train_data, allfold_val_data = trainer.fit()
    print('-------------fit finished-------------')
    print(f"Train average ACC {np.mean([allfold_train_data[k]['acc'] for k in range(params.KFold_num)]):.4f}, Val average ACC {np.mean([allfold_val_data[k]['acc'] for k in range(params.KFold_num)]):.4f},Train average AUC {np.mean([allfold_train_data[k]['auc'] for k in range(params.KFold_num)]):.4f}, Val average AUC {np.mean([allfold_val_data[k]['auc'] for k in range(params.KFold_num)]):.4f}")
    trainer.superIO(assess, allfold_train_data, allfold_val_data)
    print('-------------IO finished-------------')
