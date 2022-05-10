import joblib
import argparse
import sys
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn import preprocessing
from utils.loaddata import datadealer

class graphbuilder:
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')
        self.prj_path = Path(__file__).parent.resolve().parent.resolve()
        self.save_path = self.prj_path / 'pretrained'/ f'KFold_val_based_on_{self.params.KFold_val_type}' / f'{self.params.KFold_num}_Fold_trainval' / 'graphs' / f'epochsize_{self.params.n_epochs}' / f'hiddensize_{self.params.hidden_dim}' / f'learningrate_{self.params.lr}' / self.params.feature_type
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.dealer = datadealer(self.params)

    def buildgraph_internal(self):
        # all original single RNA and pair information
        if self.params.task_type == 'trainval':
            Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info = self.dealer.deal_trainval_RNA_data() 
        elif self.params.task_type == 'run':
            print('you give the wrong task type')
            sys.exit()

        num_mirna_node = len(mirna_node_info['mirna'])
        num_lncrna_node = len(lncrna_node_info['lncrna'])
        num_pairs = len(Label_data_pair['Label'])
        print(f"The built graph will contain {num_mirna_node} microRNA nodes, {num_lncrna_node} LncRNA nodes with {num_pairs} RNA pairs label for train and val.")        
        graph = dgl.DGLGraph()
        # add node
        mirna_ids = torch.tensor(mirna_node_info['mirna_id'].values, dtype=torch.int64).unsqueeze(-1)     
        lncrna_ids = torch.tensor(lncrna_node_info['lncrna_id'].values, dtype=torch.int64).unsqueeze(-1)
        mirna_nfea = mirna_node_info.iloc[:,mirna_merge_dim+1:].values
        lncrna_nfea = lncrna_node_info.iloc[:,lncrna_merge_dim+1:].values

        # feature normalization -- MinMaxScaler and save the scaler
        scaler_mirna = preprocessing.MinMaxScaler().fit(mirna_nfea)
        joblib.dump(scaler_mirna, self.save_path / 'trainval_scaler_mirna.save')
        scaler_lncrna = preprocessing.MinMaxScaler().fit(lncrna_nfea)
        joblib.dump(scaler_lncrna, self.save_path / 'trainval_scaler_lncrna.save')
        mirna_nfea = torch.tensor(scaler_mirna.transform(mirna_nfea), dtype=torch.float32)
        lncrna_nfea = torch.tensor(scaler_lncrna.transform(lncrna_nfea), dtype=torch.float32)

        graph.add_nodes(num_mirna_node, {'id': mirna_ids, 'nfea': mirna_nfea})
        graph.add_nodes(num_lncrna_node, {'id': lncrna_ids, 'nfea': lncrna_nfea})

        id2nodeidx_trainval = {}
        for i, key in enumerate(mirna_ids):
            id2nodeidx_trainval[key.item()] = i
        for i, key in enumerate(lncrna_ids):
            id2nodeidx_trainval[key.item()] = i + num_mirna_node
        # save id2nodeidx_trainval dictionary
        np.save(self.save_path / 'id2nodeidx_trainval.npy', id2nodeidx_trainval)

        # add edge (bothway)
        u_n = np.array([id2nodeidx_trainval[i] for i in Label_data_pair['mirna_id'].values]) 
        v_n = np.array([id2nodeidx_trainval[i] for i in Label_data_pair['lncrna_id'].values])
        labels = torch.tensor(Label_data_pair['Label'].values, dtype=torch.float32).unsqueeze(-1)
        edgeweight = torch.tensor(Label_data_pair['Label'].values, dtype=torch.float32).unsqueeze(-1)
        pair_idxs = torch.tensor(Label_data_pair['pair_index'].values, dtype=torch.float32).unsqueeze(-1)
        graph.add_edges(u_n, v_n, {'Labels': labels, 'edgeweight': edgeweight, 'pair_index': pair_idxs})
        graph.add_edges(v_n, u_n, {'Labels': labels, 'edgeweight': edgeweight, 'pair_index': pair_idxs})
        # add self-loop
        graph.add_edges(graph.nodes(), graph.nodes(),{'edgeweight': torch.ones(graph.number_of_nodes(), dtype=torch.float32).unsqueeze(-1)})
        graph.readonly()
        dgl.data.utils.save_graphs(str(self.save_path / f'{self.params.feature_type}.bin'), [graph])  # graph saved

        print('graph established')

        return graph, id2nodeidx_trainval, u_n, v_n, Label_data_pair, mirna_node_info, lncrna_node_info

    def buildgraph_test(self, graph, Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info, row, id2nodeidx):
        graph_test = copy.deepcopy(graph)
        u_name = Label_data_pair['mirna'].to_frame().iloc[[row]]
        u_id = Label_data_pair['mirna_id'].to_frame().iloc[[row]]
        v_name = Label_data_pair['lncrna'].to_frame().iloc[[row]]
        v_id = Label_data_pair['lncrna_id'].to_frame().iloc[[row]]
        Labels = Label_data_pair['Label'].to_frame().iloc[[row]]
        
        u_nfea = pd.merge(mirna_node_info, u_id, on='mirna_id', sort=False, how='right')
        u_nfea = u_nfea.iloc[:,mirna_merge_dim+1:].values 
        v_nfea = pd.merge(lncrna_node_info, v_id, on='lncrna_id', sort=False, how='right')
        v_nfea = v_nfea.iloc[:,lncrna_merge_dim+1:].values 

        # feature nalmalization -- MinMaxScaler
        # # And now to load...
        scaler_mirna = joblib.load(self.save_path / 'trainval_scaler_mirna.save')
        scaler_lncrna = joblib.load(self.save_path / 'trainval_scaler_lncrna.save')
        u_nfea = torch.tensor(scaler_mirna.transform(u_nfea), dtype=torch.float32)
        v_nfea = torch.tensor(scaler_lncrna.transform(v_nfea), dtype=torch.float32)

        if not u_id.values in graph.ndata['id'].numpy():
            graph_test.add_nodes(1, {'id': torch.tensor(u_id.values, dtype=torch.int64), 'nfea': torch.tensor(u_nfea, dtype=torch.float32)})
            u_idx = graph_test.number_of_nodes()
            print('add a new micRNA')
            graph_test.add_edges(u_idx, u_idx, {'edgeweight': torch.ones(1, dtype=torch.float32).unsqueeze(-1)})
        else : u_idx = id2nodeidx[u_id.values[0][0]]

        if not v_id.values in graph.ndata['id'].numpy():
            graph_test.add_nodes(1, {'id': torch.tensor(v_id.values, dtype=torch.int64), 'nfea': torch.tensor(v_nfea, dtype=torch.float32)})
            v_idx = graph_test.number_of_nodes()
            print('add a new LncRNA')
            graph_test.add_edges(v_idx, v_idx, {'edgeweight': torch.ones(1, dtype=torch.float32).unsqueeze(-1)})
        else : v_idx = id2nodeidx[v_id.values[0][0]]

        graph_test.add_edges(u_idx, v_idx, {'edgeweight': torch.zeros(1, dtype=torch.float32).unsqueeze(-1)})

        return graph_test, u_name.values, v_name.values, Labels.values



