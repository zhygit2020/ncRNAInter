import numpy as np
import pandas as pd
from pathlib import Path
import sys
import copy

class dataloader:
    def __init__(self, params):
        self.params = params
        self.prj_path = Path(__file__).parent.resolve().parent.resolve()
        self.fea_path_mirna = self.prj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_feature' / 'csv'
        self.fea_path_lncrna = self.prj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'lncrna' / 'lncrna_feature' / 'csv'
        self.RNA_unique_mirna_path = self.prj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'mirna' 
        self.RNA_unique_lncrna_path = self.prj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'lncrna' 

    def load_RNA_fea_csv(self):
        print(f'feature_type_selected: csv')
        RNA_unique_mirna = pd.read_csv(self.RNA_unique_mirna_path / 'mirna_names_unique_template.csv')['A'].map(lambda Seqname: Seqname[1:]).to_frame(name = 'Seqname') 
        RNA_unique_lncrna = pd.read_csv(self.RNA_unique_lncrna_path / 'lncrna_names_unique_template.csv')['B'].map(lambda Seqname: Seqname[1:]).to_frame(name = 'Seqname') 

        file_name_a1 = 'Codon related (1D).csv'
        file_name_a2 = 'Open reading frame (1D).csv'
        file_name_a3 = 'Guanine-cytosine related (1D).csv'
        file_name_a4 = 'Transcript related (1D).csv'
        file_name_b1 = 'Pseudo protein related (1D).csv'
        file_name_b2 = 'EIIP based spectrum (1D).csv'
        file_name_c1 = 'Secondary structure (1D).csv'
        RNAfea_a1_mirna = pd.read_csv(self.fea_path_mirna / file_name_a1, index_col=0)
        RNAfea_a1_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_a1, index_col=0)
        # print(RNAfea_a1_lncrna.shape) #(3301, 6) # 5-->6
        RNAfea_a2_mirna = pd.read_csv(self.fea_path_mirna / file_name_a2, index_col=0) 
        RNAfea_a2_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_a2, index_col=0) 
        # print(RNAfea_a2_lncrna.shape) #(3301, 32) # 31-->32
        RNAfea_a3_mirna = pd.read_csv(self.fea_path_mirna / file_name_a3, index_col=0) 
        RNAfea_a3_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_a3, index_col=0)  
        # print(RNAfea_a3_lncrna.shape) # 7-->8
        RNAfea_a4_mirna = pd.read_csv(self.fea_path_mirna / file_name_a4, index_col=0) 
        RNAfea_a4_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_a4, index_col=0)
        # print(RNAfea_a4_lncrna.shape) # 126-->127 
        RNAfea_b1_mirna = pd.read_csv(self.fea_path_mirna / file_name_b1, index_col=0) 
        RNAfea_b1_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_b1, index_col=0)
        # print(RNAfea_b1_lncrna.shape) # 5-->6
        RNAfea_b2_mirna = pd.read_csv(self.fea_path_mirna / file_name_b2, index_col=0) 
        RNAfea_b2_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_b2, index_col=0)
        # print(RNAfea_b2_lncrna.shape) # 8-->9  
        RNAfea_c1_mirna = pd.read_csv(self.fea_path_mirna / file_name_c1, index_col=0) 
        RNAfea_c1_lncrna = pd.read_csv(self.fea_path_lncrna / file_name_c1, index_col=0) 
        # print(RNAfea_c3_lncrna) # 8-->9
        
        # add feature in dataframe 
        RNAfea_unique_mirna_fea = RNA_unique_mirna['Seqname'].to_frame()
        RNAfea_unique_lncrna_fea = RNA_unique_lncrna['Seqname'].to_frame()
        if self.params.feature_type == 'RNA_intrinsic':
            for i in [RNAfea_a1_mirna,RNAfea_a2_mirna,RNAfea_a3_mirna,RNAfea_a4_mirna,RNAfea_b1_mirna,RNAfea_b2_mirna,RNAfea_c1_mirna]:
                RNAfea_unique_mirna_fea = pd.merge(RNAfea_unique_mirna_fea, i, on = 'Seqname', how = 'left', sort=False)
            for i in [RNAfea_a1_lncrna,RNAfea_a2_lncrna,RNAfea_a3_lncrna,RNAfea_a4_lncrna,RNAfea_b1_lncrna,RNAfea_b2_lncrna,RNAfea_c1_lncrna]:
                RNAfea_unique_lncrna_fea = pd.merge(RNAfea_unique_lncrna_fea, i, on = 'Seqname', how = 'left', sort=False)
            print('feature_selected = RNA_intrinsic') # 191 D
        # elif self.params.feature_type == 'similarity':
        #     pass
        return RNAfea_unique_mirna_fea, RNAfea_unique_lncrna_fea

    def load_pair_trainval(self):
        file_name = 'trainval_pairs_run.csv'
        label_path = self.prj_path / 'data' / 'processed_data' / 'pair_trainval' / 'run_info'
        Label_data = pd.read_csv(label_path / file_name)[['mirna','lncrna','Label','mirna_id','lncrna_id','pair_index']]
        return Label_data
    
    def load_pair_test(self):
        file_name = 'test_pairs_run.csv'
        label_path = self.prj_path / 'data' / 'processed_data' / 'pair_test' / 'run_info'
        Label_data = pd.read_csv(label_path / file_name)[['mirna','lncrna','Label','mirna_id','lncrna_id','pair_index']]
        return Label_data

    def load_target_rna(self):
        target_rna_path = self.prj_path / 'data' / 'run_data'
        target_rna = pd.read_csv(target_rna_path / 'target_rna.csv')
        mirna_id_o = pd.read_csv(self.prj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_seq.csv')
        mirna_id_o.columns = ['rna','rna_id_original','rna_seq','rna_id']
        lncrna_id_o = pd.read_csv(self.prj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'lncrna' / 'lncrna_names_unique_seq.csv')
        lncrna_id_o.columns = ['rna','rna_id_original','rna_seq','rna_id']
        rna_id_o = pd.concat([mirna_id_o,lncrna_id_o])
        target_rna = pd.merge(target_rna, rna_id_o, on='rna', how='left', sort=False)
        return target_rna, rna_id_o

class datadealer:
    def __init__(self,params):
        self.params = params
        self.prj_path = Path(__file__).parent.resolve().parent.resolve()  # 根据__file__的path设定多少个parent.resolve()
        self.loader = dataloader(params=self.params)

    def deal_trainval_RNA_data(self):
        mirna_fea, lncrna_fea = self.loader.load_RNA_fea_csv()
        print('>>>>RNA fea loaded<<<<')
        if self.params.task_type == 'trainval':
            Label_data = self.loader.load_pair_trainval()
            Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info = self.deal_trainval_graph_dataset(mirna_fea=mirna_fea, lncrna_fea=lncrna_fea, Label_data=Label_data)
        elif self.params.task_type == 'run':
            print('you give the wrong task type')
            sys.exit()
        print('>>>>Label data loaded<<<<')
        return Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info

    def deal_test_RNA_data(self):
        mirna_fea, lncrna_fea = self.loader.load_RNA_fea_csv()
        print('>>>>RNA fea loaded<<<<')
        if self.params.task_type == 'run':
            Label_data = self.loader.load_pair_test()
            Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info = self.deal_test_graph_dataset(mirna_fea=mirna_fea, lncrna_fea=lncrna_fea, Label_data=Label_data)
        elif self.params.task_type == 'trainval':
            print('you give the wrong task type')
            sys.exit()
        print('>>>>Label data loaded<<<<')
        return Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info

    def deal_trainval_graph_dataset(self, mirna_fea, lncrna_fea, Label_data):
        print('dealing with trainval pair data')
        Label_data_pair = copy.deepcopy(Label_data)
        mirna_node = Label_data_pair.drop(columns=['Label', 'pair_index', 'lncrna', 'lncrna_id']).drop_duplicates(subset = ['mirna'], keep='first').copy()
        lncrna_node = Label_data_pair.drop(columns=['Label', 'pair_index', 'mirna', 'mirna_id']).drop_duplicates(subset = ['lncrna'], keep='first').copy()
        mirna_node_info = pd.merge(mirna_node, mirna_fea, left_on='mirna', right_on='Seqname', how='left', sort=False)
        mirna_node_info = mirna_node_info.dropna(subset=['Seqname'])
        lncrna_node_info = pd.merge(lncrna_node, lncrna_fea, left_on='lncrna', right_on='Seqname', how='left', sort=False)
        lncrna_node_info = lncrna_node_info.dropna(subset=['Seqname'])

        # merge dimension for feature to minus
        mirna_merge_dim = mirna_node.shape[1]
        lncrna_merge_dim = lncrna_node.shape[1]

        Label_data_pair = Label_data_pair[Label_data_pair['mirna'].isin(mirna_node_info['mirna'])]
        Label_data_pair = Label_data_pair[Label_data_pair['lncrna'].isin(lncrna_node_info['lncrna'])]

        # save data
        self.superIO(Label_data_pair, mirna_node_info, lncrna_node_info, datatype='trainval')
        
        return Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info

    def deal_test_graph_dataset(self, mirna_fea, lncrna_fea, Label_data):
        print('dealing with test pair data')
        Label_data_pair = copy.deepcopy(Label_data)
        mirna_node = Label_data_pair.drop(columns=['Label', 'pair_index', 'lncrna', 'lncrna_id']).drop_duplicates(subset = ['mirna'], keep='first').copy()
        lncrna_node = Label_data_pair.drop(columns=['Label', 'pair_index', 'mirna', 'mirna_id']).drop_duplicates(subset = ['lncrna'], keep='first').copy()
        mirna_node_info = pd.merge(mirna_node, mirna_fea, left_on='mirna', right_on='Seqname', how='left', sort=False)
        mirna_node_info = mirna_node_info.dropna(subset=['Seqname'])
        lncrna_node_info = pd.merge(lncrna_node, lncrna_fea, left_on='lncrna', right_on='Seqname', how='left', sort=False)
        lncrna_node_info = lncrna_node_info.dropna(subset=['Seqname'])

        # merge dimension for feature to minus
        mirna_merge_dim = mirna_node.shape[1]
        lncrna_merge_dim = lncrna_node.shape[1]

        Label_data_pair = Label_data_pair[Label_data_pair['mirna'].isin(mirna_node_info['mirna'])]
        Label_data_pair = Label_data_pair[Label_data_pair['lncrna'].isin(lncrna_node_info['lncrna'])]

        # save data
        self.superIO(Label_data_pair, mirna_node_info, lncrna_node_info, datatype='test')
        
        return Label_data_pair, mirna_merge_dim, lncrna_merge_dim, mirna_node_info, lncrna_node_info

    def superIO(self, Label_data_pair, mirna_node_info, lncrna_node_info, datatype):
        dataset_save_path = self.prj_path / 'data' / f'{datatype}_data' / self.params.feature_type
        if not dataset_save_path.exists():
            dataset_save_path.mkdir(parents=True)
        Label_data_pair.to_csv(dataset_save_path  / 'RNA-RNA-Interacting.csv')
        mirna_node_info.to_csv(dataset_save_path / f'{self.params.filetype}_mi.csv')
        lncrna_node_info.to_csv(dataset_save_path / f'{self.params.filetype}_lnc.csv')

