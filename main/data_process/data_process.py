import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path
import Bio.SeqIO as Seq
from sklearn.model_selection import train_test_split
import argparse
import random

class data_process:
    def __init__(self, params):
        self.proj_path = Path(__file__).parent.resolve().parent.resolve()
        self.params = params
        if params.data_define == 'self':
            # seq_lib 
            mi_lib, lnc_lib = self.seq_lib()
            # merge seq information and remove NaN
            pos_pairs = self.drop_nan(mi_lib=mi_lib, lnc_lib=lnc_lib)
            # dealing with effective pair and rna information to original data file
            mi_unique, lnc_unique, pos_pairs_end = self.rna_data(pos_pairs_final=pos_pairs, mi_lib=mi_lib, lnc_lib=lnc_lib)
            # Negative sampling
            self.posneg_pairs = self.negative_pair_set(mirna=mi_unique, lncrna=lnc_unique, pos_pairs=pos_pairs_end)

        elif params.data_define == 'user':
            # read user-defined data
            pairs = pd.read_csv(self.proj_path / 'data' / 'original_data' / 'rna_pairs_user.csv')
            mi_unique = pairs['mirna','mirna_seq','mirna_id'].drop_duplicates(subset = ['mirna_id'], keep='first')
            lnc_unique = pairs['lncrna','lncrna_seq','lncrna_id'].drop_duplicates(subset = ['lncrna_id'], keep='first')
            mi_unique.to_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_seq.csv', index=False)
            lnc_unique.to_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'lncrna' / 'mirna_names_unique_seq.csv', index=False)

            if params.negative_sampling:
                self.posneg_pairs = self.negative_pair_set(mirna=mi_unique, lncrna=lnc_unique, pos_pairs=pairs)
            else:
                self.posneg_pairs = pairs
                pairs.drop(index=pairs.index).to_csv(self.proj_path / 'data' / 'original_data' / 'negative_rna_pairs.csv', index=False)
                pairs.to_csv(self.proj_path / 'data' / 'original_data' / 'pos_plus_neg_rna_pairs.csv', index=False)

    def split_test(self):
        ########## split trainval and test based on pair ##########
        trainval_idxs, test_idxs, trainval_idxs_label, test_idxs_label = train_test_split(np.arange(self.posneg_pairs.shape[0]), self.posneg_pairs['Label'].values, test_size=self.params.test_rate, random_state=self.params.random_seed,stratify = self.posneg_pairs['Label'].values)
        trainval_pairs_run = self.posneg_pairs.iloc[trainval_idxs]
        test_pairs_run = self.posneg_pairs.iloc[test_idxs]
        trainval_pos = trainval_pairs_run['Label'].value_counts()[1.0]
        trainval_neg = trainval_pairs_run['Label'].value_counts()[0.0]
        test_pos = test_pairs_run['Label'].value_counts()[1.0]
        test_neg = test_pairs_run['Label'].value_counts()[0.0]
        ########## split trainval and test based on pair ##########
        save_path_trainval = self.proj_path / 'data' / 'processed_data' / 'pair_trainval' / 'run_info' 
        if not save_path_trainval.exists():
            save_path_trainval.mkdir(parents=True)
        save_path_test = self.proj_path / 'data' / 'processed_data' / 'pair_test' / 'run_info'
        if not save_path_test.exists():
            save_path_test.mkdir(parents=True)
        trainval_pairs_run.to_csv(save_path_trainval / 'trainval_pairs_run.csv')
        test_pairs_run.to_csv(save_path_test / 'test_pairs_run.csv')
        print( f'{trainval_pairs_run.shape[0]} RNA pairs separated for trainval, {test_pairs_run.shape[0]} RNA pairs separated for test.')
        print( f'where trainval pairs have {trainval_pos} positive RNA pairs and {trainval_neg} negative RNA pairs, test pairs have {test_pos} positive RNA pairs and {test_neg} negative RNA pairs. Dataset is basically balanced')

    def template_forfea(self):
        mirna = pd.read_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_seq.csv')
        lncrna = pd.read_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'lncrna' /'lncrna_names_unique_seq.csv')
        mirna = mirna['mirna', 'mirna_seq']
        lncrna = lncrna['lncrna', 'lncrna_seq']
        # template
        mi_unique_template = pd.DataFrame()
        mi_unique_template['A'] = pd.Series([f'>{i}' for i in mirna['mirna']])
        mi_unique_template['Label'] = np.ones(mirna['mirna'].shape[0])
        mi_unique_template.to_csv(self.proj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_template.csv', index=False)
        lnc_unique_template = pd.DataFrame()
        lnc_unique_template['B'] = pd.Series([f'>{i}' for i in lncrna['lncrna']])
        lnc_unique_template['Label'] = np.ones(lncrna['lncrna'].shape[0])
        lnc_unique_template.to_csv(self.proj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'lncrna' / 'lncrna_names_unique_template.csv', index=False)
        # seq fasta
        mi_unique_seq_fasta = open(self.proj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_seq.fasta','w')
        for rna in mirna['mirna']:
            # print(mirna['mirna_seq'][mirna['mirna'].isin([rna])].values[0])
            mi_unique_seq_fasta.write(f'>{rna}'+'\n')
            mi_unique_seq_fasta.write(mirna['mirna_seq'][mirna['mirna'].isin([rna])].values[0]+'\n')
        mi_unique_seq_fasta.close()
        lnc_unique_seq_fasta = open(self.proj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'lncrna' / 'lncrna_names_unique_seq.fasta','w')
        for rna in lncrna['lncrna']:
            lnc_unique_seq_fasta.write(f'>{rna}'+'\n')
            lnc_unique_seq_fasta.write(lncrna['lncrna_seq'][lncrna['lncrna'].isin([rna])].values[0]+'\n')
        lnc_unique_seq_fasta.close()

    def seq_lib(self):
        # micr
        milib_iterator = Seq.parse(self.proj_path / 'data' / '_original_data' / 'mature.fa' , "fasta")
        # Lncr
        lnclib_iterator = Seq.parse(self.proj_path / 'data' / '_original_data' / 'gencode.v38.lncRNA_transcripts.fa' , "fasta")
        # load sequences
        mi_seqname = []
        mi_seqseq = []
        for seq in milib_iterator:
            seqseq = str(seq.seq)
            mi_seqseq.append(seqseq)
            seqid = seq.id
            mi_seqname.append(seqid)
        mi_lib = pd.DataFrame(data=mi_seqseq, index=mi_seqname, columns={'mirna_seq'})

        lnc_seqname = []
        lnc_seqseq = []
        for seq in lnclib_iterator:
            seqseq = str(seq.seq)
            lnc_seqseq.append(seqseq)
            seqid = seq.id.split('|',1)[0] 
            lnc_seqname.append(seqid)
        lnc_lib = pd.DataFrame(data=lnc_seqseq, index=lnc_seqname, columns={'lncrna_seq'}) 
        return mi_lib, lnc_lib

    def drop_nan(self, mi_lib, lnc_lib):
        ########## deal with original pairs ##########
        _pos_pairs_o = pd.read_csv(self.proj_path / 'data' / '_original_data' / 'mirnas_lncrnas_validated.txt', sep='\t', index_col=False, names=['lncrna','mirna','x'])
        x = _pos_pairs_o['x'].str.split(',',expand=True)
        for index, col in x.iteritems():
            _pos_pairs_o[index] = col[col.str.contains('ENST', na=False)]

        _pos_pairs = DataFrame()
        for index, row in _pos_pairs_o.iterrows():
            row = row.dropna(axis=0).to_frame().reset_index(drop=True).T # ENST for the first time
            _pos_pairs = pd.concat([_pos_pairs,row]) 

        _pos_pairs.columns = ['lncrna_NONECODE','mirna','x']+[f'lncrna_{i}' for i in range(len(_pos_pairs.columns)-3)]
        _pos_pairs.to_csv(self.proj_path / 'data_process' / 'positive_rna_pairs.csv' , index=False)
        ########## deal with original pairs ##########

        ########## merge seq information and remove NaN ##########
        # add mirna_seq
        pos_pairs = pd.merge(_pos_pairs, mi_lib, left_on=_pos_pairs.columns.to_list()[1], right_index=True, how='left', sort=False)
        # add lncrna_seq
        for i in range(len(_pos_pairs.columns)-3):
            pos_pairs = pd.merge(pos_pairs, lnc_lib, left_on=_pos_pairs.columns.to_list()[i+3], right_index=True, how='left', sort=False)
        pos_pairs.columns = _pos_pairs.columns.to_list() + ['mirna_seq'] + [f'lncrna_seq{i}' for i in range(len(_pos_pairs.columns)-3)] # (29919, 12)
        
        # remove mirna seq == NaN
        pos_pairs = pos_pairs.dropna(subset=['mirna_seq']) 
        # remove lncrna seq == NaN
        pos_pairs = pos_pairs.dropna(subset=[f'lncrna_seq{i}' for i in range(len(_pos_pairs.columns)-3)], how='all')
        # unfold
        pos_pairs_final = DataFrame()
        for i in range(len(_pos_pairs.columns)-3):
            pos_pairs_i = pos_pairs[['lncrna_NONECODE','mirna','x',f'lncrna_{i}','mirna_seq',f'lncrna_seq{i}']].dropna(subset=[f'lncrna_{i}',f'lncrna_seq{i}'], how='all')
            pos_pairs_i.columns = ['lncrna_NONECODE','mirna','x','lncrna','mirna_seq','lncrna_seq']
            pos_pairs_final = pd.concat([pos_pairs_final,pos_pairs_i])
        
        # drop no seq
        pos_pairs_final = pos_pairs_final.dropna(subset=['lncrna_seq']) 
        # drop_duplicates
        pos_pairs_final = pos_pairs_final.drop_duplicates(subset=['mirna','lncrna'],keep='first') 

        ### 29919-->28882-->15459-->15525-->15355 ###

        pos_pairs_final.to_csv(self.proj_path / 'data_process' / 'positive_rna_pairs_addseq_final.csv', index=False)
        
        pos_pairs_final['pair_index_original'] = pos_pairs_final.index
        pos_pairs_final['Label'] = np.ones(pos_pairs_final.shape[0])

        return pos_pairs_final

    def rna_data(self, pos_pairs_final, mi_lib, lnc_lib):
        ########## dealing with rna information ###########
        pos_mi_names = pos_pairs_final['mirna']
        pos_lnc_names = pos_pairs_final['lncrna'] 

        _mi_unique = pos_mi_names.drop_duplicates(keep='first').copy().to_frame()
        _mi_unique['mirna_id_original'] = _mi_unique.reset_index(drop=True).index + 1
        _lnc_unique = pos_lnc_names.drop_duplicates(keep='first').copy().to_frame()
        _lnc_unique['lncrna_id_original'] = - _lnc_unique.reset_index(drop=True).index - 1
        # print(_mi_unique) # [266 rows x 2 columns]
        # print(_lnc_unique) # [1642 rows x 2 columns]

        mi_unique = pd.merge(_mi_unique, mi_lib, left_on='mirna', right_index=True, how='left', sort=False) 
        mi_unique = mi_unique.dropna(subset=['mirna_seq']) # 266
        mi_unique['mirna_id'] = mi_unique.reset_index(drop=True).index + 1
        # [266 rows x 4 columns]
        # #                  mirna  mirna_id_original                mirna_seq  mirna_id
        # # 46      hsa-miR-136-5p                  1  ACUCCAUUUGUUUUGAUGAUGGA         1  
        mi_unique.to_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_names_unique_seq.csv', index=False)

        lnc_unique = pd.merge(_lnc_unique, lnc_lib, left_on='lncrna', right_index=True, how='left', sort=False)
        lnc_unique = lnc_unique.dropna(subset=['lncrna_seq']) # 1642
        lnc_unique['lncrna_id'] = - lnc_unique.reset_index(drop=True).index - 1
        # [1642 rows x 4 columns]
        # #                   lncrna  lncrna_id_original                                         lncrna_seq  lncrna_id
        # # 46     ENST00000448179.1                  -1  GTGCTCACACGCAGCTGGCAATGGCAAGTCAGTGGATTTAGGATAA...         -1 
        lnc_unique.to_csv(self.proj_path / 'data' / 'original_data' / 'pair_unique_RNAs' / 'lncrna' / 'lncrna_names_unique_seq.csv', index=False)
        ########## dealing with rna information ##########

        ########## dealing with pair information ##########
        pos_pairs_end = pd.merge(pos_pairs_final,mi_unique[['mirna','mirna_id_original','mirna_id']],on='mirna', how='right', sort=False)
        pos_pairs_end = pd.merge(pos_pairs_end,lnc_unique[['lncrna','lncrna_id_original','lncrna_id']],on='lncrna', how='right', sort=False)
        pos_pairs_end_dropna = pos_pairs_end.drop(columns='lncrna_NONECODE')
        pos_pairs_end_dropna['pair_index'] = pos_pairs_end_dropna.reset_index(drop=True).index
        pos_pairs_end_dropna = pos_pairs_end_dropna[['lncrna','mirna','x','pair_index_original','Label','lncrna_id_original','lncrna_seq','lncrna_id','mirna_id_original','mirna_seq','mirna_id','pair_index']]
        # #                   lncrna            mirna                                                  x  pair_index_original  Label  ...  lncrna_id mirna_id_original                mirna_seq  mirna_id pair_index
        # # 0      ENST00000448179.1   hsa-miR-136-5p  lnc-SAMD11-1,XLOC_000011,linc-SAMD11-1,ENSG000...                   46    1.0  ...         -1                 1  ACUCCAUUUGUUUUGAUGAUGGA         1          0
        # # 1      ENST00000448179.1  hsa-miR-148a-3p  lnc-SAMD11-1,XLOC_000011,linc-SAMD11-1,ENSG000...                   47    1.0  ...         -1                 2   UCAGUGCACUACAGAACUUUGU         2          1
        # # ...                  ...              ...                                                ...                  ...    ...  ...        ...               ...                      ...       ...        ...
        # # 15353  ENST00000529811.5     hsa-miR-4500  PCF11-AS1,ENSG00000247137,RP11-727A23.5,ENSG00...                 5327    1.0  ...      -1642               163        UGAGGUAGUAGUUUCUU       163      15353
        # # 15354  ENST00000529811.5    hsa-miR-98-5p  PCF11-AS1,ENSG00000247137,RP11-727A23.5,ENSG00...                 5328    1.0  ...      -1642               165   UGAGGUAGUAAGUUGUAUUGUU       165      15354
        # # [15355 rows x 12 columns]
        pos_pairs_end_dropna.to_csv(self.proj_path / 'data' / 'original_data' / 'rna_pairs_self.csv', index=False)
        print(f'Total valid num of positive RNA pairs is {pos_pairs_end_dropna.shape[0]}.')
        ########## dealing with pair information ##########
        return mi_unique, lnc_unique, pos_pairs_end_dropna

    def negative_pair_set(self, mirna, lncrna, pos_pairs):
        pair_positive = pos_pairs[['lncrna','mirna']].values.tolist()
        pair_selected = []
        for i in range(0,len(pair_positive)*2,1):
            pair_sel = [random.sample(lncrna['lncrna'].values.tolist(), 1)[0], random.sample(mirna['mirna'].values.tolist(), 1)[0]]
            pair_selected.append(pair_sel)
        # print(len(pair_selected)) # 83062
        for negi in pair_selected:
            if negi in pair_positive:
                pair_selected.remove(negi)
        # print(len(pair_selected)) # ~= 79280
        pair_selected01 = pair_selected[:len(pair_positive)]
        pair_negative = pd.DataFrame(pair_selected01,columns=['lncrna','mirna'])
        pair_negative['Label'] = np.zeros(pair_negative.shape[0])
        pair_negative['pair_index'] = np.arange(len(pair_positive),len(pair_positive)+pair_negative.shape[0])

        pair_negative = pd.merge(pair_negative,mirna,on='mirna', how='left', sort=False)
        pair_negative = pd.merge(pair_negative,lncrna,on='lncrna', how='left', sort=False)

        pair_negative.to_csv(self.proj_path / 'data' / 'original_data' / 'negative_rna_pairs.csv', index=False)
        pos_plus_neg_rna_pairs_dropna = pd.concat([pos_pairs, pair_negative], sort=False)
        pos_plus_neg_rna_pairs_dropna.to_csv(self.proj_path / 'data' / 'original_data' / 'pos_plus_neg_rna_pairs.csv', index=False)

        return pos_plus_neg_rna_pairs_dropna

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_define", type=str, default='self', choices=['self', 'user'], help="LMI defined by user or not, the default is self-defined")
    parser.add_argument("--negative_sampling", type=bool, default=True, choices=[True, False], help="run negative sampling if needed")
    parser.add_argument("--test_rate", type=float, default=0.1, help="rate for test dataset splitting, it will not split out test data when set as 0")



    params = parser.parse_args()
    print(vars(params))

    data_processer = data_process(params)
    data_processer.split_test()
    data_processer.template_forfea()