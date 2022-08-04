import sys
sys.path.append("..")
from pathlib import Path
import pandas as pd

prj_path = Path(__file__).parent.resolve().parent.resolve()


importance = pd.read_csv('./importance.csv', index_col=0)

miimp = {}
lncimp = {}
for index, row in importance.iterrows():
    rowindex = index.split('rna_')
    if rowindex[0]=='mi':
        miimp[rowindex[1]]=row['importance']
    elif rowindex[0]=='lnc':
        lncimp[rowindex[1]]=row['importance']

miimp = pd.DataFrame(miimp,index=['importance']).T
lncimp = pd.DataFrame(lncimp,index=['importance']).T
miimp = miimp.sort_values(by=['importance'], ascending=False)
miimp['bits'] = miimp.index
lncimp = lncimp.sort_values(by=['importance'], ascending=False)
lncimp['bits'] = lncimp.index
# print(miimp)
# print(lncimp)

fea_path = prj_path / 'data' / 'processed_data' / 'pair_unique_RNAs' / 'mirna' / 'mirna_feature' / 'csv'
file_name_a1 = 'Codon related (1D).csv'
file_name_a2 = 'Open reading frame (1D).csv'
file_name_a3 = 'Guanine-cytosine related (1D).csv'
file_name_a4 = 'Transcript related (1D).csv'
file_name_b1 = 'Pseudo protein related (1D).csv'
file_name_b2 = 'EIIP based spectrum (1D).csv'
file_name_c1 = 'Secondary structure (1D).csv'
RNAfea_a1_mirna = pd.read_csv(fea_path / file_name_a1, index_col=0)
RNAfea_a2_mirna = pd.read_csv(fea_path / file_name_a2, index_col=0) 
RNAfea_a3_mirna = pd.read_csv(fea_path / file_name_a3, index_col=0) 
RNAfea_a4_mirna = pd.read_csv(fea_path / file_name_a4, index_col=0) 
RNAfea_b1_mirna = pd.read_csv(fea_path / file_name_b1, index_col=0) 
RNAfea_b2_mirna = pd.read_csv(fea_path / file_name_b2, index_col=0) 
RNAfea_c1_mirna = pd.read_csv(fea_path / file_name_c1, index_col=0) 

a1_bits = RNAfea_a1_mirna.columns[1:].values
a2_bits = RNAfea_a2_mirna.columns[1:].values
a3_bits = RNAfea_a3_mirna.columns[1:].values
a4_bits = RNAfea_a4_mirna.columns[1:].values
b1_bits = RNAfea_b1_mirna.columns[1:].values
b2_bits = RNAfea_b2_mirna.columns[1:].values
c1_bits = RNAfea_c1_mirna.columns[1:].values

a1_group = ['Codon related']*5
a2_group = ['ORF-related']*31
a3_group = ['GC-related']*7
a4_group = ['Transcript-related']*126
b1_group = ['Physicochemical-property']*5
b2_group = ['Physicochemical-property']*8
c1_group = ['Secondary-structure']*9
a1_subgroup = ['Fickett score']+['Stop codon related properties']*4
a2_subgroup = ['Basic ORF properties']*4+['EDP scores on ORF']*20+['Hexamer scores on ORF']*7
a3_subgroup = ['GC content properties']*7
a4_subgroup = ['Hexamer scores on transcript']*1+['Basic transcript property']*1+['UTR related properties']*4+['EDP scores on transcript']*20+['CTD descriptors']*30+['K-mer (k=3)']*64+['Hexamer scores on transcript']*6
b1_subgroup = ['Pseudo-protein properties']*5
b2_subgroup = ['EIIP spectrum scores']*8
c1_subgroup = ['Multi-scale secondary scores']*6+['Secondary structure descriptors']*3

group_dic,subgroup_dic = {},{}
for i,j in zip(a1_bits, a1_group):
    group_dic[i]=j
for i,j in zip(a1_bits, a1_subgroup):
    subgroup_dic[i]=j

for i,j in zip(a2_bits, a2_group):
    group_dic[i]=j
for i,j in zip(a2_bits, a2_subgroup):
    subgroup_dic[i]=j

for i,j in zip(a3_bits, a3_group):
    group_dic[i]=j
for i,j in zip(a3_bits, a3_subgroup):
    subgroup_dic[i]=j

for i,j in zip(a4_bits, a4_group):
    group_dic[i]=j
for i,j in zip(a4_bits, a4_subgroup):
    subgroup_dic[i]=j

for i,j in zip(b1_bits, b1_group):
    group_dic[i]=j
for i,j in zip(b1_bits, b1_subgroup):
    subgroup_dic[i]=j

for i,j in zip(b2_bits, b2_group):
    group_dic[i]=j
for i,j in zip(b2_bits, b2_subgroup):
    subgroup_dic[i]=j

for i,j in zip(c1_bits, c1_group):
    group_dic[i]=j
for i,j in zip(c1_bits, c1_subgroup):
    subgroup_dic[i]=j


# print(group_dic)
# print(subgroup_dic)
miimp['group'] = miimp['bits'].replace(group_dic)
lncimp['group'] = lncimp['bits'].replace(group_dic)
miimp['subgroup'] = miimp['bits'].replace(subgroup_dic)
lncimp['subgroup'] = lncimp['bits'].replace(subgroup_dic)

miimp.to_csv('miimp.csv')
lncimp.to_csv('lncimp.csv')


dfm = miimp.head(50)
dfl = lncimp.head(50)

print('mirna', dfm.value_counts(subset=['group','subgroup']))
print('lncrna',dfl.value_counts(subset=['group','subgroup']))
