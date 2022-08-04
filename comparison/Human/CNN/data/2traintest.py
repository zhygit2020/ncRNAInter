import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path


proj_path = Path(__file__).parent.resolve()
tem = pd.read_csv(proj_path/'train'/'RNA-RNA-Interacting.csv')
m = pd.read_csv(proj_path/'train'/'csv_mi.csv',index_col=0).drop(columns=['mirna','mirna_id'])
l = pd.read_csv(proj_path/'train'/'csv_lnc.csv',index_col=0).drop(columns=['lncrna','lncrna_id'])
tem=pd.merge(tem,m, left_on='mirna', right_on='Seqname', how='left').drop(columns=['Seqname'])
tem=pd.merge(tem,l, left_on='lncrna', right_on='Seqname', how='left').drop(columns=['Seqname'])
tem.drop(columns=['mirna','lncrna'], inplace=True)

label = tem.pop('Label')

tem.insert(len(tem.columns), 'Label' ,label)
tem.to_csv(proj_path/'train.csv')

proj_path = Path(__file__).parent.resolve()
tem = pd.read_csv(proj_path/'test'/'RNA-RNA-Interacting.csv')
m = pd.read_csv(proj_path/'test'/'csv_mi.csv',index_col=0).drop(columns=['mirna','mirna_id'])
l = pd.read_csv(proj_path/'test'/'csv_lnc.csv',index_col=0).drop(columns=['lncrna','lncrna_id'])
tem=pd.merge(tem,m, left_on='mirna', right_on='Seqname', how='left').drop(columns=['Seqname'])
tem=pd.merge(tem,l, left_on='lncrna', right_on='Seqname', how='left').drop(columns=['Seqname'])
tem.drop(columns=['mirna','lncrna'], inplace=True)

label = tem.pop('Label')

tem.insert(len(tem.columns), 'Label' ,label)
tem.to_csv(proj_path/'test.csv')