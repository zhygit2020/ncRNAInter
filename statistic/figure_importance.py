import sys
from pathlib import Path
prj_path = Path(__file__).parent.resolve().parent.resolve()
sys.path.append(str(prj_path))
from collections import defaultdict
from tqdm import tqdm
from copy import copy
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rdkit import Chem
import argparse
import os

datapath= prj_path / 'main' / 'feature_importance'
epochsize = 512
hidden = 256
lr = 0.0005
fea = 'RNA_intrinsic'
K = 5
sns.set(style='white',  font='sans-serif', font_scale=1.2)
# group = 'subgroup'
group = 'group'
# group = 'bits'

miimp = pd.read_csv(datapath/'miimp.csv', index_col=0)
lncimp = pd.read_csv(datapath/'lncimp.csv', index_col=0)

labels = miimp[f'{group}'].drop_duplicates()
colorlib = sns.color_palette('rainbow', len(labels)).as_hex()
colormap={}
for k,c in zip(labels,colorlib):
    colormap[k]=c 

f, (ax1, ax2 ) = plt.subplots(2, 1, figsize=(10,10))

for bit in miimp.head(50).index:
    label= miimp.loc[bit][f'{group}']
    rects1 = ax1.bar(miimp.loc[bit]['bits'], miimp.loc[bit]['importance'], label=label, width = 0.8, lw = 0.8, align='center', color=colormap[label], edgecolor=colormap[label])
for bit in lncimp.head(50).index:
    label= lncimp.loc[bit][f'{group}']
    rects2 = ax2.bar(lncimp.loc[bit]['bits'], lncimp.loc[bit]['importance'], label=label, width = 0.8, lw = 0.8, align='center', color=colormap[label], edgecolor=colormap[label])

ax1.set_xticks([])
ax2.set_xticks([])

ax1.set_xlabel('Top 50 important features of miRNA', fontsize='medium')
ax2.set_xlabel('Top 50 important features of lncRNA', fontsize='medium')

ax1.set_ylabel('Feature importance score', fontsize='medium')
ax2.set_ylabel('Feature importance score', fontsize='medium')



from collections import OrderedDict
 
handles_1, labels_1 = ax1.get_legend_handles_labels()
by_label_1 = OrderedDict(zip(labels_1, handles_1))
ax1.legend(by_label_1.values(), by_label_1.keys(),title='<Type of The Feature>')
handles_2, labels_2 = ax2.get_legend_handles_labels()
by_label_2 = OrderedDict(zip(labels_2, handles_2))
ax2.legend(by_label_2.values(), by_label_2.keys(),title='<Type of The Feature>')


plt.savefig('./feature_importance_rank.tif',dpi=600,format='tif',bbox_inches="tight")