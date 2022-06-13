from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json


prj_path = Path(__file__).parent.resolve().parent.resolve()
datapath= prj_path / 'main'
with open(prj_path / 'statistic' / 'color.json', 'r', encoding='utf-8') as f:
    color = json.load(f)

epochsize = 512
hidden = 192
lr = 0.0005
fea = 'RNA_intrinsic'
K = 5

data_path = datapath / 'pretrained' / 'KFold_val_based_on_pair' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}'
fig, axs = plt.subplots(1, 1, figsize = (5,5))
for k in range(K):
    data_kth =  pd.read_csv(data_path / 'ROC_data' / f'{k}th_Fold' / f'val_ROC_for_{k}th_Fold.csv', index_col=0)
    x_k = data_kth['fprs'].values
    y_k = data_kth['tprs'].values
    axs.plot(x_k, y_k, linewidth=1.0, linestyle='-', label=f'Fold_{k}')

axs.plot([0,1], [0,1], c = 'black', linewidth=1.0, linestyle='--')
axs.set_xlim(0, 1)
axs.set_ylim(0, 1)
axs.set_xlabel('FPR', size=12)
axs.set_ylabel('TPR', size=12)
axs.grid(False)
axs.legend(loc='best')

# fig.tight_layout()
plt.savefig(prj_path / 'statistic' / 'figure_ROC.tif',dpi=600,format='tif')