from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json


prj_path = Path(__file__).parent.resolve().parent.resolve()
datapath= prj_path / 'main'
# with open(prj_path / 'statistic' / 'color.json', 'r', encoding='utf-8') as f:
#     color = json.load(f)

epochsize = 512
hidden = 256
lr = 0.0005
fea = 'RNA_intrinsic'
K = 5
uplim = 81

fig, axs = plt.subplots(2, 1, figsize = (8,8))

data_path_noleak = datapath / 'pretrained' / 'KFold_val_based_on_pair' / '5_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}'
data_path_leak = datapath / 'pretrained' / 'KFold_val_based_on_pair_infoleak' / '5_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}'

loss_noleak =  pd.read_csv(data_path_noleak / 'process_data' / 'train_process_loss.csv', index_col=0)
acc_noleak =  pd.read_csv(data_path_noleak / 'process_data' / 'val_process_acc.csv', index_col=0)
loss_leak =  pd.read_csv(data_path_leak / 'process_data' / 'train_process_loss.csv', index_col=0)
acc_leak =  pd.read_csv(data_path_leak / 'process_data' / 'val_process_acc.csv', index_col=0)

x = loss_noleak.columns.astype(int).values[:uplim]
y_acc_noleak = acc_noleak.mean()[:uplim]
y_loss_noleak = loss_noleak.mean()[:uplim]
y_loss_leak = loss_leak.mean()[:uplim]
y_acc_leak = acc_leak.mean()[:uplim]


axs[0].plot(x, y_loss_noleak, linewidth=1.0, linestyle='-', c='b', label='loss: noleak')
axs[0].plot(x, y_loss_leak, linewidth=1.0, linestyle='-', c='r', label='loss: leak')
axs[0].set_xlim(0, uplim)
# axs[0].set_ylim(0, 0.67)
axs[0].set_xlabel('epoch')
axs[0].set_xticks(np.arange(0,uplim,10))
axs[0].set_ylabel('train_loss')
axs[0].set_yticks(np.arange(0.1,0.7,0.1))

axs[0].legend(loc='best')

axs[1].plot(x, y_acc_noleak, linewidth=1.0, linestyle='-', c='b', label='acc: noleak')
axs[1].plot(x, y_acc_leak, linewidth=1.0, linestyle='-', c='r', label='acc: leak')
axs[1].set_xlim(0, uplim)
axs[1].set_xlabel('epoch')
axs[1].set_xticks(np.arange(0,uplim,10))
axs[1].set_ylabel('val_acc')
axs[1].set_yticks(np.arange(0.6,1,0.05))

axs[1].legend(loc=4)

plt.savefig(prj_path / 'statistic' / 'figure_infoleak.tif',dpi=600,format='tif')


