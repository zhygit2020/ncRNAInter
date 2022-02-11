from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
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


split_val = 'pair'
data_path = datapath / 'pretrained' / f'KFold_val_based_on_{split_val}' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}' 
data_pair = pd.read_csv(data_path / 'result_data' / 'val_result.csv', index_col=0)

fig, ax = plt.subplots(figsize = (5,5))

labels = data_pair.columns[6:]

x_pair = np.arange(len(labels))
y_pair = data_pair.mean().values[6:]

std_pair = data_pair.std().values[6:]

width = 0.8  # the width of the bars

rects1 = ax.bar(x_pair, y_pair, width, align='center', color=(1,1,1), edgecolor='black')

ax.errorbar(x_pair, y_pair, std_pair, fmt='none', capsize=10, color='black')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.8, 1, 0.05)
ax.set_xticks(x_pair, ['ACC','AUC','MCC','PRE','REC','SPC','F1'])

# ax.legend(loc=1)
ax.grid(False)

ax.bar_label(rects1, fmt='%.4f', padding=-30,size=8)


plt.savefig(prj_path / 'statistic' / 'figure_pair.tif',dpi=600,format='tif')
