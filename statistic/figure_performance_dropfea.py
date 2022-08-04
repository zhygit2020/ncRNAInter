from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import json
import seaborn as sns

prj_path = Path(__file__).parent.resolve().parent.resolve()
datapath= prj_path / 'main'
# with open(prj_path / 'statistic' / 'color.json', 'r', encoding='utf-8') as f:
#     color = json.load(f)

epochsize = 512
hidden = 256
lr = 0.0005
fea = 'RNA_intrinsic'
K = 5
sns.set(style='white',  font='sans-serif', font_scale=1.2)
colorlib = sns.color_palette('rainbow', 3).as_hex()
# colorlib = ['#c82423', '#2878b5', '#9ac9db', '#f8ac8c', '#ff8884']


data_path_ru = datapath / 'pretrained' / f'KFold_val_based_on_pair_rmuselessmifea' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}' 
data_pair_ru = pd.read_csv(data_path_ru / 'result_data' / 'val_result.csv', index_col=0)
data_path_ps = datapath / 'pretrained' / f'KFold_val_based_on_pair_premistruc' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}' 
data_pair_ps = pd.read_csv(data_path_ps / 'result_data' / 'val_result.csv', index_col=0)
data_path_o = datapath / 'pretrained' / f'KFold_val_based_on_pair' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}' 
data_pair_o = pd.read_csv(data_path_o / 'result_data' / 'val_result.csv', index_col=0)

data_pair_ru = data_pair_ru[['acc','mcc','precision','recall','specificity','f1','auc','prauc']]
data_pair_ps = data_pair_ps[['acc','mcc','precision','recall','specificity','f1','auc','prauc']]
data_pair_o = data_pair_o[['acc','mcc','precision','recall','specificity','f1','auc','prauc']]

fig, ax = plt.subplots(figsize = (10,5))

# labels_o = data_pair_o.columns[6:11,12:-1]
# labels_ps = data_pair_ps.columns[6:11,12:-1]
# labels_ru = data_pair_ru.columns[6:11,12:-1]

# y_pair_o = data_pair_o.mean().values[6:11,12:-1]
# y_pair_ps = data_pair_ps.mean().values[6:11,12:-1]
# y_pair_ru = data_pair_ru.mean().values[6:11,12:-1]

# std_pair_o = data_pair_o.std(ddof=0).values[6:11,12:-1]
# std_pair_ps = data_pair_ps.std(ddof=0).values[6:11,12:-1]
# std_pair_ru = data_pair_ru.std(ddof=0).values[6:11,12:-1]


labels_ru = data_pair_ru.columns
labels_ps = data_pair_ps.columns
labels_o = data_pair_o.columns

y_pair_ru = data_pair_ru.mean().values
y_pair_ps = data_pair_ps.mean().values
y_pair_o = data_pair_o.mean().values

print(pd.DataFrame([y_pair_ru,y_pair_ps,y_pair_o,]).std(ddof=0))


std_pair_ru = data_pair_ru.std(ddof=0).values
std_pair_ps = data_pair_ps.std(ddof=0).values
std_pair_o = data_pair_o.std(ddof=0).values
# print(std_pair)

width = 0.25  # the width of the bars

x_pair_ru = np.arange(len(labels_ru))-width
x_pair_ps = np.arange(len(labels_ps))
x_pair_o = np.arange(len(labels_o))+width


rects1 = ax.bar(x_pair_ru, y_pair_ru, width, align='center', label='model 1', color=colorlib[2], edgecolor='black')
rects2 = ax.bar(x_pair_ps, y_pair_ps, width, align='center', label='model 2', color=colorlib[1], edgecolor='black')
rects3 = ax.bar(x_pair_o, y_pair_o, width, align='center', label='model 3', color=colorlib[0], edgecolor='black')

# ax.errorbar(x_pair_o, y_pair_o, std_pair_o, fmt='none', capsize=4, color='black')
# ax.errorbar(x_pair_ru, y_pair_ru, std_pair_ru, fmt='none', capsize=4, color='black')
# ax.errorbar(x_pair_ps, y_pair_ps, std_pair_ps, fmt='none', capsize=4, color='black')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.8, 1, 0.05)

ax.set_xticks(np.arange(len(labels_o)))
ax.set_xticklabels(['ACC','MCC','PRE','REC','SPC','F1','AUROC','AUPRC'], size=12)
# ax.legend(loc=1)
ax.grid(False)

# ax.bar_label(rects1, fmt='%.4f', padding=-40,size=12)
ax.set_ylabel('value', size=12)
# stdlabel = np.around(std_pair, 4)
# stdlabel = np.array(['±'+str(i) for i in stdlabel])
# ax.bar_label(rects1, labels=stdlabel, fmt='%.4f', padding=-20,size=6)

ax.legend()

plt.gcf().subplots_adjust(left=0.14)

plt.savefig(prj_path / 'statistic' / 'figure_performance_dropfea_noerrorbar.tif',dpi=600,format='tif')
