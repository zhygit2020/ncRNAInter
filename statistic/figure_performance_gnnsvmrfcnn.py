from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import json
import seaborn as sns
from sympy import O

prj_path = Path(__file__).parent.resolve().parent.resolve()
datapath= prj_path / 'main'
# with open(prj_path / 'statistic' / 'color.json', 'r', encoding='utf-8') as f:
#     color = json.load(f)

epochsize = 512
hidden = 256
lr = 0.0005
fea = 'RNA_intrinsic'
K = 5


# split_val = 'pair'
# data_path = datapath / 'pretrained' / f'KFold_val_based_on_{split_val}' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{hidden}' / f'learningrate_{lr}' / f'{fea}' 
# data_pair = pd.read_csv(data_path / 'result_data' / 'val_result.csv', index_col=0)
# data_pair = data_pair[['acc','mcc','precision','recall','specificity','f1','auc','prauc']]

ncRNAInter = [0.9309,0.8619,0.9342,0.9272,0.9346,0.9307,0.9715,0.9741]
SVM = [0.6486,0.2977,0.6611,0.6377,0.6600,0.6492,0.7132,0.7110]
RF = [0.8127,0.6955,0.8203,0.8011,0.8243,0.8106,0.8089,0.8570]
CNN = [0.8220,0.6312,0.7980,0.8623,0.7818,0.8289,0.8949,0.8899]

original = pd.Series(ncRNAInter)
other = pd.DataFrame([SVM,RF,CNN]).max()
deviation = original-other
print(original)
print(deviation)
print(other)
print(deviation/other)



fig, ax = plt.subplots(figsize = (10,5))

labels = ['ACC','MCC','PRE','REC','SPC','F1','AUROC','AUPRC']
# labels = data_pair.columns[6:-1]


# y_pair = data_pair.mean().values
# y_pair = data_pair.mean().values[6:-1]

# std_pair = data_pair.std(ddof=0).values
# std_pair = data_pair.std(ddof=0).values[6:-1]
# print(std_pair)

width = 0.2  # the width of the bars
x_pair_1 = np.arange(len(labels))-1.5*width
x_pair_2 = np.arange(len(labels))-0.5*width
x_pair_3 = np.arange(len(labels))+0.5*width
x_pair_4 = np.arange(len(labels))+1.5*width

sns.set(style='white',  font='sans-serif', font_scale=1.2)
colorlib = sns.color_palette('rainbow', 4).as_hex()
# colorlib = ['#c82423', '#2878b5', '#9ac9db', '#f8ac8c', '#ff8884']


rects1 = ax.bar(x_pair_1, ncRNAInter, width, align='center', label='ncRNAInter', color=colorlib[0], edgecolor='black')
rects2 = ax.bar(x_pair_2, SVM, width, align='center', label='SVM', color=colorlib[1], edgecolor='black')
rects3 = ax.bar(x_pair_3, RF, width, align='center', label='RF', color=colorlib[2], edgecolor='black')
rects4 = ax.bar(x_pair_4, CNN, width, align='center', label='CNN', color=colorlib[3], edgecolor='black')

# ax.errorbar(x_pair, y_pair, std_pair, fmt='none', capsize=12, color='black')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim(0.2, 1, 0.1)

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(['ACC','MCC','PRE','REC','SPC','F1','AUROC','AUPRC'], size=12)
# ax.legend(loc=1)
ax.grid(False)

# ax.bar_label(rects1, fmt='%.4f', padding=-40,size=12)
ax.set_ylabel('value', size=12)
# stdlabel = np.around(std_pair, 4)
# stdlabel = np.array(['±'+str(i) for i in stdlabel])
# ax.bar_label(rects1, labels=stdlabel, fmt='%.4f', padding=-20,size=6)
ax.legend(loc='lower right')

plt.gcf().subplots_adjust(left=0.14)

plt.savefig(prj_path / 'statistic' / 'figure_performance_gnnsvmrfcnn.tif',dpi=600,format='tif')
