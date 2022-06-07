from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from pathlib import Path
import json
from sklearn import preprocessing
import copy

prj_path = Path(__file__).parent.resolve().parent.resolve()
datapath= prj_path / 'main'
# with open(prj_path / 'statistic' / 'color.json', 'r', encoding='utf-8') as f:
#     color = json.load(f)

epochsize = 512

# lr = ['5e-05','0.0001','0.0005','0.001','0.005','0.01']
# hidden = [16,64,128,192,256,512] #16的colorbar要重设
lr = ['5e-05','0.0001','0.0005','0.001','0.005']
hidden = [64,128,192,256,512] #16的colorbar要重设
# targeth = 192
# targetlr = '0.0005'
fea = 'RNA_intrinsic'
K = 5


 
# setup the figure and axes
fig, ax1 = plt.subplots(1, 1, figsize = (5,5), subplot_kw={'projection':'3d'})

x=[]
y=[]
top_acc=[]
top=[]
width = depth = 0.1
key = 1*0.4/width
zlolims = 0.75
zuplims = 0.95

for i,l in enumerate(lr):
    for j,h in enumerate(hidden):
        data_path = datapath / 'pretrained' / 'KFold_val_based_on_pair' / f'{K}_Fold_trainval' / 'models' / f'epochsize_{epochsize}' / f'hiddensize_{h}' / f'learningrate_{l}' / f'{fea}'
        data_lh =  pd.read_csv(data_path / 'result_data' / 'val_result.csv', index_col=0)
        acc_lh = data_lh['acc'].mean()
        acc_std_lh = data_lh['acc'].std()
        x.append(i/key-width)
        y.append(j/key-depth)
        top_acc.append(acc_lh)
        top.append(acc_lh-zlolims)

bottom = np.zeros_like(top)+zlolims

topaccmin = 0.78-zlolims
topaccmax = 0.94-zlolims
scalerlimmin = 0.1
scalerlimmax = 0.5
min_max_scaler = preprocessing.MinMaxScaler((scalerlimmin,scalerlimmax))
topsc = copy.deepcopy(top)
topsc.append(topaccmin)
topsc.append(topaccmax)
topsc = min_max_scaler.fit_transform(np.expand_dims(topsc, axis=1))

for idx, value in enumerate(top):
    rgb = 1-topsc[idx][0]
    color = np.array([rgb, rgb, rgb])
    im = ax1.bar3d(x[idx], y[idx], bottom[idx], width*2, depth*2, top[idx], shade=True, color=color)

ax1.set_zlim(zlolims, zuplims)
ax1.set_xlabel('learning rate')
ax1.set_ylabel('hidden size')
ax1.set_zlabel('acc')

ax1.set_xticks(np.array(range(len(lr)))/key, lr)
ax1.set_yticks(np.array(range(len(hidden)))/key, hidden)

ax1.grid(True)

plt.show()

# plt.savefig(prj_path / 'statistic' / 'figure_hyper_a.tif',dpi=600,format='tif')



# for colobar

fig, ax2 = plt.subplots(1, 1, figsize = (5,5))
index = 1
# a=3
for i in np.arange(scalerlimmin,scalerlimmax+0.01,0.01):
    ax2.barh(index, width=0.01, height=0.1, left = i, color = np.array([1-i, 1-i, 1-i]))

ax2.set_xlim(scalerlimmin, scalerlimmax)


ax2.set_xticks(np.arange(scalerlimmin,scalerlimmax+0.1,0.1), [0.78,0.82,0.86,0.90,0.94])
ax2.set_ylim(1, 2)
ax2.get_yaxis().set_visible(False)
plt.savefig(prj_path / 'statistic' / 'colorbar_hyper_a.tif',dpi=600,format='tif')