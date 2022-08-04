#生成相同顺序字典
import sys
import pandas as pd
import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


file = './outdir/testRNA2vec-20220709-1358-k3to6-100d-10c-712935350Mbp-sliding-qr5.w2v'        #输入要处理的数据
# file = './outdir/testRNA2vec-20220709-1505-k3to6-100d-10c-38661960Mbp-sliding-tiL.w2v'        #输入要处理的数据
print(file)
temp = 'null'
for i  in range(0,100):
    temp = temp + ' 0'


df = pd.read_csv(file,header=None)
#df = df.drop(index=0)
df.loc[0] = temp
print(df)
print(df.columns)
print(df.index)


new_df = pd.DataFrame()


new_df['name'] = df[0].apply(lambda x:x.split(' ')[0])
new_df['list'] = df[0].apply(lambda x:x.split(' ')[1:])
new_df.index = list(new_df['name'])

print(new_df)

f = ['A', 'C', 'G', 'T']
# c = itertools.product(f, f, f)
c = itertools.product(f, f, f, f, f, f)
res = []
res.append('null')
for i in c:
    temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
    # temp = i[0]+i[1]+i[2]
    res.append(temp)
res = np.array(res)

order = res

aim_df = pd.DataFrame(index=order)
aim_df = pd.merge(aim_df, new_df, left_index=True, right_index=True, how='left')
# aim_df["name"].replace(to_replace=np.nan, value="null", inplace=True)

# aim_df = new_df.loc[order]
print(aim_df.columns)
# print(aim_df)

print(aim_df['list'])

yxy_arr = []

for i in aim_df['list']:
    if str(i)=='nan':
        i=aim_df['list'][0]
    li = [float(j) for j in i]
    yxy_arr.append(li)

yxy_arr = np.array(yxy_arr)
print(yxy_arr)
print(yxy_arr.shape)
np.save("./outdir/lnc6mers.npy",yxy_arr)

b = np.load("./outdir/lnc6mers.npy", allow_pickle=True)
print(b)

