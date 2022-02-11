import pandas as pd

data = pd.read_csv(r'D:\潘子祺Workspace\Python\复现LMIINGI\positive_rna_pairs_dropna.csv')

lid = data['lncrna_id'].drop_duplicates().tolist()
mid = data['mirna_id'].drop_duplicates().tolist()

ldf = pd.DataFrame(lid).set_index([0])
ldf['id'] = list(range(1,1643))
mdf = pd.DataFrame(mid).set_index([0])
mdf['id'] = list(range(1,267))
mdf['name'] = data['mirna'].drop_duplicates().tolist()

pair = pd.DataFrame()
pair['lnc'] = data['lncrna_id']
pair['mi'] = data['mirna_id']

print(mdf)
mdf.to_csv('aaaaa.csv')
# for i in range(15355):
#     pair['lnc'][i] = ldf['id'][pair['lnc'][i]]
#     pair['mi'][i] = mdf['id'][pair['mi'][i]]
#
#
#
# pair.to_csv("pair.csv")