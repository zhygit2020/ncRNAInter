import swalign
import pandas as pd
import numpy as np
import time


def smith(seq_1, seq_2):
    match = 1
    mismatch = 0
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, gap_penalty=0)
    if seq_1 == seq_2:
        return 1
    else:
        alignment = sw.align(seq_1, seq_2)
        match = alignment.matches
        score = alignment.score
        score = match / (max(len(seq_1), len(seq_2)))
        # score = score / (5*(max(len(seq_1), len(seq_2))))
        return score


def calc(x, i):
    seq_1 = ldf['seq'][x]
    seq_2 = ldf['seq'][i]
    return smith(seq_1, seq_2)

start = time.perf_counter()
print('start')

data = pd.read_csv(r'D:\潘子祺Workspace\Python\复现LMIINGI\positive_rna_pairs_dropna.csv')
num = 266
mode = 'mi'

lnc_id = data[mode+'rna_id'].drop_duplicates().tolist()
lnc_sq = data[mode+'rna_seq'].drop_duplicates().tolist()
ldf = pd.DataFrame(lnc_id)
ldf['seq'] = pd.DataFrame(lnc_sq)

df = pd.DataFrame(np.arange(num).reshape(1, num))
x = pd.DataFrame()
x = x.append([df]*num, ignore_index=True)
df = pd.DataFrame(x.values.T, index=x.columns, columns=x.index, dtype=float)

for i in range(0, num):
    df[i] = df[i].apply(calc, args=(i,))
    df.to_csv('D:\潘子祺Workspace\Python\复现LMIINGI\matlab复现\Datasets\seqSimilarity\mirna_seq_similarity_matrix.csv')

end = time.perf_counter()
print('done', '\n', 'Running time: %s Seconds' % (end - start))