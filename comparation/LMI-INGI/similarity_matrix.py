import swalign
import pandas as pd
import numpy as np
import time

def smith(seq_1, seq_2):
    match = 2
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring)
    if seq_1 == seq_2:
        return 1
    else:
        alignment = sw.align(seq_1, seq_2)
        match = alignment.matches
        score = match/(max(len(seq_1),len(seq_2)))
        return score


if __name__ == '__main__':
    print('start')
    start = time.perf_counter()

    data = pd.read_csv(r'D:\潘子祺Workspace\Python\复现LMIINGI\positive_rna_pairs_dropna.csv')

    lncrna = pd.DataFrame(index=None)
    lncrna['seq'] = data['lncrna_seq'].drop_duplicates()
    lncrna['id'] = data['lncrna_id'].drop_duplicates()
    lncrna.set_index(['id'], inplace=True)

    seq = lncrna['seq'].tolist()
    ar = np.arange(1642*1642).reshape(1642,1642)
    df = pd.DataFrame(ar, dtype=float)

    for i in range(1642):
        for j in range(1642):
            seq_1 = seq[i]
            seq_2 = seq[j]
            smith(seq_1, seq_2)
            df[i][j] = smith(seq_1, seq_2)

    df.to_csv('lsm.csv')

    end =time.perf_counter()
    print("done")
    print('Running time: %s Seconds' % (end - start))


