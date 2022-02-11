import swalign
import pandas as pd
import time
start = time.perf_counter()
# Setup your scoring matrix
# (this can also be read from a file like BLOSUM, etc)
#
# Or you can choose your own values.
# 2 and -1 are common for an identity matrix.

match = 2
mismatch = -1
scoring = swalign.NucleotideScoringMatrix(match, mismatch)

# This sets up the aligner object. You must set your scoring matrix, but
# you can also choose gap penalties, etc...
sw = swalign.LocalAlignment(scoring, gap_penalty=-1)

# Using your aligner object, calculate the alignment between
# ref (first) and query (second)
data = pd.read_csv(r'D:\潘子祺Workspace\Python\复现LMIINGI\positive_rna_pairs_dropna.csv')

seq1 = data['mirna_seq'][10279]
seq2 = data['mirna_seq'][14967]
print(len(seq1),len(seq2))
alignment = sw.align(seq1, seq2)
end = time.perf_counter()
alignment.dump()
print('\n', 'matches = ', alignment.matches)
print('\n', 'Running time: %s Seconds' % (end - start))

