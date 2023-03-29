import numpy as np
import Bio.SeqIO as Seq
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(os.path.join(os.path.dirname(__file__), 'repDNA'))
import _01_ORF_code as ORF_code
import _02_CTDcode as CTDcode
import _03_Fickettcode as Fickettcode
import _04_kmer_counts as kmer_counts
import _06_proparcoder as proparcoder
import _07_GCcounts as GCcounts
import _08_edpfeature as edpfeature
import _09_StopCodon as StopCodon
import _18_SStructure as SStructure
import ac as RNA_ac
import psenac as RNA_psenac


import pandas as pd
from Bio import SeqIO


def switch_meth(fun, textPath):
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>RUNNING {fun}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    if fun == 'Open reading frame (1D)':
        ORF = ORF_code.ORF_count(textPath).get_ORF()
        ORFedp = edpfeature.EDPcoder(textPath).getEDP_orf()
        hexmerORF = SStructure.makeORFEucDist(textPath)

        T1 = pd.concat([ORF, ORFedp], axis=1, join='outer')
        T1 = pd.concat([T1, hexmerORF], axis=1, join='outer')
        return T1.fillna(0)
    
    elif fun == 'Codon related (1D)':
        Fickett = Fickettcode.Fickettcoder(textPath).get_fickett()
        StopCod = StopCodon.get_stop(textPath)
        T1 = pd.concat([Fickett, StopCod], axis=1, join='outer')
        return T1.fillna(0)

    elif fun == 'Guanine-cytosine related (1D)':
        return GCcounts.GCconder(textPath).get_gc()

    elif fun == 'Transcript related (1D)':
        # hexmer_trans(1col)+Transcript lengt(1col)+L5UTRL3UTR(2col)+C5UTRC3UTR(2col)+edptrans(20col)+ctd(30col)+3mer(64col)+Dist_trans(6col)
        tran_len = edpfeature.EDPcoder(textPath).get_tran_len()
        UTR_len = edpfeature.EDPcoder(textPath).getUTR_len()
        UTR_cov = edpfeature.EDPcoder(textPath).getUTR_cov()
        EDP = edpfeature.EDPcoder(textPath).getEDP()
        CTD = CTDcode.CTDcoder(textPath).get_ctd()
        # Kmer1 = kmer_counts.BasicCounter(textPath, int(1)).get_counts()
        # Kmer2 = kmer_counts.BasicCounter(textPath, int(2)).get_counts()
        Kmer3 = kmer_counts.BasicCounter(infasta=textPath, k=int(3), mean=True, std=False).get_counts()        
        # T1 = pd.concat([Kmer1, Kmer2], axis=1, join='inner')
        # T1 = pd.concat([T1, Kmer3], axis=1, join='inner')
        Dist_trans, hexmer_trans = SStructure.makeEucDist(textPath)
        return pd.concat([hexmer_trans, tran_len, UTR_len, UTR_cov, EDP, CTD, Kmer3, Dist_trans,], axis=1, join='outer').fillna(0)
    
    elif fun == 'Pseudo protein related (1D)':
        return proparcoder.ProtPar(textPath).get_protper()
    
    elif fun == 'EIIP based spectrum (1D)':
        SStruc = SStructure.makeEIIP(textPath)
        return SStruc
    
    elif fun == 'Secondary structure (1D)':
        seqname = []
        for seq in Seq.parse(textPath,'fasta'):
            seqid = seq.id
            seqname.append(seqid)
        SStruc = SStructure.extract_SSfeatures(textPath)
        SStruc.index = seqname
        tran_len = edpfeature.EDPcoder(textPath).get_tran_len()
        SStruc["nMFE"] = SStruc.iloc[:,-2]/tran_len.iloc[:,0]
        return SStruc
    
    else:
        return None

