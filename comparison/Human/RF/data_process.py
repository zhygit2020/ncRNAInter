import numpy as np
import pandas as pd
import torch
from pathlib import Path
import Bio.SeqIO as Seq
from data_utils import _04_kmer_counts as kmer_counts
from data_utils import _07_GCcounts as GC_counts
# from data_utils import _18_SStructure as SStructure
import RNA
import copy
import tqdm



class data_process:
    def __init__(self):
        self.proj_path = Path(__file__).parent.resolve()
        # print(self.proj_path)
        # print(self.proj_path / 'data' / 'original_data' / 'mirna.fa')

        # micr
        self.micRNAdata_iterator = Seq.parse(self.proj_path / 'data' / 'original_data' / 'mirna.fa' , "fasta")#从路径读取fasta文件
        # Lncr
        self.LncRNAdata_iterator = Seq.parse(self.proj_path / 'data' / 'original_data' / 'lncrna.fa' , "fasta")#从路径读取fasta文件
        # print(Seq.to_dict(self.micRNAdata)) # 'hsa-miR-30d-5p': SeqRecord(seq=Seq('TGTAAACATCCCCGACTGGAAG'), id='hsa-miR-30d-5p', name='hsa-miR-30d-5p', description='hsa-miR-30d-5p', dbxrefs=[])

        # for Sequence data
        self.trainval_seq_data = pd.read_csv(self.proj_path / 'data' / 'trainval_data' / 'RNA-RNA-Interacting.csv')
        self.pair_label = self.trainval_seq_data['Label'].replace(-1,0)
        self.test_seq_data = pd.read_csv(self.proj_path / 'data' / 'test_data' / 'RNA-RNA-Interacting.csv')
        self.pair_label_test = self.test_seq_data['Label'].replace(-1,0)
        # print(self.pair_label)
        self.tranval_A = self.trainval_seq_data['A']
        self.tranval_B = self.trainval_seq_data['B']
        self.test_A = self.test_seq_data['A']
        self.test_B = self.test_seq_data['B']

        # print(self.micRNAdata_dict) # 'hsa-miR-30d-5p': SeqRecord(seq=Seq('TGTAAACATCCCCGACTGGAAG'), id='hsa-miR-30d-5p', name='hsa-miR-30d-5p', description='hsa-miR-30d-5p', dbxrefs=[])
        # print(micR_name) # ['>hsa-miR-103a-3p' '>hsa-miR-103a-3p' '>hsa-miR-186-5p' ... '>hsa-miR-653-5p' '>hsa-miR-506-3p' '>hsa-miR-186-5p']

        self.mic_numBasepairs, self.lnc_numBasepairs, self.mic_nMFE, self.lnc_nMFE = self.get_MFE_basepairs()

    def seq(self):
        # for sequence data
        A_seqname = []
        A_seqseq = []
        for seq in self.micRNAdata_iterator:
            seqseq = str(seq.seq)
            A_seqseq.append(seqseq)
            seqid = '>'+seq.id
            A_seqname.append(seqid)
        Aresult = pd.DataFrame(data=A_seqseq, index=A_seqname, columns={'A_seq'})
            # print(Aresult)
            # #                                    A_seq
            # # >hsa-miR-377-3p    ATCACACAAAGGCAACTTTTGT
            # # >hsa-miR-376a-3p    ATCATAGAGGAAAATCCACGT
            # # ...                                  ...
            # # >hsa-miR-29a-3p    TAGCACCATCTGAAATCGGTTA
            # # >hsa-miR-30d-5p    TGTAAACATCCCCGACTGGAAG

            # # [258 rows x 1 columns]

        B_seqname = []
        B_seqseq = []
        for seq in self.LncRNAdata_iterator:
            seqseq = str(seq.seq)
            B_seqseq.append(seqseq)
            seqid = '>'+seq.id
            B_seqname.append(seqid)
        Bresult = pd.DataFrame(data=B_seqseq, index=B_seqname, columns={'B_seq'})
            # print(Bresult)
            # #                                                               B_seq
            # # >ENST00000562919.1  CCTGGTTCAGCCGCCTCTGCCTGCGAGACGGTTCAATTCCTGCGTG...
            # # >ENST00000561191.1  GCAGTGCGCGCAGGCGTGGGCAGCAGCGCAGCAGGCGCGCCAGGCA...
            # # ...                                                              ...
            # # >ENST00000560800.5  AATAGACAGCAAAAAGGTAAACGTTTGCCTGAAATCCGCACGCCAG...
            # # >ENST00000443587.5  TGATCTCATCAATCTAGCGGGAGAGACAGGATAACCTGTCCGAGAG...

            # # [1663 rows x 1 columns]
        
        # trainval
        A_seq = pd.merge(self.tranval_A, Aresult, left_on='A', right_index=True, how='left', sort=False)
        B_seq = pd.merge(self.tranval_B, Bresult, left_on='B', right_index=True, how='left', sort=False)
            # print(A_seq)
            # #                       A                    A_seq
            # # 0      >hsa-miR-103a-3p  AGCAGCATTGTACAGGGCTATGA
            # # 1      >hsa-miR-103a-3p  AGCAGCATTGTACAGGGCTATGA
            # # ...                 ...                      ...
            # # 27692   >hsa-miR-506-3p    TAAGGCACCCTTCTGAGTAGA
            # # 27693   >hsa-miR-186-5p   CAAAGAATTCTCCTTTTGGGCT

            # # [27694 rows x 2 columns]
        df = pd.concat([A_seq, B_seq, self.pair_label], axis=1, join='inner')[['A','B','A_seq','B_seq','Label']]
        df['pair_seq'] = df['A_seq']+df['B_seq']
        # df = df.drop(['A_seq','B_seq'],axis=1)
            # print(df)
            # #                       A                   B                                             pair_seq
            # # 0      >hsa-miR-103a-3p  >ENST00000562391.5  ...  AGCAGCATTGTACAGGGCTATGATGCGACTCTGAGTCATACTGAAT...
            # # 1      >hsa-miR-103a-3p  >ENST00000418062.1  ...  AGCAGCATTGTACAGGGCTATGACGTCTTTCATCCAGGTTGTAATT...
            # # ...                 ...                 ...  ...                                                ...
            # # 27692   >hsa-miR-506-3p  >ENST00000421255.5  ...  TAAGGCACCCTTCTGAGTAGAGGGATGGACCTCCAGATTTCCTTAC...
            # # 27693   >hsa-miR-186-5p  >ENST00000560706.5  ...  CAAAGAATTCTCCTTTTGGGCTAAGCTGCGAAGATGGCGGAGTAAG...

            # # [27694 rows x 5 columns]
        df.to_csv(self.proj_path / 'data' / 'processed_data' / 'pair_seq_trainval.csv')

        # test
        A_seq_test = pd.merge(self.test_A, Aresult, left_on='A', right_index=True, how='left', sort=False)
        B_seq_test = pd.merge(self.test_B, Bresult, left_on='B', right_index=True, how='left', sort=False)
            # print(A_seq)
            # #                       A                    A_seq
            # # 0      >hsa-miR-103a-3p  AGCAGCATTGTACAGGGCTATGA
            # # 1      >hsa-miR-103a-3p  AGCAGCATTGTACAGGGCTATGA
            # # ...                 ...                      ...
            # # 27692   >hsa-miR-506-3p    TAAGGCACCCTTCTGAGTAGA
            # # 27693   >hsa-miR-186-5p   CAAAGAATTCTCCTTTTGGGCT

            # # [27694 rows x 2 columns]
        df_test = pd.concat([A_seq_test, B_seq_test, self.pair_label_test], axis=1, join='inner')[['A','B','A_seq','B_seq','Label']]
        df_test['pair_seq'] = df_test['A_seq']+df_test['B_seq']
        # df = df.drop(['A_seq','B_seq'],axis=1)
            # print(df)
            # #                       A                   B                                             pair_seq
            # # 0      >hsa-miR-103a-3p  >ENST00000562391.5  ...  AGCAGCATTGTACAGGGCTATGATGCGACTCTGAGTCATACTGAAT...
            # # 1      >hsa-miR-103a-3p  >ENST00000418062.1  ...  AGCAGCATTGTACAGGGCTATGACGTCTTTCATCCAGGTTGTAATT...
            # # ...                 ...                 ...  ...                                                ...
            # # 27692   >hsa-miR-506-3p  >ENST00000421255.5  ...  TAAGGCACCCTTCTGAGTAGAGGGATGGACCTCCAGATTTCCTTAC...
            # # 27693   >hsa-miR-186-5p  >ENST00000560706.5  ...  CAAAGAATTCTCCTTTTGGGCTAAGCTGCGAAGATGGCGGAGTAAG...

            # # [27694 rows x 5 columns]
        df_test.to_csv(self.proj_path / 'data' / 'processed_data' / 'pair_seq_test.csv')
        
    def fea(self):
        # for feature data
        mic_1mer = kmer_counts.BasicCounter(self.proj_path / 'data' / 'original_data' / 'mirna.fa', int(1)).get_counts()
        lnc_1mer = kmer_counts.BasicCounter(self.proj_path / 'data' / 'original_data' / 'lncrna.fa', int(1)).get_counts()
        mic_2mer = kmer_counts.BasicCounter(self.proj_path / 'data' / 'original_data' / 'mirna.fa', int(2)).get_counts()
        lnc_2mer = kmer_counts.BasicCounter(self.proj_path / 'data' / 'original_data' / 'lncrna.fa', int(2)).get_counts()
        lnc_3mer = kmer_counts.BasicCounter(self.proj_path / 'data' / 'original_data' / 'lncrna.fa', int(3)).get_counts()

        # print(lnc_3mer)
        # #                    KMAAA: Transcript k-mer AAA content  KMAAG: Transcript k-mer AAG content  ...  KMCCT: Transcript k-mer CCT content  KMCCC: Transcript k-mer CCC content
        # # ENST00000562919.1                            21.915821                            -4.826000  ...                            -9.989075                           -14.076126
        # # ENST00000561191.1                            -5.729914                            -7.048238  ...                            -0.797873                             4.721872
        # # ...                                                ...                                  ...  ...                                  ...                                  ...
        # # ENST00000560800.5                             2.434963                             8.109646  ...                             3.724016                             3.852524
        # # ENST00000443587.5                            32.865532                             5.916103  ...                            -7.592497                           -15.168520

        # # [1663 rows x 64 columns]


        # print(pd.concat([mic_1mer, mic_2mer], axis=1, join='inner'))
        # #                  KMA: Transcript k-mer A content  KMG: Transcript k-mer G content  KMT: Transcript k-mer T content  ...  KMCG: Transcript k-mer CG content  KMCT: Transcript k-mer CT content  KMCC: Transcript k-mer CC content
        # # hsa-miR-377-3p                        110.242523                      -113.501389                       -15.795197  ...                         -17.541845                         -20.514526                         -47.095634
        # # hsa-miR-376a-3p                       175.177612                       -59.388824                       -98.046265  ...                          30.077204                         -65.969070                           0.523415
        # # ...                                          ...                              ...                              ...  ...                                ...                                ...                                ...
        # # hsa-miR-29a-3p                         64.787994                       -68.046844                       -15.795197  ...                          27.912699                         -20.514526                          -1.641090
        # # hsa-miR-30d-5p                         64.787994                       -22.592300                      -106.704285  ...                          27.912699                         -20.514526                          89.267998

        # # [258 rows x 20 columns]



        mic_GCcontent = GC_counts.GCconder(self.proj_path / 'data' / 'original_data' / 'mirna.fa').get_GC()
        # print(mic_GCcontent)
        # #                  GCCoW: GC content of whole sequence
        # # hsa-miR-377-3p                              0.363636
        # # hsa-miR-376a-3p                             0.380952
        # # ...                                              ...
        # # hsa-miR-29a-3p                              0.409091
        # # hsa-miR-30d-5p                              0.500000

        # # [258 rows x 1 columns]
        lnc_GCcontent = GC_counts.GCconder(self.proj_path / 'data' / 'original_data' / 'lncrna.fa').get_GC()

        mic_numBasepairs, lnc_numBasepairs, mic_nMFE, lnc_nMFE = self.mic_numBasepairs, self.lnc_numBasepairs, self.mic_nMFE, self.lnc_nMFE
        # print(mic_numBasepairs)
        # #                  number_of_basepairs
        # # hsa-miR-377-3p                     7
        # # hsa-miR-376a-3p                    3
        # # ...                              ...
        # # hsa-miR-29a-3p                     3
        # # hsa-miR-30d-5p                     3

        # # [258 rows x 1 columns]

        Aresult = pd.concat([mic_1mer, mic_2mer, mic_GCcontent, mic_numBasepairs, mic_nMFE], axis=1, join='inner')
        Aresult.index = ['>'+i for i in Aresult.index]
        # print('Aresult', Aresult)
        Bresult = pd.concat([lnc_1mer, lnc_2mer, lnc_3mer, lnc_GCcontent, lnc_numBasepairs, lnc_nMFE], axis=1, join='inner')
        Bresult.index = ['>'+i for i in Bresult.index]
        # print('Bresult', Bresult)

        
        # trainval
        A_fea = pd.merge(self.tranval_A, Aresult, left_on='A', right_index=True, how='left', sort=False)
        # print(A_fea)
        # #                       A  KMA: Transcript k-mer A content  KMG: Transcript k-mer G content  ...  KMCG: Transcript k-mer CG content  KMCT: Transcript k-mer CT content  KMCC: Transcript k-mer CC content  GCCoW: GC content of whole sequence                  number_of_basepairs             minimum_free_energy(MFE)
        # # 0      >hsa-miR-103a-3p                        50.954010                        54.482819  ...                         -17.541845                         -22.490810                         -47.095634                             0.478260                                  xxx                                  xxx
        # # 1      >hsa-miR-103a-3p                        50.954010                        54.482819  ...                         -17.541845                         -22.490810                         -47.095634                             0.478260                                  xxx                                  xxx
        # # ...                 ...                              ...                              ...  ...                                ...                                ...                                ...                                  ...                                  ...                                  ...
        # # 27692   >hsa-miR-506-3p                        32.320465                       -11.769775  ...                         -17.541845                          29.269028                          48.142464                             0.476190                                  xxx                                  xxx
        # # 27693   >hsa-miR-186-5p                       -26.121109                       -68.046844  ...                         -17.541845                          70.394562                          -1.641090                             0.409090                                  xxx                                  xxx

        # # [27694 rows x 24 columns]
        B_fea = pd.merge(self.tranval_B, Bresult, left_on='B', right_index=True, how='left', sort=False)

        A_fea = pd.concat([A_fea, self.pair_label], axis=1)
        B_fea = pd.concat([B_fea, self.pair_label], axis=1)

        A_fea.to_csv(self.proj_path / 'data' / 'processed_data' / 'A_fea_trainval.csv')
        B_fea.to_csv(self.proj_path / 'data' / 'processed_data' / 'B_fea_trainval.csv')

        # test
        A_fea_test = pd.merge(self.test_A, Aresult, left_on='A', right_index=True, how='left', sort=False)
        # print(A_fea)
        # #                       A  KMA: Transcript k-mer A content  KMG: Transcript k-mer G content  ...  KMCG: Transcript k-mer CG content  KMCT: Transcript k-mer CT content  KMCC: Transcript k-mer CC content  GCCoW: GC content of whole sequence                  number_of_basepairs             minimum_free_energy(MFE)
        # # 0      >hsa-miR-103a-3p                        50.954010                        54.482819  ...                         -17.541845                         -22.490810                         -47.095634                             0.478260                                  xxx                                  xxx
        # # 1      >hsa-miR-103a-3p                        50.954010                        54.482819  ...                         -17.541845                         -22.490810                         -47.095634                             0.478260                                  xxx                                  xxx
        # # ...                 ...                              ...                              ...  ...                                ...                                ...                                ...                                  ...                                  ...                                  ...
        # # 27692   >hsa-miR-506-3p                        32.320465                       -11.769775  ...                         -17.541845                          29.269028                          48.142464                             0.476190                                  xxx                                  xxx
        # # 27693   >hsa-miR-186-5p                       -26.121109                       -68.046844  ...                         -17.541845                          70.394562                          -1.641090                             0.409090                                  xxx                                  xxx

        # # [27694 rows x 24 columns]
        B_fea_test = pd.merge(self.test_B, Bresult, left_on='B', right_index=True, how='left', sort=False)

        A_fea_test = pd.concat([A_fea_test, self.pair_label_test], axis=1)
        B_fea_test = pd.concat([B_fea_test, self.pair_label_test], axis=1)

        A_fea_test.to_csv(self.proj_path / 'data' / 'processed_data' / 'A_fea_test.csv')
        B_fea_test.to_csv(self.proj_path / 'data' / 'processed_data' / 'B_fea_test.csv')

    def get_MFE_basepairs(self):
        A_seqname = []
        A_numpairs = []
        A_MFE = []
        print('micRNA SS start')
        with tqdm.tqdm(self.micRNAdata_iterator) as tq:
            for step,seq in enumerate(tq):
                A_seqname.append(seq.id)
                seqseq = str(seq.seq)
                seq2DF = RNA.fold(seqseq)
                seq_numpairs = seq2DF[0].count('(') # number of based pairs, not paired bases
                A_numpairs.append(seq_numpairs)
                seq_MFE=seq2DF[1]
                A_MFE.append(seq_MFE)
                # tq.set_postfix({'seq2DF': seq2DF}, refresh=True)

        mic_numpairs = pd.DataFrame(data=A_numpairs, index=A_seqname, columns={'number_of_basepairs'})
        mic_MFE = pd.DataFrame(data=A_numpairs, index=A_seqname, columns={'minimum_free_energy(MFE)'})
        print('micRNA SS finished')

        B_seqname = []
        B_numpairs = []
        B_MFE = []
        print('LncRNA SS start')
        with tqdm.tqdm(self.LncRNAdata_iterator) as tq:
            for step,seq in enumerate(tq):
                B_seqname.append(seq.id)
                seqseq = str(seq.seq)
                seq2DF = RNA.fold(seqseq)
                seq_numpairs = seq2DF[0].count('(') # number of based pairs, not paired bases
                B_numpairs.append(seq_numpairs)
                seq_MFE=seq2DF[1]
                B_MFE.append(seq_MFE)
                # tq.set_postfix({'seq2DF': seq2DF}, refresh=True)
        lnc_numpairs = pd.DataFrame(data=B_numpairs, index=B_seqname, columns={'number_of_basepairs'})
        lnc_MFE = pd.DataFrame(data=B_numpairs, index=B_seqname, columns={'minimum_free_energy(MFE)'})
        print('LncRNA SS finished')

        return mic_numpairs, lnc_numpairs, mic_MFE, lnc_MFE

        
    # def to_fa():
    #     pass

        
        

a = data_process()
a.seq()
a.fea()
# a.to_fa()