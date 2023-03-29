import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

def extract_SSfeatures(fastapath):
    robjects.r('''
                extract_SSfeatures <- function(fastapath){
                library(LncFinder)
                demo_DNA.seq <- seqinr::read.fasta(fastapath)
                demo_DNA.seq <- lapply(demo_DNA.seq, function(x){sub("u", "t", x)})
                Seqs <- LncFinder::run_RNAfold(demo_DNA.seq, RNAfold.path = "RNAfold", parallel.cores = 2)
                result_2 <- LncFinder::extract_features(Seqs, label = NULL, SS.features = TRUE,format = "SS", frequencies.file = "human", parallel.cores = 2)
                res2 <- result_2[,c(12:19)]
                return(res2)
                }
                '''
              )
    sstruc = robjects.r['extract_SSfeatures'](fastapath)
    sstruc = pandas2ri.rpy2py(sstruc)
    sstruc.columns = ["SLDLD: Structural logarithm distance to lncRNA of acguD", "SLDPD: Structural logarithm distance to pcRNA of acguD", "SLDRD: Structural logarithm distance acguD ratio", "SLDLN: Structural logarithm distance to lncRNA of acguACGU", "SLDPN: Structural logarithm distance to pcRNA of acguACGU", "SLDRN: Structural logarithm distance acguACGU ratio","SDMFE: Secondary structural minimum free energy", "SFPUS: Secondary structural UP frequency paired-unpaired"]
    return sstruc

def makeEIIP(fastapath):
    robjects.r('''
                makeEIIP <- function(fastapath){
                library(LncFinder)
                demo_DNA.seq <- seqinr::read.fasta(fastapath)
                demo_DNA.seq <- lapply(demo_DNA.seq, function(x){sub("u", "t", x)})
                result_1 <- compute_EIIP(
                              demo_DNA.seq,
                              label = NULL,
                              spectrum.percent = 0.1,
                              quantile.probs = seq(0, 1, 0.25)
                              )
                return(result_1)
                }''')
    sstruc = robjects.r['makeEIIP'](fastapath)
    sstruc = pandas2ri.rpy2py(sstruc)
    sstruc.columns = ['EipSP: Electron-ion interaction pseudopotential signal peak','EipAP: Electron-ion interaction pseudopotential average power','EiSNR: Electron-ion interaction pseudopotential signal/noise ratio','EiPS0: Electron-ion interaction pseudopotential spectrum 0','EiPS1: Electron-ion interaction pseudopotential spectrum 0.25','EiPS2: Electron-ion interaction pseudopotential spectrum 0.5','EiPS3: Electron-ion interaction pseudopotential spectrum 0.75','EiPS4: Electron-ion interaction pseudopotential spectrum 1']
    return sstruc

def makeORFEucDist(fastapath):
    r_script = '''
                makeORFEucDist <- function(fastapath){
                library(LncFinder)
                datapath <- getwd()
                cdspath = paste(datapath, "utils/repRNA/gencode.v34.pc_transcripts_test.fa", sep = "/")
                lncRNApath = paste(datapath, "utils/repRNA/gencode.v34.lncRNA_transcripts_test.fa", sep = "/")

                cds.seq = seqinr::read.fasta(cdspath)
                lncRNA.seq = seqinr::read.fasta(lncRNApath)
                referFreq <- make_referFreq(
                                            cds.seq,
                                            lncRNA.seq,
                                            k = 6,
                                            step = 1,
                                            alphabet = c("a", "c", "g", "t"),
                                            on.orf = TRUE,
                                            ignore.illegal = TRUE
                                            )
                write.table(referFreq,file=paste(datapath, "utils/repRNA/referFreq_orf.txt", sep = "/"))
                demo_DNA.seq <- seqinr::read.fasta(fastapath)
                demo_DNA.seq <- lapply(demo_DNA.seq, function(x){sub("u", "t", x)})
                EucDis <- compute_EucDistance(
                                              demo_DNA.seq,
                                              label = NULL,
                                              referFreq,
                                              k = 6,
                                              step = 1,
                                              alphabet = c("a", "c", "g", "t"),
                                              on.ORF = TRUE,
                                              auto.full = FALSE,
                                              parallel.cores = 2
                                              )


                LogDistance <- compute_LogDistance(
                                                  demo_DNA.seq,
                                                  label = NULL,
                                                  referFreq,
                                                  k = 6,
                                                  step = 1,
                                                  alphabet = c("a", "c", "g", "t"),
                                                  on.ORF = TRUE,
                                                  auto.full = FALSE,
                                                  parallel.cores = 2
                                                  )
                hexamerScore <- compute_hexamerScore(
                                                    demo_DNA.seq,
                                                    label = NULL,
                                                    referFreq,
                                                    k = 6,
                                                    step = 1,
                                                    alphabet = c("a", "c", "g", "t"),
                                                    on.ORF = TRUE,
                                                    auto.full = FALSE,
                                                    parallel.cores = 2
                                                    )
                hdata2<-cbind(EucDis,LogDistance)
                result <- cbind(hdata2,hexamerScore)
                return(result)
                }
                '''
    robjects.r(r_script)
    sstruc = robjects.r['makeORFEucDist'](fastapath)
    sstruc = pandas2ri.rpy2py(sstruc)
    sstruc.columns = ['EucDist.LNC_orf', 'EucDist.PCT_orf', 'EucDist.Ratio_orf', 'LogDist.LNC_orf', 'LogDist.PCT_orf', 'LogDist.Ratio_orf', 'Hexamer.Score_orf']
    return sstruc

def makeEucDist(fastapath):
    r_script = '''
                makeEucDist <- function(fastapath){
                library(LncFinder)
                datapath <- getwd()
                cdspath = paste(datapath, "utils/repRNA/gencode.v34.pc_transcripts_test.fa", sep = "/")
                lncRNApath = paste(datapath, "utils/repRNA/gencode.v34.lncRNA_transcripts_test.fa", sep = "/")

                cds.seq = seqinr::read.fasta(cdspath)
                lncRNA.seq = seqinr::read.fasta(lncRNApath)
                referFreq <- make_referFreq(
                                            cds.seq,
                                            lncRNA.seq,
                                            k = 6,
                                            step = 1,
                                            alphabet = c("a", "c", "g", "t"),
                                            on.orf = FALSE,
                                            ignore.illegal = TRUE
                                            )
                write.table(referFreq,file=paste(datapath, "utils/repRNA/referFreq_trans.txt", sep = "/"))
                demo_DNA.seq <- seqinr::read.fasta(fastapath)
                demo_DNA.seq <- lapply(demo_DNA.seq, function(x){sub("u", "t", x)})
                EucDis <- compute_EucDistance(
                                              demo_DNA.seq,
                                              label = NULL,
                                              referFreq,
                                              k = 6,
                                              step = 1,
                                              alphabet = c("a", "c", "g", "t"),
                                              on.ORF = FALSE,
                                              auto.full = FALSE,
                                              parallel.cores = 2
                                              )


                LogDistance <- compute_LogDistance(
                                                  demo_DNA.seq,
                                                  label = NULL,
                                                  referFreq,
                                                  k = 6,
                                                  step = 1,
                                                  alphabet = c("a", "c", "g", "t"),
                                                  on.ORF = FALSE,
                                                  auto.full = FALSE,
                                                  parallel.cores = 2
                                                  )
                hexamerScore <- compute_hexamerScore(
                                                    demo_DNA.seq,
                                                    label = NULL,
                                                    referFreq,
                                                    k = 6,
                                                    step = 1,
                                                    alphabet = c("a", "c", "g", "t"),
                                                    on.ORF = FALSE,
                                                    auto.full = FALSE,
                                                    parallel.cores = 2
                                                    )
                hdata2<-cbind(EucDis,LogDistance)
                return(list(hdata2, hexamerScore))
                }
                '''
    robjects.r(r_script)
    [sstruc, hexamerScore] = robjects.r['makeEucDist'](fastapath)
    sstruc = pandas2ri.rpy2py(sstruc)
    hexamerScore = pandas2ri.rpy2py(hexamerScore)
    # sstruc.columns = ['EucDist.LNC_trans', 'EucDist.PCT_trans', 'EucDist.Ratio_trans', 'LogDist.LNC_trans', 'LogDist.PCT_trans', 'LogDist.Ratio_trans', ]
    hexamerScore.columns = ['TraHS: Hexamer score on transcript']
    return sstruc, hexamerScore

