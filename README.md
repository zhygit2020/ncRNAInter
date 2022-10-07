# ncRNAInter
[![python 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/)
### ncRNAInter: A Novel Strategy Based on Graph Neural Network to Discover Interactions between lncRNA and miRNA
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Minjie Mou,  Bing Zhang, Zhanrong Li, Honglin Li* and Feng Zhu*

## Install
1. Download pretrained models and source codes of ncRNAInter.
2. ncRNAInter should be deployed on Linux.
3. The ncRNAInter tree includes three package:
```
 |- comparison
     |- Human
     	|- CNN
     	|- LMI-INGI
     	|- LncMirNet
     	|- PmliPred_human
     	|- preMLI_human
     	|- RF
     	|- SVM
     |- Plants
        |- ncRNAInter_plant
        |- PmliPred_plant
     |- Virus
     	|- ncRNAInter_virus
 |- main
     |- data
     |- feature_importance
     |- models
     |- pretrained
     |- run_result
     |- utils
     |- K_fold_trainval.py
     |- run.py
 |- statistic
    |- figure_*.py
 |- requirements.txt
 |- README.md
 |- LICENSE
```
Directory of `comparison` deposits the reproduced methods used in comparison with ncRNAInter, including their datasets and retrained models. 

Directory of `statistic` deposits the scripts we used to plot.

Directory of `main` deposits the basis of ncRNAInter, including source code, datasets and pretrained models. Usage of ncRNAInter is explained in `./main/README.md`.

## Disclaimer
For more information, please refer to https://doi.org/10.1093/bib/bbac411 and cite Zhang, Hanyu et al. ncRNAInter: a novel strategy based on graph neural network to discover interactions between lncRNA and miRNA. *Briefings in bioinformatics*, bbac411. 4 Oct. 2022, doi:10.1093/bib/bbac411. PMID:36198065

Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn