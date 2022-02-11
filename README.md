# RNAInter
[![python 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/)
### RNAInter: A Novel Strategy Based on Graph Neural Network to Discover Interactions between lncRNA and miRNA
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Honglin Li and Feng Zhu*

## Install
1. Download all pretrained models and source codes of RNAInter.
2. RNAInter should be deployed on Linux.
3. The RNAInter tree includes three package:
```
 |- comparation
     |- Plants
        |- PmliPred_plant
        |- RNAInter_plant
     |- PmliPred_human
     |- LMI-INGI
 |- main
     |- data
     |- models
     |- pretrained
     |- test_result
     |- utils
     |- K_fold_trainval.py
     |- run.py
 |- statistic
    |- figure_*.py
 |- requirements.txt
 |- README.md
 |- LICENSE
```
Directory of `comparation` deposits the reproduced methods used in comparism with RNAInter, including their datasets and retrained models. Usage is explained in `./comparation/README.md`.

Directory of `statistic` deposits the scripts we used to plot. Usage is explained in `./statistic/README.md`.

Directory of `main` deposits the basis of RNAInter, including source code, datasets and pretrained models. Usage of RNAInter is explained in `./main/README.md`.

## Disclaimer
RNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Hanyu Zhang at hanyu_zhang@zju.edu.cn
