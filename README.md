# RNAInter
[![python 3.7](https://img.shields.io/badge/python-3.7-brightgreen)](https://www.python.org/)
### RNAInter: A Novel Strategy Based on Graph Neural Network to Discover Interactions between lncRNA and miRNA
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Honglin Li and Feng Zhu
For more information, please refer to the publication in [bioRxiv 2020.05.13.094953.](https://www.biorxiv.org/content/10.1101/2020.05.13.094953v1)

## Install
1. Download all pretrained models and source codes of RNAInter.
2. the RNAInter tree includes package:
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
Directory of `comparation` deposits the reproduced methods in comparism with RNAInter, including their datasets and retrained models. Usage is explained in `./comparation/README.md`.

Directory of `statistic` deposits the scripts we used to plot. Usage is explained in `./statistic/README.md`.

Directory of `main` deposits the basis of RNAInter, including source code, datasets and pretrained models. Usage of RNAInter is explained in `./main/README.md`.

## Disclaimer
RNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Hanyu Zhang at hanyu_zhang@zju.edu.cn
