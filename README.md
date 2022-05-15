## Correct Errors
### In the manuscript, there are some mistakes in the data given in Table 2b. Please be informed of the correction below.
| (b)      | ACC | MCC | PRE | REC | SPC | F1 | AUC |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **ncRNAInter**  | **0.9435**      | **0.8870**      | **0.9402**      | **0.9472**      | **0.9397**      | **0.9437**      | **0.9856**      |
| PmliPred    | 0.9191      | 0.8511      | 0.9178      | 0.9228      | 0.9155      | 0.9193      | 0.9682      |



# ncRNAInter
[![python 3.8](https://img.shields.io/badge/python-3.8-brightgreen)](https://www.python.org/)
### ncRNAInter: A Novel Strategy Based on Graph Neural Network to Discover Interactions between lncRNA and miRNA
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Bing Zhang, Zhanrong Li, Honglin Li* and Feng Zhu*

## Install
1. Download pretrained models and source codes of ncRNAInter.
2. ncRNAInter should be deployed on Linux.
3. The ncRNAInter tree includes three package:
```
 |- comparison
     |- Plants
        |- PmliPred_plant
        |- ncRNAInter_plant
     |- PmliPred_human
     |- LMI-INGI
 |- main
     |- data
     |- data_process
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
Directory of `comparison` deposits the reproduced methods used in comparison with ncRNAInter, including their datasets and retrained models. Usage is explained in `./comparison/README.md`.

Directory of `statistic` deposits the scripts we used to plot. Usage is explained in `./statistic/README.md`.

Directory of `main` deposits the basis of ncRNAInter, including source code, datasets and pretrained models. Usage of ncRNAInter is explained in `./main/README.md`.

## Disclaimer
ncRNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn
