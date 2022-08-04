# ncRNAInter Usage
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Minjie Mou, Bing Zhang, Zhanrong Li, Honglin Li* and Feng Zhu*
## Dependencies
- ncRNAInter should be deployed on Linux.
- Dependencies can be installed using `pip install -r requirements.txt`.
- To use GPU, please install the gpu version of `dgl` and `pytorch`.
## Usage
### --- Predict new lncRNA-miRNA interactions using pre-trained models with pre-built graph and original LMI data ---
#### 1. For any miRNA or lncRNA users want to investigate
##### 1.1 Place the name of the RNA that users want to investigate into the `./data/run_data/target_rna.csv` imitating the examples. 
​		Practicable RNAs could refer to the files of `./data/processed_data/pair_unique_RNAs/mirna/mirna_names_unique_template.csv` and `./data/processed_data/pair_unique_RNAs/lncrna/lncrna_names_unique_template.csv`. By default, `./data/run_data/target_rna.csv` contains 6 miRNAs which are investigated in this study.

##### 1.2 Apply the original graph and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode predict_on_rna --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 192.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is RNA_intrinsic.

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run"

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "predict_on_rna".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/prerna/` directory, each .csv file contains ranked logits of candidates against the target RNAs. For example, the results of 6 miRNAs investigated in this study are provided.
#### 2. For any specific lncRNA-miRNA pairs users want to investigate
##### 2.1 Place the names of specific lncRNA-miRNA pairs that users want to investigate into the `./data/run_data/target_pair.csv` imitating the examples. 
​		Practicable RNAs could refer to the default files introduced above. If users have priori information of those pairs, labels should be placed under column "Label". Otherwise, fill the blanks with "-1".
##### 2.2 Apply the original graph and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode predict_on_pair --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 192.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is RNA_intrinsic.

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run".

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "predict_on_pair".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/prepair/` directory, the .csv file of `logits_of_prepair.csv` contains ranked logits of the evaluating pairs. For example, the results of 3 LM-pairs investigated in this study are provided.
#### 3. For any specific lncRNA or miRNA users want to investigate
##### 3.1 Place the names of specific lncRNA or miRNA that users want to investigate into the `./data/run_data/target_rna.csv` imitating the examples. 
​		Practicable RNAs could refer to the default files introduced above.
##### 3.2 Apply the original graph and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode predict_on_rna --gpu -1 
```
​		following parameters stipulate the specific graph and the pre-trained model to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 192.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is RNA_intrinsic.

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run".

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "predict_on_rna".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/prerna/` directory, the .csv files of `logits_of_{name of prerna}.csv` contains ranked logits of the spesific rnas. 
## Disclaimer
ncRNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Dr. Zhang at hanyu_zhang@zju.edu.cn
