# ncRNAInter Usage
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Honglin Li and Feng Zhu*
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



### --- Train your own ncRNAInter with self-defined datasets and predict ---

#### 1. ncRNAInter totally provides all codes to support users to utilize their own data to train their own models
##### 1.1 Record RNA interactions that users defined into the `./data/original_data/rna_pairs_user.csv` imitating the format of existing data. 
​		In the `rna_pairs_user.csv`, each row record one RNA pair, and users should provide following information in the columns: `lncrna`, which records lncRNA names; `mirna`, which records miRNA names; `Label`, which records "1 "or "0". "1" means the corresponding  pair is positive while "0" means negative; `lncrna_seq`, which records the sequence of corresponding lncRNA; `lncrna_id`, which records the id of corresponding lncRNA; `mirna_seq`, which records the sequence of corresponding miRNA; `mirna_id`, which records the id of corresponding miRNA; `pair_id`, which records the id of specific RNA pair.
##### 1.2 Generate negative samples randomly and split test dataset if needed by executing the following command:
```
python ./data_process/data_process.py --data_define user --negative_sampling True --test_rate 0.1
```
​		following parameters stipulate the specific data and processing strategy to apply:
- `--data_define`,  the program would run with the data defined by tool-supplier or users. Here the parameter should be set to "user".
- `--negative_sampling`, If users want to run negative sampling randomly, the parameter should be set to "True" else "False".
- `--test_rate`, this parameter rate for test dataset splitting, it will not split out any test data when test rate is set as 0"

  __Output:__  this process will generate three groups of files. __(1)__ `negative_rna_pairs.csv` and `pos_plus_neg_rna_pairs.csv` under `./data/original_data/`, which conclude the information of RNA pairs for graph building; __(2)__ files under `./data/original_data/pair_unique_RNAs/`, which conclude the information of RNAs for graph building; __(3)__ files under `./data/processed_data/pair_unique_RNAs/`, which will be used for RNA feature representing further.

##### 1.3 Generate the RNA feature representations by executing the following command:
```
Algorithm for RNA feature representation will be updated soon 
```
##### 1.4 Apply K-Fold training and validating on the processed data by executing the following command:
```
python ./K_fold_trainval.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type trainval --gpu -1 
```
​		following parameters stipulate the specific data, training and validating strategy to apply:
- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 192.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is RNA_intrinsic.

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

- `--n_layers`, set the number of hidden NodeSAGE layers, default is 2

  following parameters stipulate the mode while the program running:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "trainval".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ After K-Fold cross-validation, original graphs, pre-trained models and statistic evaluations will simultaneously be generated under `./pretrained/`.

#### 2. Test on test datasets with self-defined ncRNAInter if it's split out
​		Users can apply the pre-trained models on test datasets if it's split out.

##### 2.1 Apply the original graph and pre-trained models by executing the following command:

```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_num 5 --task_type run --run_mode test --gpu -1 
```

​		following parameters stipulate the specific graph and the pre-trained model to apply:

- `--lr`, set the learning rate while training, default is 5e-4.

- `--hidden_dim`, set the number of hidden GraphSAGE units, default is 192.

- `--feature_type`, set the type of RNA feature representation which would be applied, default is RNA_intrinsic.

- `--batch_size`, set the batch size while training, default is 32.

- `--n_epochs`, set the number of epochs for training, default is 512.

- `--KFold_num`, set the number of folds for K-Fold Cross-Validation, default is 5.

  following parameters stipulate the settings while predicting:

- `--task_type`, the program would run the task of "trainval" or "run". Here the parameter should be set to "run".

- `--run_mode`, the program would run under the mode of "test", "predict_on_rna" or "predict_on_pair". Here the parameter should be set to "test".

- `--gpu`, set the GPU id the program would run on, while -1 for CPU.

  __Output:__ the output  will be under the automatically generated `./run_result/test/` directory.


## Disclaimer
ncRNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Hanyu Zhang at hanyu_zhang@zju.edu.cn
