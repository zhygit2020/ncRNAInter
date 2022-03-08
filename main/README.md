# ncRNAInter Usage
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Honglin Li and Feng Zhu*
## Dependencies
- ncRNAInter should be deployed on Linux.
- Dependencies can be installed using `pip install -r requirements.txt`.
- To use GPU, please install the gpu version of `dgl` and `pytorch`.
## Usage
### --- Predict new lncRNA-miRNA interactions using pre-trained models with pre-built graph and original LMI data ---
#### 1. For any miRNA or lncRNA user want to investigate
##### 1.1 Place the name of the RNA that user want to investigate into the `./data/run_data/target_rna.csv` imitating the examples. 
Practicable RNAs could refer to the files of `./data/processed_data/pair_unique_RNAs/mirna/mirna_names_unique_template.csv` and `./data/processed_data/pair_unique_RNAs/lncrna/lncrna_names_unique_template.csv`. By default, `./data/run_data/target_rna.csv` contains 6 miRNAs which are investigated in this study.

##### 1.2 Apply the original gragh and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_val_type pair --KFold_num 5 --run_mode predict_on_rna --gpu -1 
```
following parameters stipulate the specific gragh and the pre-trained model to apply:
- `--lr`, type=float, default=5e-4, help="learning rate while training"
- `--hidden_dim`, type=int, default=192, help="number of hidden GraphSAGE units"
- `--feature_type`, type=str, default='RNA_intrinsic'
- `--batch_size`, type=int, default=32, help="batch size"
- `--n_epochs`, type=int, default=512, help="number of training epochs"
- `--KFold_val_type`, type=str, default='pair', help="KFold validation dataset splitting"
- `--KFold_num`, type=int, default=5, help="number of folds for K-Fold Validation, default 5"

following parameters stipulate the settings while predicting:
- `--run_mode`, type=str, default='predict_on_rna', choices=['test', 'predict_on_rna', 'predict_on_pair'], help="run mode, test,predict_on_rna or predict_on_pair")
- `--gpu`, type=int, default=-1, help="GPU id, -1 for cpu" 

__Output:__ the output  will be under the automatically generated `./run_result/prerna/` directory, each csv file contains ranked logits against the target RNAs. By default, the results of 6 miRNAs investigated in this study are provided.

#### 2. For any specific lncRNA-miRNA pair user want to investigate
##### 2.1 Place the names of specific lncRNA-miRNA pairs that user want to investigate into the `./data/run_data/target_pair.csv` imitating the examples. 
Practicable RNAs could refer to the default files introduced above. If user have priori information of those pairs, labels should be placed under column "Label". Otherwise, fill the blanks with "-1".
##### 2.2 Apply the original gragh and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_val_type pair --KFold_num 5 --run_mode predict_on_pair --gpu -1 
```
following parameters stipulate the spesific gragh and the pretrained model to apply:
- `--lr`, type=float, default=5e-4, help="learning rate while training"
- `--hidden_dim`, type=int, default=192, help="number of hidden GraphSAGE units"
- `--feature_type`, type=str, default='RNA_intrinsic'
- `--batch_size`, type=int, default=32, help="batch size"
- `--n_epochs`, type=int, default=512, help="number of training epochs"
- `--KFold_val_type`, type=str, default='pair', help="KFold validation dataset spliting"
- `--KFold_num`, type=int, default=5, help="number of folds for K-Fold Validation, default 5"

following parameters stipulate the settings while predicting:
- `--run_mode`, type=str, default='predict_on_rna', choices=['test', 'predict_on_rna', 'predict_on_pair'], help="run mode, test,predict_on_rna or predict_on_pair")
- `--gpu`, type=int, default=-1, help="GPU id, -1 for cpu"

__Output:__ the output  will be under the automatically generated `./run_result/prepair/` directory, the csv file of `logits_of_prepair.csv` contains ranked logits against the LM-pairs. By default, the results of 3 LM-pairs invesgated in this study are provided.



### --- Train your own ncRNAInter with self-defined datasets and predict ---

#### 1. ncRNAInter totally provides all codes to support users to utilize their own data to train their own models
##### 1.1 Record RNA interactions that user defined into the `./data/original_data/rna_pairs_user.csv` imitating the format of existing data. 
Therein, each row record one RNA pair, and user should provide following information: `lncrna`, which records lncRNA names; `mirna`, which records miRNA names; `Label`, `1` means positive while `0` means negative; `lncrna_seq`, which records the sequence of corresponding lncRNA; `lncrna_id`, which records the id of corresponding lncRNA; `mirna_seq`, which records the sequence of corresponding miRNA; `mirna_id`, which records the id of corresponding miRNA; `pair_id`, which records the id of spesific RNA pair.
##### 1.2 Generate negative samples randomly and split test dataset if needed by executing the following command:
```
python ./utils/data_process.py --data_define user --negative_sampling True --test_rate 0.1
```
following parameters stipulate the specific data and processing strategy to apply:
- `--data_define`,  type=str, default='self', choices=['self', 'user'], help="LMI defined by user or not, the default is self-defined"
- `--negative_sampling`, type=bool, default=True, choices=[True, False], help="run negative sampling if needed"
- `--test_rate`, type=float, default=0.1, help="rate for test dataset splitting, it will not split out any test data when test rate is set as 0"

__Output:__  this process will generate three groups of files. __(1)__ `negative_rna_pairs.csv` and `pos_plus_neg_rna_pairs.csv` under `./data/original_data/`, which conclude the information of RNA pairs for graph building; __(2)__ files under `./data/original_data/pair_unique_RNAs/`, which conclude the information of RNAs for graph building; __(3)__ files under `./data/processed_data/pair_unique_RNAs/`, which will be used for RNA feature representing further.

##### 1.3 Generate the RNA feature representations by executing the following command:
```
Algorithm for RNA feature representation will be updated later 
```
##### 1.4 Apply K-Fold training and validating on the processed data by executing the following command:
```
python ./K_fold_trainval.py --gpu -1 --lr 5e-4 --weight_decay 5e-4 --n_epochs 512 --hidden_dim 192 --n_layers 2 --task_type trainval --feature_type RNA_intrinsic --batch_size 32 --KFold_num 5
```
following parameters stipulate the specific data, training and validating strategy to apply:
- `--gpu`, type=int, default=-1, help="GPU id, -1 for CPU"
- `--lr`, type=float, default=5e-4, help="learning rate while training"
- `--weight_decay`, type=float, default=5e-4, help="Weight for L2 loss"
- `--n_epochs`, type=int, default=512, help="number of training epochs"
- `--hidden_dim`, type=int, default=192, help="number of hidden GraphSAGE units"
- `--n_layers`, type=int, default=2, help="number of hidden NodeSAGE layers"
- `--task_type`, type=str, default='trainval', choices=['trainval', 'run'], help="task type, trainval mode or run mode"
- `--feature_type`, type=str, default='RNA_intrinsic'
- `--batch_size`, type=int, default=32, help="batch size"
- `--KFold_num`, type=int, default=5, help="number of folds for K-Fold Validation, default 5"

__Output:__ After K-Fold cross-validation, original graphs, pre-trained models and statistic evaluations will simultaneously be generated under `./pretrained/`.

#### 2. Test on test datasets with self-defined ncRNAInter if it's split out
User can apply the pre-trained models on test datasets if it's split out.

##### 2.1 Apply the original gragh and pre-trained models by executing the following command:

```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_val_type pair --KFold_num 5 --run_mode test --gpu -1 
```

following parameters stipulate the spesific gragh and the pretrained model to apply:

- `--lr`, type=float, default=5e-4, help="learning rate while training"
- `--hidden_dim`, type=int, default=192, help="number of hidden GraphSAGE units"
- `--feature_type`, type=str, default='RNA_intrinsic'
- `--batch_size`, type=int, default=32, help="batch size"
- `--n_epochs`, type=int, default=512, help="number of training epochs"
- `--KFold_val_type`, type=str, default='pair', help="KFold validation dataset spliting"
- `--KFold_num`, type=int, default=5, help="number of folds for K-Fold Validation, default 5"

following parameters stipulate the settings while predicting:

- `--run_mode`, type=str, default='predict_on_rna', choices=['test', 'predict_on_rna', 'predict_on_pair'], help="run mode, test,predict_on_rna or predict_on_pair")
- `--gpu`, type=int, default=-1, help="GPU id, -1 for cpu"

__Output:__ the output  will be under the automatically generated `./run_result/test/` directory.


## Disclaimer
ncRNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Hanyu Zhang at hanyu_zhang@zju.edu.cn
