# ncRNAInter Usage
##### Hanyu Zhang, Yunxia Wang, Ziqi Pan, Xiuna Sun, Honglin Li and Feng Zhu*
## Dependencies
- ncRNAInter should be deployed on Linux.
- Dependencies can be installed using `pip install -r requirements.txt`.
- To use GPU, please install the gpu version of `dgl` and `pytorch`.
## Usage
### Predict new lncRNA-miRNA interactions using pre-trained models with default datasets
#### For any miRNA or lncRNA user want to investigate
1. Place the name of the RNA that user want to investigate into the file of `./data/run_data/target_rna.csv`. Appropriate names could refer to the files of `./data/processed_data/pair_unique_RNAs/mirna/mirna_names_unique_template.csv` and `./data/processed_data/pair_unique_RNAs/lncrna/lncrna_names_unique_template.csv`. For example, `./data/run_data/target_rna.csv` is a data file containing 6 miRNAs invesgated in literature by default.
2. Apply the original gragh and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_val_type pair --KFold_num 5 --run_mode predict --gpu -1 
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

__Output:__ the output  will be under the automatically generated `./run_result/predict/` directory, each csv file contains ranked logits against the target RNAs. Here the logits results of 6 miRNAs invesgated in literature is provided by default.

#### For any specific lncRNA-miRNA pair user want to investigate, ...
1. Place the name of specific lncRNA-miRNA pairs that user want to investigate into the file of `./data/test_data/RNA_Intrinsic/RNA-RNA-Interacing.csv`. Appropriate names could refer to the default file. If user have priori information of those pairs, labels should be placed under column "Label". Otherwise, 
2. Apply the original gragh and pre-trained models by executing the following command:
```
python ./run.py --lr 5e-4 --hidden_dim 192 --feature_type RNA_intrinsic --batch_size 32 --n_epochs 512 --KFold_val_type pair --KFold_num 5 --run_mode predict --gpu -1 
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
- `--run_mode`, type=str, default='test', choices=['test', 'predict'], help="run mode, test or predict"
- `--gpu`, type=int, default=-1, help="GPU id, -1 for cpu"
- `--random_seed`, type=int, default=42
- `--dropout`, type=float, default=0.1, help="dropout probability"


### Train your own ncRNAInter with self-defined datasets and predict

##### *Algorithm of RNA feature extraction will update later

## Disclaimer
ncRNAInter manuscript is under review, the information presented is for information purposes only. Should you have any questions, please contact Hanyu Zhang at hanyu_zhang@zju.edu.cn
