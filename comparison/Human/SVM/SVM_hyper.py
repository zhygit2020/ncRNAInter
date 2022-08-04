import os
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import optuna
import logging
from pathlib import Path
import copy
# from utils import loaddata

prj_path = Path(__file__).parent.resolve()
fea_path_mirna = prj_path / 'data' / 'trainval_data' / 'RNA_intrinsic' / 'csv_mi.csv'
fea_path_lncrna = prj_path / 'data' / 'trainval_data' / 'RNA_intrinsic' / 'csv_lnc.csv'
label = prj_path / 'data' / 'trainval_data' / 'RNA_intrinsic' / 'RNA-RNA-Interacting.csv'

#Function
################################################################
def load_tensor(type):
    pair = pd.read_csv(label, index_col=0)
    if type=='feature':
        pair_mi = pair['mirna']
        pair_lnc = pair['lncrna']
        fea_mi_lib = pd.read_csv(fea_path_mirna, index_col=0)
        fea_lnc_lib = pd.read_csv(fea_path_lncrna, index_col=0)
        mirna_fea_info = pd.merge(pair_mi, fea_mi_lib, on='mirna', how='left', sort=False)
        lncrna_fea_info = pd.merge(pair_lnc, fea_lnc_lib, on='lncrna', how='left', sort=False)
        # pair_feature = pd.concat(mirna_fea_info.iloc[:,mirna_merge_dim+1:].values,lncrna_fea_info.iloc[:,mirna_merge_dim+1:])
        pair_feature = np.concatenate(mirna_fea_info.iloc[:,3:].values, lncrna_fea_info.iloc[:,3:], axis=1)
        return pair_feature
    if type=='label':
        return pair['Label'].values

def calculate(correct_label,pred_score):
    roc_auc = roc_auc_score(correct_label, pred_score[:,1])
    precision, recall, thresholds = precision_recall_curve(correct_label, pred_score[:,1])
    prc_auc = auc(recall, precision)
    predicted_labels = np.argmax(pred_score, axis=1)
    ACC = accuracy_score(correct_label, predicted_labels)
    CM = confusion_matrix(correct_label, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(TN, FP, FN, TP))
    FPR = FP / (FP + TN)
    Pre = TP / (TP + FP)
    MCC = matthews_corrcoef(correct_label, predicted_labels)
    return ACC, FPR, Pre, MCC, roc_auc, prc_auc, TN, FP, FN, TP

def Rdsplit(total_sample, random_state = 888, split_size = 0.2):
    base_indices = np.arange(total_sample) 
    base_indices = shuffle(base_indices, random_state = random_state) 
    cv = int(len(base_indices) * split_size)
    idx_1 = base_indices[0:cv]
    idx_2 = base_indices[(cv):(2*cv)]
    idx_3 = base_indices[(2*cv):(3*cv)]
    idx_4 = base_indices[(3*cv):(4*cv)]
    idx_5 = base_indices[(4*cv):len(base_indices)]
    print(len(idx_1), len(idx_2), len(idx_3), len(idx_4), len(idx_5))
    return base_indices, idx_1, idx_2, idx_3, idx_4, idx_5

def objective(trial):
    #des
    """Load preprocessed data."""

    """Create a dataset and split it into train/dev/test."""
    feature = load_tensor('feature')
    label = load_tensor('label')
    min_max_scaler = MinMaxScaler()
    feature_scale = min_max_scaler.fit_transform(feature)

    total_sample = 24840
    base_indices, idx_1, idx_2, idx_3, idx_4, idx_5 = Rdsplit(total_sample)
    idx_all = [idx_1, idx_2, idx_3, idx_4, idx_5]

    for i, idx in enumerate(idx_all):
        print(f'Fold {i}')
        index_valid = idx
        index_train = list(set(base_indices)-set(index_valid))
        # print(len(index_train))
        feature_train = [feature_scale[i] for i in index_train]
        label_train = [label[i] for i in index_train]
        feature_valid = [feature_scale[i] for i in index_valid]
        label_valid = [label[i] for i in index_valid]
        lable0=lable1=0
        for i in label_valid:
            if i ==0:
                lable0 = lable0+1
            else:
                lable1 = lable1+1  
                         
        C=trial.suggest_int('C', 1, 30,step=1)
        gamma=trial.suggest_float('gamma', 1e-4, 1e-3,step=1e-4)  
        clf = svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True,random_state=0)
        clf.fit(feature_train,label_train)

        pred_train = clf.predict_proba(feature_train)

        pred_valid = clf.predict_proba(feature_valid)

        ACC_train, FPR_train, Pre_train, MCC_train, roc_auc_train, prc_auc_train, TN_train, FP_train, FN_train, TP_train = calculate(label_train,pred_train)
        ACC, FPR, Pre, MCC, roc_auc, prc_auc, TN, FP, FN, TP = calculate(label_valid,pred_valid)

        print('ACC_train: {}, FPR_train: {}, Pre_train: {}, MCC_train: {}, roc_auc_train: {}, prc_auc_train: {}, TN_train: {}, FP_train: {}, FN_train: {}, TP_train: {}'.format(ACC_train, FPR_train, Pre_train, MCC_train, roc_auc_train, prc_auc_train,TN_train,FP_train,FN_train,TP_train))
        print('ACC: {}, FPR: {}, Pre: {}, MCC: {}, roc_auc: {}, prc_auc: {}, TN: {}, FP: {}, FN: {}, TP: {}'.format(ACC, FPR, Pre, MCC, roc_auc, prc_auc,TN,FP,FN,TP))
        logging.info('ACC_train: {}, FPR_train: {}, Pre_train: {}, MCC_train: {}, roc_auc_train: {}, prc_auc_train: {}, TN_train: {}, FP_train: {}, FN_train: {}, TP_train: {}'.format(ACC_train, FPR_train, Pre_train, MCC_train, roc_auc_train, prc_auc_train,TN_train,FP_train,FN_train,TP_train))
        logging.info('ACC: {}, FPR: {}, Pre: {}, MCC: {}, roc_auc: {}, prc_auc: {}, TN: {}, FP: {}, FN: {}, TP: {}'.format(ACC, FPR, Pre, MCC, roc_auc, prc_auc,TN,FP,FN,TP))

    return prc_auc

################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler("SVM.log", mode="w"))
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
study = optuna.create_study(direction="maximize", sampler = optuna.samplers.RandomSampler())
logger.info("Start optimization.")

# model running
study.optimize(objective, n_trials=150)
with open("SVM.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))

