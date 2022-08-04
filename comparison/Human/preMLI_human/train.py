import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import get_model
# from model import get_model_max
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn import metrics
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
prj_path = Path(__file__).parent.resolve()

def evaluate(y_label, y_score):
    # ROC, AUC
    fprs, tprs, thresholds_auc = metrics.roc_curve(y_label, y_score)
    auc = metrics.auc(fprs, tprs)
    # PRAUC
    pres, recs, thresholds_prc = metrics.precision_recall_curve(y_label, y_score)
    prauc = metrics.auc(recs, pres)
    av_prc = metrics.average_precision_score(y_label, y_score)
    # scores' label prediction by threshold
    threshold = 0.5
    label_pred = copy.deepcopy(y_score)
    label_pred = np.where(y_score >= threshold, np.ones_like(label_pred), label_pred)
    label_pred = np.where(y_score < threshold, np.zeros_like(label_pred), label_pred)
    # TN, FP, FN, TP
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_label, y_pred=label_pred, labels=[0,1]).ravel()
    # Model Evaluation
    acc = metrics.accuracy_score(y_label, label_pred)
    mcc = metrics.matthews_corrcoef(y_label, label_pred)
    precision = metrics.precision_score(y_label, label_pred)
    recall = metrics.recall_score(y_label, label_pred)
    f1 = metrics.f1_score(y_label, label_pred)
    specificity = tn/(fp+tn)
    sensitivity = tp/(tp+fn)
    return fprs, tprs, thresholds_auc, pres, recs, np.append(thresholds_prc, [1], axis=0), tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, prauc, av_prc


class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.mi = val_data[0]
        self.lnc = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.mi,self.lnc])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        self.model.save_weights(
            "./model/2021bs64/%sModel%d.h5" % (self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
       
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

name = 'preMLI'
Data_dir=prj_path/'processData'
train = np.load(Data_dir/'train2021.npz')
# train = np.load(Data_dir+'train2021.npz')
X_mi_tra, X_lnc_tra, y_tra = train['X_mi_tra'], train['X_lnc_tra'], train['y_tra']

X_mi_tra, X_mi_val,X_lnc_tra,X_lnc_val, y_tra, y_val=train_test_split(
    X_mi_tra,X_lnc_tra,y_tra,test_size=0.2,stratify=y_tra)
# X_mi_tra, X_mi_val,X_lnc_tra,X_lnc_val, y_tra, y_val=train_test_split(
#     X_mi_tra,X_lnc_tra,y_tra,test_size=0.1,stratify=y_tra)

model = get_model()
model.summary()
print('Traing %s cell line specific model ...' % name)
print(X_mi_val.shape, X_lnc_val.shape, y_val.shape)
back = roc_callback(val_data=[X_mi_val, X_lnc_val, y_val],name=name)
# history = model.fit([X_mi_tra, X_lnc_tra], y_tra, validation_data=([X_mi_val, X_lnc_val], y_val), epochs=100, batch_size=32,
#                         callbacks=[back])
history = model.fit([X_mi_tra, X_lnc_tra], y_tra, validation_data=([X_mi_val, X_lnc_val], y_val), epochs=100, batch_size=32, verbose=2,)

vallabel = model.predict([X_mi_val, X_lnc_val])

fprs, tprs, thresholds_auc, pres, recs, thresholds_prc, tn, fp, fn, tp, acc, auc, mcc, precision, recall, specificity, sensitivity, f1, auprc, av_prc = evaluate(y_val, vallabel)

# print the results of each fold
print('The', '0', 'fold')
print('TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn)
# print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
print('AUROC:', auc,'AUPRC:', auprc, 'ACC:', acc, 'F1:', f1, 'MCC:', mcc, 'precision:', precision, 'recall:', recall, 'specificity:', specificity, 'sensitivity:', sensitivity, )

# save kth result
result_savepath = Path(__file__).parent.resolve() / 'trainval_result' 
if not result_savepath.exists():
    result_savepath.mkdir(parents=True)
pd.DataFrame({'fprs':fprs, 'tprs':tprs, 'thresholds':thresholds_auc}).to_csv(result_savepath / f'ROC_0thFold.csv') # roc
pd.DataFrame({'pres':pres, 'recs':recs, 'thresholds_prc':thresholds_prc}).to_csv(result_savepath / f'PRC_0thFold.csv') # roc
pd.DataFrame(y_val).to_csv(result_savepath / f'testlabeldl_0thFold.csv') # testlabeldl
pd.DataFrame(vallabel).to_csv(result_savepath / f'fuzzy_resultslabel_0thFold.csv') # fuzzy result
pd.DataFrame({'AUROC:': auc,'AUPRC:': auprc, 'ACC:': acc, 'F1:': f1, 'MCC:': mcc, 'precision:': precision, 'recall:': recall, 'specificity:': specificity, 'sensitivity:': sensitivity},index=[0]).to_csv(result_savepath / f'trainval_result_0thFold.csv')


t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("开始时间:"+t1+"结束时间："+t2)
