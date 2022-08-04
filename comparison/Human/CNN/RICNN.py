import os
import tensorflow as tf
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import ensemble, metrics, preprocessing, svm
from model import train_model as model_RPIPred


def mkdata(ath_feature):
    np.random.seed(1234)
    ath_feature_npy = np.array(ath_feature)
    np.random.shuffle(ath_feature_npy)
    ath_label = ath_feature_npy[:, -1]
    ath_feature_data = ath_feature_npy[:, :(ath_feature_npy.shape[1] - 1)]
    return ath_feature_data, ath_label



def  filter_data(train_pro, threshold=0.8):
    a = (train_pro == 0).sum(axis=0) / len(train_pro)
    # train_pro.to_csv('/public/home/wangyx/LncRNA/smallRNA/code/RPITER/sample/train_pro.csv')
    b = a[(a <= threshold)].index
    res = [i - a.index[0] for i in b]
    train_pro01 = train_pro.iloc[:, res]
    return train_pro01


def train_model(path, Result_path):
    Accs = []
    Sns = []
    Sps = []
    Pres = []
    MCCs = []
    AUCs = []
    AUPRCs = []
    mdlnms = []
    times = []
    test_nms = []
    train_nms = []
    methnms = []
    best_paras = []
    TNs = []
    FPs = []
    FNs = []
    TPs = []
    Accs_v = []
    Sns_v = []
    Sps_v = []
    Pres_v = []
    MCCs_v = []
    AUCs_v = []
    AUPRCs_v = []
    best_paras_v = []
    TNs_v = []
    FPs_v = []
    FNs_v = []
    TPs_v = []

    os.makedirs(Result_path, exist_ok=True)
    dir = os.path.join(path , 'data')

    train_feature_raw = pd.read_csv(os.path.join(dir,'train.csv'), header=None, low_memory=False)
    test_feature_raw = pd.read_csv(os.path.join(dir,'test.csv'), header=None, low_memory=False)
    # filter data
    data_all = pd.concat([train_feature_raw, test_feature_raw], axis=0)
    # print(data_all)
    data_pro_filters = filter_data(data_all, threshold=0.8)
    train_feature_filter = data_pro_filters.iloc[:train_feature_raw.shape[0], :]
    test_feature_filter = data_pro_filters.iloc[train_feature_raw.shape[0]:, :]
    # print(train_feature_filter)
    # print(test_feature_filter)

    traindata, trainlabel = mkdata(train_feature_filter)
    testdata, testlabel = mkdata(test_feature_filter)

    # normalization data
    scaler = preprocessing.StandardScaler().fit(traindata)
    traindata = scaler.transform(traindata)
    testdata = scaler.transform(testdata)
    # print('testdata', testdata)
    # print(0)

    # feature selection
    # if select:
    #     selected_final = feature_sel(train_feature_filter)
    #     traindata = traindata[:, selected_final]
    #     testdata = testdata[:, selected_final]

    modelnms = ['RF', 'svm', 'DNN', 'CNN', 'stackDNN']
    # for model_num in range(len(modelnms)):
    for model_num in range(3, 4, 1):
        mcc = -1
        time_start = time.time()
        modelnm = modelnms[model_num]
        print('The current methods is: {}'.format(modelnm))


        Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, best_para, TN, FP, FN, TP, Acc_v, Sn_v, Sp_v, Pre_v, MCC_v, AUC_v, AUPRC_v, best_para_v, TN_v, FP_v, FN_v, TP_v = model_RPIPred(traindata, trainlabel, testdata, testlabel, modelnm='CNN')

        # methnms.append(dirs)
        # train_nms.append(file_temp_train)
        # test_nms.append(file_temp_test)
        mdlnms.append(modelnm)
        times.append((time.time() - time_start) / 60)
        Accs_v.append(Acc_v)
        MCCs_v.append(MCC_v)
        AUCs_v.append(AUC_v)
        AUPRCs_v.append(AUPRC_v)
        Sns_v.append(Sn_v)
        Sps_v.append(Sp_v)
        Pres_v.append(Pre_v)
        best_paras_v.append(best_para_v)
        TNs_v.append(TN_v)
        FPs_v.append(FP_v)
        FNs_v.append(FN_v)
        TPs_v.append(TP_v)
        Accs.append(Acc)
        MCCs.append(MCC)
        AUCs.append(AUC)
        AUPRCs.append(AUPRC)
        Sns.append(Sn)
        Sps.append(Sp)
        Pres.append(Pre)
        best_paras.append(best_para)
        TNs.append(TN)
        FPs.append(FP)
        FNs.append(FN)
        TPs.append(TP)


        dict_value = {
            # 'method name': methnms,
            # 'train_nms': train_nms,
            # 'test_nms': test_nms,
            'mdlnms': mdlnms,
            'times': times,
            'Accs_v': Accs_v,
            'MCCs_v': MCCs_v,
            'AUCs_v': AUCs_v,
            'AUPRCs_v': AUPRCs_v,
            'Sns_v': Sns_v,
            'Sps_v': Sps_v,
            'Pres_v': Pres_v,
            'best_para_v': best_paras_v,
            'TN_v':TNs_v,
            'FP_v':FPs_v,
            'FN_v':FNs_v,
            'TP_v':TPs_v,
            'Accs': Accs,
            'MCCs': MCCs,
            'AUCs': AUCs,
            'AUPRCs': AUPRCs,
            'Sns': Sns,
            'Sps': Sps,
            'Pres': Pres,
            'best_para': best_paras,
            'TN':TNs,
            'FP':FPs,
            'FN':FNs,
            'TP':TPs,
        }
        print(dict_value)
        data_AUC = pd.DataFrame(dict_value)
        data_AUC.to_csv(os.path.join(Result_path, modelnm + 'result.csv'))


if __name__ == '__main__':
    # if tf.test.is_gpu_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    path = Path(__file__).parent.resolve()
    Result_path = path / 'result'
    train_model(path, Result_path)
