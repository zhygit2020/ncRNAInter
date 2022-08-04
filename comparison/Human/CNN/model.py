import math
import time
from pathlib import Path
from collections import Counter
import random
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class CNN1DModel(tf.keras.models.Model):
    def __init__(self, classes, shape1, shape2, dropout):
        super(CNN1DModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="valid",
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.normal1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(dropout)

        self.conv2 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding='valid',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.normal2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(dropout)

        self.conv3 = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='valid',
            activation=tf.nn.relu
        )
        self.pool3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.normal3 = tf.keras.layers.BatchNormalization()
        self.drop3 = tf.keras.layers.Dropout(dropout)

        # self.flatten = tf.keras.layers.Reshape(target_shape=(6 * 1 * 64,)) # 3mers
        # self.flatten = tf.keras.layers.Reshape(target_shape=(30 * 1 * 64,))# 4mers
        self.flatten = tf.keras.layers.Reshape(target_shape=(shape1 * 1 * 64,))  # Transcript
        # self.dense1 = tf.keras.layers.Dense(units=shape1, activation=tf.nn.relu)
        # self.drop4 = tf.keras.layers.Dropout(dropout)
        self.dense1 = tf.keras.layers.Dense(units=shape2, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs, training):
        inputs = tf.expand_dims(inputs, axis=2)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.normal1(x, training=training)
        x = self.drop1(x)  # (15, 58, 32)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.normal2(x, training=training)
        x = self.drop2(x)  # (15, 28, 32)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.normal3(x, training=training)
        x = self.drop3(x)  # (15, 13, 64)
        # print('The shape before reshape')
        # print(x.shape)
        x = self.flatten(x)
        x2 = self.dense1(x)
        output = self.dense2(x2)

        return output, x2

class Dataset_make(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        c = Counter(list(batch_y))
        # print(dict(c))

        return batch_x, batch_y

def calc_metrics(y_label, y_proba,y_predict):
    # print('y_label')
    # print(y_label)
    # print('y_predict')
    # print(y_predict)
    con_matrix = metrics.confusion_matrix(y_label, y_predict)
    TN = con_matrix[0][0]
    FP = con_matrix[0][1]
    FN = con_matrix[1][0]
    TP = con_matrix[1][1]
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    if len(y_proba.shape) == 1:
        AUC = metrics.roc_auc_score(y_label,y_proba)
    elif len(y_proba.shape)==2:
        fpr, tpr, thresholds = metrics.roc_curve(y_label, y_proba[:, 1])
        AUC = metrics.auc(fpr, tpr)
    if len(y_proba.shape) == 1:
        pre, rec, thresholds_prc = metrics.precision_recall_curve(y_label, y_proba)
        AUPRC = metrics.auc(rec, pre)
    elif len(y_proba.shape)==2:
        pre, rec, thresholds_prc = metrics.precision_recall_curve(y_label, y_proba[:, 1])
        AUPRC = metrics.auc(rec, pre)
    return Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, TN, FP, FN, TP


def train_model(X_train, y_train, X_test, y_test, modelnm='CNN'):
    # parameters
    classes = len(np.unique(y_train))
    shape1 = 128
    shape2 = 32
    Epochs = 128

    # dropouts = [0.2,0.5,0.7]
    # learnrantes = [0.001,0.01,0.1]
    # batch_sizes = [32,64,128,256]

    # 超参数选择
    dropouts = [0.2]
    learnrantes = [0.001, 0.0005]
    # learnrantes = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64]
    parameters_all = [[x, y, z] for x in dropouts for y in learnrantes for z in batch_sizes]
    parameters = random.sample(parameters_all, 4)
    # print('parameters', parameters)

    # traing, validation 8:2 分层抽样
    train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                        stratify=y_train)
    # print(train_x, test_x, train_y, test_y)
    res_dict_v = {}
    res_dict = {}
    for index, parameter in enumerate(parameters):
        print('Current parameter is {}/{} dropout, learning rate, batchsize: {}'.format(index, len(parameters),
                                                                                        parameter))
        dropout = parameter[0]
        learnrante = parameter[1]
        batch_size = parameter[2]

        time0 = time.time()

        # make the model
        if modelnm == 'DNN':
            # DNN
            pass
        elif modelnm == 'CNN':
            # CNN
            shape1 = int((int((int((train_x.shape[1] - 2) / 2) - 2) / 2) - 2) / 2)
            print('shape1:{}'.format(shape1))
            model = CNN1DModel(classes, shape1, shape2, dropout)
            # model = FC_CNN1DModel(classes, shape1, shape2, dropout)

        # define loss and optimizer
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=learnrante)

        # train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        # test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        # training model
        def train_step(data, labels):
            with tf.GradientTape() as tape:
                N = len(labels)
                logits, x2 = model(data, training=True)

                loss = loss_obj(labels, logits)

                loss = tf.reduce_mean(loss)
                loss = loss / N

                # print(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            group_t = np.argmax(logits.numpy(), axis=1)
            # train_acc(labels, logits)
            return loss, logits, group_t

        # testing model
        def test_step(data, labels):
            N = len(labels)
            logits, x2 = model(data, training=False)
            # logits,x2 = model.predict(data, batch_size=32, verbose=1)
            test_loss = loss_obj(labels, logits)
            test_loss = tf.reduce_mean(test_loss)
            test_loss = test_loss / N
            logits_ = np.argmax(logits.numpy(), axis=1)
            # test_a = test_acc(labels, logits)
            return test_loss, logits_, logits

        # data preparing
        train_dataset = Dataset_make(train_x, train_y, batch_size)
        test_dataset = Dataset_make(test_x, test_y, batch_size)

        mcc = -1
        for epoch in range(Epochs):
            # train_acc.reset_states()
            # test_acc.reset_states()

            group_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            score_all = np.zeros(shape=(1, 2))
            # score_all = np.array([])

            group_train = np.array([], dtype=int)
            labels_train = np.array([], dtype=int)
            score_trains = np.zeros(shape=(1, 2))
            # score_trains = np.array([])

            for images, labels in train_dataset:
                t_loss, score_train, group_t = train_step(images, labels)
                group_train = np.append(group_train, group_t)
                labels_train = np.append(labels_train, labels)
                score_trains = np.concatenate((score_trains, score_train), axis=0)
                # print('0')

            for images, labels in test_dataset:
                loss, group, score = test_step(images, labels)
                # print('images')
                # print(images.shape)
                group_all = np.append(group_all, group)
                labels_all = np.append(labels_all, labels)

                score_all = np.concatenate((score_all, score), axis=0)
            score_all = np.delete(score_all, 0, 0)
            score_trains = np.delete(score_trains, 0, 0)

            Acc_train, Sn_train, Sp_train, Pre_train, MCC_train, AUC_train, AUPRC_train, TN_train, FP_train, FN_train, TP_train = calc_metrics(labels_train, score_trains,group_train)
            # print(Acc_train)
            MCC_v = metrics.matthews_corrcoef(labels_all, group_all)
            # accuracy = metrics.accuracy_score(labels_all, group_all)
            # classesnms = [str(i) for i in range(0, classes)]
            # report = metrics.classification_report(labels_all, group_all, target_names=classesnms)
            # print(X_test.shape)
            # Model Evaluation on testing dataset
            score, feature_out = model(X_test, training=False)
            # score = model.predict(X_test, batch_size=32, verbose=1)
            group = np.argmax(score.numpy(), axis=1)
            # Acc_e, Sn_e, Sp_e, Pre_e, MCC_e, AUC_e = calc_metrics(y_test,score,group)

            if mcc < MCC_v or mcc == MCC_v:
                mcc = MCC_v
                _model = model
                # Model Evaluation for epoches
                # print('score_all')
                # print(score_all)

                Acc_v, Sn_v, Sp_v, Pre_v, MCC_v, AUC_v, AUPRC_v, TN_v, FP_v, FN_v, TP_v = calc_metrics(labels_all, score_all, group_all)

                last_improve = epoch
                print('last_improve: %s' % last_improve)
                Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, TN, FP, FN, TP = calc_metrics(y_test, score, group)

            if epoch - last_improve >= 10:
                break

            tmp = 'Epoch {:.3f}, traing Acc {:.3f}, validation Acc {:.3f}, Test Acc {:.3f}, train_loss{:.3f}, validation loss{:.3f}, test auc{:.3f}, test auprc{:.3f}, test mcc{:.3f}'
            print(tmp.format(epoch + 1, Acc_train, Acc_v, Acc, t_loss, loss, AUC, AUPRC, MCC))
            # print(report)
        
        print(model.summary())
        _model.save_weights(Path(__file__).parent.resolve()/'result'/f'lr{learnrante}_bs{batch_size}_model')

        res_dict_v[index] = [parameter, Acc_v, Sn_v, Sp_v, Pre_v, MCC_v, AUC_v, AUPRC_v, TN_v, FP_v, FN_v, TP_v]
        res_dict[index] = [parameter, Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, TN, FP, FN, TP]
        print('The parameter is dropout, learning rate, batchsize: {}, and test MCC: {}'.format(parameter, MCC))
        print('Each group parameters selection use time: {} min'.format((time.time() - time0) / 60))
    df_v = pd.DataFrame(res_dict_v)
    df_list_v = df_v.iloc[5, :].tolist()
    maxindex_v = df_list_v.index(max(df_list_v))
    dfres_v = df_v.iloc[:, maxindex_v].tolist()

    df = pd.DataFrame(res_dict)
    df_list = df.iloc[5, :].tolist()
    maxindex = df_list.index(max(df_list))
    dfres = df.iloc[:, maxindex].tolist()
    print('The best parameter is dropout, learning rate, batchsize: {}'.format(dfres[0]))
    Acc_v, Sn_v, Sp_v, Pre_v, MCC_v, AUC_v, AUPRC_v, TN_v, FP_v, FN_v, TP_v = dfres_v[1], dfres_v[2], dfres_v[3], dfres_v[4], dfres_v[5], dfres_v[6], dfres_v[7], dfres_v[8], dfres_v[9], dfres_v[10], dfres_v[11]
    Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, TN, FP, FN, TP = dfres[1], dfres[2], dfres[3], dfres[4], dfres[5], dfres[6], dfres[7], dfres[8], dfres[9], dfres[10], dfres[11]
    return Acc, Sn, Sp, Pre, MCC, AUC, AUPRC, dfres[0], TN, FP, FN, TP, Acc_v, Sn_v, Sp_v, Pre_v, MCC_v, AUC_v, AUPRC_v, dfres_v[0], TN_v, FP_v, FN_v, TP_v
