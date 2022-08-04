# PmliPred for K-fold cross validation
import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, TimeDistributed, RNN, Bidirectional, normalization
from keras import optimizers, regularizers
from sklearn import ensemble
from sklearn import metrics
import joblib
import argparse
import copy

# np.random.seed(1337)  # seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="PmliPred for K-fold cross validation")
args = parser.parse_args()

##########the parameter which can be adjusted#######################################################################################################
K = 5  # K-fold cross validation
WeightStrategy = 1  # weight strategy selection 1: Complete weight 2: Average weight
ThresholdStrategy = 1  # threshold strategy selection 1: variable threshold  2: constant threshold
threshold = 0.5  # threshold, can be used just on constant threshold strategy

print('WeightStrategy =',WeightStrategy)
print('ThresholdStrategy =',ThresholdStrategy)
print('threshold =',threshold)
print('GRUepochsize =',GRUepochsize)
print('KFold =', K)
##########the parameter which can be adjusted#######################################################################################################

TotalSequenceLength = 0  # the total sequence length

# Load data
proj_path = Path(__file__).parent.resolve()
sequencepath = proj_path / 'data' / 'processed_data' / 'pair_seq_trainval.fasta' # raw sequence information
listsequence = open(sequencepath, 'r',encoding='utf-8-sig').readlines()
featurepath = proj_path / 'data' / 'processed_data' / 'pair_fea_trainval.fasta' # feature information
listfeature = open(featurepath,'r',encoding='utf-8-sig').readlines()
print('data loaded')

# Get the maximum length of the sequence
for linelength in listsequence:
    miRNAname, lncRNAname, sequence, label = linelength.split(',')
    if len(sequence) > TotalSequenceLength:
        TotalSequenceLength = len(sequence)
print('Get the maximum length of the sequence done')

# Initialize evaluation criteria
TPsum, FPsum, TNsum, FNsum, TPRsum, TNRsum, PPVsum, NPVsum, FNRsum, FPRsum, FDRsum, FORsum, ACCsum, AUCsum, AUPRCsum, F1sum, MCCsum, BMsum, MKsum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
print('Initialize evaluation criteria done')

# create machine learning data
def createdatamachinelearning(data, iteration, K):

    # separate the data
    totalpartdata = len(data)
    partdata = int(totalpartdata / K)
    partdatastart = iteration * partdata
    partdataend = partdatastart + partdata
    traindataP = data[0 : partdatastart]
    traindataL = data[partdataend : totalpartdata]
    traindata = traindataP + traindataL
    testdata = data[partdatastart : partdataend]
    # print(traindata)
    # print(testdata)

    # separate the label
    rowtraindata = len(traindata)
    columntraindata = len(traindata[0].split()) - 1
    rowtestdata = len(testdata)
    columntestdata = len(testdata[0].split()) - 1
    # print(rowtraindata, columntraindata, rowtestdata, columntestdata)

    # get the training data and label
    trainfeature = [([0] * columntraindata) for p in range(rowtraindata)]
    # print(trainfeature)
    trainlabel = [([0] * 1) for p in range(rowtraindata)]
    # print(trainlabel) #[[0], [0], [0], [0], [0], [0], [0], [0], [0]]
    for linetraindata in traindata:
        # print(linetraindata)
        setraindata = re.split(r'\s', linetraindata)
        # print(setraindata)
        indextraindata = traindata.index(linetraindata)
        # print(indextraindata)
        for itraindata in range(len(setraindata) - 1):
            if itraindata < len(setraindata) - 2:
                trainfeature[indextraindata][itraindata] = float(setraindata[itraindata])
            else:
                trainlabel[indextraindata][0] = float(setraindata[itraindata])
                # trainlabel[indextraindata][0] = int(setraindata[itraindata]) 

    # get the test data and label
    testfeature = [([0] * columntestdata) for p in range(rowtestdata)]
    testlabel = [([0] * 1) for p in range(rowtestdata)]
    for linetestdata in testdata:
        setestdata = re.split(r'\s', linetestdata)
        indextestdata = testdata.index(linetestdata)
        for itestdata in range(0, len(setestdata) - 1):
            if itestdata < len(setestdata) - 2:
                testfeature[indextestdata][itestdata] = float(setestdata[itestdata])
            else:
                testlabel[indextestdata][0] = float(setestdata[itestdata])
                # testlabel[indextestdata][0] = int(setestdata[itestdata])

    # print(trainfeature, trainlabel, testfeature, testlabel)
    # #[[50.95401001, 54.4828186, -71.13114929, -34.30548096, -63.76742554, 55.78327179, 32.50906754, -7.375694275, -11.49349213, 28.3136673, -29.96363449, 79.46546936, 36.16069412, -9.139762878, -27.24342728, -54.1536293, 60.5266037, -17.54184532, -22.49081039, -47.09563446, 0.478260875, 90.90219116, -12.58544922, -23.16308594, -55.15368652, 33.42631531, 27.06999969, 24.0330658, 6.967010498, 26.50352478, -14.85263062, -15.77120972, -8.173740387, -2.650596619, -6.959747314, -5.769004822, -9.326854706, 34.00171661, -17.33906555, -25.44155884, -46.13726044, 11.89400482, 19.4069252, -0.467720032, 2.843707085, 20.86544037, -2.042869568, 0.219565392, 8.123228073, 4.885141373, 20.071064, 1.836763382, -2.694317818, 15.10614395, -2.631183147, -2.388246536, -3.056397438, 13.07761955, -0.814058304, 8.902428627, 5.441926003, 2.919713974, -5.810276031, -10.32948399, -1.539908409, 0.871088982, -11.00448799, -7.397171021, 1.802968979, 16.95141983, -6.520864964, -8.300021172, -10.2561512, -2.331388474, -0.081996918, -2.188007355, 2.05494976, 3.502141953, 0.407482147, -2.83976841, -7.942347527, -7.533951759, -2.439036369, 2.262220383, 0.198925018, 6.368017197, -2.515489578, 3.457540512, -16.57480812, 10.86924553, 8.698421478, 17.86146927, -3.292644501, -0.658481836, -7.256850243, -2.727040291, -6.680624485, -0.846134186, -13.50971603, -2.396909714, -8.598426819, -4.337425232, -7.468235016, -18.10042381, -16.16716385, 0.413602948], [-26.12110901, -68.04684448, 75.1138916, 19.05419922, 72.59620667, -29.19696808, -8.992908478, -50.85395432, -9.517208099, 32.26623535, -73.44189453, -5.514770508, -50.79582596, -50.64173889, 111.096489, 36.75545883, -24.45363617, -17.54184532, 70.39456177, -1.641090393, 0.409090906, -29.15054321, -7.638183594, -0.761444092, 37.55012512, -33.68996048, -14.77793121, 13.89025879, 2.805862427, -11.20689392, 1.200897217, -3.843364716, 6.502590179, -7.722000122, -6.073143005, -3.044166565, 16.3729744, 23.84709167, 9.300571442, -7.549797058, 12.18976593, -20.27806664, -11.38794994, -4.150100708, 2.376761436, -10.3904686, -7.563486099, 4.806287766, -1.534894943, 1.202760696, -3.831905365, -0.468418121, 17.05341339, -1.904918671, 5.176883698, -8.825025558, 8.422230721, -2.556247711, -4.957475662, -3.510091782, -0.078690529, 7.96156311, -3.97795105, -5.736850739, 3.046813965, -1.889219284, -3.657457352, -0.050140381, 1.797058105, 16.01752853, 3.12543726, -7.844895363, -4.747356415, -13.36079979, -2.842305183, 1.482552528, 7.10270977, -7.533180237, 10.50300217, -4.683914185, -4.271787643, -6.617789268, 5.363121033, 3.627597809, -5.321691513, 4.978998184, 5.292577267, 2.0685215, 4.094995499, 2.588319778, 4.549095154, 20.14300919, -6.513989449, -1.11951828, 2.389451981, 1.86559248, 6.181111813, -0.391008377, -3.86932373, -6.07929039, 2.88020134, 4.841930389, -4.252800941, 7.162014008, 4.502639771, 0.511254013], [-62.91763306, 35.84927368, 92.42993164, -65.3613739, -63.76742554, -27.03246307, 40.7906456, -50.85395432, -54.97175217, -58.64285278, 69.41525269, 91.88783264, -3.176776886, 94.37991333, 72.1354599, -54.1536293, 72.94896698, -17.54184532, -65.96907043, -47.09563446, 0.428571433, 31.5319519, 5.14012146, -21.24780273, -15.42433167, 11.11431885, 27.92123413, -4.953083038, -1.956317902, 2.771575928, 5.533065796, 9.024787903, -11.89791107, 2.816410065, -12.9404335, -19.89149857, 9.062839508, 15.2084198, -16.86574745, -5.213630676, -10.3118515, -0.760831833, 2.917842865, -3.03978157, 12.24769306, 8.683923721, 9.5148983, 9.465778351, 0.352005005, -3.674943447, 1.270429611, -2.573534012, 0.090553284, 9.962020874, -2.473410606, -1.283838272, -8.097784996, -1.888772964, 10.90148163, -5.172361851, -0.964379311, 4.970758438, -0.871623039, 1.804336548, -0.277726173, -2.332063675, 3.440885544, 1.060178757, 6.899393082, -2.164758682, -6.520864964, 4.464889526, -9.625061035, -3.380760193, 2.702874184, 6.584887505, -2.986437798, -6.422861099, 2.300754547, -1.577586174, -7.153484344, 0.923398018, -13.1529026, -13.3352623, 5.768667221, 4.427045822, -2.357717037, -4.471453667, 11.52707672, 17.22785187, 11.53833008, -3.250716209, -10.17226791, -4.334952354, -7.256850243, -0.573259592, -4.684616566, 7.926760674, -4.421276093, -4.968971252, -3.659773827, 3.070552826, -5.472227097, -3.812822342, -4.033344269, 0.471057892], [50.95401001, 54.4828186, -27.65289307, -77.78373718, 23.18909454, -74.65151215, 75.98733521, -7.375694275, 31.98476791, 71.79193115, -29.96363449, -7.491054535, 79.63896179, 34.33850098, -70.72168732, -54.1536293, -69.90818024, 25.93641472, -22.49081039, -3.61737442, 0.434782594, 43.98794556, -22.14015198, -21.07286072, -0.774978638, 20.17036438, 0.533317566, 6.072059631, 15.55921555, 7.319786072, -21.37939835, -1.668434143, -6.120700836, -9.383907318, 6.333374023, -24.34551239, 6.618289948, 26.26047516, -7.123434067, -3.163803101, -16.51068878, 2.504549026, 3.684848785, 7.707263947, 6.524310112, 3.714086533, -6.942076683, -1.003170967, 4.859844208, -4.917402744, 1.283885956, -3.062443733, 12.83360672, 11.23139381, 4.519345284, -1.772747993, 1.644528389, 2.870248795, -3.05711937, -1.920438766, 9.53148365, -7.494198799, -3.356541634, -5.42614603, -5.009835243, -1.578514576, 3.705526352, -3.923112869, 0.171277046, -0.406476974, -6.520864964, 4.980712891, -4.125946999, 4.616727829, -2.531600475, -6.884801865, -4.480079651, 13.31295013, -8.168194771, 7.17345047, -5.897568703, -4.059893608, -10.40334511, -9.580972672, -0.205900192, 13.931633, -2.106534004, -2.461990356, -2.682706833, 10.26200676, 2.576478958, 7.245146751, 4.064428329, -2.087761402, -2.762468338, -2.318084717, 0.060948372, 1.198644638, 9.577692032, -7.705071449, -6.144692421, 1.590368271, -2.973853111, -3.799365997, -11.26382637, 0.458426952], [-117.0301971, 113.7713318, 120.5684509, -117.309433, -63.76742554, 16.25757599, -8.992908478, -50.85395432, -54.97175217, 32.26623535, 108.3762817, 39.93977356, 40.11326218, 85.72189331, 20.18740082, -54.1536293, -24.45363617, -17.54184532, -20.51452637, -47.09563446, 0.454545468, 38.61911011, 5.799591064, 0.725875854, -45.1446228, 16.74182892, 9.297691345, 12.16686249, 1.006919861, 11.67887115, -1.712356567, 4.003993988, -7.879516602, 12.70851898, 6.887748718, -4.767562866, -13.807724, -2.131347656, -8.169483185, -11.93147659, -24.14321899, 8.533355713, 6.865894318, 5.014595032, -3.421408653, 4.647941589, 4.538066864, 4.603258133, -4.396206856, 0.088454247, 6.801218033, 4.012413025, 1.330365181, 5.512769699, -1.532561064, -2.040034294, -2.338378906, 1.646007538, 6.865499496, 5.933180809, -2.661425591, 8.391231537, -8.029114723, -0.42028904, -1.561502457, 9.301057816, -1.834905624, 0.303982735, -3.72253418, 0.015901566, -2.115578651, 1.598377228, -7.330091476, 5.728775024, 2.474256516, -1.100182533, 5.709826469, -2.773771286, 6.248806, 3.01235199, 0.487621307, 1.635630608, 0.476226807, -4.753307343, -2.030712128, -4.313181877, -2.885296345, -4.286802292, -2.260328293, 0.91686058, -6.768667221, 2.394381523, 1.460853577, 1.538762569, -4.319993019, -3.096847057, -2.275338173, 1.710119247, 1.522783279, -5.725167274, -9.348836899, -3.260396957, -1.594520092, -7.092605591, -12.13168526, 0.441997051], [-26.12110901, -22.59230042, 75.1138916, -26.40034485, -63.76742554, -29.19696808, 127.3707275, -50.85395432, -9.517208099, -13.18830872, 17.4671936, -5.514770508, 40.11326218, 40.26734924, -25.26714325, -8.699085236, 21.0009079, -17.54184532, -65.96907043, 43.81345367, 0.409090906, -34.12205505, -3.193496704, -26.90879822, 64.22428894, -25.9753418, -2.424850464, -0.352115631, -6.658779144, -5.054580688, -9.82604599, -5.544940948, 17.52347183, 5.872444153, -10.89932251, -24.81949615, 3.232685089, -8.585800171, 18.57749176, 4.022132874, 50.44798279, -13.52626038, -4.085100174, -5.712402344, -4.28421402, -4.419813156, 2.173646927, -1.303642273, 1.22032547, 1.523698807, -1.627728462, -7.129393578, 6.946897507, -10.44913292, -4.469418526, -0.189980507, 8.51306057, -8.66617775, -0.319015503, 7.55923748, -3.524230957, 7.271242142, -14.95644855, -4.634762287, 2.586601257, 2.979347229, -5.219758987, 2.154036522, -5.414960861, 0.812337875, 8.54504776, 1.341194153, 6.873016357, 3.588352203, 3.909501076, 3.686729431, -5.207983017, -3.445764542, -0.935709, -0.596498489, -5.834089279, -1.749222755, -3.962244034, -15.89511395, -3.11751461, -2.46312809, -0.587246656, -7.256843567, 13.60202026, -7.288088799, -1.790943146, -5.810567856, 6.438577652, -4.334952354, 4.042584419, 1.084441662, 17.80148506, 3.145363808, -0.012014389, -3.875113487, 4.854270935, 3.600566864, 15.13063431, 10.23817062, 21.54262352, 0.542372882], [-45.0605011, 41.80163574, 169.8108826, -166.5518494, -22.1007576, 8.68182373, -12.78078461, -9.187286377, 70.02824402, -16.97618484, 51.55810547, -50.96931458, -9.12915802, 70.57038879, 137.6116333, -54.1536293, -69.90818024, -17.54184532, -24.3024025, -47.09563446, 0.333333343, -66.43664551, 172.6259766, -108.5284424, 2.339050293, -33.00220108, 35.64618683, -47.70851898, -20.77790642, 13.18219757, 112.8358231, -4.983757019, 51.88312149, -33.95100403, -0.03024292, -53.67402649, -22.40621948, -12.28686142, 22.85006714, -1.947769165, -6.038860321, -15.74001312, -0.924562454, -13.35552979, -2.731488228, -3.142515182, 25.27861595, -5.34554863, 18.95100594, -11.6589756, -13.03733444, -14.71743584, -8.229187012, -10.66947556, -2.641264915, -2.458818436, -4.945040703, -1.739162445, 31.9818058, -11.32824039, -5.627812386, 19.24203491, 58.11461258, 4.285663605, 31.28620338, -4.663780212, 7.246803284, -11.08372307, 1.732397079, 13.11364365, 20.90143967, 17.2437191, 0.67244339, -11.53264618, -1.960558891, -11.37918377, -8.974461555, -7.597841263, 9.427270889, -2.9204216, 1.148012161, -11.22050381, -8.004150391, -21.65500259, -12.69896698, -12.04458046, -2.525571346, -9.470596313, 1.696645737, -3.907209396, 6.688798904, -11.5704565, -3.3632164, 4.805815697, 18.33729935, -0.908968449, 0.631989956, -6.38100338, 13.84201622, -6.143951416, -3.174455643, -2.600006104, 7.156993866, -7.151664734, -3.380170822, 0.656307101], [64.78799438, -22.59230042, 29.65936279, -71.85488892, 27.1416626, -74.65151215, 81.91618347, -5.399410248, 35.93733597, -13.18830872, -27.98735046, -5.514770508, 85.56781006, 85.72189331, -70.72168732, -54.1536293, -69.90818024, -17.54184532, 24.9400177, -1.641090393, 0.363636374, -22.44419861, 9.971572876, -11.79006958, 24.26264954, -3.939682007, -1.104820251, -10.12031555, -6.685176849, -5.221782684, -0.902656555, -2.130470276, 18.51789474, -4.15096283, -6.719230652, -0.574989319, -0.049777985, -10.3106308, 19.20230484, 1.250076294, 12.80080414, 4.443714142, -3.900295258, -4.550777435, 0.318282127, -4.56061554, 2.939264297, 1.090007782, -0.47810936, -2.313180923, -10.46311188, 2.145999908, 0.575565338, -5.521030426, -2.911786079, -4.352466583, 6.163413048, -4.714895248, 5.445104599, -5.368231773, -0.479367256, -5.40102005, 0.294269562, -4.379556656, 8.676328659, -2.089557648, 6.845293045, -3.566082001, -3.276515961, -0.560489655, 10.61309242, -0.357240677, 8.870658875, -5.572637558, 3.187886238, 4.197140694, -5.859196663, -1.703327179, -4.517383575, -3.526958466, 3.115697861, 2.527667046, -0.010961533, -0.389190674, -2.607103348, -1.952716827, 1.876804829, -3.305561066, 1.836177826, 0.429672241, -5.698223114, -4.323336601, -0.583967209, 6.56847477, 0.531311989, 4.780519009, 7.338067532, -2.249148369, -3.012876511, 1.308195114, 5.294281006, -2.18995285, 9.665721893, 9.37575531, -3.986707687, 0.515576303], [64.78799438, 22.86224365, -61.2497406, -26.40034485, 27.1416626, 16.25757599, 36.46163559, -5.399410248, -54.97175217, -13.18830872, -27.98735046, 85.39431763, 85.56781006, -50.64173889, -25.26714325, -54.1536293, 21.0009079, 73.36724091, -65.96907043, -47.09563446, 0.454545468, 19.03146362, -8.996154785, -10.61936951, 0.583999634, 10.99671936, 2.508155823, 1.183265686, 4.937538147, -7.690513611, -4.575164795, 3.688327789, -0.12739563, 0.255714417, -0.202644348, -5.843162537, -4.534172058, 15.67140198, -6.222490311, -9.433433533, 0.629112244, 5.845912933, -0.333614349, 1.565658569, 4.169371605, -0.782281876, 0.629272461, 0.021821976, 2.734714508, -0.158621788, 0.999210358, 0.085691452, 0.322573662, 5.195547104, -0.223134041, 0.13688755, -0.285384178, -0.263572693, -2.634080887, -2.009897232, -2.678572655, -3.8152771, 0.499307632, -0.313508034, -0.853009224, 2.118695259, 2.412158966, -2.119236946, 1.320318222, 4.386894226, -2.982294559, -2.629201889, 1.145330429, 0.616553307, -0.603675842, -0.055758476, 0.402750015, -1.463563919, -1.835346222, 2.093416214, 1.090114594, -0.840589523, -1.04442215, -0.541316986, -3.321432114, 1.048454285, -0.461297512, -3.985164642, -1.074048996, 4.880994797, 6.218822479, 1.758374214, 2.94798851, -1.504096031, -3.718279839, 1.981079578, -2.965125561, -0.83702755, -2.492012024, -3.19438839, -2.819627762, 5.126951218, -2.51423645, -2.845541, 0.925952911, 0.472929925]]
    # #[[1.0], [1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [1.0], [1.0]]
    # #[[50.95401001, 54.4828186, -71.13114929, -34.30548096, -63.76742554, 55.78327179, 32.50906754, -7.375694275, -11.49349213, 28.3136673, -29.96363449, 79.46546936, 36.16069412, -9.139762878, -27.24342728, -54.1536293, 60.5266037, -17.54184532, -22.49081039, -47.09563446, 0.478260875, -17.21055603, 15.41290283, 13.36743164, -11.56980896, -15.00233459, 1.906600952, -2.775215149, -1.486148834, 1.28565979, -10.02692413, 20.02944183, 4.416126251, -16.42510796, 27.63175201, 6.957023621, -4.501148224, 13.30999756, -3.594511032, -11.37020874, -9.677562714, -9.671897888, -5.478696823, 3.728902817, -4.0707798, -0.289241791, -1.165962219, 3.291461945, 0.165712357, -4.992309093, 2.565618515, 2.713670731, -2.996604919, 0.536428452, 2.197247982, -4.885523796, 0.729005814, -0.367370605, -0.293907166, -1.304542542, 3.355870247, -4.68104744, -5.72857666, -1.0566082, 1.531988144, -3.134902477, 16.0734005, 6.347383499, 0.787173271, 5.344667435, -0.594939232, 1.027320862, -1.312795639, -5.212651253, -4.087988377, -4.712517262, -2.307795048, 6.271749496, -0.910600662, 16.52842331, 5.829439163, -3.419027328, 2.855081558, 6.540542603, 1.075832367, 6.915403366, -2.131502628, -4.883962631, -4.338970184, 0.332752228, 11.9064827, -0.411949158, 1.617486954, 0.109492302, -2.071665287, 1.360650063, -2.976920843, -4.852125645, 5.474489212, -8.570656776, -3.331541061, 0.599939346, -3.023790359, -2.517633438, -4.672066689, 0.485185176]]
    # #[[0.0]]
    return np.array(trainfeature), np.array(trainlabel), np.array(testfeature), np.array(testlabel)

# RF
def RF(trainfeature, trainlabel, testfeature, iteration):

    RFStruct = ensemble.RandomForestClassifier()
    RFStruct.fit(trainfeature, np.array(trainlabel).ravel()) # training
    group = RFStruct.predict(testfeature) # test
    score = RFStruct.predict_proba(testfeature) # get the confidence probability
    print('RF_score',score)

    # save model
    model_savepath = proj_path / 'pretrained' / 'RF' / f'RF'
    if not model_savepath.exists():
        model_savepath.mkdir(parents=True)
    joblib.dump(RFStruct, model_savepath / f'RF_{iteration}thFold.pkl')

    return group, score

# calculate the results of model
def comparison(testlabel, resultslabel):

    # ROC AUC
    fprs, tprs, thresholds = metrics.roc_curve(testlabel[:,1], resultslabel[:,1], pos_label=1)
    auc = metrics.auc(fprs, tprs)

    # PRC AUC
    pres, recs, thresholds_prc = metrics.precision_recall_curve(testlabel[:,1], resultslabel[:,1], pos_label=1)
    auprc = metrics.auc(recs, pres)

    # initialization
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # formatting
    for row1 in range(resultslabel.shape[0]):
        for column1 in range(resultslabel.shape[1]):
            if resultslabel[row1][column1] < 0.5:
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1

    # TP, FP, TN, FN
    for row2 in range(testlabel.shape[0]):
        # TP
        if testlabel[row2][0] == 0 and testlabel[row2][1] == 1 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TP = TP + 1
        # FP
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] != resultslabel[row2][0] and testlabel[row2][1] != resultslabel[row2][1]:
            FP = FP + 1
        # TN
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TN = TN + 1
        # FN
        if testlabel[row2][0] == 0 and testlabel[row2][1] == 1 and testlabel[row2][0] != resultslabel[row2][0] and testlabel[row2][1] != resultslabel[row2][1]:
            FN = FN + 1

    # TPR：sensitivity, recall, hit rate or true positive rate
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 999999

    # TNR：specificity, selectivity or true negative rate
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 999999

    # PPV：precision or positive predictive value
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 999999

    # NPV：negative predictive value
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 999999

    # FNR：miss rate or false negative rate
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 999999

    # FPR：fall-out or false positive rate
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 999999

    # FDR：false discovery rate
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 999999

    # FOR：false omission rate
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 999999

    # ACC：accuracy
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999

    # F1 score：is the harmonic mean of precision and sensitivity
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999

    # MCC：Matthews correlation coefficient
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 999999

    # BM：Informedness or Bookmaker Informedness
    if TPR != 999999 and TNR != 999999:
        BM = TPR + TNR - 1
    else:
        BM = 999999

    # MK：Markedness
    if PPV != 999999 and NPV != 999999:
        MK = PPV + NPV - 1
    else:
        MK = 999999

    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, fprs, tprs, thresholds, auc, pres, recs, np.append(thresholds_prc, [1], axis=0), auprc

# K-fold cross validation
for iteration in range(K):
    print('Fold_%s'%iteration)

    # create machine learning data
    trainfeature, trainlabelml, testfeature, testlabelml = createdatamachinelearning(listfeature, iteration, K)
    print('machine learning data created')

    # RF
    RFgroup, RFscore = RF(trainfeature, trainlabelml, testfeature, iteration)
    print('RF done')
    
    # RF_Result
    RFscore_ml = copy.deepcopy(RFscore)
    ml_TP, ml_FP, ml_TN, ml_FN, ml_TPR, ml_TNR, ml_PPV, ml_NPV, ml_FNR, ml_FPR, ml_FDR, ml_FOR, ml_ACC, ml_F1, ml_MCC, ml_BM, ml_MK, ml_fprs, ml_tprs, ml_thresholds, ml_auc, ml_pres, ml_recs, ml_thresholds_prc, ml_auprc = comparison(np_utils.to_categorical(testlabelml, num_classes=2), RFscore_ml)
    
            ######################################################################################

    # obtain the results
    resultslabel = copy.deepcopy(RFscore_ml)
    TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, fprs, tprs, thresholds, auc, pres, recs, thresholds_prc, auprc = comparisondeeplearning(np_utils.to_categorical(testlabelml, num_classes=2), resultslabel)

    # print the results of each fold
    print('-----------------------------------')
    print('The', iteration + 1, 'fold')
    print('-------------RF_Result--------------')
    print('TP:', ml_TP, 'FP:', ml_FP, 'TN:', ml_TN, 'FN:', ml_FN)
    print('TPR:', ml_TPR, 'TNR:', ml_TNR, 'PPV:', ml_PPV, 'NPV:', ml_NPV, 'FNR:', ml_FNR, 'FPR:', ml_FPR, 'FDR:', ml_FDR, 'FOR:', ml_FOR)
    print('AUC:', ml_auc,'AUPRC:', ml_auprc, 'ACC:', ml_ACC, 'F1:', ml_F1, 'MCC:', ml_MCC, 'BM:', ml_BM, 'MK:', ml_MK)
    print('-------------Total_Result--------------')
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
    print('AUC:', auc, 'AUPRC:', auprc, 'ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK)

    # add the results
    TPsum += TP
    FPsum += FP
    TNsum += TN
    FNsum += FN
    TPRsum += TPR
    TNRsum += TNR
    PPVsum += PPV
    NPVsum += NPV
    FNRsum += FNR
    FPRsum += FPR
    FDRsum += FDR
    FORsum += FOR
    ACCsum += ACC
    AUCsum += auc
    AUPRCsum += auprc
    F1sum += F1
    MCCsum += MCC
    BMsum += BM
    MKsum += MK

    # save kth result
    result_savepath = proj_path / 'trainval_result' / f'RF' / f'{iteration}th_fold' 
    if not result_savepath.exists():
        result_savepath.mkdir(parents=True)
    pd.DataFrame({'fprs':fprs, 'tprs':tprs, 'thresholds':thresholds}).to_csv(result_savepath / f'ROC_{iteration}thFold.csv') # roc
    pd.DataFrame({'pres':pres, 'recs':recs, 'thresholds_prc':thresholds_prc}).to_csv(result_savepath / f'PRC_{iteration}thFold.csv') # prc
    pd.DataFrame(testlabelml).to_csv(result_savepath / f'testlabeldl_{iteration}thFold.csv') # testlabeldl
    pd.DataFrame(resultslabel).to_csv(result_savepath / f'resultslabel_{iteration}thFold.csv') # result
    pd.DataFrame({'TP:': TP, 'FP:': FP, 'TN:': TN, 'FN:': FN, 'TPR:': TPR, 'TNR:': TNR, 'PPV:': PPV, 'NPV:': NPV, 'FNR:': FNR, 'FPR:': FPR, 'FDR:': FDR, 'FOR:': FOR, 'ACC:': ACC, 'AUC:': auc, 'AUPRC:': auprc, 'F1:': F1, 'MCC:': MCC, 'BM:': BM, 'MK:': MK},index=[0]).to_csv(result_savepath / f'trainval_result_{iteration}thFold.csv')


# obtain the average results
TPaverage, FPaverage, TNaverage, FNaverage, TPRaverage, TNRaverage, PPVaverage, NPVaverage = TPsum / K, FPsum / K, TNsum / K, FNsum / K, TPRsum / K, TNRsum / K, PPVsum / K, NPVsum / K
FNRaverage, FPRaverage, FDRaverage, FORaverage, ACCaverage, AUCaverage, AUPRCaverage, F1average, MCCaverage, BMaverage, MKaverage = FNRsum / K, FPRsum / K, FDRsum / K, FORsum / K, ACCsum / K, AUCsum / K, AUPRCsum / K, F1sum / K, MCCsum / K, BMsum / K, MKsum / K

# print the results
print('\ntest average TP: ', TPaverage)
print('\ntest average FP: ', FPaverage)
print('\ntest average TN: ', TNaverage)
print('\ntest average FN: ', FNaverage)
print('\ntest average TPR: ', TPRaverage)
print('\ntest average TNR: ', TNRaverage)
print('\ntest average PPV: ', PPVaverage)
print('\ntest average NPV: ', NPVaverage)
print('\ntest average FNR: ', FNRaverage)
print('\ntest average FPR: ', FPRaverage)
print('\ntest average FDR: ', FDRaverage)
print('\ntest average FOR: ', FORaverage)
print('\ntest average ACC: ', ACCaverage)
print('\ntest average AUC: ', AUCaverage)
print('\ntest average AUPRC: ', AUPRCaverage)
print('\ntest average F1: ', F1average)
print('\ntest average MCC: ', MCCaverage)
print('\ntest average BM: ', BMaverage)
print('\ntest average MK: ', MKaverage)
