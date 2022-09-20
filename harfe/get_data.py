import numpy as np
from spgl1 import spg_bpdn
from helpers import *
from algorithms import *
from data_simulation import *
import csv

# np.random.seed(0)

'''Data input'''
def get_dataset(dataset,num_train,num_val):
    if dataset == 'housing':
        tmp = csv.reader(open("DataFolder/housing.csv"), delimiter = ",")
        data = np.array(list(tmp)).astype("float")
        
        col = list(range(1,3))+list(range(4,14))
        data_input = data[:,col]
        data_output = data[:,0]
        X = data_input
        Y = data_output.flatten()
        X_train, Y_train, X_val, Y_val = make_data_sets(X,Y,num_train,num_val)
        for i in range(X.shape[1]):
            meanX = np.mean(X_train[:,i])
            stdX = np.std(X_train[:,i])
            X_train[:,i] = (X_train[:,i] - meanX)/stdX
            X_val[:,i] = (X_val[:,i] - meanX)/stdX
        meanY = np.mean(Y_train)
        stdY = np.std(Y_train)
        Y_train = (Y_train-meanY)/stdY
        Y_val = (Y_val-meanY)/stdY
        
    elif dataset == 'skillcraft':
        tmp = csv.reader(open("DataFolder/skillcraft.csv"), delimiter = ",")
        data = np.array(list(tmp)).astype("float")
        col = [0] + list(range(2,15))+list(range(16,20))
        data_input = data[:,col]
        data_output = data[:,15]
        X = data_input
        Y = data_output.flatten()
        plt.plot(X,'.')
        plt.plot(Y,'.')
        X_train, Y_train, X_val, Y_val = make_data_sets(X,Y,num_train,num_val)
        for i in range(X.shape[1]):
            meanX = np.mean(X_train[:,i])
            stdX = np.std(X_train[:,i])
            X_train[:,i] = (X_train[:,i] - meanX)/stdX
            X_val[:,i] = (X_val[:,i] - meanX)/stdX
        meanY = np.mean(Y_train)
        stdY = np.std(Y_train)
        Y_train = (Y_train-meanY)/stdY
        Y_val = (Y_val-meanY)/stdY
        
    else:
        tmp = csv.reader(open("DataFolder/propulsion.csv"), delimiter = ",")
        data = np.array(list(tmp)).astype("float")
        data_input = data[:,1:-2]
        data_output = data[:,0]
        X = data_input
        Y = data_output.flatten()
        plt.plot(X,'.')
        plt.plot(Y,'.')
        X_train, Y_train, X_val, Y_val = make_data_sets(X,Y,num_train,num_val)
        for i in range(X.shape[1]):
            meanX = np.mean(X_train[:,i])
            stdX = np.std(X_train[:,i])
            X_train[:,i] = (X_train[:,i] - meanX)/stdX
            X_val[:,i] = (X_val[:,i] - meanX)/stdX
        meanY = np.mean(Y_train)
        stdY = np.std(Y_train)
        Y_train = (Y_train-meanY)/stdY
        Y_val = (Y_val-meanY)/stdY
        
        
        
        
    return X_train,Y_train, X_val, Y_val


def get_dataset_SALSA(dataset):
    if dataset == 'housing':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/housing_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/housing_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/housing_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/housing_Yval.csv"), delimiter = ",")
    
    elif dataset =='skillcraft':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/skillcraft_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/skillcraft_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/skillcraft_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/skillcraft_Yval.csv"), delimiter = ",")
        
    elif dataset=='propulsion':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/propulsion_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/propulsion_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/propulsion_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/propulsion_Yval.csv"), delimiter = ",")
        
    elif dataset=='forestfire':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/ffire_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/ffire_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/ffire_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/ffire_Yval.csv"), delimiter = ",")
        
    elif dataset == 'telemonitor':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/tele_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/tele_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/tele_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/tele_Yval.csv"), delimiter = ",")
    
    elif dataset == 'speech':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/speech_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/speech_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/speech_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/speech_Yval.csv"), delimiter = ",")
    elif dataset == 'airfoil':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/airfoil_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/airfoil_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/airfoil_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/airfoil_Yval.csv"), delimiter = ",")
    elif dataset == 'insulin':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/insulin_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/insulin_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/insulin_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/insulin_Yval.csv"), delimiter = ",")
    elif dataset == 'music':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/music_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/music_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/music_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/music_Yval.csv"), delimiter = ",")
    elif dataset == 'galaxy':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/galaxy_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/galaxy_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/galaxy_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/galaxy_Yval.csv"), delimiter = ",")
    elif dataset == 'ccpp':
        xtr = csv.reader(open("DataFolder/DataSetSALSA/ccpp_Xtrain.csv"), delimiter = ",")
        xte = csv.reader(open("DataFolder/DataSetSALSA/ccpp_Xval.csv"), delimiter = ",")
        ytr = csv.reader(open("DataFolder/DataSetSALSA/ccpp_Ytrain.csv"), delimiter = ",")
        yte = csv.reader(open("DataFolder/DataSetSALSA/ccpp_Yval.csv"), delimiter = ",")
        
    elif dataset == 'superconduct':
        tmp = csv.reader(open("DataFolder/superconduct/train.csv"), delimiter = ",")
        x = list(tmp)
        data = np.array(x[1:]).astype("float")
        print(data.shape)
        X = data[:,0:81]
        Y = data[:,81]
        xtr, ytr, xte, yte = make_data_sets(X,Y,4000,17263)
        for i in range(X.shape[1]):
            meanX = np.mean(xtr[:,i])
            stdX = np.std(xtr[:,i])
            xtr[:,i] = (xtr[:,i] - meanX)/stdX
            xte[:,i] = (xte[:,i] - meanX)/stdX
        meanY = np.mean(ytr)
        stdY = np.std(ytr)
        ytr = (ytr-meanY)/stdY
        yte = (yte-meanY)/stdY
        
        
        
    
    else:
        print('dataset does not exist!')
    
    X_train = np.array(list(xtr)).astype("float")
    X_val = np.array(list(xte)).astype("float")
    Y_train = (np.array(list(ytr)).astype("float")).flatten()
    Y_val = (np.array(list(yte)).astype("float")).flatten()
    
    print('shape of input training and validation data is:',X_train.shape, X_val.shape)
    print('shape of output training and validation data is:',Y_train.shape, Y_val.shape)
    
    return X_train, Y_train, X_val, Y_val



