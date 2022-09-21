#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 10:12:21 2021

@author: eshasaha
"""

import numpy as np

def generate_omega_bias(rows,columns,weight,par1,par2,distribution,bool_bias,sparsity):
    mask = np.zeros(shape=(rows,columns)) #(N,d)
    for i in range(rows):
        idx = np.random.choice(columns,sparsity,replace=False)
        mask[i,idx] = 1.
    if distribution=='uniform':
        Omega = weight*np.random.uniform(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.uniform(par1,par2,rows)
        else: 
            bias = np.zeros(rows)
    elif distribution=='norm-uni':
        Omega = weight*np.random.normal(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.uniform(0,2*np.pi,rows)
        else:
            bias = np.zeros(rows)
    else:
        Omega = weight*np.random.normal(par1,par2,(rows,columns))
        Omega *= mask
        if bool_bias == True:
            bias = np.random.normal(par1,par2,rows)
        else:
            bias = np.zeros(rows)
    return Omega,bias
    

'''Define dictionary'''
def feature_matrix(X,Omega,bias,activation,dictType):
    ''' Input: X of size(#measurements,dimension)
    - Omega: matrix of size (rows,columns) i.e., (#features,dimension)
    - bias: vector of size (#features,)
    - activation: 'sin' for sin; 'tanh' for tanh; else for 'relu'.
    -dictType: Type of dictionary - 'SRF' for SRF dictionary; 'poly' for polynomial dictiionary upto order 2;
    'SRFpoly' for SRF+polynomial dictionary
    -idty: add constant times identity; 'yes' or 'no'.'''
    
    if dictType == 'SRF':
        if activation=='sin':
            A = np.sin(np.matmul(X,Omega.T) + bias)
        elif activation=='tanh':
            A = np.tanh(np.matmul(X,Omega.T) + bias)
        else:
            A = np.maximum(np.matmul(X,Omega.T) + bias,0)
                            
    elif dictType == 'poly':
        A = dictionarypoly(X,'legendre')
        Adict = A
    else:
        polydict = dictionarypoly(X,'legendre')
        if activation=='sin':
            Asrf = np.sin(np.matmul(X,Omega.T) + bias)
        elif activation=='tanh':
            Asrf = np.tanh(np.matmul(X,Omega.T) + bias)
        else:
            Asrf = np.maximum(np.matmul(X,Omega.T) + bias,0)
        A = np.hstack((polydict,Asrf))
    return A


#functions for function approximation
    
def make_data_sets(X,Y,num_data_train,num_data_val):
    '''
    IN:
        - X = input data; 2D array
        - Y = output data; 2D array
        - num_data_train = number of data points for training set; int
        - num_data_val = number of data points for validation set; int

    OUT:
        - X_train = input training data; 2D array
        - Y_train = output training data; 1D array
        - X_val = input training data; 2D array
        - Y_val = output training data; 1D array
    '''
    num_data_tot = X.shape[0]
    ind = list(range(num_data_tot)) # define indices to choose data points
#     np.random.shuffle(ind) # shuffle indices to choose data points at random

    # choose training and validation indices from shuffled list ind
    ind_train = ind[:num_data_train]; del ind[:num_data_train] # delete used indices
    ind_val = ind[:num_data_val]; del ind[:num_data_val] # delete used indices
#     print(ind_train,ind_val)

    # choose training and validation data points defined by their respective indices
    X_train = X[ind_train,:]; Y_train = Y[ind_train] # note Y is a 1D array
    X_val = X[ind_val,:] ; Y_val = Y[ind_val] # note Y is a 1D array
    return X_train, Y_train, X_val, Y_val


