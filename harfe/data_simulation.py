#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 10:12:21 2021

@author: eshasaha
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib

#functions for random tests
def generate_A(par1,par2,distribution,rows,columns):
    '''Input: 
    -par1,par2: parameters of the distribution
    -distribution: 'normal' or 'uniform'
    -rows: number of rows desired (or #measurements)
    -columns: number of columns (or #features)
    Output: Matrix A of size (rows,columns) with ||A||_2<1'''
        
    if distribution=='normal':
        A = np.random.normal(par1,par2,(rows,columns))
    else:
        A = np.random.uniform(par1,par2,(rows,columns))
    norm = np.linalg.norm(A,2)
    A_normalized = A/(norm+0.1)
    return A_normalized

def generate_true_data(par1,par2,length,s,method):
    '''Input:
        -length: length of vector same as columns of A
        -s : sparsity level << length
        -method: 'ones' for x[j] = 1 for all j={0,1,...,s-1}; 'uniform' for s-sparse x[j]=uniform(-1,1); 
        'normal' for s-sparse x[j]=normal(0,1); else x[j] = (s+1-i)/s for all j={0,1,...,s-1}.
        Output: vector of length (length,)'''
    x = np.zeros(length)
    if method=='ones':
        for i in range(s):
            x[i] = 1
    elif method=='uniform':
        for i in range(s):
            x[i] = np.random.uniform(par1,par2)
        np.random.shuffle(x)
    elif method=='normal':
        for i in range(s):
            x[i] = np.random.normal(par1,par2)
        np.random.shuffle(x)
    else:
        for i in range(s):
            x[i] = (s+1-i)/s
    return x

def create_data_random(par1,par2,par1_x,par2_x,rows,columns,s,distribution,method):
    '''Input: 
    -par1,par2: parameters of the distribution
    -distribution: 'normal' or 'uniform'
    -s: sparsity wanted in values to be recovered
    -rows: number of rows desired (or #measurements)
    -columns: number of columns (or #features)
    
    Output: Matrix A of size (rows,columns) with ||A||_2<1
    - x: vector to be recovered with s-sparsity of size (#columns,) or (#features,)
    - y: given input data (generated using np.matmul(A,x)) of size (#rows,) or (#measurements,)'''
    
    A = generate_A(par1,par2,distribution,rows,columns)
    x = generate_true_data(par1_x,par2_x,columns,s,method)
    y = np.matmul(A,x)
    return A,x,y


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

# def make_data_sets_ODE(X,Y,num_data_train,num_data_val):
#     '''
#     IN:
#         - X = input data; 2D array
#         - Y = output data; 2D array
#         - num_data_train = number of data points for training set; int
#         - num_data_val = number of data points for validation set; int

#     OUT:
#         - X_train = input training data; 2D array
#         - Y_train = output training data; 1D array
#         - X_val = input training data; 2D array
#         - Y_val = output training data; 1D array
#     '''
#     num_data_tot = X.shape[0]
#     ind = list(range(num_data_tot)) # define indices to choose data points
#     np.random.shuffle(ind) # shuffle indices to choose data points at random


#     # choose training and validation indices from shuffled list ind
#     ind_train = ind[:num_data_train]; del ind[:num_data_train] # delete used indices
#     ind_val = ind[:num_data_val]; del ind[:num_data_val] # delete used indices
# #     print(ind_train,ind_val)

#     # choose training and validation data points defined by their respective indices
#     X_train = X[ind_train,:]; Y_train = Y[ind_train] # note Y is a 1D array
#     X_val = X[ind_val,:] ; Y_val = Y[ind_val] # note Y is a 1D array
#     return X_train, Y_train, X_val, Y_val



    
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
    elif distribution=='SRF':
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
    

def dictionarypoly(U,option=[]):
    #% Description: Construct the dictionary matrix phiX containing all multivariate monomials up to degree two for the Lorenz 96
    # % Input: U = [x1(t1) x2(t1) .... xn(t1)
    # %             x1(t2) x2(t2) .... xn(t2)
    # %                    ......
    # %             x1(tm) x2(tm) .... xn(tm)]
    # %        option = [] (monomial) or 'legendre'
    # % Output: the dictionary matrix phiX of size m by N, where m= #measurements and N = (n^2+3n+2)/2
    #if nargin ==1
    #if str(option) == 'mon' or str(option=='legendre):
    #end
    #U=np.array([[1,2,3],[4,5,6],[7,8,9]])

    m=int(np.size(U,0))
    n=int(np.size(U,1))
    phiX=np.zeros([m,int((n+1)*(n+2)/2)])
    phiX[:,0]=np.ones([1,m])
    phiX[:,1:n+1]=np.sqrt(3)*U
    for k in range(1,n+1):
        phiX[:,int(((k)*(2*(n)-k+3)/2)) : int(((k+1)*(n) -(k**2)/2 + k/2 +1 ))]=3*np.multiply(np.matlib.repmat(np.array([U[:,k-1]]).T,
                                                                                                               1,n+1-k),U[:,k-1:n])
        if option=='legendre':
            phiX[:, int((k*(2*n-k+3)/2))]=(np.sqrt(5)/2.0)*(3*np.array([np.square(U[:,k-1])]).T-np.ones([m,1])).T
          
    return phiX



# '''Define dictionary'''
# def dictionary_SRF(X,Omega,bias,activation,dictType,idty,b):
#     ''' Input: X of size(#measurements,dimension)
#     - Omega: matrix of size (rows,columns) i.e., (#features,dimension)
#     - bias: vector of size (#features,)
#     - activation: 'sin' for sin; 'tanh' for tanh; else for 'relu'.
#     -dictType: Type of dictionary - 'SRF' for SRF dictionary; 'poly' for polynomial dictiionary upto order 2;
#     'SRFpoly' for SRF+polynomial dictionary
#     -idty: add constant times identity; 'yes' or 'no'.'''
    
#     if dictType == 'SRF':
#         if activation=='sin':
#             A = np.sin(np.matmul(X,Omega.T) + bias)
#         elif activation=='tanh':
#             A = np.tanh(np.matmul(X,Omega.T) + bias)
#         else:
#             A = np.maximum(np.matmul(X,Omega.T) + bias,0)
#         if idty == 'yes':
#             ident = np.sqrt(b)*np.identity(A.shape[1])
#             Adict = np.vstack((A,ident))
#         else:
#             Adict = A
                            
#     elif dictType == 'poly':
#         A = dictionarypoly(X,'legendre')
#         Adict = A
#     else:
#         polydict = dictionarypoly(X,'legendre')
#         if activation=='sin':
#             Asrf = np.sin(np.matmul(X,Omega.T) + bias)
#         elif activation=='tanh':
#             Asrf = np.tanh(np.matmul(X,Omega.T) + bias)
#         else:
#             Asrf = np.maximum(np.matmul(X,Omega.T) + bias,0)
#         A = np.hstack((polydict,Asrf))
#         if idty == 'yes':
#             ident = np.sqrt(b)*np.identity(A.shape[1])
#             Adict = np.vstack((A,ident))
#         else:
#             Adict = A
#     return Adict,A


'''Define dictionary'''
def dictionary_SRF(X,Omega,bias,activation,dictType):
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




def make_input_data(rows,columns,par1,par2,par3,par4,ratio,method):
    '''Input: rows-number of measurements
    -columns: input dimension
    -par1,par2,par3,par4: parameters for the distribution; par3=par4=ratio=None for method=='uni' or 'nor'.
    -ratio: ratio of input data taken between two distributions; 0<ratio<1
    -method: distribution to be used to sample input; 'uni' for uniform dist; 'nor' for normal dist; 'uni-norm' for mixed distribution
    Output: input data x of size (rows,columns)'''
    if method=='uni':
        x = np.random.uniform(par1,par2,(rows,columns))
    elif method=='nor':
        x = np.random.normal(par1,par2,(rows,columns))
    else:
        num1 = int(ratio*rows)
        num2 = int(rows-num1)
        x1 = np.random.uniform(par1,par2,(num1,columns))
        x2 = np.random.normal(par3,par4,(num2,columns))
        xx = np.vstack((x1,x2))
        ind = list(range(rows)) # define indices to choose data points
        np.random.shuffle(ind)
        x= xx[ind,:]
    return x

