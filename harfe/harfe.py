""" Hard Ridge Random Feature Expansion in Python
Implements Hard Ridge Random Feature Expansion algorithm
harfe(b, x):
    Gives coefficients c that bests represents given data of the form y ~ c^T*phi(Wx+b)
"""

__all__ = ['harfe']

import logging
import numpy as np
from spgl1 import spg_bpdn
import time



def harfe(y,c0 = None,s = None,tot_iter = None,thresh = None,mu = None,lam = None):
    
    """Implimentation of the Hard Ridge Random Feature Expansion algorithm
    **HARFE**
    
    Given data {(x_k,y_k)}_{k=1}^m such that x_k in R^d and d is large. Find function f such that f(x_k)~y_k for all k.
    Assume that f is of the form f(x) = c^T\phi(Wx+b) where W and b are weights and bias sampled randomly and fixed.
    
    Method
    ------
    harfe(b, ...) solves the problem of representing y with a sparse random feature basis i.e.,
    
    y(t) ~ sum_j(c_j) * phi(<x,w_j> + b_j)), where phi is a nonlinear activation function. 
    
    Let A = phi(Wx+b). Then the vector c is obtained by solving the minimization problem,
    min_c ||Ac-y||_2^2 + m*lam||c||_2^2
    where lam is the regularization hyperparameter.
    
    Solve this iteratively using the Hard Thresholding Pursuit algorithm i.e.,
    Start with s-sparse vector c^0 = 0 and iterate:
    
    S^{n+1} = {s largest entry indices of (1-mu*m*lam)*c^n + mu A^*(y - A c^n) }
    c^{n+1}= argmin {||b - Az||_2^2 + m*lam||z||_2^2, supp(z) subseteq S^{n+1}}.
    
    Input
    ------
    
    y : numpy array of size (m,)
        True values of the function i.e., true iutput values.
        
    x : numpy array of size (m,d)
        The input data
        
        
    Output
    -------
    
    c : numpy array of size (N,)
    
    error : float value for relative training error
    
    iter_req : int, total number of iterations required
    
    time :  float, time required to run the algorithm in seconds
    
    Hyperparameters
    ---------------
    N : int, (default: None)
        Number of features to generate. If None is given, use 10*len(y) as default
    s : int, (default: None)
        Sparsity level to be used for hard thresholding. If None is given, use int(0.15*N) as default
    c0 : ndarray, (N,)
        Initial vector c. If None is given, use c0 = np.zeros(N) as default
    mu : float, (default: None)
        Step size for first step of the algorithm. If None is given, default is mu = 0.1
    thres : float, (default: None)
        Threshold for convergence. If None is given, use thresh = 1e-3 as default
    lam: float, (default: None)
        Regularization parameter for the l2 min problem. If None os given, default is lam = 0.0001??????????
    
    verbosity : int, (default: 0)
        If 1 will print out feature info throughout the execution of the
        function. If 2, will print out progress and parameters in addition to
        feature info. Defaults to 0 (no printing).
    """

    
    # Define useful constants
    m = len(y)       # Number of data points

    # Default parameter Handeling
    if N is None:
        N = 10 * m
        
    if s is None:
        s = int(0.15 * m)
        
    if mu is None:
        mu = 0.1
        
    if lam is None:
        lam = 0.0001
    
    if thresh is None:
        thresh = 1e-3
        
    if c0 is None:
        t = np.zeros(N)
    elif len(c0) != N:
        raise ValueError('Initial c0 should be of size N')

    
    error = np.zeros(tot_iter)
    b = b.flatten()
    C = np.zeros((A.shape[1],tot_iter))
    C[:,0] = c0
    i = 0
    iter_req = i
    z1 = np.matmul(A.T,b)
    z2 = np.matmul(A.T,A)
    rel_err = np.linalg.norm(np.matmul(A,c0).flatten()-b)/np.linalg.norm(b)
    start_time = time.time()
    while rel_err>thresh:
        c_tilde = C[:,i] + mu*(z1 - np.matmul(z2,C[:,i])) - ((mu*lam)*C[:,i])
        c_tilde_sort = np.sort(abs(c_tilde))
        idx = c_tilde >= c_tilde_sort[-s]
        A_pruned = A[:,idx]
        z1_pruned = np.matmul(A_pruned.T,b)
        z2_pruned = np.matmul(A_pruned.T,A_pruned)
        c_pruned = np.matmul(np.linalg.pinv((z2_pruned + lam*np.identity(z2_pruned.shape[1])),rcond=1e-15)
                                 ,z1_pruned)
        c_pruned = c_pruned.flatten()
        erlst = 'nd'
        C[idx,i+1] = c_pruned
        error[i+1] = np.linalg.norm(np.matmul(A,C[:,i+1]).flatten()-b)/np.linalg.norm(b)
        iter_req = i+1
#         print(np.count_nonzero(C[:,i+1]))
#         error[i+1] = np.linalg.norm(C[:,i+1].flatten()-C[:,i].flatten())
        print(np.linalg.norm(C[:,i+1].flatten()-C[:,i].flatten()),error[i+1])
        if error[i+1]<=5e-3: #or np.linalg.norm(C[:,i+1].flatten()-C[:,i].flatten())<=1e-3:
#             print(error[i+1],5e-3*np.linalg.norm(b))
#             print(np.linalg.norm(C[:,i+1]))
            break
        elif np.linalg.norm(C[:,i+1].flatten()-C[:,i].flatten())<=1e-5:
            break
        rel_err = error[i+1]
#         iter_req = i+1
#         print('iter reqc-----------------------------------',iter_req)
        i = i+1
#         iter_req = i
#         print('iter req',iter_req)
        if i+1==tot_iter:
#             print('Warning: You might want to use more iterations as maximum number of iterations of',tot_iter)
            iter_req = i
            break
                
    end_time = time.time()  
#     print("total time taken for HARFE ", end_time - start_time,'s')
    print('Finally the total number of iterations the algorithm ran for was',iter_req+1)
    return C[:,iter_req],error[0:iter_req+1],iter_req+1,erlst,end_time - start_time

