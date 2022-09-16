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
    N_features : int, (default: None)
        Number of features to generate. If None is given, will use 10*len(y)
    eps : float, (default: None)
        Radius of a feature's neighbourhood in time-frequency space. If None is
        given, will default to: 0.2 * (t[-1] - t[0])
    max_frq : float, (default: None)
        Maximum possible frequency a feature could have. If None is given, will
        use half the sample rate: 0.5 * (len(t)+1) / (t[-1] - t[0])
    w : float, (default: 0.1)
        Window size of the features in seconds. Defaults to 0.1s or 100ms.
    r : float, (default: 0.05)
        Maximum relative error in the representation of the signal. Defaults to
        5%. Should be between [0.01, 0.5] for sensible results.
    threshold : float, (default: None)
        Bottom quantile of nonzero-coefficients to prune after the
        coefficients are learned. If None is given, will skip this step. Should
        be in the range [0, 1].
    frq_scale : float, (default: None)
        Amount to scale the frequencies of the features before the clustering
        algorithm. If None is given, will default to: (t[-1] - t[0]) / max_frq
    min_samples : int, (default: 4)
        Number of features in a neighbourhood required to be considered a core
        point in the clustering algorithm. Should be 3, 4, or 5 for sensible
        results.
    seed : int, (default: None)
        Seed to use in the random generation of the features. This is useful for
        repeatability. If None is given, a random seed will be used.
    n_modes : int, (default: None)
        Number of modes in the input signal if known. Will merge extra modes
        with the smallest L2-norm so at most n_modes are returned. If None
        is given, will not merge any modes.
    verbosity : int, (default: 0)
        If 1 will print out feature info throughout the execution of the
        function. If 2, will print out progress and parameters in addition to
        feature info. Defaults to 0 (no printing).
    return_features : bool, (default: False)
        If True, will return the learned modes in addition to weights,
        tau, frq, and phs of the features, and the features' label (which mode
        each feature belongs to).
    cutoff : float, (default: None)
        If given, will *not* use DBSCAN to cluster features and instead separate
        features into two modes: features with frequency above and below cutoff."""
    
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

