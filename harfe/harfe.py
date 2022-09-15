"""Sparse Random Mode Decomposition in Python
Hosts the Sparse Random Mode Decomposition algorithm
SRMD(y, t):
    Decomposes a time series y(t) = sum_k s_k(t) into modes s_k(t) that are well
    connected in time-frequency space, and mutualy disjoint from each other.
"""

__all__ = ['SRMD']

import logging
import numpy as np
from spgl1 import spg_bpdn
import time



def hard_thresh_pur_tik(A,b,c0,s,tot_iter,thresh,mu,lam):
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

