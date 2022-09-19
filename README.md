# HARFE : Hard Ridge Random Feature Expansion
## Problem Statement

Given data $\{(\mathbf{x}_k,y_k)\}_{k=1}^m$ such that $x_k\in\mathbb{R}^d$ and $d$ is large. Find function $f$ such that $f(\mathbf{x}_k)\approx y_k$ for all $k$.
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
