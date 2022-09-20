# HARFE : Hard Ridge Random Feature Expansion
## Problem Statement

Given data $(x_k,y_k)_{k=1}^m$ such that $x_k\in\mathbb{R}^d$ and $d$ is large. Find function $f$ such that $f(\mathbf{x}_k)\approx y_k$ for all $k$.
Assume that $f$ is of the form $f(x) = c^T \phi(Wx+b)$ where $W$ and $b$ are weights and bias sampled randomly and fixed.

## Method

HARFE solves the problem of representing $y$ with a sparse random feature basis i.e.,
    
$y(t) \approx \sum_j (c_j) * \phi(\langle x,\omega_j\rangle + b_j))$, where phi is a nonlinear activation function. 
    
Let $A = \phi(Wx+b)$. Then the vector $c$ is obtained by solving the minimization problem,
$\min_c$ $||Ac-y||_2^2$ + $m$ $\lambda$ $||c||_2^2$
where \lambda is the regularization hyperparameter.
    
We solve this iteratively using the Hard Thresholding Pursuit algorithm i.e.,

Start with $s$-sparse vector $c^0 = 0$ and iterate:
    
1. $S^{n+1}$ = { $s$ largest entry indices of (1 - $\mu$ $m$ $\lambda$) $c^n$ + $\mu$ $A^{*}$ $(y - A c^n)$ }
2. $c^{n+1}$ = argmin { $||b - Az||_2^2$ + $m$ $\lambda$ $||z||_2^2$, supp(z) $\subset$ $S^{n+1}$ }.


## Examples

1. func_harfe.ipynb: Exhibits approximation of Friedman function of type 1 using HARFE.
2. data_harfe.ipynb: Exhibits function approximation of propulsion dataset using HARFE.

## Installation

Download the source by clicking Code -> Download ZIP and run the notebooks in the tests file.

## About 

Created by Esha Saha in September 2022 based on HARFE algorithm.

### Contact

Email esaha@uwaterloo.ca if you have any questions, comments or suggestions.

### Citation

Please cite the associated paper if you found the code useful in any way:

    @misc{https://doi.org/10.48550/arxiv.2202.02877,
      doi = {10.48550/ARXIV.2202.02877},
      url = {https://arxiv.org/abs/2202.02877},
      author = {Saha, Esha and Schaeffer, Hayden and Tran, Giang},
      keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Optimization and Control (math.OC), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Mathematics, FOS: Mathematics},
      title = {HARFE: Hard-Ridge Random Feature Expansion},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }
