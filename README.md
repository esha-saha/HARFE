# HARFE : Hard Ridge Random Feature Expansion

## Installation

Use the steps below for a local installation of HARFE. Requires Python >= 3.6.

```
$ git clone https://github.com/esaha2703/HARFE.git
$ pip install - e .
```
## Quickstart

Use the code below to learn the function $f( \mathbf{x} )$ = $x_2^2 + x_6 x_8 + \cos(x_{10})$ for $\mathbf{x} \in \mathbb{R}^{10}$ using HARFE algorithm. For more details refer to [Method](#method) section below.

```python
import numpy as np
from harfe import harfe
from harfe.utils import generate_omega_bias, feature_matrix

dimension = 10       # input dimension
num_points = 500     # number of sampling points

inp = np.random.uniform(-1, 1, (num_points, dimension))        # Input data
out = inp[:,1]**2 + inp[:,5]*inp[:,7] + np.cos(inp[:,9])       # Output function f(x) = x2^2 + x6*x8 + cos(x10) 

# generate random weights and biases 
weights, bias = generate_omega_bias(rows = 5000, columns = d, weight = 1, par1 = -1, par2 = 1,
                                 distribution = 'norm-uni', bool_bias = True, sparsity = 2)  

#build random feature matrix of the form A = sin(weights*inp + bias) and normalize the columns
A = feature_matrix(inp, weights, bias, activation = 'sin', dictType = 'SRF')
scale_A = np.linalg.norm(A, axis = 0) 
A /= scale_A 

# Implement HARFE algorithm 
coeff, rel_error, iterations, _ = harfe(out, A, N = 5000)

#recover the function using learnt coefficients using out ~ A*coefficients
out_recovered = np.matmul(A, coeff)
print('Relative error:', rel_error[-1],'\nIterations required:', iterations)
```


## Problem Statement

Given data $(x_k, y_k)_{k=1}^m$ such that $\mathbf{x}_k\in\mathbb{R}^d$ and $y_k\in\mathbb{R}$ where $d$ is large. 
Find function $f$ such that $f(\mathbf{x}_k)\approx y_k$ for all $k$.
Assume that $f$ is of the form $f(\mathbf{x}) = \mathbf{c}^T \phi(W\mathbf{x}+b)$ where $W$ and $b$ are weights and bias sampled randomly and fixed.

## Method

HARFE solves the problem of representing $y$ with a sparse random feature basis i.e.,
    
$y \approx \sum_j (c_j) * \phi(\langle \mathbf{x},\omega_j\rangle + b_j))$, where phi is a nonlinear activation function. 
    
Let $A = \phi(W\mathbf{x}+b)$. Then the vector $\mathbf{c}$ is obtained by solving the minimization problem,
$\min_c$ $||A\mathbf{c}-y||_2^2$ + $m$ $\lambda$ $||\mathbf{c}||_2^2$
where $\lambda$ is the regularization hyperparameter.
    
We solve this iteratively using the Hard Thresholding Pursuit algorithm i.e.,

Start with $s$-sparse vector $\mathbf{c}^0 = 0$ and iterate:
    
1. $S^{n+1}$ = { $s$ largest entry indices of (1 - $\mu$ $m$ $\lambda$) $\mathbf{c}^n$ + $\mu$ $A^{*}$ $(y - A \mathbf{c}^n)$ }
2. $c^{n+1}$ = argmin { $||b - A\mathbf{z}||_2^2$ + $m$ $\lambda$ $||\mathbf{z}||_2^2$, supp( $\mathbf{z}$ ) $\subset$ $S^{n+1}$ }.


## Examples

1. func_harfe.ipynb: Exhibits approximation of Friedman function of type 1 using HARFE.
2. data_harfe.ipynb: Exhibits function approximation of propulsion dataset using HARFE.


## Contact and citation

Email esaha@uwaterloo.ca if you have any questions, comments or suggestions. Please cite the associated paper if you found the code useful in any way:

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
