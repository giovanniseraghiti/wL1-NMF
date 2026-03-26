# wL1-NMF

### This code is compatible with Julia 1.11

This is my personal setting:  

wL1-NMF/
├── Project.toml
├── src/
│ └── ... # algorithms here
│
├── tests/ # tests presented in the paper
│ ├── Project.toml
│ └── ...
│
└── Dataset/
└── ...

**src** folder contains the implemented algorithms; here a complete list:  

For solving L1-NMF:  
1. Coordinate Descent (CD): optimizes alternatively over each entry of the factors by solving a one-dimensional nonnegative Least Absolute Deviation (LAD) subproblem using the weighted median algorithm[1].  
2. Sparse Coordinate Descent (sCD): modification of the CD algorithm for sparse data. This code can solve wL1-NMF and has complexity that scales linearly with the number of nonzero entries in the data.  
3. Nesterov Smoothing (NS): Block Coordinate Descent (BCD) approach solving one subproblem for each column of H (row of W) by Nesterov smoothing method[2].
4. Projected subgradient (SUB): subrgadient approach with diminuishing step size, optimizing both the factor W and H simultaneuosly[3]  

Other NMF models:
1. FroNMF: computes Frobenius NMF by the Hierarchical Alternate Least Square (HALS) algorithm. We followed the implementation from [4]
2. L21NMF: computes NMF with L21-norm as error measure by the Multiplicative Update (MU) algorithm from [5].
3. KLNMF:  computes NMF with KL divergence as error measure by the MU algorithm. We followed the implementation from [4]  

Others:
- Weighted median algorithm: solves nonnegative one-dimensional LAD problems.

**tests** folder containing the experiments presented in [6]. The names correspond to the labels in the paper. The setting used in the paper is in the code
- Example: computes the rank-2 L1-NMF, FroNMF, L21-NMF, and KLNMF of a 6 x 6 binary matrix with two clusters in the diagonal block. It is a small example to see the impact of the choice of the error measure in NMF models.
- Table_7_1: comparison of CD and sCD in terms of CPU time per iteration on randomly generated synthetic data for different dimensions and sparsities of the data.
- Figure_7_1: compares  L1-NMF, FroNMF, L21-NMF, and KLNMF on the MNIST dataset affected by sparse noise in terms of relative error with ground truth data.
- Tab_Fig_7_2: compares CD, sCD, NS, and SUB to compute L1-NMF on the same noisy MNIST instance of Figure_7_1. Results are evaluated in terms of L1 relative error.
- Figure_7_3: compares different regularization parameters in the wL1-NMF model on a randomly generated matrix completion instance with false zeros.
- Table_7_3_4: compares sCD for wL1-NMF and FroNMF on the tdt2 data set for topic modeling. Results are evaluated qualitatively by looking at the 10 most relevant words for each topic (words corresponding to the 10 largest entries in each column of W) and in terms of sparsity of the extracted topic. 


**Dataset**
- MNIST_all.mat: MNIST grey scale digit dataset x784.
- tdt2_top30.mat: words x documents dataset, containing 19528 words and 9394 documents. The (i,j)th entry in the data corresponds to the frequency of word i in   document j.

References  
[1] Ke, Q., Kanade, T.: Robust `1-norm factorization in the presence of outliers and missing data by alternative convex programming. In: IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1, pp.739–746. IEEE (2005)  
[2] Guan, N., Tao, D., Luo, Z., Shawe-Taylor, J.: MahNMF: Manhattan non-negative matrix factorization. arXiv preprint arXiv:1207.3438 (2012)  
[3] Rahimi, M., Ghaderi, S., Moreau, Y., Ahookhosh, M.: Projected subgradient methods for paraconvex optimization: Application to robust low-rank matrix recovery. arXiv preprint arXiv:2501.00427 (2024)
[4] Gillis, N.: Nonnegative Matrix Factorization. SIAM, Philadelphia (2020)
[5] Kong, D., Ding, C., Huang, H.: Robust nonnegative matrix factorization using L21-norm. In: Proceedings of the 20th ACM International Conference on Information and Knowledge Management, pp. 673–682 (2011)
[6] Our paper (to update)
