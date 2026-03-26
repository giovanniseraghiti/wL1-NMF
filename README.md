# wL1-NMF

**src** folder contains the implemented algorithms; here a complete list:  

For solving L1-NMF:  
1. Coordinate Descent (CD): optimizes alternatively over each entry of the factors by solving a one-dimensional nonnegative Least Absolute Deviation (LAD) subproblem using the weighted median algorithm[1].  
2. Sparse Coordinate Descent (sCD): modification of the CD algorithm for sparse data. This code can solve wL1-NMF and has complexity that scales linearly with the number of nonzero entries in the data.  
3. Nesterov Smoothing (NS): Block Coordinate Descent (BCD) approach solving one subproblem for each column of H (row of W) by Nesterov smoothing method[2].
4. Projected subgradient (SUB): subrgadient approach with diminuishing step size, optimizing both the factor W and H simultaneuosly[3]  

Other NMF models:
1. FroNMF: computes Frobenius NMF by the Hierarchical Alternate Least Square (HALS) algorithm.
2. L21NMF: computes NMF with L21-norm as error measure by the Multiplicative Update (MU) algorithm.
3. KLNMF:  computes NMF with KL divergence as error measure by the MU algorithm.  

Others
- Weighted median algorithm: solves nonnegative one-dimensional LAD problems.

**tests** folder contains the experiments 

**Dataset**
- MNIST_all.mat: MNIST grey scale digit dataset x784.
- tdt2_top30.mat: words x documents dataset, containing 19528 words and 9394 documents. The (i,j)th entry in the data corresponds to the frequency of word i in   document j.

References  
[1] Ke, Q., Kanade, T.: Robust `1-norm factorization in the presence of outliers and missing data by alternative convex programming. In: IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1, pp.739–746. IEEE (2005)  
[2] Guan, N., Tao, D., Luo, Z., Shawe-Taylor, J.: MahNMF: Manhattan non-negative matrix factorization. arXiv preprint arXiv:1207.3438 (2012)  
[3] Rahimi, M., Ghaderi, S., Moreau, Y., Ahookhosh, M.: Projected subgradient methods for paraconvex optimization: Application to robust low-rank matrix recovery. arXiv preprint arXiv:2501.00427 (2024)
