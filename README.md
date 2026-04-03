# wL1-NMF

This code implements the paper *Nonnegative Matrix Factorization in the Component-Wise L1 Norm for Sparse Data* (2026, Seraghiti et al.) [1]. 
It is compatible with Julia 1.11 and above.

## Project Structure 

wL1-NMF/    
├── Project.toml    
├── src/  
│ └── ... # algorithms here  
│  
├── tests/   
│ ├── Project.toml  
│ └── ... # tests presented in the paper  
│  
└── Dataset/  
└── ...  

### Algorithms

The **src** folder contains the implemented algorithms; here a complete list:  

#### NMF Models 
- NMF with Frobenius norm (FroNMF): computes Frobenius NMF by the Hierarchical Alternate Least Square (HALS) algorithm. We followed the implementation from [2]
- NMF with L21 norm (L21NMF): computes NMF with L21-norm as error measure by the Multiplicative Update (MU) algorithm from [3].
- NMF with Kullback-Leibler Divergence (KLNMF):  computes NMF with KL divergence as error measure by the MU algorithm. We followed the implementation from [2]  
- NMF with L1 norm (L1NMF) : computes NMF with L1-norm using the algorithms described below.

#### Algorithm solving L1-NMF 
- Coordinate Descent (CD): optimizes alternatively over each entry of the factors by solving a one-dimensional nonnegative Least Absolute Deviation (LAD) subproblem using the weighted median algorithm [4].  
- Sparse Coordinate Descent (sCD): modification of the CD algorithm for sparse data (Algorithm 6.1 and 6.2 of [1]). This code can solve wL1-NMF and has complexity that scales linearly with the number of nonzero entries in the data.  
- Nesterov Smoothing (NS): Block Coordinate Descent (BCD) approach solving one subproblem for each column of H (row of W) by Nesterov smoothing method [5].
- Projected subgradient (SUB): subrgadient approach with diminuishing step size, optimizing both the factor W and H simultaneuosly [6].
- Weighted median algorithm (weighted-median) : solves nonnegative one-dimensional LAD problems (Algorithm 2.1 of [1]).

### Experiments
The **tests** folder contains the experiments presented in [1]. The names correspond to the labels in the paper. The setting used in the paper is in the code. Figures are saved in the root directory and table are printed in 
- Example_prob: computes the rank-2 L1-NMF, FroNMF, L21-NMF, and KLNMF of a 6 x 6 binary matrix with two clusters in the diagonal block. It is a small example to see the impact of the choice of the error measure in NMF models.
- Table_7_1: comparison of CD and sCD in terms of CPU time per iteration on randomly generated synthetic data for different dimensions and sparsities of the data.
- Figure_7_1_2: compares  L1-NMF, FroNMF, L21-NMF, and KLNMF on the MNIST dataset affected by sparse noise in terms of relative error with ground truth data.
- Table_7_2: compares CD, sCD, NS, and SUB to compute L1-NMF on the same noisy MNIST instance of Figure_7_1. Results are evaluated in terms of L1 relative error.
- Figure_7_3: compares different regularization parameters in the wL1-NMF model on a randomly generated matrix completion instance with false zeros.
- Table_7_3_4: compares sCD for wL1-NMF and FroNMF on the tdt2 data set for topic modeling. Results are evaluated qualitatively by looking at the 10 most relevant words for each topic (words corresponding to the 10 largest entries in each column of W) and in terms of sparsity of the extracted topic. 


## Datasets
- MNIST_all.mat: MNIST grey scale 28 x 28 digit dataset.
- tdt2_top30.mat: words x documents dataset, containing 19528 words and 9394 documents. The (i,j)th entry in the data corresponds to the frequency of word i in document j.


## References    
[1] Seraghiti, G., Dubrulle, K., Vandaele, A., Gillis, N.: Nonnegative Matrix Factorization in the Component-Wise L1 Norm for Sparse Data. arXiv preprint arXiv:2603.29715 (2026)  
[2] Gillis, N.: Nonnegative Matrix Factorization. SIAM, Philadelphia (2020)  
[3] Kong, D., Ding, C., Huang, H.: Robust nonnegative matrix factorization using L21-norm. In: Proceedings of the 20th ACM International Conference on Information and Knowledge Management, pp. 673–682 (2011)  
[4] Ke, Q., Kanade, T.: Robust `1-norm factorization in the presence of outliers and missing data by alternative convex programming. In: IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), vol. 1, pp.739–746. IEEE (2005)  
[5] Guan, N., Tao, D., Luo, Z., Shawe-Taylor, J.: MahNMF: Manhattan non-negative matrix factorization. arXiv preprint arXiv:1207.3438 (2012)  
[6] Rahimi, M., Ghaderi, S., Moreau, Y., Ahookhosh, M.: Projected subgradient methods for paraconvex optimization: Application to robust low-rank matrix recovery. arXiv preprint arXiv:2501.00427 (2024)
