"""
    In this Script we compare different NMF model on a small 6 x 6  synthetic example. We consider
    a binary matrix containing two clusters on the diagonal blocks and few entries in the non
    diagonal blocks. We compute the NMF of X, , that is X≈ W * H, of rank 2 in order to recover the
    two original clusters.
    
    We aim at showing the importance of the choice of the error measure in NMF models according to the data 
    and the task at hand.    
"""
#Activate the test enviroment
using Pkg
Pkg.activate(@__DIR__)

using Random
using L1NMF  
using MAT

Random.seed!(10)

#Create the matricx for the example
X=[1 1 1 0 0  1 ;
1 1 1 0 0  0  ;
1 1 1 0 1 0 ;
0 1 0 1 1 1 ;
0 0 0 1 1 1 ;
1 0 0 1 1 1 ]

X=Matrix{Float64}(X)
m,n=size(X)

#Set the rank
r=2;

tol=-1000000.0
maxiter=10

#Random initialization 
W0=rand(m,r); #L1NMF.normalize(W0)
H0=rand(r,n)
XHt = X*H0'; 
HHt = H0*H0'; 
scaling = sum(XHt.*W0)/sum( HHt.*(W0'*W0)); 
W0 = W0*scaling;
#Scale W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k. 
W0, H0 =L1NMF.rescale(W0,H0)

##
# L2NMF model
W, H, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=maxiter)

# initialization with few steps of HALS for FroNMF
W_init, H_init, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=3)

# L1NMF model
lambda=1.0  #lambda=0 gets worse
W2, H2, times2,errors2 = sCD(X, r, lambda = lambda, W0=W_init, H0=H_init, maxiter=maxiter)  

#FroNMF model
W_l21, H_l21, errors_l21 = l21NMF(X, r, W0=W_init, H0=H_init, maxiter=maxiter)

#KL-NMF model
W_kl, H_kl, errors_kl = KLNMF(X, r, W0=W_init, H0=H_init, maxiter=maxiter)

display(W2*H2)
display(W*H)
display(W_l21*H_l21)
display(W_kl*H_kl)

