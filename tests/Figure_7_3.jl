"""
    In this Script we compare different choices of the regularization paramter in 
    the wL1-NMF model. We use the sCD algorithm to compute the factorization.

    We consider a matrix completion problem with false zeros on synthetic data.
    We generate a random low rank matrix X=WH, we add Laplacian noise by 
    X*=X+N and we replace a proportion of q_1 in [0,1) entries of X to zero,
    siulating missing values. There are two types of zeros: a proportion of q_2 of type I, 
    and of (1-q_2) for type II for q_2 in [0,1]: 
        -type I sets the smallest entries of X to zero, for a total of q_1 q_2 mn entries. 
         These entries can be seen as small entries thresholded to zero. We call them false zeros. 
        -type II sets randomly selected entries of X to zero, for a total of q_1 (1-q_2) mn entries. 
         We call them missing entries, as they do not bring any useful information since they were 
         picked at random.  
    These entries can be considered as missing. 
    
    We evaluate the results in terms of relative error with the ground truth low-rank matrix
"""
#Activate the test enviroment
using Pkg
Pkg.activate(@__DIR__)

using SparseArrays
using LinearAlgebra
using Plots
using DataFrames
using CSV
using Distributions
using Random
using L1NMF       #import the module, everything is used in the module is called like L1NMF.l2nmf

#Set parameters for stopping criteria
timelim=5.0
maxiter=10
tol=1e-6

#Choose dimensions of the problem (paper setting: m=100, n=50, r=20)
m=100; n=50; r=20

#Number of runs (paper setting: rep=10)
rep=1

#Regularization parameter of the wL1-NMF model
lambda_vec=[0.0,0.01,0.03,0.05,0.07,0.1,1.0]
n_lam=length(lambda_vec)

#Percentage of missing values over the total entries (both random and smallest missing)
q=0.6 

#Ratio between random missing entries and smallest missing entries
ratio=[0.0,0.25,0.50,0.75,1.0]
n_r=length(ratio)

#Initialize the variable that stores the relative ground truth error
res=zeros(n_r,n_lam)

for k in 1:rep
    #Fix the seed
    Random.seed!(k)

    #Generate the groundtruth matrix
    W=rand(m,r); H=rand(r,n); X_t=W*H

    #Set the parameter of the Laplacian noise
    mu = 0.0   # location
    b = 0.1   # scale
    lap = Laplace(mu, b) #define Laplace distribution
    # Generate a nosise matrix of size m x n
    N = rand(lap, m, n)
    #Noisy observation
    X_n=X_t+N
    #Check that the matrix we obtained is still nonnegative 
    nonneg_check=sum(X_n.<0)
    
    #Cycle on the possible ratio between random missing values and smallest missing values
    for t in 1:n_r
        Random.seed!(1000+k)
        X=copy(X_n)
        #Create the matrix with missing values
        p1=ratio[t] #Fraction of the missing values completely random
        p2=1.0-p1 #Fraction of the missing the smallest values
        num_zeros = round(Int, p1*q * m*n) #explicit number of indices to remove for random missing values
        num_mis = round(Int, p2*q * m*n)   #explicit number of indices to remove for smallest missing values
        
        #Remove randomly missing data
        entries = vcat(ones(Int, m*n - num_zeros), zeros(Int, num_zeros))
        # Shuffle the entries randomly
        shuffled = shuffle(entries)
        # Reshape into m x n matrix
        M=reshape(shuffled, m, n)   #mask of the random data to 
        X=M.*X

        #Remove the smallest num_mis entries among the remaining
        xv=vec(X)
        pos=findall(>(0.0), xv)  #remaining indices
        ms=length(xv[pos])
        idx = sortperm(vec(xv[pos]))[1:num_mis]  #take the indices associated to the smallest values
        xv[pos[idx]].=0.0  
        X=reshape(xv,m,n)  #Matrix containing both types of missing values

        #Random initialization 
        W0=rand(m,r);
        H0=rand(r,n)

        #Rescale the starting point for FroNMF
        XHt = X*H0'; 
        HHt = H0*H0'; 
        scaling = sum(XHt.*W0)/sum( HHt.*(W0'*W0)); 
        W0 = W0*scaling;
        #Scale W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k. 
        W0, H0 =L1NMF.rescale(W0,H0)

        # FroNMF for initialization (10 iter for the setting in the paper)
        W, H, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=10)

        #wL1-NMF with sBCD algorithm
        global rep0=copy(rep)
        #Cicle on the regularization parameter of the wL1-NMF model
        for j in 1: n_lam
            #Set regularization parameter
            lambda=lambda_vec[j] 
            #Call the sCD algorithm
            W_l1, H_l1, times_l1,errors_l1 = sCD(X, r, lambda = lambda, W0=W, H0=H, timelim=timelim, maxiter=maxiter, tol=tol) 
            #Compute and store the error with the ground truth
            global res[t,j]+=sqrt(sum((X_t-W_l1*H_l1).^2))/sqrt(sum((X_t).^2)) 
        end
        
    end
end
#Average the results
res./=rep

#Save the results in a .csv file (uncomment if needed)
#df = DataFrame(res, :auto) 
#for j in 1:n_lam
#    lam=lambda_vec[j]
#    names(df)[j]="lambda=$lam"
#end
#Save on a CSV file
#CSV.write("Rec_system_$q.csv", df)

#Plot the result: lambda vs rel. err. ground truth, each line is
#a different ratio between random missing values and sallest missing values
res[res .> 1] .= 1 #Avoid too large values for displaying purposes
pl_t=nothing
for t in 1:n_r
    rr=ratio[t]
    if t==1
        global pl_t=plot(res[t,:], label="q_2=$rr", title="Sparsity q_1=$(q*100)%", xlabel="lambda", ylabel="rel. err. groud truth",
        xticks=(1:n_lam, lambda_vec))
    else
        global pl_t=plot!(res[t,:], label="q_2=$rr",xticks=(1:n_lam, lambda_vec))
    end
end
display(pl_t)


