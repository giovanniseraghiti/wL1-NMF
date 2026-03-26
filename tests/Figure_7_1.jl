"""
    In this Script we compare different NMF models on the MNIST data sets.
    In particular, we test:
        -L1NMF:  computed by the sCD algorithm
        -FroNMF: computed by the HALS algorithm 
        -L21NMF: computed by the MU algorithm
        -KLNMF:  computed by the MU algorithm
    We consider a noisy version of the MNIST data set where each white pixel is turned into black
    and each grey is turned into white with probability p. Different percentages of noise are 
    analyzed and the results are evaluated in terms of relative error with the ground truth in 
    Frobenius norm ans Peak Signal to Noise Ratio (PSNR) 
"""
#Activate the test enviroment
using Pkg
Pkg.activate(@__DIR__)

using MAT
using SparseArrays
using LinearAlgebra
using Plots
using Statistics
using Random
using L1NMF      

# 7679 for image plot in the paper plus rng(4110) in matlab 
Random.seed!(7679)

#Open the data set
data  = matopen("Dataset/MNIST_all.mat")
X_ts = Matrix{Float64}(sparse(read(data,"X"))); 
dim=28  #images are 28x28

#Select a portion of the Dataset
samp=300   #number of samples (samp=300 for the results in the paper)
ind_samp=randperm(60000)[1:samp];   #select random 
X_t=X_ts[:,ind_samp]
m, n      = size(X_t)
X_t=X_t./255.0   #normalize
normX=sqrt(sum(X_t.^2))

#Algorithms options
timelim=20.0  #for results in the paper use 90.0
maxiter=200000
tol=1e-6

#Choose the rank of the factorization
r=50 

#Choose the penalization parameter of the wL1-NMF model
lambda=1.0

#Choose the smoothing parameter in the BCD algorithm with Nesterov smoothing 
sigma=0.5 

#Set the number of runs for each level of noise
rep=1; 

#Define the percentages of noise to add
#noise_vec=[0.0,0.04,0.08,0.12,0.16,0.20] #choice in the paper
noise_vec=[0.08,0.12,0.16]
n_v=length(noise_vec)

#Instantiate global variables
true_err_l1=[]; true_err_l2=[]; true_err_l21=[];  true_err_KL=[]
true_err_l1_vec=[]; true_err_l2_vec=[]; true_err_l21_vec=[]; 
true_err_KL_vec=[]; true_err_n_vec=[];

psnr_l1_vec=[]; psnr_l2_vec=[]; psnr_l21_vec=[]; 
psnr_KL_vec=[];  psnr_n_vec=[]

sparsity=[]; gain=[]

#Outer loop on the percentage of noise
for i in 1:n_v
    #add sparse noise noise: each white pixel has probability noise to 
    #become black and each grey pixel has probability noise to turn white 
    X=copy(X_t)
    noise=noise_vec[i]
    pos=findall(>(0.0), X_t)
    np=Int64(length(pos)); 
    ind_p=randperm(np)[1:Int64(trunc(noise*np))]
    zer=findall(==(0.0), X_t)
    nz=Int64(length(zer));
    ind_z=randperm(nz)[1:Int64(trunc(noise*nz))]
    X[pos[ind_p]].=0.0; X[zer[ind_z]].=1.0;   #X contains the noisy images
    
    #Sparsity of the matrix
    nonz=nnz(sparse(X))
    push!(sparsity,(m*n-nonz)/(m*n)*100)
    
    #theoretical gain
    push!(gain,(m*n*log(m*n))/(nonz*log(nonz)))
    
    #compute peak signal 
    mse_n = mean((X_t .- X).^2); 
    push!(psnr_n_vec,10 * log10(1.0 / mse_n))
    
    #compute the error with groundtruth
    push!(true_err_n_vec,sqrt(sum((X_t-X).^2))/normX)
    
    #Prepare the data set for the display of the images
    n_disp=2;
    X_disp=zeros(m,n_disp)
    for k in 1:n_disp
        X_disp[:,k]=reshape(reverse(reshape(X[:,k],dim,dim)'),m)
    end
    
    psnr_l1=0; psnr_l2=0; psnr_l21=0; psnr_KL=0;
    true_err_l1=0; true_err_l2=0; true_err_l21=0; true_err_KL=0;
    T_l1=[]; T_l2=[]; T_l21=[]; T_KL=[];
    #Inner loop: several runs for each method to average
    for j in 1:rep
        Random.seed!(j)
        #Random initialization 
        W0=rand(m,r); 
        H0=rand(r,n)

        #Rescale the initial point to match the norm of X
        XHt = X*H0'; 
        HHt = H0*H0'; 
        scaling = sum(XHt.*W0)/sum( HHt.*(W0'*W0)); 
        W0 = W0*scaling;
        #Scale W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k. 
        W0, H0 =L1NMF.rescale(W0,H0)

        #Compute FroNMF 
        W_l2, H_l2 = FroNMF(X, r, W0=W0, H0=H0, timelim=timelim, maxiter=maxiter, tol=tol)
        T_l2=W_l2*H_l2;
        #Relative error with the true data on average
        true_err_l2+=sqrt(sum((X_t-T_l2).^2))/normX/rep
        #Compute PSNR
        mse_l2 = mean((X_t .- T_l2).^2); 
        psnr_l2+=10 * log10(1.0 / mse_l2)/rep

        #Compute L21-NMF
        W_l21, H_l21 = l21NMF(X, r, W0=W0, H0=H0,timelim=timelim, maxiter=maxiter, tol=tol)
        T_l21=W_l21*H_l21;
        #Relative error with the true data
        true_err_l21+=sqrt(sum((X_t-T_l21).^2))/normX/rep
        #Compute PSNR
        mse_l21 = mean((X_t .- T_l21).^2); 
        psnr_l21+=10 * log10(1.0 / mse_l21)/rep

        #Compute KL-NMF
        W_KL, H_KL = KLNMF(X, r, W0=W0, H0=H0,timelim=timelim, maxiter=maxiter, tol=tol)
        T_KL=W_KL*H_KL;
        #Relative error with the true data
        true_err_KL+=sqrt(sum((X_t-T_KL).^2))/normX/rep
        #Compute PSNR
        mse_KL = mean((X_t .- T_KL).^2); 
        psnr_KL+=10 * log10(1.0 / mse_KL)/rep

        #Run a few iterations of HALS for FroNMF to initialize L1-NMF algorithms
        W, H, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=10)

        #Compute wL1-NMF 
        W_l1, H_l1 = sCD(X, r, lambda = lambda, W0=W, H0=H, timelim=timelim, benchmark=benchmark, maxiter=maxiter,tol=tol)
        
        #Relative error with the true data
        T_l1=W_l1*H_l1
        true_err_l1+=sqrt(sum((X_t-T_l1).^2))/normX/rep
        #Compute PSNR
        mse_l1 = mean((X_t .- T_l1).^2); 
        psnr_l1+=10 * log10(1.0 / mse_l1)/rep

        #Save the matrices for displaying
        #Uncomment if you need to store the result (adjust the path)
        #if j==1
            #Uncomment if you need to store the result
            #matwrite("./results/MNIST_paper/app_l1$noise.mat",Dict("mat" => T_l1))
            #matwrite("./results/MNIST_paper/app_l2$noise.mat",Dict("mat" => T_l2))
            #matwrite("./results/MNIST_paper/app_l21$noise.mat",Dict("mat" => T_l21))
            #matwrite("./results/MNIST_paper/app_KL$noise.mat",Dict("mat" => T_KL))
        #end 
    end

    #Save the averaged values for each noise level
    push!(true_err_l1_vec,true_err_l1);  push!(true_err_l2_vec,true_err_l2); 
    push!(true_err_l21_vec,true_err_l21); push!(true_err_KL_vec,true_err_KL);

    push!(psnr_l1_vec,psnr_l1); push!(psnr_l2_vec,psnr_l2); 
    push!(psnr_l21_vec,psnr_l21); push!(psnr_KL_vec,psnr_KL);
 
end
##

#Plot the true error figure
pl_t1=plot(noise_vec, true_err_l1_vec, label="L1-NMF", legend=:outertopright)
pl_t1=plot!(noise_vec, true_err_l2_vec, label="L2-NMF")
pl_t1=plot!(noise_vec, true_err_l21_vec, label="L21-NMF")
pl_t1=plot!(noise_vec, true_err_KL_vec, label="KL-NMF")
title!(pl_t1, "Comparison of NMF models")
xlabel!(pl_t1, "Noise level (p)")
ylabel!(pl_t1, "Relative Error with ground truth")
display(pl_t1)

#Plot the PSNR figure
pl_t2=plot(noise_vec, psnr_l1_vec, label="L1-NMF", legend=:outertopright)
pl_t2=plot!(noise_vec, psnr_l2_vec, label="L2-NMF")
pl_t2=plot!(noise_vec, psnr_l21_vec, label="L21-NMF")
pl_t2=plot!(noise_vec, psnr_KL_vec, label="KL-NMF")
title!(pl_t2, "Comparison of NMF models")
xlabel!(pl_t2, "Noise level (p)")
ylabel!(pl_t2, "PSNR")
display(pl_t2)

##
#Prepare the variables for storing
psnr_mat1=hcat(noise_vec,psnr_l1_vec,psnr_l2_vec,psnr_l21_vec,psnr_KL_vec,psnr_n_vec)
psnr_mat = Dict("matp" => psnr_mat1)


true_err_mat1=hcat(noise_vec,true_err_l1_vec,true_err_l2_vec,true_err_l21_vec,true_err_KL_vec,true_err_n_vec)
true_err_mat = Dict("mate" => true_err_mat1)


#Uncomment if you need to store the result (adjust the path)
#matwrite("./results/MNIST_paper/data_original.mat",Dict("Xt" => X_t))
#matwrite("./results/MNIST_paper/noise_data$noise.mat",Dict("mat" => X))
#matwrite("./results/MNIST_paper/sparsity.mat",Dict("spar" => sparsity))
#matwrite("./results/MNIST_paper/gain.mat",Dict("g" => gain))
#matwrite("./results/MNIST_paper/psnr_mat1.mat",psnr_mat)
#matwrite("./results/MNIST_paper/true_err_mat1.mat",true_err_mat)


