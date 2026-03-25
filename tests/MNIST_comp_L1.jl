"""
    In this Script we test different algorithms to solve L1NMF on the MNIST data sets.
    In particular, we compare:
        -sCD: coordinate descent algorithm for sparse data with complexity 
              per iteration of O(rnnz(X)log(nnz(X))), where r is the rank of the 
              factorization and nnz(X) the nonzero entries in the original data X.
        -CD:  original coordinate descent algorithm for L1-NMF
        -NS:  Nesterov smoothing gradient method, which is a BCD approach solving one 
              subproblem for each column of H (row of W) by using (accelerated) gradient
              method on the dual of L1NMF
        -SUB: Projected subgradient approach with diminishing step size optiizing both 
              the factor W and H at the same time
    We consider a noisy version of the MNIST data set where each white pixel is turned into black
    and each grey is turned into white with probability p. Different percentages of noise are 
    analyzed and the results are evaluated in terms of final relative error 
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

# 7679 for image plot in the paper plus rng(610) in matlab 
Random.seed!(7679)

#Open the data set
data  = matopen("Dataset/MNIST_all.mat")
X_ts = Matrix{Float64}(sparse(read(data,"X"))); 
dim=28  #images are 28x28

#Select a portion of the Dataset
samp=300   #number of samples
ind_samp=randperm(60000)[1:samp];   #select random 
X_t=X_ts[:,ind_samp]
m, n      = size(X_t)
X_t=X_t./255.0   #normalize
normX=sqrt(sum(X_t.^2))
true_err_l1=[]; true_err_l2=[]; true_err_l21=[]; true_err_rri=[]; true_err_m=[]; true_err_sub=[];  true_err_KL=[]


#Algorithms options
timelim=20.0  #timelim=90 for the results in the paper
maxiter=1000000000
tol=1e-6

#Choose the rank of the factorization
r=50 

#Choose the penalization parameter of the wL1-NMF model
lambda=1.0

#Choose the smoothing parameter in the BCD algorithm with Nesterov smoothing 
sigma=0.5 

#Choose the step parameter for the projected subgradient method
beta= 0.01

#Set the number of runs for each level of noise
rep=1; 

#Define the percentages of noise to add
#noise_vec=[0.0,0.04,0.08,0.12,0.16]
noise_vec=[0.04,0.08]
n_v=length(noise_vec)

#Instantiate global variables
iter_l1=zeros(n_v); iter_rri=zeros(n_v); iter_m=zeros(n_v); iter_sub=zeros(n_v);
errl1_l1=zeros(n_v); errl1_rri=zeros(n_v); errl1_m=zeros(n_v); errl1_sub=zeros(n_v);
cpu_l1=zeros(n_v); cpu_rri=zeros(n_v); cpu_m=zeros(n_v); cpu_sub=zeros(n_v);
p_vec=[]
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
    
    #Instantiate local variables to compute the average of the decrease of the error
    times_l1_vec=[]; times_rri_vec=[]; times_m_vec=[]; times_sub_vec=[]; 
    errors_l1_vec=[]; errors_rri_vec=[]; errors_m_vec=[]; errors_sub_vec=[];
    T_l1=[]; T_m=[]; T_rri=[]; T_sub=[]; T_l2=[]; T_l21=[]; T_KL=[];

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

        #Run a few iterations of HALS for FroNMF to initialize L1-NMF algorithms
        W, H, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=10)


        #Compute wL1-NMF 
        W_l1, H_l1, times_l1,errors_l1, it_l1 = sCD(X, r, lambda = lambda, W0=W, H0=H, timelim=timelim, maxiter=maxiter,tol=tol)
        #Save iteration number on average
        iter_l1[i]+=it_l1/rep
        #Save final relative error
        errl1_l1[i]+=errors_l1[end]/rep
        #Save CPU time
        cpu_l1[i]+=times_l1[end]/rep
        #Save the error and time vectors for each 
        push!(times_l1_vec,times_l1); push!(errors_l1_vec,errors_l1);
        T_l1=W_l1*H_l1
        
        #Original BCD algorithm for L1-NMF
        W_rri, H_rri, times_rri,errors_rri, it_rri = CD(X, r, lambda = lambda, W0=W, H0=H, timelim=timelim, maxiter=maxiter,tol=tol)
        #Save iteration number on average
        iter_rri[i]+=it_rri/rep
        #Save final relative error
        errl1_rri[i]+=errors_rri[end]/rep
        #Save CPU time
        cpu_rri[i]+=times_rri[end]/rep
        #Save the error and time vectors for each 
        push!(times_rri_vec,times_rri); push!(errors_rri_vec,errors_rri);
        
        #BCD with Nesterov smoothing for L1-NMF
        Wm, Hm, times_m,errors_m, it_m = NS(X, r, sigma = sigma, lambda = lambda,  W0=W, H0=H, maxiter=maxiter,timelim=timelim,tol=tol)
        #Save iteration number on average
        iter_m[i]+=it_m/rep
        #Save final relative error
        errl1_m[i]+=errors_m[end]/rep
        #Save CPU time
        cpu_m[i]+=times_m[end]/rep
        #Save the error and time vectors for each 
        push!(times_m_vec,times_m); push!(errors_m_vec,errors_m);

        #Projected subgradient method for L1-NMF
        Wsub, Hsub, times_sub,errors_sub,it_sub = SUB(X, r,  W0=W, H0=H, beta=beta, maxiter=maxiter,tol=tol,timelim=timelim)
        #Save iteration number on average
        iter_sub[i]+=it_sub/rep
        #Save final relative error
        errl1_sub[i]+=errors_sub[end]/rep
        #Save CPU time
        cpu_sub[i]+=times_sub[end]/rep
        #Save the error and time vectors for each 
        push!(times_sub_vec,times_sub); push!(errors_sub_vec,errors_sub);
    end

    #interpolate and compute averaged values for Plots
    mx=maximum([maximum(length.(errors_l1_vec)),maximum(length.(errors_rri_vec)),maximum(length.(errors_m_vec)),maximum(length.(errors_sub_vec))])
    errors_l1_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in errors_l1_vec]
    times_l1_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in times_l1_vec]
    global errors_l1_int = [mean(getindex.(errors_l1_int_vec, i)) for i in 1:mx]
    global times_l1_int = [mean(getindex.(times_l1_int_vec, i)) for i in 1:mx] 

    errors_rri_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in errors_rri_vec]
    times_rri_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in times_rri_vec]
    global errors_rri_int = [mean(getindex.(errors_rri_int_vec, i)) for i in 1:mx]
    global times_rri_int = [mean(getindex.(times_rri_int_vec, i)) for i in 1:mx]   
    
    errors_m_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in errors_m_vec]
    times_m_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in times_m_vec]
    global errors_m_int = [mean(getindex.(errors_m_int_vec, i)) for i in 1:mx]
    global times_m_int = [mean(getindex.(times_m_int_vec, i)) for i in 1:mx] 

    errors_sub_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in errors_sub_vec]
    times_sub_int_vec = [L1NMF.interpolate_to_length(x, mx) for x in times_sub_vec]
    global errors_sub_int = [mean(getindex.(errors_sub_int_vec, i)) for i in 1:mx]
    global times_sub_int = [mean(getindex.(times_sub_int_vec, i)) for i in 1:mx] 
    
    #Plot one graph of averaged quantities for each value of noise
    #legend=:outertopright,
    pl_t=plot(times_l1_int,errors_l1_int, label="sCD",  title="Noise=$(noise_vec[i])")
    pl_t=plot!(times_sub_int,errors_sub_int, label="SUB")
    pl_t=plot!(times_m_int, errors_m_int, label="NS")
    pl_t=plot!(times_rri_int, errors_rri_int, label="CD")
    push!(p_vec,pl_t)

end

#Save matrix for tables
#E=hcat(errl1_l1,errl1_rri,errl1_m,errl1_sub)
#CPU=hcat(cpu_l1,cpu_rri,cpu_m,cpu_sub)
#IT=hcat(iter_l1,iter_rri,iter_m,iter_sub)
#matwrite("C:/Users/Giovanni/OneDrive - UMONS/L1NMF/results/MNIST_paper/tab_cpu_new.mat",Dict("CPU" => CPU))
#matwrite("C:/Users/Giovanni/OneDrive - UMONS/L1NMF/results/MNIST_paper/tab_err_new.mat",Dict("E" => E))
#matwrite("C:/Users/Giovanni/OneDrive - UMONS/L1NMF/results/MNIST_paper/tab_it_new.mat",Dict("IT" => IT))


##
# Layout 2x2 (change according to the number of dimensions considered)
pp=plot(p_vec..., xlabel="CPU time", ylabel="Relative error",
 layout=(ceil(Int, n_v / 2), 2))
display(pp)
##


