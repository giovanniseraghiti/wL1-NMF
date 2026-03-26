"""
    In this Script we compare the sCD and the originale CD a algorithm on synthetic data.
    We analyze different dimensions of random input matrices and sparsity level.
    We evaluate the results in terms of average CPU time per iteration.
"""
#Activate the test enviroment
using Pkg
Pkg.activate(@__DIR__)

using SparseArrays
using Random
using L1NMF       #import the module, everything is used in the module is called like L1NMF.l2nmf
using MAT
using Plots

#Set parameters
rescale=false
tol=-1.0
timelim=1800000000.0
maxiter=15  #maxiter=30 used in the paper

#Choose the number of runs per problem instance (rep=10 for the paper results)
rep=1

#Set the lambda paraeter in the wL1-nMF model
lambda=1.0

#Set the rank
r=20;

#Define dimensions and sparsity level for this experiment
#Setting for the results in the paper
#dimen=[[100,200],[300,400],[500,600],[800,1000]]; nd=length(dimen);
#sparse_param=[0.25,0.50,0.80]; ns=length(sparse_param);

dimen=[[100,200],[300,400]]; nd=length(dimen);
sparse_param=[0.25,0.50,0.7]; ns=length(sparse_param);

#Allocate variables to save the results
t_l1=zeros(nd,ns); t_l2=zeros(nd,ns);
e_l1=zeros(nd,ns); e_l2=zeros(nd,ns);
it_l1=zeros(nd,ns); it_l2=zeros(nd,ns);

for k=1:nd

    #Define the dimensions of the problem
    m=dimen[k][1]; n=dimen[k][2]; 

    for j=1:ns

        #Select the sparsity level
        sp=sparse_param[j];
        X=rand(m,n)
        sx = round(Int, sp * n*m)
        indici = randperm(n*m)[1:sx]    # scegli k posizioni casuali
        X[indici] .= 0               # mettile a zero
        sparsity=(m*n-nnz(sparse(X)))/(m*n)*100

        for i=1:rep
            Random.seed!(i)
            #Random initialization 
            W0=rand(m,r); #L1NMF.normalize(W0)
            H0=rand(r,n)
            
            #Rescaling of the initial point
            XHt = X*H0'; 
            HHt = H0*H0'; 
            scaling = sum(XHt.*W0)/sum( HHt.*(W0'*W0)); 
            W0 = W0*scaling;
            #Scale W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k. 
            W0, H0 =L1NMF.rescale(W0,H0)

            #Run FroNMF as warm start
            W0, H0 = FroNMF(X, r, W0=W0, H0=H0, benchmark=false, maxiter=10)

            #Run sBCD for L1-NMF
            W2, H2, times2,errors2, it2 = sCD(X, r, lambda = lambda, W0=W0, H0=H0, maxiter=maxiter, timelim=timelim,rescale=rescale, tol=tol)
            #Run standard BCD
            W4, H4, times4,errors4, it4 = CD(X, r, lambda = lambda, W0=W0, H0=H0, timelim=timelim, maxiter=maxiter, tol=tol)
            
            #Save averaged CPU time
            t_l1[k,j]+=times2[end]/rep; t_l2[k,j]+=times4[end]/rep;
            #Save averaged final relative error
            e_l1[k,j]+=errors2[end]/rep; e_l2[k,j]+=errors4[end]/rep;
            #Save average iteration number
            it_l1[k,j]+=it2/rep; it_l2[k,j]+=it4/rep; 
        end
    end
end
avg_t_it_l1=t_l1./it_l1; avg_t_it_l2=t_l2./it_l2; 

##
#Display Array with the results
rows = Vector{Vector{Any}}()
r=20
for k=1:nd
    m=dimen[k][1]; n=dimen[k][2];
    first = true
    for j in 1: ns
        #r = results[(m,n,j)]
        sp=sparse_param[j];
        push!(rows, [
            first ? "$(m)×$(n)" : "";
            "$(Int(sp*100))%";
            avg_t_it_l1[k,j];
            avg_t_it_l2[k,j];
            r*log(m*n)/((1-sp)*r*log((1-sp)*m*n));
            avg_t_it_l2[k,j]/avg_t_it_l1[k,j]
        ])
        first = false
    end
end
header = reshape(["m×n", "sparsity", "sCD", "CD", "σ", "gain"], 1, :)
table = permutedims(hcat(rows...))
table = vcat(header, table)
show(stdout, "text/plain", table)

##
#Plot average CPU time for different dimensions and noise
p_vec=[]
for k=1:nd
    p=[]
    p=plot(avg_t_it_l1[k,:], title="m=$(dimen[k][1]), n=$(dimen[k][2])", label="sCD", marker = :circle)
    p = plot!(avg_t_it_l2[k,:],label="CD", marker = :diamond)
    push!(p_vec,p)
end

# Layout 2x2 (change according to the number of dimensions considered)
pp=plot(p_vec..., xlabel="Noise level", ylabel="CPU time per iter",
xticks = (1:length(sparse_param), sparse_param), layout=(ceil(Int, nd / 2),2))
display(pp)
##
#Uncomment if you need to store variables
#Save the results in a .mat file
#IT=hcat(hcat(it_l1,zeros(nd)),it_l2)
#T=hcat(hcat(t_l1,zeros(nd)),t_l2)
#E=hcat(hcat(e_l1,zeros(nd)),e_l2)
#T_IT=hcat(hcat(avg_t_it_l1,zeros(nd)),avg_t_it_l2)
#matwrite("./results/Synth_results/time_synt.mat", Dict("T" => T))
#matwrite("./results/Synth_results/err_synt.mat", Dict("E" => E))
#matwrite("./results/Synth_results/it_synt.mat", Dict("IT" => IT))
#matwrite("./results/Synth_results/t_it_synt.mat", Dict("T_IT" => T_IT))

