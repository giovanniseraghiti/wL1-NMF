"""
    In this Script we compare the wL1-NMF model for different values of the regularization parameter
    with FroNMF for topic modeling. We consider the tdt2 dataset containing a collection of articles 
    from various sources, collected from January to June of 1998, on r = 30 selected topics. 
    It consists of a matrix X m x n with m =19528 and n = 9394, where the (i,j)th entry is the number 
    of occurrences of word i in document j. 

    We compute the NMF of X, , that is X≈ W * H, where the columns of W correspond to the ain topic discussed
    by the documents in the data set. 
"""
#Activate the test enviroment
using Pkg
Pkg.activate(@__DIR__)

using MAT
using SparseArrays
using DelimitedFiles
using LinearAlgebra
using Random
using DataFrames
using L1NMF       #import the module, everything is used in the module is called like L1NMF.l2nmf
using Statistics
Random.seed!(100)

#Open the data set
data  = matopen("Dataset/tdt2_top30.mat")
X = sparse(read(data,"X"))'   #word x doc dataset 
words=read(data,"words");
m, n      = size(X)

#Preprocess the data (largest element in each columns equal to 1)
L1NMF.preprossessing!(X)

#Choose the rank of the factorization (r=30 is suggested since there are 30 topics)
r=30

#Set the paraeters for the stopping criteria
tol=1e-6
timelim=300000.0
maxiter=3

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
W, H, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=20, timelim=timelim)

#count the words per topic (each column of W)
w_per_topic_l2=mean(vec(sum(W.>1e-5, dims=1)))
w_rel=10; k=100;
Bag_rel_l2=Dict();
#ind_rel_l2_k=zeros(k,r); 
count_null_l2=0;
#Take the most relevant words per topic (largest entries in each column of W)
for i =1:r
    ind_w_l2=findall(x -> x>1e-5, W[:,i])
    ind_rel_t2=sortperm(W[:,i]); 
    nw_l2=length(ind_w_l2)
    if nw_l2>=w_rel
        ind_rel_l2=ind_rel_t2[end:-1:end-w_rel+1]
        #ind_rel_l2_k[:,i]=ind_rel_t2[end:-1:end-k+1]
        global Bag_rel_l2["rel_topic$i"]=words[ind_rel_l2]
    else
        ind_rel_l2=ind_w_l2
        mis_l2=fill("",w_rel-nw_l2)
        global Bag_rel_l2["rel_topic$i"]=[words[ind_rel_l2];mis_l2]
    end 

    #Count the number of topics described by less than 5 words
    if sum(W[:,i])<=5.0
        global count_null_l2+=1.0
    end
end

# Initialization
W0, H0, times1,errors1 = FroNMF(X, r, W0=W0, H0=H0, maxiter=3)

# wL1-NMF model
lambda_vec=[0.08,0.1] #define the parameters of the model
#lambda_vec=[0.0,0.01,0.04,0.08,0.1,1.0] #define the parameters of the model
p=length(lambda_vec)

#initialize counting variables
sum_l1=zeros(p) 
count_null_l1=zeros(p)
w_per_topic=zeros(p); 
#Cycle on the number of regularization parameters
#Bag=Dict(); 
#Bag_rel=Dict(); 
topic_L2=zeros(r,w_rel); topic_L1=zeros(r,w_rel);  
G_Bag=Any[];
zz_vec=[];
count_null_l1=zeros(p)
for j in 1:p
    #Set regularization parameter
    lambda=lambda_vec[j]  

    #Call the sCD algorithm
    W2, H2, times2,errors2 = sCD(X, r, lambda = lambda, W0=W0, H0=H0, maxiter=maxiter, timelim=timelim,tol=tol)
    
    #Compute average word per topic
    w_per_topic[j]=mean(vec(sum(W2.>1e-5, dims=1)))
    
    Bag_rel=Dict();
    zz=[]
    for i in 1:r
        #Identify relevant words
        ind_w=findall(x -> x>1e-5, W2[:,i])
        ind_rel_t=sortperm(W2[:,i]); 
        nw=length(ind_w)
        #Construct the bag of word of the relevant words
        if nw>=w_rel
            ind_rel=ind_rel_t[end:-1:end-w_rel+1]
            global Bag_rel["rel_topic$i"]=words[ind_rel]
        else
            ind_rel=ind_w
            mis=fill("",w_rel-nw) #fill with "" if needed
            global Bag_rel["rel_topic$i"]=[words[ind_rel];mis]
        end 

        #Count the number of topics described by less 5 words
        if sum(W2[:,i])<=5.0
            global count_null_l1[j]+=1.0
        end

    end
    
    #Global dictionary for each value of lambda
    push!(G_Bag,Bag_rel)
    push!(zz_vec,mean(zz))
    
end

#Merge wL1-NMF and FroNMF results for saving purposes
push!(w_per_topic,w_per_topic_l2)
push!(G_Bag,Bag_rel_l2)
push!(count_null_l1,count_null_l2)

#This will save the bag of words in an html file which is better for visualization
#Uncomment if you need to store the results (changin the path)
#=
push!(lambda_vec,2.0)
using PrettyTables
nd=length(G_Bag)
for i=1:nd
    pp=lambda_vec[i]
    open("l1_topics_$pp.html", "w") do io
        pretty_table(io, G_Bag[i]; backend = :html)
    end
end

=#
