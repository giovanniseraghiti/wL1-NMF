export NS

function NS(X::AbstractMatrix{T},
    r::Integer;
    sigma::Float64=0.5,
    maxiter::Integer = 30,
    timelim::Float64 = 30.0,
    tol::Float64 = 1e-6,
    W0::AbstractMatrix{T} = zeros(T, 0, 0),
    H0::AbstractMatrix{T} = zeros(T, 0, 0),
    updater::Function = L1NMF.smoothing_gradient_acc,
    objfunction::Function = L1NMF.wl1_norm_loss,
    args...
    ) where T <: AbstractFloat

    """
        Computes the wL1-NMF of X employing the sCD algorithm that updates each entries of W and H alternatively by
        solving a one-dimensional Least Absolute Deviation (LAD) problem using the weighted median algorithm. This code
        implement a variant of the standard CD algorithm that has complexity scaling with the nonzero entries in the data.
        Stopping criteria are: maximum number of iterations, timelimit, or progresses in the relative error below a 
        threshold tol between two consecutive iterations.

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
        # Optional Arguments
            - sigma::Float64            : Smoothing parameter parameter.                        (default: 0.5)
            - maxiter::Integer          : Maximum number of iterations for the factorization.   (default: 30)
            - timelim::Float64          : Fixed timelimit in seconds.                           (default: 30.0)
            - tol::Float64              : Tolerance on consecutive relative residuals.          (default: 1e-6)
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: HALS warm start)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: HALS warm start)
            - updater::Function         : Function for updating matrix W.                       (default: smoothing_gradient)                    
            - objfunction::Function     : Objective function for assessing convergence.         (default: wl1_norm_loss (lambda=1.0))
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - W::AbstractMatrix{T}      : Factorized matrix W.
            - H::AbstractMatrix{T}      : Factorized matrix H.
            - times::Vector{Float64}    : Array of execution times for each iteration. 
            - errors::Vector{Float64}   : Array of relative loss function values for each iteration. 
            - it:: Integer              : Total iteration number
            
        # Usage Example
            >> X = rand(1000,500)
            >> r = 5
            >> W, H, times, errors = L1NMF.NS(X, r, maxiter=20)
    """
    b = @elapsed begin
    #norm of the original matrix
    normX=sum(abs.(X))
    
    #Instantiate time and error vectors
    times = []
    errors = []
    # If not provided, initiate W0 and H0: 10 iterations of HALS for FroNMF
    W0, H0 = (length(W0) == 0 || length(H0) == 0) ? L1NMF.FroNMF(X,r,maxiter=10) : (W0, H0)

    #Instantiate the variables
    W, H = copy(W0), copy(H0)

    #Compute first error
    push!(errors,objfunction(X, W0, H0, 1.0))
    #first CPU time saved
    end
    push!(times,b)

    println("Starting NS loop")
    

    sigma0=sigma; sigma_t=sigma
    # Main l1 NMF loop with error & time calc
    iter=0
    for it in 1:maxiter
        iter+=1
            b=@elapsed begin
            #Update W
            flag=false
            W = updater(X',H',W',sigma_t,flag; args...)'

            ##Update H
            flag=true
            H,err = updater(X, W, H, sigma_t,flag; args...)

            #Decrease the smoothing parameter
            sigma_t=sigma0/(it+1)
            
            push!(errors,err/normX)
            end
            push!(times,times[end]+b)

            
            if times[end]>timelim
                pop!(times)
                pop!(errors)
                println("Stopped for timelimit at iteration $(it-1)")
                break
            end

            # Early break if there is no progress
            if errors[it]-errors[it+1] < tol
                println("Iteration $it, relative error $(errors[it])")
                println("No progess: end at iteration $it")
                break
            end

            println("Iteration $it, relative error $(errors[it])")

    end

    
    return W, H, times, errors, iter 
    

end


function smoothing_gradient(X::AbstractMatrix{T},
    W::AbstractMatrix{T},
    H::AbstractMatrix{T},
    sigma::Float64,
    flag::Bool,
    maxiter::Integer = 10;
    args...
    ) where T <: AbstractFloat

"""
        Code for the update of the H factor in the Nesterov smoothing algorithm. For each column of H, it computes gradient steps 
        for the dual problem of L1-NMF    

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - W::                       : Value for matrix W of dimension (m x r) at the current iterate.      
            - H::                       : Value for matrix H of dimension (r x n) at the current iterate.      
            - sigma:Float64             : smoothing parameter
            - flag::Bool                : True is the subproblem for H, false for W, needed for the error coputation
            - lambda::Float64           : Regularization parameter in the wL1-NMF model
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - H::AbstractMatrix{T}      : Update of the matrix H.
            - err:Float64               : error, only if the code is updating H
    """


    # Init
m, =size(W)
r,n  = size(H)


#Compute the m-vector q and the Lipschitz constant
q=[]; L=0; 

#Compute the Liepschitz constant for the step size
for j in 1:m
    rr=sqrt(sum(W[j,:].^2))
    L+=rr
    push!(q,rr)
end
#Liepschitz constant
L/=sigma

#Quantity we need to compute the gradient
Q=q*ones(n)'

#initialize the residual
res=W*H-X;
for k in 1:maxiter

    #Compute gradient
    U=min.(1,max.(-1,res./(sigma.*Q)))
    U[isnan.(U)] .= 0.0
    grad=W'*U;

    #Projected gradient step
    H=max.(0,H-grad./L)

    #Update residual
    res=W*H-X

end

if flag
    err=sum(abs.(res))
    return H,err
else
    return H
end

end

function smoothing_gradient_acc(X::AbstractMatrix{T},
    W::AbstractMatrix{T},
    H::AbstractMatrix{T},
    sigma::Float64,
    flag::Bool,
    maxiter::Integer = 30;
    args...
    ) where T <: AbstractFloat

"""
        Code for the update of the H factor in the Nesterov smoothing algorithm. For each column of H, it computes gradient steps 
        for the dual problem of L1-NMF using Nesterov accelerated gradient  

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - W::                       : Value for matrix W of dimension (m x r) at the current iterate.      
            - H::                       : Value for matrix H of dimension (r x n) at the current iterate.      
            - sigma:Float64             : smoothing parameter
            - flag::Bool                : True is the subproblem for H, false for W, needed for the error coputation
            - lambda::Float64           : Regularization parameter in the wL1-NMF model
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - H::AbstractMatrix{T}      : Update of the matrix H.
            - err:Float64               : error, only if the code is updating H
    """


    # Init
m, =size(W)
r,n  = size(H)


#Compute the m-vector q and the Lipschitz constant
q=[]; L=0; 

#Compute the Liepschitz constant for the step size
for j in 1:m
    rr=sqrt(sum(W[j,:].^2))
    L+=rr
    push!(q,rr)
end
#Liepschitz constant
L/=sigma

#Quantity we need to compute the gradient
Q=q*ones(n)'

#initialize the residual
res=zeros(r,n)
H_old=copy(H)
for k in 1:maxiter

    #Compute gradient
    Y=2/(k+3)*H+(k+1)/(k+3)*H_old
    res=W*Y-X
    U=min.(1,max.(-1,res./(sigma.*Q)))
    U[isnan.(U)] .= 0.0
    grad=W'*U;
    H_old=H
    H=max.(0,Y-grad./L)

    #Update residual
    if k==maxiter
        res=W*H-X
    end

end

if flag
    err=sum(abs.(res))
    return H,err
else
    return H
end

end