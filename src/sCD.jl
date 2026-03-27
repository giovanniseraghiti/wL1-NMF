export sCD

function sCD(X::AbstractMatrix{T},
                        r::Integer;
                        lambda::Float64 = 1.0,
                        maxiter::Integer = 30,
                        timelim::Float64 = 30.0,
                        tol::Float64 = 1e-6,
                        W0::AbstractMatrix{T} = zeros(T, 0, 0),
                        H0::AbstractMatrix{T} = zeros(T, 0, 0),
                        updater::Function = L1NMF.updateH_l1_sparse,
                        objfunction::Function = L1NMF.wl1_norm_loss,
                        args...
                            ) where T <: AbstractFloat
    
     """
        Computes the wL1-NMF of X, that is X≈ W * H, employing the sCD algorithm. The wL1-NMF model is a generalization of L1-NMF suitable for sparse data with false zeros. The sCD algorithm updates each entries of W and H alternatively by
        solving a one-dimensional Least Absolute Deviation (LAD) problem using the weighted median algorithm. This code
        implement a variant of the standard CD algorithm that has complexity scaling with the nonzero entries in the data.
        Stopping criteria are: maximum number of iterations, timelimit, or progresses in the relative error below a 
        threshold tol between two consecutive iterations.

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
        # Optional Arguments
            - lambda::Float64           : Regularization parameter.                             (default: 1.0)
            - maxiter::Integer          : Maximum number of iterations for the factorization.   (default: 30)
            - timelim::Float64          : Fixed timelimit in seconds.                           (default: 30.0)
            - tol::Float64              : Tolerance on consecutive relative residuals.          (default: 1e-6)
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: HALS warm start)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: HALS warm start)
            - updater::Function         : Function for updating matrix W and H.                 (default: updateH_l1_sparse)
            - objfunction::Function     : Objective function for computing first error.         (default: wl1_norm_loss)
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
            >> W, H, times, errors = L1NMF.sCD(X, r, maxiter=20)
    """
    
    b = @elapsed begin
    #norm of the original matrix
    normX=sum(abs.(X))

    #Create variables
    times = []
    errors = []
    
    # If not provided, initiate W0 and H0: 10 iterations of HALS for FroNMF
    W0, H0 = (length(W0) == 0 || length(H0) == 0) ? L1NMF.FroNMF(X,r,maxiter=10) : (W0, H0)

    W, H = copy(W0), copy(H0)

    #vector of vectors containing the indices per columns (HK) and per rows (WK) of the nonnegative entries in X
    HK = L1NMF.find_cols_null(X);   
    WK = L1NMF.find_cols_null(X')
    
    #Compute first error
    push!(errors,objfunction(X, W0, H0, lambda))
    #first CPU time saved
    end
    push!(times,b)

    println("Starting sCD loop ...")
    # Main l1 NMF loop with error & time calc
    iter=0
    for it in 1:maxiter
        iter+=1
        b=@elapsed begin
        #Update W
        flag=false
        W = updater(X',H',W', WK,flag, lambda; args...)'
        #Update H
        flag=true
        H, err = updater(X, W, H, HK,flag, lambda; args...)

        #Update the error and the CPU time
        push!(errors,err/normX)
        end
        push!(times,times[end]+b)
        
        #Stopping criteria: timelimit
        if times[end]>timelim
            if length(times)>2
                pop!(times)
                pop!(errors)
                println("Stopped for timelimit at iteration $(it-1)")
            else
                println("Warning: the first iteration exceeds the timelimit")
                println("Iteration $it, relative error $(errors[it])") 
            end
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


function updateH_l1_sparse(X::AbstractMatrix{T},
                           W,
                           H,
                           KH::Vector{Any},
                           flag::Bool,
                           lambda::Float64;
                           args...
                           ) where T <: AbstractFloat
    
    
    """
        Code for the update of the H factor in the wL1-NMF model. For each column of H it that updates each entries 
        alternatively by solving a one-dimensional Least Absolute Deviation (LAD) problem using the weighted median algorithm.
        This code has coplexity that scales linearly with the nonzero entries in the data

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
            - W::                       : Value for matrix W of dimension (m x r) at the current iterate.      
            - H::                       : Value for matrix H of dimension (r x n) at the current iterate.      
            - KH::Vector{Any}           : Vector of vectors containing the indices where X is nonzero for each column.
            - flag::Bool                : True is the subproblem for H, false for W, needed for the error coputation
            - lambda::Float64           : Regularization parameter in the wL1-NMF model
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - H::AbstractMatrix{T}      : Update of the matrix H.
    """
    
    #size of the factor H
    r,n  = size(H)
    Hold = copy(H) #instantiate the variable

    # Calculate sums of rows of W
    S=vec(sum(W, dims=1))
    
    #initialize the error
    err=0;

    # Outer Loop on columns of H
    for q in 1:n
        # Find index where X isn't 0 for each columns
        K = KH[q];
        # Calculate actual values H*K for the comlumn
        v = W[K,:]*H[:,q]

        # Inner loop to update each element in the column
        for p in 1:r
            # Calculate coefficient 'a', 'b' and 'c' of |X-WH|_[q,p] = sum(|a_i-b_i*H[q,p]|) + |0-c*H[q,p]|
            b = W[K,p]
            #Better to use a full vector than a sparse one in the weighted median algo
            a = Vector{Float64}(X[K,q] - v + W[K,p]*H[p,q])
            c = S[p] - sum(W[K,p])

            # Apply tolerance lambda on c
            c *= lambda
            
            a=[a;0.0]
           
            b=[b;c]
            
            # Solve scalar subproblem in H[p,q]: piecwise linear function
            # global optimum found by the weighted median algorithm
            H[p,q] = weigthed_median(a, b)
            
            # Compute the new residual
            v = v + W[K,p]*(H[p,q]-Hold[p,q])
            
        end
        #compute the first part of the error column by column
        if flag
            err+=sum(abs.(X[K,q] - v))-lambda*sum(v)
        end
    end 
    
    #Add last part of the error after the loops
    if flag
        err+=lambda*sum(sum(W, dims=1) .* sum(H, dims=2)')
        return H, err
    else
        return H
    end
    
end