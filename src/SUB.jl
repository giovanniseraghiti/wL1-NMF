export SUB

function SUB(X::AbstractMatrix{T},
                            r::Integer;
                            beta::Float64 = 0.001,
                            maxiter::Integer = 20,
                            timelim::Float64 = 30.0,
                            W0::AbstractMatrix{T} = zeros(T, 0, 0),
                            H0::AbstractMatrix{T} = zeros(T, 0, 0),
                            objfunction::Function = L1NMF.wl1_norm_loss,
                            args...
                            ) where T <: AbstractFloat

    
    """
        Computes the L1-NMF of X employing a projected subgradient method with 
        decaying stepsize.

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
        # Optional Arguments
            - beta::Float64             : Initial step size
            - maxiter::Integer          : Maximum number of iterations for the factorization.   (default: 30)
            - timelim::Float64          : Fixed timelimit in seconds.                           (default: 10.0)
            - tol::Float64              : Tolerance on consecutive relative residuals.          (default: 1e-6)
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: HALS warm start)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: HALS warm start)
            - objfunction::Function     : Objective function for assessing convergence.         (default: wl1_norm_loss(lambda=1.0))
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
            >> W, H, times, errors = L1NMF.SUB(X, r, maxiter=20)
    """

    
    #Create variables
    b=@elapsed begin
    #norm of the original matrix
    normX=sum(abs.(X))

    times = []
    errors = []

   # If not provided, initiate W0 and H0: 10 iterations of HALS for FroNMF
    W0, H0 = (length(W0) == 0 || length(H0) == 0) ? L1NMF.FroNMF(X,r,maxiter=10) : (W0, H0)
    
    W, H = copy(W0), copy(H0)
    m, =size(W)

    #Compute first error
    push!(errors,objfunction(X, W0, H0, 1.0))
    #first CPU time saved
    end
    push!(times,b)

    println(" Starting SUB loop...")
    
    # Main l1 NMF loop with error & time calc
    iter=0
    G=X-W*H
    for it in 1:maxiter
        iter+=1
        
            b=@elapsed begin

            #Compute the subgradient
            indp=findall(G.>0)
            indn=findall(G.<0)
            G[indp].=1.0; G[indn].=-1.0; 
            subgrad_W=-G*H'; subgrad_H=-W'*G

            #Update step size
            step=beta/(it+1);
            
            W =max.(0,W-step*subgrad_W)
            
            H = max.(0,H-step*subgrad_H)
            
            #Evaluate the new residual
            G=X-W*H

            #Compute new error and time
            push!(errors,sum(abs.(G))/normX)
            end
            push!(times,times[end]+b)
            
            #Stopping criteria: timelimit
            if times[end]>timelim
                println("Iteration $it, relative error $(errors[it])")
                println("Stopped for timelimit at iteration $it")
                break
            end

            if it == 1 || it % 10 == 0
                println("Iteration $it, relative error $(errors[it])")
            end
    end

   return W, H, times, errors, iter
end


