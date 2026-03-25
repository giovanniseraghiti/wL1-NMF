export KLNMF
function KLNMF(X::AbstractMatrix{T},
             r::Integer;
             W0::AbstractMatrix{T}  = zeros(T, 0, 0),
             H0::AbstractMatrix{T}  = zeros(T, 0, 0),
             maxiter::Integer       = 100,
             timelim:: Float64 = 30.0,
             tol::Float64 = 1e-6,
             objfunction::Function  = KL_norm_loss,
             updater::Function = L1NMF.MU_update,
             args...
             ) where T <: AbstractFloat

 """
        Perform Non-negative Matrix Factorization (NMF) using the KL divergence and the MU algorithm to 
        minimize it.

        This function factorizes the input matrix X into two non-negative matrices W and H
        such that X ≈ W * H. This is done by iteratively updating W and H using specified
        update functions until convergence or reaching the maximum number of iterations.
        
        More details in "Gillis, Nicolas. Nonnegative matrix factorization. Society for Industrial and Applied Mathematics, 2020."
        
        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
        # Optional Arguments
            - maxiter::Integer          : Maximum number of iterations for the factorization.   (default: 100)
            - timelim::Float64          : Fixed timelimit in seconds.                           (default: 30.0)
            - tol::Float64              : Tolerance on consecutive relative residuals.          (default: 1e-6)
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: zeros)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: zeros)
            - updater::Function         : Function for updating matrix W.                       (default: MU_update)
            - objfunction::Function     : Objective function for assessing convergence.         (default: KL_norm_loss)
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - W::AbstractMatrix{T}      : Factorized matrix W.
            - H::AbstractMatrix{T}      : Factorized matrix H.
            - times::Vector{Float64}    : Array of execution times for each iteration. 
            - errors::Vector{Float64}   : Array of loss function values for each iteration. 

        # Usage Example
            >> X = rand(1000,500)
            >> r = 5
            >> W, H, times, errors = L1NMF.KLNMF(X, r, maxiter=20)
    """
    
    b=@elapsed begin
    m, n = size(X)
    # If not provided, init W and H randomly
    W0 = length(W0) == 0 ? rand(m, r) : W0
    H0 = length(H0) == 0 ? rand(r, n) : H0
    W, H = copy(W0), copy(H0)

    #Tolerance to avoid completly zero values in the factors
    epsil=1e-16
    end
    errors = []
    times=[]
    push!(errors, objfunction(X, W, H))
    push!(times,b)
    for it in 1:maxiter
        b=@elapsed begin
        #Updating W
        W=updater(X',H',W',epsil); W=W'

        #Updating H
        H=updater(X,W,H,epsil);

        #compute the error
        push!(errors,objfunction(X,W,H))
        end
        push!(times,times[end]+b)

        if times[end]>timelim
                break
        end

        # Early break if there is no progress
        if errors[it]-errors[it+1] < tol
            println("Iteration $it, relative error $(errors[it])")
            println("No progess: end at iteration $it")
            break
        end

        if it == 1 || it % 10 == 0
            println("Iteration $it, relative error $(errors[it])")
        end

    end
    return W, H, times, errors
end

function MU_update(X::AbstractMatrix{T},
             W::AbstractMatrix{T}  = zeros(T, 0, 0),
             H::AbstractMatrix{T}  = zeros(T, 0, 0),
             epsilon:: Float64 = 2^(-10),
             args...
             ) where T <: AbstractFloat

    XdWH = X./(W*H.+epsilon);
    N = W'*XdWH; 
    D = repeat(sum(W, dims=1)', 1, size(X, 2)) .+ epsilon
    H = max.(epsilon, H .* (N./(D.+epsilon))); 
    
    return H
end