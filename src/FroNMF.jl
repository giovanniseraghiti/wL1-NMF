export FroNMF

function FroNMF(X::AbstractMatrix{T},
             r::Integer;
             maxiter::Integer       = 100,
             timelim:: Float64 = 30.0,
             tol::Float64 = 1e-6,
             W0::AbstractMatrix{T}  = zeros(T, 0, 0),
             H0::AbstractMatrix{T}  = zeros(T, 0, 0),
             updaterW::Function     = hals_updtW,
             updaterH::Function     = hals_updtH,
             benchmark::Bool        = true,
             objfunction::Function  = l2_norm_loss,
             args...
             ) where T <: AbstractFloat
    """
        Perform Non-negative Matrix Factorization (NMF) using the L2-norm loss.

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
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: rand)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: rand)
            - updaterW::Function        : Function for updating matrix W.                       (default: hals_updtW)
            - updaterH::Function        : Function for updating matrix H.                       (default: hals_updtH)
            - benchmark::Bool           : Whether to benchmark the factorization process.       (default: false)
            - objfunction::Function     : Objective function for assessing convergence.         (default: l2_norm_loss)
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - W::AbstractMatrix{T}      : Factorized matrix W.
            - H::AbstractMatrix{T}      : Factorized matrix H.
        # Optional Returns
            - times::Vector{Float64}    : Array of execution times for each iteration. Returs only if benchmark == true.
            - errors::Vector{Float64}   : Array of loss function values for each iteration. Returs only if benchmark == true.

        # Usage Example
            >> X = rand(1000,500)
            >> r = 5
            >> W, H = L1NMF.l2nmf(X, r)
            >> W, H, times, errors = L1NMF.l2nmf(X, r, benchmark=true, maxiter=20)
    """
    b=@elapsed begin
    # Constants
    m, n = size(X)

    # If not provided, init W and H randomly
    W0 = length(W0) == 0 ? rand(m, r) : W0
    W = copy(W0)
    H0 = length(H0) == 0 ? rand(r, n) : H0
    # Work on Ht to work along columns instead of rows (faster)
    H = copy(H0)
    Ht = copy(H0')

    times = []
    errors = []
    
    #Compute first error
    push!(errors, objfunction(X, W, Ht))
    #first CPU time saved
    end
    push!(times,b)

    # Main NMF loop
    println("Starting FroNMF loop ...")
    for it in 1:maxiter
        if benchmark
            b=@elapsed begin
                updaterW(X, W, Ht; args...)
                updaterH(X, W, Ht; args...)
            end
            
            push!(errors, objfunction(X, W, Ht))

            # Early break if there is no progress
            if errors[it]-errors[it+1] < tol
                println("Iteration $it, relative error $(errors[it])")
                println("No progess: end at iteration $it")
                break
            end

            #Stopping criteria: timelimit
            push!(times,times[end]+b)
            if times[end]>timelim
                println("Iteration $it, relative error $(errors[it])")
                println("Stopped for timelimit at iteration $it")
                break
            end

            if it == 1 || it % 10 == 0
                println("Iteration $it, relative error $(errors[it])")
            end
        else
            updaterW(X, W, Ht; args...)
            updaterH(X, W, Ht; args...)
        end
    end
    if benchmark
        return W, Ht', times, errors
    else
        return W, Ht'
    end
end


function hals_updtW(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XHt::AbstractMatrix{T} = zeros(T, 0, 0),
                    HHt::AbstractMatrix{T} = zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    """
    Perform Hierarchical  Alternating Least Squares (HALS) to update the matrix W.

    W =  W ◦ (X * H') / (H * H' * W) 

    # Arguments
        - X::AbstractMatrix{T}      : The input data matrix to be factorized.
        - W::AbstractMatrix{T}      : Actual value of the matrix W.
        - Ht::AbstractMatrix{T}     : Actual value of the matrix H transposed.
    # Optional Arguments
        - XHt::AbstractMatrix{T}    : Value of the product X * H'.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - HHt::AbstractMatrix{T}    : Value of the product H * H'.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - args...                   : Additional arguments that won't affect the function.
    """

    # Init
    r = size(W, 2)

    # If needed, compute intermediary values
    if length(XHt) == 0
        XHt = X * Ht
    end
    if length(HHt) == 0
        HHt = Ht' * Ht
    end

    # Loop on columns of W
    for j in 1:r
        jcolW = view(W, :, j)
        deltaW = max.((XHt[:,j] - W * HHt[:,j]) / max(HHt[j,j],1e-16), -jcolW)
        jcolW .+= deltaW
    end
end


function hals_updtH(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XtW::AbstractMatrix{T} = zeros(T, 0, 0),
                    WtW::AbstractMatrix{T} = zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    """
    Perform Hierarchical  Alternating Least Squares (HALS) to update the matrix H.

    H' =  H' ◦ (X' * H) / (W' * W * H') 

    # Arguments
        - X::AbstractMatrix{T}      : The input data matrix to be factorized.
        - W::AbstractMatrix{T}      : Actual value of the matrix W.
        - Ht::AbstractMatrix{T}     : Actual value of the matrix H transposed.
    # Optional Arguments
        - XtW::AbstractMatrix{T}    : Value of the product X' * W.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - WtW::AbstractMatrix{T}    : Value of the product W' * W.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - args...                   : Additional arguments that won't affect the function.
    """
    # Init
    r = size(Ht, 2)

    # If needed, compute intermediary values
    if length(XtW) == 0
        XtW = X' * W
    end
    if length(WtW) == 0
        WtW = W' * W
    end

    # Loop on rows of H (columns of Ht)
    for i in 1:r
        irowH = view(Ht, :, i)
        deltaH = max.((XtW[:,i] - Ht * WtW[:,i]) /  max(WtW[i,i],1e-16), -irowH)
        irowH .+= deltaH
    end
end
