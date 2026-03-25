export weigthed_median

function weigthed_median(x::AbstractVector{T},
                         y::AbstractVector{T},
                         args...
                         ) where T <: AbstractFloat

    """
        Weighted median algorithm that solves a one dimensional least absolute deviation problem with nonnegativity constraint:
               
            ||x-alpha*y||_1  with  alpha >=0


        # Arguments
            - x::AbstractVector{T}      : vector of inputs.
            - y::AbstractVector{T}      : vector of inputs.

        # Returns
            - alpha::Float64            : Scalar global minimum (not unique)
    """                            
                         
    # Find index where y isn't null and only keep these values
    k = findall(x -> x > 1e-14, y)
    y = y[k]
    x = x[k]
    
    # Return 0 if there is only 0
    if length(y) == 0
        return 0.0
    end

    # Calculate the breakpoints x/y
    S = []
    for i in eachindex(y)
        push!(S,x[i]/y[i])
    end

    # Sort the breakpoints and y following the same ordering
    Inds = sortperm(S)
    S    = S[Inds]
    y    = y[Inds]

    # Find smallest index k such that sum_(i->k)y_i >= valseuil/2 (half of the sum of all y)
    s        = cumsum(y)
    valseuil = last(s)/2
    k        = findfirst(x -> x >= valseuil,s)

    #return the corresponding breakpoints
    return max(S[k],1e-12)
end