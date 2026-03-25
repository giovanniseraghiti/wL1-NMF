#Evaluate the wL1-NMF (and L1-NMF for lambda=1) loss
function wl1_norm_loss(X, W, H, lambda)
    WH = W * H
    n, m = size(X)
    total_errors = 0.0
    
    for i in 1:n
        for j in 1:m
            if X[i, j] == 0
                total_errors += lambda * abs(WH[i, j])
            else
                total_errors += abs(X[i, j] - WH[i, j])
            end
        end
    end

    return total_errors / sum(X)
end

#Evaluate the Frobenius loss
function l2_norm_loss(X, W, Ht)
    return (sum(X.*X) - 2*sum(W.*(X*Ht)) + sum((W'*W).*(Ht'*Ht)))/sum(X)
end

#Evaluate the L21 loss
function l21_norm_loss(X, W, H)
    return (sum(sqrt.(sum((X-W*H).^2,dims=1))))/sum(X)
end

#Evaluate the KL loss
function KL_norm_loss(X, W, H)
    eps=1e-16;
     WH = W * H .+ eps
    return sum(X .* log.(X ./ WH .+ eps) .- X .+ WH)/sum(X)
end

# Scale W and H so that columns/rows have the same norm, that is,  ||W(:,k)|| = ||H(k,:)|| for all k. 
function rescale(
    W::AbstractMatrix{T},
    H::AbstractMatrix{T}) where T <: AbstractFloat
    r, = size(H)
    d=zeros(r)
    for k = 1 : r
        normW=norm(W[:,k],1); normH=norm(H[k,:],1)
        W[:,k] = W[:,k]./sqrt(normW).*sqrt(normH);
        d[k] = sqrt(normW)/sqrt(normH); 
        H[k,:] = H[k,:].*d[k];
    end
    return W, H
end

#Finds the indices per columns in X corresponding to nonzero entries
function find_cols_null(X::AbstractMatrix{T}) where T <: AbstractFloat
    cols_null = []
    _, n = size(X)

    for q in 1:n
        indices = findall(x -> x > 0, X[:, q])
        if typeof(indices) != Vector{Int}
            indices = [indices]
        end
        push!(cols_null, indices)
    end
    
    return cols_null
end

#Display images if needed
function affichage(trueH, nrows, ncols, filename;
    bw=true, nbimgperrow=0, notebook=false)
    # Copy matrix to avoid changing the original one
    H = copy(trueH)
    if sum(H) != 0 && maximum(H)>1 normalize!(H,Inf) end
        # Get  dimensions and check them
        r, n = size(H)
        nrows*ncols == n || error("The dimensions of the image don't match the parameters")

        # Init array of subimages
        subimages = []

        # Create one subimage per material
        for row in eachrow(H)
            # Normalize so that values are in [0, 1]
            # max = maximum(row)
            # if max != 0
            #     row ./= max
            # end
            # Create the subimage
            img = reshape(row, nrows, ncols)
            # Add borders (bottom and right)
            img = hcat(img, ones(nrows))
            img = vcat(img, ones(ncols+1)')
            # Revert black and white if needed
            if bw
                img = ones(size(img)) - img
            end
            # Save it
            push!(subimages, img)
        end

    # If number of subimages per row is not defined, set it to r (all in one line)
    if nbimgperrow == 0
        nbimgperrow = r
    end

    # Compute number of lines to display
    nblines, remainder = divrem(r, nbimgperrow)

    # If the number of subimages is not a multiple of nbimgperrow, add white subimages
    if remainder != 0
        nblines += 1
        for i in 1:(nbimgperrow - remainder)
            push!(subimages, ones(nrows+1, ncols+1))
        end
    end

    # Build lines
    lines = []
    for l in 1:nblines
        push!(lines, hcat(subimages[(l-1)*nbimgperrow+1:l*nbimgperrow]...))
    end

    # Concatenate lines
    imgarr = vcat(lines...)

    # Remove last column and last line (useless borders)
    imgarr = imgarr[1:end-1,1:end-1]

    # Save image
    # imgview = colorview(Gray, imgarr)
    imgview = imgarr
    PNGFiles.save(filename, imgview)
    if notebook
        return imgview
    end
end

#Interpolating function needed for plots
function interpolate_to_length(x::Vector{Any}, N::Int64)
    # coordinate originali (da 1 a lunghezza originale)
    orig_x = range(1, length(x), length=length(x))
    itp = LinearInterpolation(orig_x, x)

    # nuove coordinate uniformi da 1 a lunghezza originale
    new_x = range(1, length(x), length=N)
    return itp.(new_x)
end

#Preprocess function that make every column of X with maximu element 1 (for topic modeling)
function preprossessing!(X)
    m,r = size(X)
    for p in 1:r
        col_max = maximum(X[:, p])
        #col_max = sum(X[:, p])
        if col_max > 0
            X[:, p] ./= col_max
        end
    end
    return X
end

