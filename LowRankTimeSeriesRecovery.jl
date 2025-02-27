module LowRankTimeSeriesRecovery
    using Random
    using NPZ
    using LinearAlgebra
    using KrylovKit
# ========================================== Helper Functions ========================================== #
    """
        generateBatches(T::Int, k::Int) -> Vector{Vector{Int}}

    Splits a range of indices from `1` to `T` into `k` random batches.

    # Arguments
    - `T::Int`: Total number of elements to split.
    - `k::Int`: Number of batches.

    # Returns
    - A vector of `k` vectors, where each inner vector contains indices belonging to a batch.
    """
    function generateBatches(T::Int, k::Int)
        # Generate a random permutation of indices from 1 to T-d
        indices = randperm(T)

        # Calculate the size of each batch
        batch_size = div(length(indices), k)

        # Initialize an array to store batches
        batches = Vector{Vector{Int}}(undef, k)

        # Split indices into batches
        for i in 1:k
            start_idx = (i - 1) * batch_size + 1
            end_idx = min(i * batch_size, length(indices))
            batches[i] = indices[start_idx:end_idx]
        end

        return batches
    end

    # ========================================== Link Functions ========================================== #
    """
    Link Functions for Monotone VI

    These link functions act inplace

    # Reference:
        Signal Recovery by Stochastic Optimization
        A. Juditsky & A. Nemirovski
    """

    function Id!(x::Vector{Float64})
        x .= x
    end

    function reLU!(x::Vector{Float64})
        x[x.<0] .= 0
    end

    function tanh!(x::Vector{Float64})
        x .= tanh.(x)
    end

    function sigmoid!(x::Vector{Float64})
        x = exp.(x) ./ (1 .+ exp.(x))
    end

    function clip!(x::Vector{Float64})
        x[x.>1] .= 1
        x[x.<0] .= 0
    end

    function log!(x::Vector{Float64})
        x .= log.(x)
    end


    # From Flux.jl
    fast_maximum(x::AbstractArray{T}; dims) where {T} = @fastmath reduce(max, x; dims, init=float(T)(-Inf))


    # From Flux.jl
    softmax!(x::AbstractArray; dims=1) = softmax!(x, x; dims)

    # From Flux.jl
    function softmax!(out::AbstractArray{T}, x::AbstractArray; dims=1) where {T}
        max_ = fast_maximum(x; dims)
        if all(isfinite, max_)
            @fastmath out .= exp.(x .- max_)
        else
            _zero, _one, _inf = T(0), T(1), T(Inf)
            @fastmath @. out = ifelse(isequal(max_, _inf), ifelse(isequal(x, _inf), _one, _zero), exp(x - max_))
        end
        tmp = dims isa Colon ? sum(out) : sum!(max_, out)
        out ./= tmp
    end

    """
        struct multichannelSoftmax!

    Softmax across `c` channels in a vector of values.

    # Fields
    - `c::Integer`: The number of channels per softmax computation.
    """
    struct multichannelSoftmax!
        c::Integer
    end

    """
        (self::multichannelSoftmax!)(x::Vector{Float64})

    Applies in-place softmax operation to `x` accroding to `c` channels.
    - `x` is divided into `c`-sized segments.
    - (`softmax!`) applied to each segment.

    # Arguments
    - `x::Vector{Float64}`: A vector containing the input values to which softmax is applied.
    """
    function (self::multichannelSoftmax!)(x::Vector{Float64})
        for i in 1:length(x)÷self.c
            softmax!(@view x[(i-1)*self.c+1:(i)*self.c])
        end
    end

    # ========================================== Datastructures for Timeseries VI============================== #
    """
        MultivariateTimeSeriesData(X, d)

    # Fields
    - `X::Array{Float64,3}`: Observations of the time series, stored as a 3D array with dimensions `(N, T, c)`, where:
      - `N`: Number of trajectories.
      - `T`: Number of samples per trajectory.
      - `c`: Number of channels in each trajectory.
    - `d::Integer`: The order of the time series.
    - `T::Integer`: Number of samples per trajectory.
    - `N::Integer`: Number of trajectories.
    - `c::Integer`: Number of channels in each trajectory.
    """
    struct MultivariateTimeSeriesData
        X::Array{Float64,3}
        d::Integer
        T::Integer
        N::Integer
        c::Integer
    end

    """
        MultivariateTimeSeriesData(X::Array{Float64,3}, d::Integer) -> MultivariateTimeSeriesData

    Construct `MultivariateTimeSeriesData` from a 3D array of time series data.

    # Arguments
    - `X::Array{Float64,3}`: A 3D containing time series observations with dimensions `(N, T, c)`, where:
      - `N`: Number of trajectories.
      - `T`: Number of samples per trajectory.
      - `c`: Number of channels per trajectory.
    - `d::Integer`: The order of the time series.
    """
    function MultivariateTimeSeriesData(X::Array{Float64,3}, d::Integer)
        N, T, c = size(X)
        return MultivariateTimeSeriesData(X, d, T, N, c)
    end

    struct MultivariateTimeSeriesStochasticGenerator
        X::Array{Float64,3}
        d::Integer
        T::Integer
        N::Integer
        c::Integer
    end

    function Base.show(io::IO, ::MIME"text/plain", s::MultivariateTimeSeriesData)
        print(io, "MultivariateTimeseries(N=$(s.N), T=$(s.T), c=$(s.c), d=$(s.d))")
    end

    # ========================================== Monotone VI: Vector Case ========================================== #

    struct MultivariateTimeSeriesEmpiricalMonotoneVI
        f!::Union{Function,multichannelSoftmax!}
        data::Union{MultivariateTimeSeriesData,MultivariateTimeSeriesStochasticGenerator}
    end


    """
        `MultivariateTimeSeriesEmpiricalMonotoneVI(X::Array{Float64,3}, d::Integer, f!) -> MultivariateTimeSeriesEmpiricalMonotoneVI`

    Construct a `MultivariateTimeSeriesEmpiricalMonotoneVI` from a time series data array, lookback `d` and a link function `f!`.

    # Arguments
    - `X::Array{Float64,3}`: A 3D array containing time series observations with dimensions `(N, T, c)`, where:
      - `N`: Number of trajectories.
      - `T`: Number of samples per trajectory.
      - `c`: Number of channels per trajectory.
    - `d::Integer`: The order of the time series.
    - `f!`: In place link function
    """
    function MultivariateTimeSeriesEmpiricalMonotoneVI(X::Array{Float64,3}, d::Integer, f!)
        myData = MultivariateTimeSeriesData(X, d)
        return MultivariateTimeSeriesEmpiricalMonotoneVI(f!, myData)
    end

    function MultivariateTimeSeriesEmpiricalMonotoneVI(X::MultivariateTimeSeriesStochasticGenerator, f!)
        return MultivariateTimeSeriesEmpiricalMonotoneVI(f!, X)
    end


    function Base.show(io::IO, ::MIME"text/plain", s::MultivariateTimeSeriesEmpiricalMonotoneVI)
        print(io, "MultivariateTimeSeriesEmpiricalMonotoneVI(f!=$(s.f!), N=$(s.data.N), T=$(s.data.T), c=$(s.data.c) | d=$(s.data.d))")
    end


    function applyEta_k_H!(self::Union{MultivariateTimeSeriesData,MultivariateTimeSeriesStochasticGenerator},
        ηHβ, β, k)
        """
        Apply η_k^H: ηHβ <- η_k^H(β).
        """

        N, T = size(self.X)
        d = self.d
        c = self.c
        # define an alternate indexer into β
        params = reshape(β, (N, c, (d * c + 1)))
        # define an alternate indexer into ηHβ
        preds = reshape(ηHβ, (c, N))

        for i in 1:N
            preds[:, i] = params[i, :, :] * [1; self.X[i, k:k+d-1, :][:]]
        end

        return ηHβ
    end

    function applyEta_k_H(self::Union{MultivariateTimeSeriesData,MultivariateTimeSeriesStochasticGenerator}, paramvec, k)
        N, T = size(self.X)
        d = self.d
        c = self.c

        predvec = zeros(c * N)

        applyEta_k_H!(self, predvec, paramvec, k)
        return predvec

    end

    function applyEta_k!(self::Union{MultivariateTimeSeriesData,MultivariateTimeSeriesStochasticGenerator}, ηpredvec, predvec, k)
        N, T = size(self.X)
        d = self.d
        c = self.c
        # define an alternate indexer into ηpred
        params = reshape(ηpredvec, (N, c, (d * c + 1)))

        preds = reshape(predvec, (c, N))

        for i in 1:N
            for j in 1:c
                params[i, j, :] = [1; self.X[i, k:k+d-1, :][:]] .* preds[:, i][j]
            end
        end
    end

    function applyEta_k(self::Union{MultivariateTimeSeriesData,MultivariateTimeSeriesStochasticGenerator}, predvec, k)
        N, T = size(self.X)
        d = self.d
        c = self.c

        ηpredvec = zeros(N * c * (d * c + 1))
        applyEta_k!(self, ηpredvec, predvec, k)
        return ηpredvec
    end

    function stochasticReshuffle!(data::MultivariateTimeSeriesData)
        """
        This function should do nothing if we are not taking random sub-batches
        """
        return
    end

    function MultivariateTimeSeriesStochasticGenerator(data, d, T)
        N = length(data)
        c = size(data[1])[end]
        return MultivariateTimeSeriesStochasticGenerator(data, d, T, N, c)
    end

    function stochasticReshuffle!(myStoch::MultivariateTimeSeriesStochasticGenerator)
        function getFirstDim(obj)
            return size(obj)[1]
        end

        lengths = map(getFirstDim, myStoch._underlying) .- (myStoch.T - 1)
        for n in 1:myStoch.N
            g = rand(1:lengths[n])
            myStoch.X[n, :, :] = myStoch._underlying[n][g:myStoch.T+g-1, :]
        end
    end

    """
        (self::MultivariateTimeSeriesEmpiricalMonotoneVI)(params::Vector{Float64}, t::Union{Vector{Int64},Int64,Nothing})

    Applies the `MultivariateTimeSeriesEmpiricalMonotoneVI` model to a given parameter vector.

    # Arguments
    - `params::Vector{Float64}`: A vector of parameters to be processed by the model.
    - `t::Union{Vector{Int64}, Int64, Nothing}`: Specifies the time index selection:
      - If `t` is `nothing`, the function averages across all time steps.
      - If `t` is an integer, it returns the result for that specific observation.
      - If `t` is a vector of indices, it averages across the selected time indices.
    """
    function (self::MultivariateTimeSeriesEmpiricalMonotoneVI)(params::Vector{Float64}, t::Union{Vector{Int64},Int64,Nothing})
        # (N, T, c)      data "vector"
        # (c, N)         prediction "vector"
        # (N, c, dc + 1) parameter "vector"

        # Equivalently B: (N, c,dc+1), so we can compute the nuclear norm
        # Equivalently β: (Nc(cd+1),), so we can build the monotone VI
        N, T = size(self.data.X)
        d = self.data.d
        c = self.data.c

        @debug "MultivariateTimeSeriesEmpiricalMonotoneVI: Setup"
        pred = zeros(N * c)
        imm = zeros(N * c * (d * c + 1))
        output = zeros(N * c * (d * c + 1))

        if isnothing(t)
            iter = 1:T-d
        elseif typeof(t) == Int64
            iter = (t,)
        else
            iter = t
        end

        # Shuffle the data, or do nothing.
        stochasticReshuffle!(self.data)

        @debug "MultivariateTimeSeriesEmpiricalMonotoneVI: $(iter)"

        for τ in t
            @debug "MultivariateTimeSeriesEmpiricalMonotoneVI: Step $(τ). applyEta_k_H!"
            # pred <- η^T z
            applyEta_k_H!(self.data, pred, params, τ)
            # pred <- f(η^T z)
            @debug "MultivariateTimeSeriesEmpiricalMonotoneVI: Step $(τ). Apply link function"
            self.f!(pred)
            # pred <- f(η^T z) - yt
            pred -= transpose(self.data.X[:, τ+d, :])[:]
            # imm <- η(f(η^T z) - yt)
            @debug "MultivariateTimeSeriesEmpiricalMonotoneVI: Step $(τ). applyEta_k!"
            applyEta_k!(self.data, imm, pred, τ)

            # Normalize by the number of samples.
            if isnothing(t)
                imm ./= (T - d)
            else
                imm ./= length(t)
            end
            output += imm
        end
        return output
    end

    # ========================================== Conversions ========================================== #
    function params2mat(params::AbstractArray{Float64,3}, VI::MultivariateTimeSeriesEmpiricalMonotoneVI)
        return params2mat(params, VI.data.N)
    end

    function params2mat(params::AbstractArray{Float64,3}, N)
        return transpose(reshape(params, N, :))
    end

    function mat2params(mat::AbstractArray{Float64,2}, VI::MultivariateTimeSeriesEmpiricalMonotoneVI)
        return reshape(transpose(mat), (VI.data.N, VI.data.c, VI.data.d * VI.data.c + 1))
    end

    function vec2params(vec::AbstractArray{Float64}, VI::MultivariateTimeSeriesEmpiricalMonotoneVI)
        return reshape(vec, VI.data.N, VI.data.c, (VI.data.d * VI.data.c + 1))
    end

    function vec2params(vec::AbstractArray{Float64}, N, c, d)
        return reshape(vec, N, c, (d * c + 1))
    end

    function params2vec(params::AbstractArray{Float64,3})
        return params[:]
    end

    function matrix2vec(mat::AbstractArray{Float64,2}, VI::MultivariateTimeSeriesEmpiricalMonotoneVI)
        return params2vec(mat2params(mat, VI))
    end

    function vec2mat(vec::AbstractArray{Float64}, VI::MultivariateTimeSeriesEmpiricalMonotoneVI)
        return params2mat(vec2params(vec, VI), VI)
    end

    # ========================================== Driver Codes ========================================== #
    function serialize(self::MultivariateTimeSeriesData, fp::String)
        npzwrite(fp, X=self.X, d=self.d, T=self.T, N=self.N, c=self.c)
    end

    function deseralizeMultivariateTimeSeriesData(fp::String; d=nothing)
        data = npzread(fp)
        if d == nothing
            d = data["d"]
        end

        return MultivariateTimeSeriesData(
            data["X"], d
        )
    end

    function deseralizeCategories(fp::String)
        data = npzread(fp)
        return data["cats"]
    end


    """
        mutable struct MatrixSVDIterator{T}

    Iterator for computing the Singular Value Decomposition (SVD) of a matrix tieratively

    # Fields
    - `A::AbstractMatrix{T}`: The original input matrix.
    - `σ::Vector{T}`: Singular values computed up to `upto`.
    - `lvecs::Vector{Vector{T}}`: Left singular vectors computed up to `upto`.
    - `rvecs::Vector{Vector{T}}`: Right singular vectors computed up to `upto`.
    - `upto::Int`: The number of computed singular values so far.
    - `A_deflate::Matrix{T}`: The deflated version of `A` after removing computed singular components.
    - `k0::Int`: Initial guess for the number of singular values to compute per iteration (default 10).
    - `g::Float64`: Ramp factor for adjusting `k` dynamically.
    - `k::Int`: Current number of singular values to compute per iteration.
    - `ncalls::Int`: Number of times the SVD algorithm has been invoked.
    """
    mutable struct MatrixSVDIterator{T}
        A::AbstractMatrix{T}  # The original matrix
        σ::Vector{T}  # Singular values computed up to `upto`
        lvecs::Vector{Vector{T}}  # Left singular vectors computed up to `upto`
        rvecs::Vector{Vector{T}}  # Right singular vectors computed up to `upto`
        upto::Int  # Up to which eigenvalue have we computed
        A_deflate::Matrix{T}  # Deflated A, up to `upto`
        k0::Int  # Guess parameter, default 10
        g::Float64 # Ramp factor for `k`
        k::Int  # Actual k
        ncalls::Int # Number of calls made to SVD algorithm
    end


    """
        MatrixSVDIterator(A::AbstractMatrix{T}; k0::Int=10, g::Float64=1.00) where {T} -> MatrixSVDIterator{T}

    Creates a `MatrixSVDIterator` for performing iterative singular value decomposition.

    # Arguments
    - `A::AbstractMatrix{T}`: The matrix for which the SVD will be computed iteratively.
    - `k0::Int=10`: Initial number of singular values to compute per iteration.
    - `g::Float64=1.00`: Growth factor for adjusting `k` dynamically.

    # Example
    ```julia
        A = randn(100, 50)
        iter = MatrixSVDIterator(A)

        for sval in iter
            println("Singular value: ", sval)
        end
    ```
    """

    function MatrixSVDIterator(A::AbstractMatrix{T}; k0::Int=10, g::Float64=1.00) where {T}
        MatrixSVDIterator(
            A,            # Original matrix
            T[],          # Singular values (empty initially)
            Vector{T}[],  # Left singular vectors (empty initially)
            Vector{T}[],  # Right singular vectors (empty initially)
            0,            # `upto` starts from 0 (no eigenvalues computed)
            copy(A),      # Initial deflated matrix is just A
            k0,           # User-defined or default `k`
            g,            # Ramp factor for `k`
            k0,           # Actual k used at current step (starts at k)
            0             # Number of calls made to SVD algorithm (none to start)
        )
    end

    function Base.iterate(iter::MatrixSVDIterator, state=0)
        # State is Union[0, (n_sval, sval), nothing]
        # The evolution is 0 -> (n_sval, sval) -> nothing. evolution to nothing occurs when 
        # the eigenvalue we want to compute
        n_sval = state + 1

        rnk_max = minimum(size(iter.A))
        if n_sval > rnk_max
            return nothing
        end

        sv = 0
        # Check if we have the singular values on hand
        if n_sval > length(iter.σ)
            # Clip number of values to compute
            nk = min(iter.k, rnk_max - length(iter.σ))
            # @info "Computing nk eigenvalues" nk
            # Compute the next singular values and vectors using Krylov Method
            vals, lvecs, rvecs, _ = svdsolve(iter.A_deflate, nk, :LR)
            iter.ncalls += 1

            append!(iter.σ, vals)
            append!(iter.lvecs, lvecs)
            append!(iter.rvecs, rvecs)

            # Deflate the matrix in O(rmn) time        
            iter.A_deflate = iter.A_deflate - (hcat(lvecs...) * Diagonal(vals) * hcat(rvecs...)')
            # Ramp
            iter.k = Int(ceil(iter.k * iter.g))
        end

        # Now gaurenteed to have the eigenvalue on hand
        return (iter.σ[n_sval], n_sval)
    end

    """
        ProjNuc(A::AbstractMatrix{T}, λ::Float64; k0::Int=10, g::Float64=1.00) where {T} -> AbstractMatrix{T}

    Project the matrix `A` onto the λ-nuclear norm ball by iteratively computing singular values 
    and vectors using an exponential scheduling rule defined by `k0` and `g`. The algorithm terminates 
    the singular value decomposition once sufficient values have been computed.

    # Arguments
    - `A::AbstractMatrix{T}`: The matrix to be projected.
    - `λ::Float64`: The nuclear norm constraint.
        - If λ is Inf, then return A itself with no projection.
    - `k0::Int=10`: Initial number of singular values to compute per iteration.
    - `g::Float64=1.05`: Growth factor for dynamically adjusting `k` values to compute.
    """
    function ProjNuc(A::AbstractMatrix{T}, λ::Float64; k0::Int=10, g::Float64=1.00) where {T}
        if λ == Inf
            @info "Skipping Projection"
            return A
        end

        # Get SVD Iterator
        svdValIter = MatrixSVDIterator(A, k0=k0, g=g)

        # This projection is only valid in the nuclear ball
        cs_j = 0.0
        j = 1
        s_mem = Float64[]
        for μ_j in svdValIter
            @assert μ_j >= 0.0

            cs_j += μ_j

            # Check if enough singular values have been found
            if μ_j * j <= cs_j - λ
                j -= 1
                cs_j -= μ_j

                θ = (cs_j - λ) / j
                @info "Singular Value Computation Terminated (nsvs, ncomputed)" j length(svdValIter.σ)
                return hcat((svdValIter.lvecs[1:j])...) * Diagonal(s_mem .- θ) * hcat((svdValIter.rvecs[1:j])...)'
            end

            push!(s_mem, μ_j)
            j += 1
        end

        # Inside nuclear ball
        @info "Matrix A is outside nuclear ball with |A|_*" sum(svdValIter.σ)
        return A
    end

    """
        struct IterationSolution

    # Fields
    - `params::Array{Float64,3}`: A 3D array containing the recovered parameters with dimensions `(N, c, cd+1)`, where:
      - `N`: Number of trajectories.
      - `c`: Number of channels per trajectory.
      - `d`: Autoregressive order of the model.
    - `lambda::Float64`: The nuclear norm constraint applied to the solution.
    - `nOuter::Int64`: The number of outer iterations performed.
    - `nBatches::Int64`: The number of batches used to divide the data.
    - `termthresh::Float64`: The termination threshold used for convergence.
    """
    struct IterationSolution
        params::Array{Float64,3}
        lambda::Float64
        nOuter::Int64
        nBatches::Int64
        termthresh::Float64
        runtime_log::Vector{Any}
    end

    function serialize(self::IterationSolution, fp::String)
        npzwrite(fp, params=self.params, lambda=self.lambda, nOuter=self.nOuter, nBatches=self.nBatches, termthresh=self.termthresh,
        runtime_log = hcat(map(t -> [Float64(t[1]), Float64(t[2]), Float64(t[3]), Float64(Int(t[4])), t[5]], self.runtime_log)...)'
        )
    end


    @enum TimeCodes vi proj 
    """
        ExtraGradientNuclearBallMonotoneVI(
            VI::MultivariateTimeSeriesEmpiricalMonotoneVI, λ::Float64;
            nBatches::Int64=1, 
            startingPoint::Union{Array{Float64,3},Nothing}=nothing,
            ITER_MAX_OUTER::Int64=100,
            LINE_SEARCH_ITER_MAX::Int64=100,
            TERMINATION_THRESHOLD=5e-3,
            α_hat=1.0,
            θ=0.75,
            ν=0.3,
        ) -> IterationSolution

    Performs an extra-gradient method for solving variational inequalities (VI) on the nuclear ball constraint.

    # Arguments
    - `VI::MultivariateTimeSeriesEmpiricalMonotoneVI`: The variational inequality operator.
    - `λ::Float64`: The nuclear norm constraint.
    - `nBatches::Int64=1`: Number of batches for stochastic optimization.
    - `startingPoint::Union{Array{Float64,3},Nothing}=nothing`: Optional starting point for the iteration.
    - `ITER_MAX_OUTER::Int64=100`: Maximum number of outer iterations.
    - `LINE_SEARCH_ITER_MAX::Int64=100`: Maximum number of iterations for line search.
    - `TERMINATION_THRESHOLD=5e-3`: Convergence threshold for stopping criterion.
    - `α_hat=1.0`: Initial step size, must be in the range `(0, 1]`.
    - `θ=0.75`: Step size decay parameter, must be in `(0,1]`.
    - `ν=0.3`: Line Search parameter, must be between `(0, 1/(2√2)]`.
    """
    function ExtraGradientNuclearBallMonotoneVI(VI::MultivariateTimeSeriesEmpiricalMonotoneVI, λ::Float64;
            nBatches::Int64=1, 
            startingPoint::Union{Array{Float64,3},Nothing}=nothing,
            ITER_MAX_OUTER::Int64 = 100,
            LINE_SEARCH_ITER_MAX::Int64 = 100,
            TERMINATION_THRESHOLD=5e-3,
            α_hat = 1.0,  # Between (0,1]
            θ = 0.5,     # Between (0,1]
            ν = 0.3,      # Between (0, 1/(2sqrt(2))]
            )

        c = VI.data.c
        d = VI.data.d
        N = VI.data.N
        T = VI.data.T

        if nBatches == 0 
            nBatches = T-d
        end

        x_cur_params = isnothing(startingPoint) ? zeros(N, c, (d*c+1)) : startingPoint;

        termcond = Inf

        @info "Running Extragradient Method with Line Search for Stochastic VI\n" *
              "nOuter: $(ITER_MAX_OUTER). nBatches: $(nBatches). " *
              "VIData: c=$(c), d=$(d), N=$(N), T=$(T)\n"

        k_used = 0
        runtime_log = []
    
        for k in 1:ITER_MAX_OUTER
            @info "Epoch $(k)/$(ITER_MAX_OUTER)."
            
            batches = generateBatches(T-d, nBatches)
            
            for (τ, batch) in enumerate(batches)
                @info "Begin Epoch $(k)/$(ITER_MAX_OUTER): Batch $(τ)/$(nBatches)"
                
                
                t = @elapsed (F_hat_x_cur_vec = VI(x_cur_params[:], batch))
                push!(runtime_log, (k,τ, 0, vi, t))
            
                F_hat_x_cur_params = vec2params(F_hat_x_cur_vec, VI);

                @info "Norm of the VI Field:" norm(F_hat_x_cur_vec)

                α_cand = 0.0            
                z_cand_α = nothing      
                @info "Begin Line Search"
                for l in 1:LINE_SEARCH_ITER_MAX
                    α_cand = α_hat * (θ^l)
                    @info "Line Search Iteration:" l , α_cand
                
                    t = @elapsed (z_cand_α = ProjNuc(params2mat(x_cur_params - (α_cand * F_hat_x_cur_params), VI),λ))
                    push!(runtime_log, (k,τ, l, proj, t))
                
                    t = @elapsed (F_hat_z_cand_α = VI(matrix2vec(z_cand_α, VI), batch))
                    push!(runtime_log, (k,τ, l, vi, t))

                    LHS = α_cand * norm(F_hat_z_cand_α - F_hat_x_cur_vec)
                    RHS = ν * norm((z_cand_α - params2mat(x_cur_params, VI))[:])

                    if LHS <= RHS
                        @info "Line Search Terminated:" α_cand
                        break
                    end
                    if l == LINE_SEARCH_ITER_MAX
                        @warn "Line Search Iteration Limit Hit"
                    end
                end
            
                t = @elapsed (F_hat_z_cand_α =  VI(matrix2vec(z_cand_α, VI), batch))
                push!(runtime_log, (k,τ, -1, vi, t))
            
                t = @elapsed (x_next_mat = ProjNuc(vec2mat(params2vec(x_cur_params) - α_cand * F_hat_z_cand_α, VI), λ))
                push!(runtime_log, (k,τ, -1, proj, t))

                termcond = norm((x_next_mat-params2mat(x_cur_params, VI))[:])/norm(x_next_mat[:])

                @info "|x_cur - x_next|/|x_next|" termcond

                x_cur_params = mat2params(x_next_mat, VI)

                @info "End Epoch $(k)/$(ITER_MAX_OUTER): Batch $(τ)/$(nBatches)"

                if termcond <= TERMINATION_THRESHOLD
                    @info "Termination Condition Hit"
                    break
                end
            end

            k_used = k
            # Break outer loop too if needed     
            if termcond <= TERMINATION_THRESHOLD
                break
            end
        end

        return IterationSolution(x_cur_params, λ, k_used, nBatches, TERMINATION_THRESHOLD, runtime_log)
    end
end # module LowRankTimeSeriesRecovery
