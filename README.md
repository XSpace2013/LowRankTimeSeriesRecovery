# Source code release for *Nonlinear Sequence Data Embedding by Monotone Variational Inequality*

Conference paper for ICLR 2025 [Link](https://arxiv.org/abs/2406.06894).

J. Y. Zhou and Y. Xie, “Nonlinear sequence embedding by monotone variational inequality,” in *The Thirteenth International Conference on Learning Representations*, 2025.

## Contents

This directory contains the following datasets and Julia source code files:

- `LowRankTimeSeriesRecovery.jl` contains datastructures and algorithms implementing the timeseries VI Field, Operators A_t and A_t^\*, Nuclear Projection, and Extragradient Method with Backtracing.
- `Example.jl` contains some basic usage examples using `AR (linear)`, and `Lang (sigmoid)` ([Caroll](https://www.gutenberg.org/ebooks/author/7)+[arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)) data.
- `./Data` contains the following:
  - `./Data/AR_Mixed` Mixed AR Sequence data.
  - `./Data/Lang` Encoded Natural Language data (in order: 228 from Alice's Adventures in Wonderland, 316 from Through the Looking Glass, 600 from arXiv).

## Dependencies

`Random`, `NPZ`, `LinearAlgebra`, and `KrylovKit`.

### Selected Structures and Routines

`MultivariateTimeSeriesData(X::Array{Float64,3}, d::Integer) -> MultivariateTimeSeriesData`

Construct `MultivariateTimeSeriesData` from a 3D array of time series data.

##### Arguments

- `X::Array{Float64,3}`: A 3D array containing time series observations with dimensions `(N, T, c)`, where:
  - `N`: Number of trajectories.
  - `T`: Number of samples per trajectory.
  - `c`: Number of channels per trajectory.
- `d::Integer`: The order of the time series.

`MultivariateTimeSeriesEmpiricalMonotoneVI(X::Array{Float64,3}, d::Integer, f!) -> MultivariateTimeSeriesEmpiricalMonotoneVI`

Construct a `MultivariateTimeSeriesEmpiricalMonotoneVI` from a time series data array, lookback `d` and a link function `f!`.

##### Arguments

- `X::Array{Float64,3}`: A 3D array containing time series observations with dimensions `(N, T, c)`, where:
  - `N`: Number of trajectories.
  - `T`: Number of samples per trajectory.
  - `c`: Number of channels per trajectory.
- `d::Integer`: The order of the time series.
- `f!`: In place link function

`(self::MultivariateTimeSeriesEmpiricalMonotoneVI)(params::Vector{Float64}, t::Union{Vector{Int64},Int64,Nothing})`

Applies the `MultivariateTimeSeriesEmpiricalMonotoneVI` model to a given parameter vector.

##### Arguments

- `params::Vector{Float64}`: A vector of parameters to be processed by the model.
- `t::Union{Vector{Int64}, Int64, Nothing}`: Specifies the time index selection:
  - If `t` is `nothing`, the function averages across all time steps.
  - If `t` is an integer, it returns the result for that specific observation.
  - If `t` is a vector of indices, it averages across the selected time indices.

`ExtraGradientNuclearBallMonotoneVI`
    ExtraGradientNuclearBallMonotoneVI(
        VI::MultivariateTimeSeriesEmpiricalMonotoneVI, λ::Float64;
        nBatches::Int64=1)

Performs an extra-gradient method for solving variational inequalities (VI) on the nuclear ball constraint.

##### Arguments

- `VI::MultivariateTimeSeriesEmpiricalMonotoneVI`: The variational inequality operator.
- `λ::Float64`: The nuclear norm constraint.
- `nBatches::Int64=1`: Number of batches for stochastic optimization.

`ProjNuc(A::AbstractMatrix{T}, λ::Float64; k0::Int=10, g::Float64=1.00) where {T} -> AbstractMatrix{T}`

Project the matrix `A` onto the λ-nuclear norm ball by iteratively computing singular values
and vectors using an exponential scheduling rule defined by `k0` and `g`. The algorithm terminates
the singular value decomposition once sufficient values have been computed.

##### Arguments

- `A::AbstractMatrix{T}`: The matrix to be projected.
- `λ::Float64`: The nuclear norm constraint.
- `k0::Int=10`: Initial number of singular values to compute per iteration.
- `g::Float64=1.05`: Growth factor for dynamically adjusting `k` values to compute.

`MatrixSVDIterator(A::AbstractMatrix{T}; k0::Int=10, g::Float64=1.00) where {T} -> MatrixSVDIterator{T}`

Create a `MatrixSVDIterator` for performing iterative singular value decomposition.

##### Arguments

- `A::AbstractMatrix{T}`: The matrix for which the SVD will be computed iteratively.
- `k0::Int=10`: Initial number of singular values to compute per iteration.
- `g::Float64=1.00`: Growth factor for adjusting `k` dynamically.

##### Example

```julia
    A = randn(100, 50)
    iter = MatrixSVDIterator(A)

    for sval in iter
        println("Singular value: ", sval)
    end
```
