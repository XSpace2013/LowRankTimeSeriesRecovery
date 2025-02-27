include("LowRankTimeSeriesRecovery.jl")
using .LowRankTimeSeriesRecovery

using NPZ
using LinearAlgebra

## Example Using the Time Series Data

Ts_Data = NPZ.npzread("./Data/AR/AR(15).Decayed.Pos.Gau.D-0.npy")[1:500,:]
Ts_Data = reshape(Ts_Data, size(Ts_Data)..., 1)
Ts_TimeSeries = LowRankTimeSeriesRecovery.MultivariateTimeSeriesData(Ts_Data, 15)
Ts_Link = LowRankTimeSeriesRecovery.Id!

Ts_VI = LowRankTimeSeriesRecovery.MultivariateTimeSeriesEmpiricalMonotoneVI(Ts_Link, Ts_TimeSeries)

TS_λ = 7.0

TS_Solution = LowRankTimeSeriesRecovery.ExtraGradientNuclearBallMonotoneVI(Ts_VI, TS_λ, TERMINATION_THRESHOLD=1e-3,ITER_MAX_OUTER=10)

LowRankTimeSeriesRecovery.serialize(TS_Solution, "TS_Soln.$(TS_λ).npz")

## Natural Language Example to Reproduce Figure 2
Lang_fp = "./Data/Lang/Arxiv/X.Caroll.Arxiv.npz"
Lang_λ = 10.0
Lang_fp_out = "./Lang_Soln$(Lang_λ).npz"

Lang_TimeSeries = LowRankTimeSeriesRecovery.deseralizeMultivariateTimeSeriesData(Lang_fp)
Lang_Softmax = LowRankTimeSeriesRecovery.multichannelSoftmax!(Lang_TimeSeries.c)
Lang_VI = LowRankTimeSeriesRecovery.MultivariateTimeSeriesEmpiricalMonotoneVI(Lang_Softmax, Lang_TimeSeries)

Lang_Solution = LowRankTimeSeriesRecovery.ExtraGradientNuclearBallMonotoneVI(Lang_VI, λ, nBatches = 3)

LowRankTimeSeriesRecovery.serialize(Lang_Solution, Lang_fp_out)
