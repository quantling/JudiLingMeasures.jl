module JudiLingMeasures

using StatsBase
using LinearAlgebra
using Statistics
using Distances
using StringDistances
using JudiLing
using DataFrames

greet() = print("Hello World!")

include("measures.jl")
include("helpers.jl")

end # module
