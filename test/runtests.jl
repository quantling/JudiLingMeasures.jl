# [test/runtests.jl]
using JudiLingMeasures
using DataFrames
using StatsBase
using JudiLing
using LinearAlgebra
using Statistics
using Test
using Distances
using PyCall
import Conda

# if !haskey(Conda._installed_packages_dict(),"pandas")
#     Conda.add("pandas")
# end
# if !haskey(Conda._installed_packages_dict(),"numpy")
#     Conda.add("numpy")
# end
# if !haskey(Conda._installed_packages_dict(),"pyldl")
#     Conda.add("pyldl", channel="https://github.com/msaito8623/pyldl")
# end

# Test scripts
include("test_helpers.jl")
include("test_measures.jl")
