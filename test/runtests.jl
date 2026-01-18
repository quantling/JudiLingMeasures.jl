# [test/runtests.jl]
using JudiLingMeasures
using DataFrames
using StatsBase
using JudiLing
using LinearAlgebra
using Statistics
using Test
using Distances

!Sys.iswindows()
  using PythonCall
end

# Test scripts
include("test_helpers.jl")
include("test_measures.jl")
