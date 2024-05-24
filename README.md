# JudiLingMeasures.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://quantling.github.io/JudiLingMeasures.jl/dev)
[![Build Status](https://github.com/quantling/JudiLingMeasures.jl/workflows/CI/badge.svg)](https://github.com/quantling/JudiLingMeasures.jl/actions)

JudiLingMeasures enables easy calculation of measures in Discriminative Lexicon Models developed with [JudiLing](https://github.com/quantling/JudiLing.jl) (Luo, Heitmeier, Chuang and Baayen, 2024).

Most measures in JudiLingMeasures are based on R implementations in WpmWithLdl (Baayen et al., 2018) and [LdlConvFunctions](https://github.com/dosc91/LDLConvFunctions) (Schmitz, 2021) and the Python implementation in [pyldl](https://github.com/msaito8623/pyldl) (Saito, 2022) (but all errors are my own). The conceptual work behind this package is therefore very much an effort of many people (see [Bibliography](https://quantling.github.io/JudiLingMeasures.jl/dev/index.html#Bibliography)). I have tried to acknowledge where each measure is used/introduced, but if I have missed anything, or you find any errors please let me know: maria dot heitmeier at uni dot tuebingen dot de.

You can find the documentation [here](https://quantling.github.io/JudiLingMeasures.jl/dev/index.html).

## Installation

```
using Pkg
Pkg.add("https://github.com/quantling/JudiLingMeasures.jl")
```

Note: Requires JudiLing 0.5.5. Update your JudiLing version by running

```
using Pkg
Pkg.update("JudiLing")
```

If this step does not work, i.e. the version of JudiLing is still not 0.5.5, refer to [this forum post](https://discourse.julialang.org/t/general-registry-delays-and-a-workaround/67537) for a workaround.

## How to use

For a demo of this package, please see `notebooks/measures_demo.ipynb`.

## Measures in this package

For an overview over all measures in this package and how to use them, please refer to the documentation, which can be found [here](https://quantling.github.io/JudiLingMeasures.jl/dev/index.html).

For a comparison of measures in JudiLingMeasures and WpmWithLDL, [LDLConvFunctions](https://github.com/dosc91/LDLConvFunctions) and [pyldl](https://github.com/msaito8623/pyldl) see `notebooks/compare_JudiLingMeasures_with_WpmWithLdl_LDLConvFunctions_pyldl.ipynb`.
