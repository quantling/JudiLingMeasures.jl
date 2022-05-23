# JudiLingMeasures.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MariaHei.github.io/JudiLingMeasures.jl/dev)
[![Build Status](https://github.com/MariaHei/JudiLingMeasures.jl/workflows/CI/badge.svg)](https://github.com/MariaHei/JudiLingMeasures.jl/actions)

This is code for JudiLingMeasures. Most measures are based on R implementations in WpmWithLdl (Baayen et al., 2018) and [LdlConvFunctions](https://github.com/dosc91/LDLConvFunctions) (Schmitz, 2021) and the python implementation in (pyldl)[https://github.com/msaito8623/pyldl] (Saito, 2022) (but all errors are my own). The conceptual work behind this package is therefore very much an effort of many people (see [References](https://mariahei.github.io/JudiLingMeasures.jl/dev/index.html#References)). I have tried to acknowledge where each measure is used/introduced, but if I have missed anything, or you find any errors please let me know: maria dot heitmeier at uni dot tuebingen dot de.

You can find the documentation [here](https://mariahei.github.io/JudiLingMeasures.jl/dev/index.html).

PLEASE NOTE THAT THIS PACKAGE IS WORK IN PROGRESS. MAJOR CHANGES TO THE CODE ARE POSSIBLE AT ANY POINT AND NEW MEASURES ARE STILL BEING ADDED.

## Installation

```
using Pkg
Pkg.add("https://github.com/MariaHei/JudiLingMeasures.jl")
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

For an overview over all measures in this package and how to use them, please refer to the documentation, which can be found in `docs/build/index.html`.

For a comparison of measures in JudiLingMeasures and WpmWithLDL/[LDLConvFunctions](https://github.com/dosc91/LDLConvFunctions) see `notebooks/compare_JudiLingMeasures_with_WpmWithLdl_and_LDLConvFunctions.ipynb`.
