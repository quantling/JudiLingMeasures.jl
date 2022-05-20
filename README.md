# JudiLingMeasures.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MariaHei.github.io/JudiLingMeasures.jl/dev)
[![Build Status](https://github.com/MariaHei/JudiLingMeasures.jl/workflows/CI/badge.svg)](https://github.com/MariaHei/JudiLingMeasures.jl/actions)

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
