# JudiLingMeasures.jl


This is code for JudiLingMeasures. Most measures are based on R implementations in WpmWithLdl (Baayen et al., 2018) and [LdlConvFunctions](https://github.com/dosc91/LDLConvFunctions) (Schmitz, 2021) and the python implementation in [pyldl](https://github.com/msaito8623/pyldl) (Saito, 2022) (but all errors are my own). The conceptual work behind this package is therefore very much an effort of many people (see [Bibliography](@ref)). I have tried to acknowledge where each measure is used/introduced, but if I have missed anything, or you find any errors please let me know: maria dot heitmeier at uni dot tuebingen dot de.

PLEASE NOTE THAT THIS PACKAGE IS WORK IN PROGRESS. MAJOR CHANGES TO THE CODE ARE POSSIBLE AT ANY POINT AND NEW MEASURES ARE STILL BEING ADDED.

## Installation

```
using Pkg
Pkg.add("https://github.com/MariaHei/JudiLingMeasures.jl")
```

Requires JudiLing 0.5.5. Update your JudiLing version by running

```
using Pkg
Pkg.update("JudiLing")
```

If this step does not work, i.e. the version of JudiLing is still not 0.5.5, refer to [this forum post](https://discourse.julialang.org/t/general-registry-delays-and-a-workaround/67537) for a workaround.

## How to use

For a demo of this package, please see `notebooks/measures_demo.ipynb`.

## Calculating measures in this package

The following gives an overview over all measures available in this package. For a closer description of the parameters, please refer to [Measures](@ref). All measures come with examples. In order to run them, first run the following piece of code, taken from the [Readme of the JudiLing package](https://github.com/MegamindHenry/JudiLing.jl). For a detailed explanation of this code please refer to the [JudiLing Readme](https://github.com/MegamindHenry/JudiLing.jl) and [documentation](https://megamindhenry.github.io/JudiLing.jl/stable/).

```
using JudiLing
using CSV # read csv files into dataframes
using DataFrames # parse data into dataframes
using JudiLingMeasures

# if you haven't downloaded this file already, get it here:
download("https://osf.io/2ejfu/download", "latin.csv")

latin =
    DataFrame(CSV.File(joinpath(@__DIR__, "latin.csv")));

cue_obj = JudiLing.make_cue_matrix(
    latin,
    grams = 3,
    target_col = :Word,
    tokenized = false,
    keep_sep = false
);

n_features = size(cue_obj.C, 2);
S = JudiLing.make_S_matrix(
    latin,
    ["Lexeme"],
    ["Person", "Number", "Tense", "Voice", "Mood"],
    ncol = n_features
);

G = JudiLing.make_transform_matrix(S, cue_obj.C);
F = JudiLing.make_transform_matrix(cue_obj.C, S);

Chat = S * G;
Shat = cue_obj.C * F;

A = cue_obj.A;
max_t = JudiLing.cal_max_timestep(latin, :Word);
```

Make sure that you set `check_gold_path=true`.
```
res_learn, gpi_learn, rpi_learn = JudiLing.learn_paths_rpi(
    latin,
    latin,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.f2i, # api changed in 0.3.1
    gold_ind = cue_obj.gold_ind,
    Shat_val = Shat,
    check_gold_path = true,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    sep_token = "_",
    keep_sep = false,
    target_col = :Word,
    issparse = :dense,
    verbose = false,
);
```

Almost all available measures can be simply computed with

```
all_measures = JudiLingMeasures.compute_all_measures_train(latin, # the data of interest
                                                     cue_obj, # the cue_obj of the training data
                                                     Chat, # the Chat of the data of interest
                                                     S, # the S matrix of the data of interest
                                                     Shat, # the Shat matrix of the data of interest
                                                     F, # the F matrix
                                                     G, # the G matrix
                                                     res_learn_train=res_learn, # the output of learn_paths for the data of interest
                                                     gpi_learn_train=gpi_learn, # the gpi_learn object of the data of interest
                                                     rpi_learn_train=rpi_learn); # the rpi_learn object of the data of interest
```

It's also possible to not compute measures based on the `learn_paths` algorithm:

```
all_measures = JudiLingMeasures.compute_all_measures_train(latin, # the data of interest
                                                     cue_obj, # the cue_obj of the training data
                                                     Chat, # the Chat of the training data
                                                     S, # the S matrix of the training data
                                                     Shat, # the Shat matrix of the training data
                                                     F, # the F matrix
                                                     G, # the G matrix); #
```

If `low_cost_measures_only` is set to `true`, only measures which are computationally relatively lean are computed.

The only measures not computed in `JudiLingMeasures.compute_all_measures_train` are those which return multiple values for each wordform. These are
- "Functional Load"
- "Semantic Support for Form" with `sum_supports=false`

It is also possible to compute measures for validation data, please see the `measures_demo.ipynb` notebook for details.

## Overview over all available measures

### Measures capturing comprehension (processing on the semantic side of the network)

#### Measures of semantic vector length/uncertainty/activation
- **L1Norm**

  Computes the L1-Norm (city-block distance) of the predicted semantic vectors $\hat{S}$:

  Example:
  ```
  JudiLingMeasures.L1Norm(Shat)
  ```

  Used in Schmitz et al. (2021), Stein and Plag (2021) (called Semantic Vector length in their paper), Saito (2022) (called VecLen)

- **L2Norm**

  Computes the L2-Norm (euclidean distance) of the predicted semantic vectors $\hat{S}$:

  Example:
  ```
  JudiLingMeasures.L2Norm(Shat)
  ```

  Used in Schmitz et al. (2021)


#### Measures of semantic neighbourhood

- **Density**

  Computes the average correlation/cosine similarity of each predicted semantic vector in $\hat{S}$ with the $n$ most correlated/closest semantic vectors in $S$:

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  correlation_density = JudiLingMeasures.density(cor_s, 10)

  cosine_sims = JudiLingMeasures.cosine_similarity(Shat, S)
  cosine_density = JudiLingMeasures.density(cosine_sim, 10)
  ```

  Used in Heitmeier et al. (2022) (called Semantic Density, based on Cosine Similarity), Schmitz et al. (2021), Stein and Plag (2021) (called Semantic Density, based on correlation)

- **ALC**

  Average Lexical Correlation. Computes the average correlation between each predicted semantic vector and all semantic vectors in $S$.

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  JudiLingMeasures.ALC(cor_s)
  ```

  Used in Schmitz et al. (2021), Chuang et al. (2020)

- **EDNN**

  Euclidean Distance Nearest Neighbour. Computes the euclidean distance between each predicted semantic vector and all semantic vectors in $S$ and returns for each predicted semantic vector the distance to the closest neighbour.

  Example:
  ```
  JudiLingMeasures.EDNN(Shat, S)
  ```

  Used in Schmitz et al. (2021), Chuang et al. (2020)

- **NNC**

  Nearest Neighbour Correlation. Computes the correlation between each predicted semantic vector and all semantic vectors in $S$ and returns for each predicted semantic vector the correlation to the closest neighbour.

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  JudiLingMeasures.NNC(cor_s)
  ```

  Used in Schmitz et al. (2021), Chuang et al. (2020)

- **Total Distance (F)**

  Summed Euclidean distances between predicted semantic vectors of trigrams in the target form.
  Code by Yu-Ying Chuang.

  Example:
  ```
  JudiLingMeasures.total_distance(cue_obj, F, :F)
  ```

  Used in Chuang et al. (to appear)

#### Measures of comprehension accuracy/uncertainty

- **TargetCorrelation**

  Correlation between each predicted semantic vector and its target semantic vector in $S$.

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  JudiLingMeasures.TargetCorrelation(cor_s)
  ```

  Used in Stein and Plag (2021) and Saito (2022) (but called PredAcc there)

- **Rank**

  Rank of the correlation with the target semantics among the correlations between the predicted semantic vector and all semantic vectors in $S$.

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  JudiLingMeasures.rank(cor_s)
  ```

- **Recognition**

  Whether a word form was correctly comprehended. Not currently implemented.

  NOT YET IMPLEMENTED

- **Comprehension Uncertainty**

  Sum of production of correlation/mse/cosine cosimilarity of shat with all vectors in S and the ranks of this correlation/mse/cosine similarity.

  Note: the current version of Comprehension Uncertainty is not completely tested against its original implementation in [pyldl](https://github.com/msaito8623/pyldl).

  Example:
  ```
  JudiLingMeasures.uncertainty(S, Shat, method="corr") # default
  JudiLingMeasures.uncertainty(S, Shat, method="mse")
  JudiLingMeasures.uncertainty(S, Shat, method="cosine")
  ```

  Used in Saito (2022).

- **Functional Load**

  Correlation/MSE of rows in F of triphones in word w and the target semantic vector of w.

  Note: the current version of Functional Load is not completely tested against its original implementation in [pyldl](https://github.com/msaito8623/pyldl).

  Example:
  ```
  JudiLingMeasures.functional_load(F, Shat, cue_obj, method="corr")
  JudiLingMeasures.functional_load(F, Shat, cue_obj, method="mse")
  ```

  Instead of returning the functional load for each cue in each word, a list of cues can also be specified. In this case it is assumed that cues are specified in the same order as the words they are to be compared to are specified in F and Shat.

  ```
  JudiLingMeasures.functional_load(F[:,1:6], Shat[1:6,:], cue_obj, cue_list = ["#vo", "#vo", "#vo","#vo","#vo","#vo"])
  JudiLingMeasures.functional_load(F[:,1:6], Shat[1:6,:], cue_obj, cue_list = ["#vo", "#vo", "#vo","#vo","#vo","#vo"], method="mse")
  ```

  Used in Saito (2022).

### Measures capturing production (processing on the form side of the network)

#### Measures of production accuracy/support/uncertainty for the predicted form

- **SCPP**

  The correlation between the predicted semantics of the word form produced by the path algorithm and the target semantics.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.SCPP(df, latin)
  ```

  Used in Chuang et al. (2020) (based on WpmWithLDL)

- **PathSum**

  The summed path supports for the highest supported predicted form, produced by the path algorithm. Path supports are taken from the $\hat{Y}$ matrices.

  Example:
  ```
  pred_df = JudiLing.write2df(rpi_learn)
  JudiLingMeasures.path_sum(pred_df)
  ```

  Used in Schmitz et al. (2021) (but based on WpmWithLDL)

- **TargetPathSum**

  The summed path supports for the target word form, produced by the path algorithm. Path supports are taken from the $\hat{Y}$ matrices.

  Example:
  ```
  JudiLingMeasures.target_path_sum(gpi_learn)
  ```
  Used in Chuang et al. (2022) (but called Triphone Support)

- **PathSumChat**

  The summed path supports for the highest supported predicted form, produced by the path algorithm. Path supports are taken from the $\hat{C}$ matrix.

  Example:
  ```
  JudiLingMeasures.path_sum_chat(res_learn, Chat)
  ```

- **C-Precision**

  Correlation between the predicted form vector and the target form vector.

  Example:
  ```
  JudiLingMeasures.c_precision(Chat, cue_obj.C)
  ```

  Used in Heitmeier et al. (2022), Gahl and Baayen (2022) (called Semantics to Form Mapping Precision)

- **L1Chat**

  L1-Norm of the predicted $\hat{c}$ vectors.

  Example:
  ```
  JudiLingMeasures.L1Norm(Chat)
  ```

  Used in Heitmeier et al. (2022)

- **Semantic Support for Form**

  Sum of activation of ngrams in the target wordform.

  Example:
  ```
  JudiLingMeasures.semantic_support_for_form(cue_obj, Chat)
  ```

  Instead of summing the activations, the function can also return the activation for each ngram:
  ```
  JudiLingMeasures.semantic_support_for_form(cue_obj, Chat, sum_supports=false)
  ```

  Used in Gahl and Baayen (2022) (unclear which package this was based on?)
  The activation of individual ngrams was used in Saito (2022).

#### Measures of production accuracy/support/uncertainty for the target form

- **Production Uncertainty**

  Sum of production of correlation/mse/cosine similarity of chat with all vectors in C and the ranks of this correlation/mse/cosine similarity.

  Note: the current version of Production Uncertainty is not completely tested against its original implementation in [pyldl](https://github.com/msaito8623/pyldl).

  Example:
  ```
  JudiLingMeasures.uncertainty(cue_obj.C, Chat, method="corr") # default
  JudiLingMeasures.uncertainty(cue_obj.C, Chat, method="mse")
  JudiLingMeasures.uncertainty(cue_obj.C, Chat, method="cosine")
  ```

  Used in Saito (2022)

- **Total Distance (G)**

  Summed Euclidean distances between predicted form vectors of trigrams in the target form.
  Code by Yu-Ying Chuang.

  Example:
  ```
  JudiLingMeasures.total_distance(cue_obj, G, :G)
  ```

  Used in Chuang et al. (to appear)

#### Measures of support for the predicted path, focusing on the path transitions and components of the path

- **LastSupport**

  The support for the last trigram of each target word in the Chat matrix.

  Example:
  ```
  JudiLingMeasures.last_support(cue_obj, Chat)
  ```

  Used in Schmitz et al. (2021) (called Support in their paper).

- **WithinPathEntropies**

  The entropy over path supports for the highest supported predicted form, produced by the path algorithm. Path supports are taken from the $\hat{Y}$ matrices.

  Example:
  ```
  pred_df = JudiLing.write2df(rpi_learn)
  JudiLingMeasures.within_path_entropies(pred_df)
  ```

- **MeanWordSupport**

  Summed path support divided by each word form's length. Path supports are taken from the $\hat{Y}$ matrices.

  Example:
  ```
  pred_df = JudiLing.write2df(rpi_learn)
  JudiLingMeasures.mean_word_support(res_learn, pred_df)
  ```

- **MeanWordSupportChat**

  Summed path support divided by each word form's length. Path supports are taken from the $\hat{C}$ matrix.

  Example:
  ```
  JudiLingMeasures.mean_word_support_chat(res_learn, Chat)
  ```

  Used in Stein and Plag (2021) (but based on WpmWithLDL)

- **lwlr**

  The ratio between the predicted form's length and its weakest support from the production algorithm. Supports taken from the $\hat{Y}$ matrices.

  Example:
  ```
  pred_df = JudiLing.write2df(rpi_learn)
  JudiLingMeasures.lwlr(res_learn, pred_df)
  ```

- **lwlrChat**

  The ratio between the predicted form's length and its weakest support. Supports taken from the $\hat{C}$ matrix.

  Example:
  ```
  JudiLingMeasures.lwlr_chat(res_learn, Chat)
  ```

#### Measures of support for competing forms

- **PathCounts**

  The number of candidates predicted by the path algorithm.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.PathCounts(df)
  ```

  Used in Schmitz et al. (2021) (but based on WpmWithLDL)

- **PathEntropiesChat**

  The entropy over the summed path supports for the candidate forms produced by the path algorithm. Path supports are taken from the $\hat{C}$ matrix.

  Example:
  ```
  JudiLingMeasures.path_entropies_chat(res_learn, Chat)
  ```

  Used in Schmitz et al. (2021) (but based on WpmWithLDL), Stein and Plag (2021) (but based on WpmWithLDL)

- **PathEntropiesSCP**

  The entropy over the semantic supports for the candidate forms produced by the path algorithm.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.path_entropies_scp(df)
  ```

- **ALDC**

  Average Levenstein Distance of Candidates. Average of Levenshtein distance between each predicted word form candidate and the target word form.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.ALDC(df)
  ```

  Used in Schmitz et al. (2021), Chuang et al. (2020) (both based on WpmWithLDL)

## Bibliography

Baayen, R. H., Chuang, Y. Y., and Blevins, J. P. (2018). Inflectional morphology with linear mappings. The Mental Lexicon, 13 (2), 232-270.

Chuang, Y. Y., Kang, M., Luo, X. F. and Baayen, R. H. (to appear). Vector Space Morphology with Linear Discriminative Learning. In Crepaldi, D. (Ed.) Linguistic morphology in the mind and brain.

Chuang, Y-Y., Vollmer, M-l., Shafaei-Bajestan, E., Gahl, S., Hendrix, P., and Baayen, R. H. (2020). The processing of pseudoword form and meaning in production and comprehension: A computational modeling approach using Linear Discriminative Learning. Behavior Research Methods, 1-51.

Gahl, S., and Baayen, R. H. (2022). Time and thyme again: Connecting spoken word duration to models of the mental lexicon. OSF, January 22, 1-41.

Saito, Motoki (2022): pyldl - Linear Discriminative Learning in Python. URL: https://github.com/msaito8623/pyldl

Schmitz, Dominic. (2021). LDLConvFunctions: Functions for measure computation, extraction, and other handy stuff. R package version 1.2.0.1. URL: https://github.com/dosc91/LDLConvFunctions

Schmitz, D., Plag, I., Baer-Henney, D., & Stein, S. D. (2021). Durational differences of word-final/s/emerge from the lexicon: Modelling morpho-phonetic effects in pseudowords with linear discriminative learning. Frontiers in psychology, 12.

Stein, S. D., & Plag, I. (2021). Morpho-phonetic effects in speech production: Modeling the acoustic duration of English derived words with linear discriminative learning. Frontiers in Psychology, 12.
