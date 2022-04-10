# JudiLingMeasures.jl


This is code for JudiLingMeasures.

Requires JudiLing 0.5.5. Update your JudiLing version by running

```
using Pkg
Pkg.update("JudiLing")
```

If this step does not work, i.e. the version of JudiLing is still not 0.5.5, refer to [this forum post](https://discourse.julialang.org/t/general-registry-delays-and-a-workaround/67537) for a workaround.

## How to use

For a demo of this package, please see `notebooks/measures_demo.ipynb`.

## Measures in this package

The following gives an overview over all measures available in this package. For a closer description of the parameters, please refer to the documentation provided with the code. All measures come with examples. In order to run them, first run the following piece of code, taken from the [Readme of the JudiLing package](https://github.com/MegamindHenry/JudiLing.jl). For a detailed explanation of this code please refer to the [JudiLing Readme](https://github.com/MegamindHenry/JudiLing.jl) and [documentation](https://megamindhenry.github.io/JudiLing.jl/stable/).

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

res_learn, gpi_learn, rpi_learn = JudiLingMeasures.learn_paths_rpi(
    latin,
    latin,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.f2i, # api changed in 0.3.1
    #gold_ind = cue_obj.gold_ind,
    #Shat_val = Shat,
    #check_gold_path = false,
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

### Measures capturing comprehension (processing on the semantic side of the network)

#### Measures of semantic vector length/uncertainty/activation
- **L1Norm**

  Computes the L1-Norm (city-block distance) of the predicted semantic vectors $\hat{S}$:

  Example:
  ```
  JudiLingMeasures.L1Norm(Shat)
  ```

  Used in Schmitz et al. (2021), Stein and Plag (2021) (called Semantic Vector length in their paper)

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

#### Measures of comprehension accuracy

- **TargetCorrelation**

  Correlation between each predicted semantic vector and its target semantic vector in $S$.

  Example:
  ```
  _, cor_s = JudiLing.eval_SC(Shat, S, R=true)
  JudiLingMeasures.TargetCorrelation(cor_s)
  ```

  Used in Stein and Plag (2021)

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
  JudiLingMeasures.PathSumChat(res_learn, Chat)
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

  Used in Gahl and Baayen (2022) (unclear which package this was based on?)


#### Measures of support for the predicted path, focusing on the path transitions and components of the path

- **LastSupport**

  The support for the last trigram of each target word in the Chat matrix.

  Example:
  ```
  JudiLingMeasures.last_support(latin, cue_obj, Chat)
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
  JudiLingMeasures.lwlr(pred_df)
  ```

- **lwlrChat**

  The ratio between the predicted form's length and its weakest support. Supports taken from the $\hat{C}$ matrix.

  Example:
  ```
  JudiLingMeasures.lwlrChat(res_learn, Chat)
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
  JudiLingMeasures.PathEntropiesChat(res_learn, Chat)
  ```

  Used in Schmitz et al. (2021) (but based on WpmWithLDL), Stein and Plag (2021) (but based on WpmWithLDL)

- **PathEntropiesSemanticSupport**

  The entropy over the semantic supports for the candidate forms produced by the path algorithm.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.path_entropies_semantic_support(df)
  ```

- **ALDC**

  Average Levenstein Distance of Candidates. Average of Levenshtein distance between each predicted word form candidate and the target word form.

  Example:
  ```
  df = JudiLingMeasures.get_res_learn_df(res_learn, latin, cue_obj, cue_obj)
  JudiLingMeasures.ALDC(df)
  ```

  Used in Schmitz et al. (2021), Chuang et al. (2020) (both based on WpmWithLDL)