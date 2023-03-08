"""
    l1_rowwise(M::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L1 Norm of each row of `M`.
# Example
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> l1_rowwise(ma1)
3×1 Matrix{Int64}:
 6
 6
 6
```
"""
function l1_rowwise(M::Union{JudiLing.SparseMatrixCSC, Matrix})
    sum(abs.(M), dims=2)
end

"""
    l2_rowwise(M::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L2 Norm of each row of `M`.
# Example
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> l2_rowwise(ma1)
3×1 Matrix{Float64}:
 3.7416573867739413
 3.7416573867739413
 3.7416573867739413
```
"""
function l2_rowwise(M::Union{JudiLing.SparseMatrixCSC, Matrix})
    sqrt.(sum(M.^2, dims=2))
end

"""
    correlation_rowwise(S1::Union{JudiLing.SparseMatrixCSC, Matrix},
                        S2::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the correlation between each row of S1 with all rows in S2.
# Example
```jldoctest
julia> ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> correlation_rowwise(ma2, ma3)
4×4 Matrix{Float64}:
  0.662266   0.174078    0.816497  -0.905822
 -0.41762    0.29554    -0.990148   0.988623
 -0.308304   0.0368355  -0.863868   0.862538
  0.207514  -0.0909091  -0.426401   0.354787
```
"""
function correlation_rowwise(S1::Union{JudiLing.SparseMatrixCSC, Matrix},
                             S2::Union{JudiLing.SparseMatrixCSC, Matrix})
    if (size(S1,1) > 0) & (size(S1,2) > 0) & (size(S2,1) > 0) & (size(S2,2) > 0)
        cor(S1, S2, dims=2)
    else
        missing
    end
end

"""
    sem_density_mean(s_cor::Union{JudiLing.SparseMatrixCSC, Matrix},
                     n::Int)
Compute the average semantic density of the predicted semantic vector with its
n most correlated semantic neighbours.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
- `n::Int`: the number of highest semantic neighbours to take into account
# Example
```jldoctest
julia> ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> cor_s = correlation_rowwise(ma2, ma3)
julia> sem_density_mean(cor_s, 2)
4-element Vector{Float64}:
 0.7393813797301239
 0.6420816485652429
 0.4496869233815781
 0.281150888376636
```
"""
function sem_density_mean(s_cor::Union{JudiLing.SparseMatrixCSC, Matrix},
                          n::Int)
    if n > size(s_cor,2)
        throw(MethodError("n larger than the dimension of the semantic vectors"))
    end
    sems = Vector{Union{Missing, Float32}}(missing, size(s_cor,1))
    for i in 1:size(s_cor)[1]
        sems[i] = mean(s_cor[i,:][partialsortperm(s_cor[i, :], 1:n, rev=true)])
    end
    sems
end

"""
    mean_rowwise(S::Union{JudiLing.SparseMatrixCSC, Matrix})
Calculate the mean of each row in S.
# Examples
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> mean_rowwise(ma1)
3×1 Matrix{Float64}:
  2.0
 -2.0
  2.0
```
"""
function mean_rowwise(S::Union{JudiLing.SparseMatrixCSC, Matrix})
    if (size(S,1) > 0) & (size(S,2) > 0)
        map(mean, eachrow(S))
    else
        missing
    end
end

"""
    euclidean_distance_rowwise(Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
                             S::Union{JudiLing.SparseMatrixCSC, Matrix})
Calculate the pairwise Euclidean distances between all rows in Shat and S.

Throws error if missing is included in any of the arrays.
# Examples
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
julia> euclidean_distance_rowwise(ma1, ma4)
3×3 Matrix{Float64}:
 1.0     7.2111  1.0
 6.7082  2.0     7.28011
 1.0     7.2111  1.0
```
"""
function euclidean_distance_rowwise(Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
                                    S::Union{JudiLing.SparseMatrixCSC, Matrix})
    Distances.pairwise(Euclidean(), Shat', S', dims=2)
end

"""
    get_nearest_neighbour_eucl(eucl_sims::Matrix)
Get the nearest neighbour for each row in `eucl_sims`.
# Examples
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
julia> eucl_sims = euclidean_distance_array(ma1, ma4)
julia> get_nearest_neighbour_eucl(eucl_sims)
3-element Vector{Float64}:
 1.0
 2.0
 1.0
```
"""
function get_nearest_neighbour_eucl(eucl_sims::Matrix)
    lowest,_ = findmin(eucl_sims, dims=2)
    vec(lowest)
end

"""
    max_rowwise(S::Union{JudiLing.SparseMatrixCSC, Matrix})
Get the maximum of each row in S.
# Examples
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> max_rowwise(ma1)
3×1 Matrix{Int64}:
 3
 -1
 3
```
"""
function max_rowwise(S::Union{JudiLing.SparseMatrixCSC, Matrix})

    function findmax_custom(x)
        if any(ismissing.(x))
            missing
        else
            findmax(x)[1]
        end
    end
    cor_nnc = map(findmax_custom, eachrow(S));
    cor_nnc
end

"""
    count_rows(dat::DataFrame)
Get the number of rows in dat.
# Examples
```jldoctest
julia> dat = DataFrame("text"=>[1,2,3])
julia> count_rows(dat)
 3
```
"""
function count_rows(dat::Any)
    size(dat,1)
end

"""
    get_avg_levenshtein(targets::Array, preds::Array)
Get the average levenshtein distance between two lists of strings.
# Examples
```jldoctest
julia> targets = ["abc", "abc", "abc"]
julia> preds = ["abd", "abc", "ebd"]
julia> get_avg_levenshtein(targets, preds)
 1.0
```
"""
function get_avg_levenshtein(targets::Union{Array, SubArray}, preds::Union{Array, SubArray})
    if (length(targets) > 0) & (length(preds) > 0)
        mean(StringDistances.Levenshtein().(targets, preds))
    else
        missing
    end
end

"""
    entropy(ps::Union{Missing, Array, SubArray})
Compute the Shannon-Entropy of the values in ps bigger than 0.

Note: the result of this is entropy function is different to other entropy measures as a) the values are scaled between 0 and 1 first, and b) log2 instead of log is used
# Examples
```jldoctest
julia> ps = [0.1, 0.2, 0.9]
julia> entropy(ps)
1.0408520829727552
```
"""
function entropy(ps::Union{Missing, Array, SubArray})
    if ((!any(ismissing.(ps))) && (length(ps) > 0))
        ps = ps[ps.>0]
        if length(ps) == 0
            missing
        else
            p = ps./sum(ps)
            -sum(p.*log2.(p))
        end
    else
        missing
    end
end


"""
    get_res_learn_df(res_learn_val, data_val, cue_obj_train, cue_obj_val)
Wrapper for JudiLing.write2df for easier use.
"""
function get_res_learn_df(res_learn_val, data_val, cue_obj_train, cue_obj_val)
    JudiLing.write2df(res_learn_val,
    data_val,
    cue_obj_train,
    cue_obj_val,
    grams = cue_obj_val.grams,
    tokenized = cue_obj_val.tokenized,
    sep_token = cue_obj_val.sep_token,
    start_end_token = cue_obj_val.start_end_token,
    output_sep_token = "",
    path_sep_token = ":",
    target_col = cue_obj_val.target_col)
end


"""
    function make_measure_preparations(data_train, S_train, Shat_train,
                                       res_learn_train, cue_obj_train,
                                       rpi_learn_train)
Returns all additional objects needed for measure calculations if the data of interest is the training data.
# Arguments
- `data_train`: The data for which the measures are to be calculated (training data).
- `S_train`: The semantic matrix of the training data
- `Shat_train`: The predicted semantic matrix of the training data.
- `res_learn_train`: The first object return by the `learn_paths_rpi` algorithm for the training data.
- `cue_obj_train`: The cue object of the training data.
- `rpi_learn_train`: The second object return by the `learn_paths_rpi` algorithm for the training data.
# Returns
- `results::DataFrame`: A deepcopy of `data_train`.
- `cor_s::Matrix`: Correlation matrix between `Shat_train` and `S_train`.
- `df::DataFrame`: The output of `res_learn_train` (of the training data) in form of a dataframe
- `rpi_df::DataFrame`: Stores the path information about the predicted forms (from `learn_paths`), which is needed to compute things like PathSum, PathCounts and PathEntropies.
"""
function make_measure_preparations(data_train, S_train, Shat_train,
                                   res_learn_train, cue_obj_train,
                                   rpi_learn_train)
    # make a copy of the data to not change anything in there
    results = deepcopy(data_train)

    # compute the accuracy and the correlation matrix
    acc_comp, cor_s = JudiLing.eval_SC(Shat_train, S_train, R=true)

    # represent the res_learn object as a dataframe
    df = get_res_learn_df(res_learn_train, results, cue_obj_train, cue_obj_train)
    missing_ind = df.utterance[ismissing.(df[!,:pred])]
    df_sub = df[Not(ismissing.(df.pred)),:]


    rpi_df = JudiLing.write2df(rpi_learn_train)
    rpi_df[:, :pred] = Vector{Union{Missing, String}}(missing, size(data_train,1))
    rpi_df[Not(missing_ind),:pred] = df_sub[df_sub.isbest .== true,:pred]

    results, cor_s, df, rpi_df
end

"""
    function make_measure_preparations(data_val, S_train, S_val, Shat_val,
                                       res_learn_val, cue_obj_train, cue_obj_val,
                                       rpi_learn_val)
Returns all additional objects needed for measure calculations if the data of interest is the validation data.
# Arguments
- `data_val`: The data for which the measures are to be calculated (validation data).
- `S_train`: The semantic matrix of the training data
- `S_val`: The semantic matrix of the validation data
- `Shat_val`: The predicted semantic matrix of the validation data.
- `res_learn_val`: The first object return by the `learn_paths_rpi` algorithm for the validation data.
- `cue_obj_train`: The cue object of the training data.
- `cue_obj_val`: The cue object of the data of interest.
- `rpi_learn_val`: The second object return by the `learn_paths_rpi` algorithm for the validation data.
# Returns
- `results::DataFrame`: A deepcopy of `data_val`.
- `cor_s::Matrix`: Correlation matrix between `Shat_val` and `S_val`.
- `df::DataFrame`: The output of `res_learn_val` (of the validation data) in form of a dataframe
- `rpi_df::DataFrame`: Stores the path information about the predicted forms (from `learn_paths`), which is needed to compute things like PathSum, PathCounts and PathEntropies.
"""
function make_measure_preparations(data_val, S_train, S_val, Shat_val,
                                   res_learn_val, cue_obj_train, cue_obj_val,
                                   rpi_learn_val)
    # make a copy of the data to not change anything in there
    results = deepcopy(data_val)

    # compute the accuracy and the correlation matrix
    acc_comp, cor_s = JudiLing.eval_SC(Shat_val, S_val, S_train, R=true)

    # represent the res_learn object as a dataframe
    df = JudiLingMeasures.get_res_learn_df(res_learn_val, results, cue_obj_train, cue_obj_val)
    missing_ind = df.utterance[ismissing.(df[!,:pred])]
    df_sub = df[Not(ismissing.(df.pred)),:]


    rpi_df = JudiLing.write2df(rpi_learn_val)
    rpi_df[:, :pred] = Vector{Union{Missing, String}}(missing, size(data_val,1))
    rpi_df[Not(missing_ind),:pred] = df_sub[df_sub.isbest .== true,:pred]

    results, cor_s, df, rpi_df
end

"""
    function correlation_diagonal_rowwise(S1, S2)
Computes the pairwise correlation of each row in S1 and S2, i.e. only the
diagonal of the correlation matrix.
# Example
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
julia> correlation_diagonal_rowwise(ma1, ma4)
3-element Array{Float64,1}:
 0.8660254037844387
 0.9607689228305228
 0.9819805060619657
```
"""
function correlation_diagonal_rowwise(S1, S2)
    if size(S1) != size(S2)
        error("both matrices must have same size")
    else
        diag = zeros(Float64, size(S1)[1])
        for i in 1:size(S1)[1]
            diag[i] = cor(S1[i,:], S2[i,:])
        end
        diag
    end
end

"""
    cosine_similarity(s_hat_collection, S)
Calculate cosine similarity between all predicted and all target semantic vectors
# Example
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
julia> cosine_similarity(ma1, ma4)
3×3 Array{Float64,2}:
  0.979958  -0.857143   0.963624
 -0.979958   0.857143  -0.963624
  0.979958  -0.857143   0.963624
```
"""
function cosine_similarity(s_hat_collection, S)
    dists = Distances.pairwise(CosineDist(), s_hat_collection', S', dims=2)
    sims = - dists .+1
    sims
end

"""
    safe_sum(x::Array)
Compute sum of all elements of x, if x is empty return missing
# Example
```jldoctest
julia> safe_sum([])
missing
julia> safe_sum([1,2,3])
6
```
"""
function safe_sum(x::Union{Missing, Array})
    if ismissing(x)
        missing
    elseif length(x) > 0
        sum(x)
    else
        missing
    end
end

"""
    safe_length(x::Union{Missing, String})
Compute length of x, if x is missing return missing
# Example
```jldoctest
julia> safe_length(missing)
missing
julia> safe_length("abc")
3
```
"""
function safe_length(x::Union{Missing, String})
    if ismissing(x)
        missing
    else
        length(x)
    end
end

function indices_length(res)
    lengths = []
    for i = 1:size(res)[1]
        if isempty(res[i])
            append!(lengths, 0)
        else
            append!(lengths, length(res[i][1].ngrams_ind))
        end
    end
    lengths
end


"""
    function compute_all_measures_train(data_train::DataFrame,
                                        cue_obj_train::JudiLing.Cue_Matrix_Struct,
                                        Chat_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                        S_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                        Shat_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                        F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                        G_train::Union{JudiLing.SparseMatrixCSC, Matrix};
                                        res_learn_train::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing,
                                        gpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                        rpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                        sem_density_n::Int64=8,
                                        calculate_production_uncertainty::Bool=false,
                                        low_cost_measures_only::Bool=false)
Compute all measures currently available in JudiLingMeasures for the training data.
# Arguments
- `data_train::DataFrame`: The data for which measures should be calculated (the training data).
- `cue_obj_train::JudiLing.Cue_Matrix_Struct`: The cue object of the training data.
- `Chat_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: The Chat matrix of the training data.
- `S_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: The S matrix of the training data.
- `Shat_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: The Shat matrix of the training data.
- `F_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: Comprehension mapping matrix for the training data.
- `G_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: Production mapping matrix for the training data.
- `res_learn_train::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing`: The first output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `gpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing`: The second output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `rpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing`: The third output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `sem_density_n::Int64=8`: Number of neighbours to take into account in Semantic Density measure.
- `calculate_production_uncertainty`: "Production Uncertainty" is computationally very heavy for large C matrices, therefore its computation is turned off by default.
- `low_cost_measures_only::Bool=false`: Only compute measures which are not computationally heavy. Recommended for very large datasets.
# Returns
- `results::DataFrame`: A dataframe with all information in `data_train` plus all the computed measures.
"""
function compute_all_measures_train(data_train::DataFrame,
                                    cue_obj_train::JudiLing.Cue_Matrix_Struct,
                                    Chat_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                    S_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                    Shat_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                    F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                    G_train::Union{JudiLing.SparseMatrixCSC, Matrix};
                                    res_learn_train::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing,
                                    gpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                    rpi_learn_train::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                    sem_density_n::Int64=8,
                                    calculate_production_uncertainty::Bool=false,
                                    low_cost_measures_only::Bool=false)
    # MAKE PREPARATIONS

    # generate additional objects for the measures such as
    # - results: copy of data_val for storing the measures in
    # - cor_s: the correlation matrix between Shat and S
    # - df: DataFrame of res_learn, the output of learn_paths
    # - pred_df: DataFrame with path supports for the predicted forms produced by learn_paths
    if (!ismissing(res_learn_train) && !ismissing(gpi_learn_train) && !ismissing(rpi_learn_train))
        results, cor_s, df, pred_df = make_measure_preparations(data_train, S_train, Shat_train,
                                        res_learn_train, cue_obj_train, rpi_learn_train)
    else
        results = deepcopy(data_train)

        # compute the accuracy and the correlation matrix
        acc_comp, cor_s = JudiLing.eval_SC(Shat_train, S_train, R=true)
    end

    # CALCULATE MEASURES

    # vector length/activation/uncertainty
    results[!,"L1Shat"] = L1Norm(Shat_train)
    results[!,"L2Shat"] = L2Norm(Shat_train)

    # semantic neighbourhood
    results[!,"SemanticDensity"] = density(cor_s, n=sem_density_n)
    results[!,"ALC"] = ALC(cor_s)
    results[!,"EDNN"] = EDNN(Shat_train, S_train)
    results[!,"NNC"] = NNC(cor_s)

    if !low_cost_measures_only
        results[!,"DistanceTravelledF"] = total_distance(cue_obj_train, F_train, :F)
    end

    # comprehension accuracy
    results[!,"TargetCorrelation"] = target_correlation(cor_s)
    results[!,"rank"] = rank(cor_s)
    results[!,"recognition"] = recognition(data_train)

    if !low_cost_measures_only
        results[!,"ComprehensionUncertainty"] = vec(uncertainty(S_train, Shat_train, method="cosine"))
    end

    # Measures of production accuracy/support/uncertainty for the target form
    if calculate_production_uncertainty && !low_cost_measures_only
        results[!,"ProductionUncertainty"] = vec(uncertainty(cue_obj_train.C, Chat_train, method="cosine"))
    end
    if !low_cost_measures_only
        results[!,"DistanceTravelledG"] = total_distance(cue_obj_train, G_train, :G)
    end

    # production accuracy/support/uncertainty for the predicted form
    results[!,"C-Precision"] = c_precision(Chat_train, cue_obj_train.C)
    results[!,"L1Chat"] = L1Norm(Chat_train)
    results[!,"SemanticSupportForForm"] = semantic_support_for_form(cue_obj_train, Chat_train)

    # support for the predicted path, focusing on the path transitions and components of the path
    results[!,"Support"] = last_support(cue_obj_train, Chat_train)

    if (!ismissing(res_learn_train) && !ismissing(gpi_learn_train) && !ismissing(rpi_learn_train))
        # production accuracy/support/uncertainty for the predicted form
        results[!,"SCPP"] = SCPP(df, results)
        results[!,"PathSum"] = path_sum(pred_df)
        results[!,"TargetPathSum"] = target_path_sum(gpi_learn_train)
        results[!,"PathSumChat"] = path_sum_chat(res_learn_train, Chat_train)

        # support for the predicted path, focusing on the path transitions and components of the path
        results[!,"WithinPathEntropies"] = within_path_entropies(pred_df)
        results[!,"MeanWordSupport"] = mean_word_support(res_learn_train, pred_df)
        results[!,"MeanWordSupportChat"] = mean_word_support_chat(res_learn_train, Chat_train)
        results[!,"lwlr"] = lwlr(res_learn_train, pred_df)
        results[!,"lwlrChat"] = lwlr_chat(res_learn_train, Chat_train)

        # support for competing forms
        results[!,"PathCounts"] = path_counts(df)
        results[!,"ALDC"] = ALDC(df)
        results[!,"PathEntropiesSCP"] = path_entropies_scp(df)
        results[!,"PathEntropiesChat"] = path_entropies_chat(res_learn_train, Chat_train)
    end

    results
end

"""
    function compute_all_measures_val(data_val::DataFrame,
                                      cue_obj_train::JudiLing.Cue_Matrix_Struct,
                                      cue_obj_val::JudiLing.Cue_Matrix_Struct,
                                      Chat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                      S_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                      S_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                      Shat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                      F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                      G_train::Union{JudiLing.SparseMatrixCSC, Matrix};
                                      res_learn_val::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing,
                                      gpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                      rpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                      sem_density_n::Int64=8,
                                      calculate_production_uncertainty::Bool=false,
                                      low_cost_measures_only::Bool=false)
Compute all measures currently available in JudiLingMeasures for the validation data.
# Arguments
- `data_val::DataFrame`: The data for which measures should be calculated (the validation data).
- `cue_obj_train::JudiLing.Cue_Matrix_Struct`: The cue object of the training data.
- `cue_obj_val::JudiLing.Cue_Matrix_Struct`: The cue object of the validation data.
- `Chat_val::Union{JudiLing.SparseMatrixCSC, Matrix}`: The Chat matrix of the validation data.
- `S_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: The S matrix of the training data.
- `S_val::Union{JudiLing.SparseMatrixCSC, Matrix}`: The S matrix of the validation data.
- `Shat_val::Union{JudiLing.SparseMatrixCSC, Matrix}`: The Shat matrix of the data of interest.
- `F_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: Comprehension mapping matrix for the training data.
- `G_train::Union{JudiLing.SparseMatrixCSC, Matrix}`: Production mapping matrix for the training data.
- `res_learn_val::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing`: The first output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `gpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing`: The second output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `rpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing`: The third output of JudiLing.learn_paths_rpi (with `check_gold_path=true`)
- `low_cost_measures_only::Bool=false`: Only compute measures which are not computationally heavy. Recommended for very large datasets.
# Returns
- `results::DataFrame`: A dataframe with all information in `data_val` plus all the computed measures.
"""
function compute_all_measures_val(data_val::DataFrame,
                                  cue_obj_train::JudiLing.Cue_Matrix_Struct,
                                  cue_obj_val::JudiLing.Cue_Matrix_Struct,
                                  Chat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                  S_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                  S_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                  Shat_val::Union{JudiLing.SparseMatrixCSC, Matrix},
                                  F_train::Union{JudiLing.SparseMatrixCSC, Matrix},
                                  G_train::Union{JudiLing.SparseMatrixCSC, Matrix};
                                  res_learn_val::Union{Array{Array{JudiLing.Result_Path_Info_Struct,1},1}, Missing}=missing,
                                  gpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                  rpi_learn_val::Union{Array{JudiLing.Gold_Path_Info_Struct,1}, Missing}=missing,
                                  sem_density_n::Int64=8,
                                  calculate_production_uncertainty::Bool=false,
                                  low_cost_measures_only::Bool=false)
    # MAKE PREPARATIONS

    # generate additional objects for the measures such as
    # - results: copy of data_val for storing the measures in
    # - cor_s: the correlation matrix between Shat and S
    # - df: DataFrame of res_learn, the output of learn_paths
    # - pred_df: DataFrame with path supports for the predicted forms produced by learn_paths
    if (!ismissing(res_learn_val) && !ismissing(gpi_learn_val) && !ismissing(rpi_learn_val))
        results, cor_s, df_val, pred_df_val = make_measure_preparations(data_val, S_train, S_val, Shat_val,
                                        res_learn_val, cue_obj_train, cue_obj_val, rpi_learn_val)
    else
        results = deepcopy(data_val)

        # compute the accuracy and the correlation matrix
        acc_comp, cor_s = JudiLing.eval_SC(Shat_val, S_val, S_train, R=true)
    end

    # CALCULATE MEASURES

    # vector length/activation/uncertainty
    results[!,"L1Shat"] = JudiLingMeasures.L1Norm(Shat_val)
    results[!,"L2Shat"] = JudiLingMeasures.L2Norm(Shat_val)

    # semantic neighbourhood
    results[!,"SemanticDensity"] = JudiLingMeasures.density(cor_s, n=sem_density_n)
    results[!,"ALC"] = JudiLingMeasures.ALC(cor_s)
    results[!,"EDNN"] = EDNN(Shat_val, S_val, S_train)
    results[!,"NNC"] = JudiLingMeasures.NNC(cor_s)
    if !low_cost_measures_only
        results[!,"DistanceTravelledF"] = total_distance(cue_obj_val, F_train, :F)
    end

    # comprehension accuracy
    results[!,"TargetCorrelation"] = JudiLingMeasures.target_correlation(cor_s)
    results[!,"rank"] = JudiLingMeasures.rank(cor_s)
    results[!,"recognition"] = JudiLingMeasures.recognition(data_val)
    if !low_cost_measures_only
        results[!,"ComprehensionUncertainty"] = vec(JudiLingMeasures.uncertainty(S_val, Shat_val, S_train, method="cosine"))
    end

    # Measures of production accuracy/support/uncertainty for the target form
    if calculate_production_uncertainty && !low_cost_measures_only
        results[!,"ProductionUncertainty"] = vec(JudiLingMeasures.uncertainty(cue_obj_val.C, Chat_val, cue_obj_train.C, method="cosine"))
    end
    if !low_cost_measures_only
        results[!,"DistanceTravelledG"] = JudiLingMeasures.total_distance(cue_obj_val, G_train, :G)
    end

    # production accuracy/support/uncertainty for the predicted form
    results[!,"C-Precision"] = JudiLingMeasures.c_precision(Chat_val, cue_obj_val.C)
    results[!,"L1Chat"] = JudiLingMeasures.L1Norm(Chat_val)
    results[!,"SemanticSupportForForm"] = JudiLingMeasures.semantic_support_for_form(cue_obj_val, Chat_val)

    # support for the predicted path, focusing on the path transitions and components of the path
    results[!,"Support"] = JudiLingMeasures.last_support(cue_obj_val, Chat_val)

    if (!ismissing(res_learn_val) && !ismissing(gpi_learn_val) && !ismissing(rpi_learn_val))
        # production accuracy/support/uncertainty for the predicted form
        results[!,"SCPP"] = SCPP(df_val, results)
        results[!,"PathSum"] = path_sum(pred_df_val)
        results[!,"TargetPathSum"] = target_path_sum(gpi_learn_val)
        results[!,"PathSumChat"] = path_sum_chat(res_learn_val, Chat_val)

        # support for the predicted path, focusing on the path transitions and components of the path
        results[!,"WithinPathEntropies"] = within_path_entropies(pred_df_val)
        results[!,"MeanWordSupport"] = mean_word_support(res_learn_val, pred_df_val)
        results[!,"MeanWordSupportChat"] = mean_word_support_chat(res_learn_val, Chat_val)
        results[!,"lwlr"] = lwlr(res_learn_val, pred_df_val)
        results[!,"lwlrChat"] = lwlr_chat(res_learn_val, Chat_val)

        # support for competing forms
        results[!,"PathCounts"] = path_counts(df_val)
        results[!,"ALDC"] = ALDC(df_val)
        results[!,"PathEntropiesSCP"] = path_entropies_scp(df_val)
        results[!,"PathEntropiesChat"] = path_entropies_chat(res_learn_val, Chat_val)
    end

    results
end

function safe_divide(x, y)
    if (y != 0) & (!ismissing(y)) & (!ismissing(x))
        x/y
    else
        missing
    end
end

function mse_rowwise(X::Union{Matrix,JudiLing.SparseMatrixCSC},
                     Y::Union{Matrix,JudiLing.SparseMatrixCSC})
    mses = zeros(size(X, 1), size(Y,1))
    for (index_x, x) in enumerate(eachrow(X))
        for (index_y, y) in enumerate(eachrow(Y))
            mses[index_x, index_y] = StatsBase.msd(convert(Vector{Float64}, x),
                                                   convert(Vector{Float64}, y))
        end
    end
    mses
end

function normalise_vector(x)
    x = vec(x)
    if length(x) > 0
        x_min, _ = findmin(x)
        x_max, _ = findmax(x)
        (x .- x_min) ./ (x_max-x_min)
    else
        x
    end
end

function normalise_matrix_rowwise(X::Union{Matrix,JudiLing.SparseMatrixCSC})
    if (size(X, 1) > 0) & (size(X,2) > 0)
        mapreduce(permutedims, vcat, map(normalise_vector, eachrow(X)))
    else
        X
    end
end
