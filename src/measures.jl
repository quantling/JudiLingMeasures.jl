# L1NORM = SEMANTIC VECTOR LENGTH and L2NORM
"""
    L1Norm(M::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L1 Norm of each row of a matrix.
# Examples
```jldoctest
julia> Shat = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> L1Norm(Shat)
3-element Vector{Int64}:
 6
 6
 6
```
"""
function L1Norm(M::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(l1_rowwise(M))
end

"""
    L2Norm(M::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L2 Norm of each row of a matrix.
# Examples
```jldoctest
julia> Shat = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> L2Norm(Shat)
3-element Vector{Float64}:
 3.7416573867739413
 3.7416573867739413
 3.7416573867739413
```
"""
function L2Norm(M::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(l2_rowwise(M))
end

"""
    density(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix};
            n::Int=8, ignore_missing::Bool=false)
Compute the average correlation of each predicted semantic vector with its n most correlated neighbours.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
- `n::Int`: the number of highest semantic neighbours to take into account
# Example
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> acc, cor_s = JudiLing.eval_SC(Shat, S, R=true)
julia> density(cor_s, n=2)
4-element Vector{Float64}:
 0.7393813797301239
 0.6420816485652429
 0.4496869233815781
 0.281150888376636
```
"""
function density(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix};
                 n=8)
    vec(sem_density_mean(cor_s, n))
end

"""
    ALC(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the Average Lexical Correlation (ALC) between the predicted vectors
in Shat and all semantic vectors in S.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
# Examples
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> acc, cor_s = JudiLing.eval_SC(Shat, S, R=true)
julia> ALC(cor_s)
4-element Vector{Float64}:
  0.1867546970250672
 -0.030901103469572838
 -0.0681995247218424
  0.011247813283240052
```
"""
function ALC(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(mean_rowwise(cor_s))
end

"""
    EDNN(Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
         S::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the Euclidean Distance nearest neighbours between the predicted semantic
vectors in Shat and the semantic vectors in S.
# Examples
# Examples
```jldoctest
julia> ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
julia> ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
julia> EDNN(ma1, ma4)
3-element Vector{Float64}:
 1.0
 2.0
 1.0
```
"""
function EDNN(Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
              S::Union{JudiLing.SparseMatrixCSC, Matrix})
    eucl_sims = euclidean_distance_array(Shat, S)
    ednn = get_nearest_neighbour_eucl(eucl_sims)
    vec(ednn)
end

"""
    NNC(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
For each predicted semantic vector get the highest correlation with the semantic vectors in S.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
# Examples
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> acc, cor_s = JudiLing.eval_SC(Shat, S, R=true)
julia> NNC(cor_s)
4-element Vector{Float64}:
 0.8164965809277259
 0.9886230654859615
 0.8625383733289683
 0.35478743759344955
```
"""
function NNC(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(max_rowwise(cor_s))
end

"""
    last_support(cue_obj::JudiLing.Cue_Matrix_Struct,
                 Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
Return the support in `Chat` for the last ngram of each target word.
"""
function last_support(cue_obj::JudiLing.Cue_Matrix_Struct,
                      Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
    ngrams = cue_obj.gold_ind
    support = []
    for (index, n) in enumerate(ngrams)
        l = n[end]
        s = Chat[index, l]
        append!(support, [s])
    end
    vec(support)
end

"""
    semantic_support_for_form(cue_obj::JudiLing.Cue_Matrix_Struct,
                 Chat::Union{JudiLing.SparseMatrixCSC, Matrix};
                 sum_supports::Bool=true)
Return the support in `Chat` for all target ngrams of each target word.
"""
function semantic_support_for_form(cue_obj::JudiLing.Cue_Matrix_Struct,
                      Chat::Union{JudiLing.SparseMatrixCSC, Matrix};
                      sum_supports::Bool=true)
    ngrams = cue_obj.gold_ind
    support = []
    for (index, n) in enumerate(ngrams)
        s = Chat[index, n]
        if sum_supports
            append!(support, sum(s))
        else
            append!(support, [s])
        end
    end
    vec(support)
end

"""
    path_counts(df::DataFrame)
Return the number of possible paths as returned by `learn_paths`.
# Arguments
- `df::DataFrame`: DataFrame of the output of `learn_paths`.
"""
function path_counts(df::DataFrame)
    g = groupby(df, :utterance)
    c = combine(g, [:pred] => count_rows => :num_preds)
    c[df[ismissing.(df.pred),:utterance],:num_preds] .= 0
    vec(c.num_preds)
end

"""
    path_sum(pred_df::DataFrame)
Compute the summed path support for each predicted word with highest support in dat_val.
# Arguments
- `pred_df::DataFrame`: The output of `get_predicted_path_support`
"""
function path_sum(pred_df::DataFrame)
    map(safe_sum, pred_df.timestep_support)
end

"""
    target_path_sum(gpi)
Compute the summed path support for each target word.
Code by Yu-Ying Chuang.
"""
function target_path_sum(gpi)
    all_support = Vector{Float64}(undef, length(gpi))
    for i in 1:length(all_support)
        all_support[i] = sum(gpi[i].ngrams_ind_support)
    end
    return(all_support)
end

"""
    within_path_entropies(pred_df::DataFrame)
Compute the Shannon Entropy of the path supports for each word in dat_val.
# Arguments
- `pred_df::DataFrame`: The output of `get_predicted_path_support`
"""
function within_path_entropies(pred_df::DataFrame)
    map(entropy, pred_df.timestep_support)
end

"""
    ALDC(df::DataFrame)
Compute the Average Levenshtein Distance of all candidates (ALDC) with the correct word form.
# Arguments
- `df::DataFrame`: DataFrame of the output of `learn_paths`.
"""
function ALDC(df::DataFrame)

    g = groupby(df, :utterance)
    c = combine(g, [:identifier, :pred] => get_avg_levenshtein => :avg_levenshtein)

    vec(c.avg_levenshtein)
end

"""
    mean_word_support(res_learn, pred_df::DataFrame)
Compute the summed path support divided by each word form's length for each word in dat_val.
# Arguments
- `res_learn`: The output of learn_paths
- `pred_df::DataFrame`: The output of `get_predicted_path_support`
"""
function mean_word_support(res_learn, pred_df::DataFrame)
    lengths = indices_length(res_learn)
    path_sums = path_sum(pred_df)
    res = map(safe_divide, path_sums, lengths)
    res
end

"""
    target_correlation(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
Calculate the correlation between each predicted semantic vector and its target semantic vector.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
# Examples
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> acc, cor_s = JudiLing.eval_SC(Shat, S, R=true)
julia> target_correlation(cor_s)
4-element Vector{Float64}:
  0.6622661785325219
  0.2955402316445243
 -0.86386842558136
  0.35478743759344955
```
"""
function target_correlation(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(diag(cor_s))
end

"""
    rank(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
Return the rank of the correct form among the comprehension candidates.
# Arguments
- `s_cor::Union{JudiLing.SparseMatrixCSC, Matrix}`: the correlation matrix between S and Shat
# Examples
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> acc, cor_s = JudiLing.eval_SC(Shat, S, R=true)
julia> rank(cor_s)
4-element Vector{Any}:
 2
 2
 4
 1
```
"""
function rank(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix})
    d = diag(cor_s)
    rank = []
    for row in 1:size(cor_s,1)
        sorted = sort(cor_s[row,:], rev=true)
        c = findall(x->x==d[row], sorted)
        append!(rank, c[1])
    end
    rank
end

"""
    recognition(data::DataFrame)
Return a vector indicating whether a wordform was correctly understood.
Not implemented.
"""
function recognition(data::DataFrame)
    println("Recognition not implemented")
    repeat([missing], size(data,1))
end

# LWLR (Length-Weakest-Link-Ratio from the WpmWithLDL package)
# needs changes to the JudiLing learn path function
"""
    lwlr(res_learn, pred_df::DataFrame)
The ratio between the predicted form's length and its weakest support from `learn_paths`.
# Arguments
- `pred_df::DataFrame`: The output of `get_predicted_path_support`
"""
function lwlr(res_learn, pred_df::DataFrame)
    lengths = indices_length(res_learn)
    wl = pred_df.weakest_support
    lengths./wl
end

"""
    c_precision(c_hat_collection, cue_obj)
Calculate the correlation between the predicted and the target cue vector.
# Examples
```jldoctest
julia> c = [[1. 1. 0.]; [0. 0. 1.]; [1. 0. 1.]]
julia> chat = [[0.9 0.9 0.1]; [0.9 0.1 1.]; [0.9 -0.1 0.8]]
julia> c_precision(chat, c)
3-element Array{Float64,1}:
 1.0
 0.5852057359806527
 0.9958705948858222
```
"""
function c_precision(c_hat_collection, c)
    vec(correlation_diagonal_rowwise(c_hat_collection, c))
end


"""
    SCPP(df::DataFrame, results::DataFrame)
Semantic Correlation of Predicted Production. Returns the correlation of the predicted semantic vector of the predicted path with the target semantic vector.
# Arguments
- `df::DataFrame`: The output of learn_paths as DataFrame.
- `results::DataFrame`: The data of interest.
"""
function SCPP(df::DataFrame, results::DataFrame)
    id = df.utterance[ismissing.(df.isbest)]
    res = Vector{Union{Missing, Float64}}(missing, size(results,1))
    remaining = df[Not(ismissing.(df.isbest)),:]
    res[Not(id)] = remaining[remaining.isbest .== true,:support]
    res
end

"""
    path_sum_chat(res_learn,
                 Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
"""
function path_sum_chat(res_learn,
                      Chat::Union{JudiLing.SparseMatrixCSC, Matrix})

    n = size(res_learn)
    ngrams = JudiLing.make_ngrams_ind(res_learn, n)
    sums = []
    for (index, n) in enumerate(ngrams)
        s = Chat[index, n]
        append!(sums, sum(s))
    end
    vec(sums)
end

"""
    mean_word_support_chat(res_learn, Chat)
Compute the summed path support, taken from Chat, divided by each word form's length for each word in dat_val.
"""
function mean_word_support_chat(res_learn, Chat)
    lengths = indices_length(res_learn)
    path_sums = path_sum_chat(res_learn, Chat)
    map(safe_divide, path_sums, lengths)
end

"""
    path_entropies_chat(res_learn,
                        Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
"""
function path_entropies_chat(res_learn,
                             Chat::Union{JudiLing.SparseMatrixCSC, Matrix})

    entropies = Vector{Union{Missing, Float32}}(missing, size(res_learn, 1))
    for i=1:size(res_learn)[1]
        sums = Vector{Union{Missing, Float32}}(missing, size(res_learn[i], 1))
        for (j, cand) in enumerate(res_learn[i])
            if !ismissing(cand.ngrams_ind)
                s = Chat[i, cand.ngrams_ind]
                sums[j] = sum(s)
            end

        end
        entropies[i] = entropy(sums)
    end
    vec(entropies)
end
"""
    path_entropes_scp(df::DataFrame)
Computes the entropy over the semantic supports for all candidates per target word form.
# Arguments
- `df::DataFrame`: DataFrame of the output of `learn_paths`.
"""
function path_entropies_scp(df::DataFrame)
    g = groupby(df, :utterance)
    c = combine(g, [:support] => entropy => :entropy)
    c[df[ismissing.(df.pred),:utterance],:entropy] .= 0
    vec(c.entropy)
end

# LWLR (Length-Weakest-Link-Ratio from the WpmWithLDL package)
# needs changes to the JudiLing learn path function
"""
    lwlr_chat(res_learn, Chat)
The ratio between the predicted form's length and its weakest support in Chat.
"""
function lwlr_chat(res_learn, Chat)
    n = size(res_learn)

    ngrams = JudiLing.make_ngrams_ind(res_learn, n)

    weakest_links = Vector{Union{Missing, Float32}}(missing, n)
    lengths = Vector{Union{Missing, Int64}}(missing, n)

    for (i, n) in enumerate(ngrams)
        if (!ismissing(n) && !(length(n) < 1))
            lengths[i] = length(n)
            l = Chat[i, n]
            weakest_links[i] = findmin(l)[1]
        end
    end
    vec(lengths./weakest_links)
end


"""
    total_distance(cue_obj::JudiLing.Cue_Matrix_Struct,
                   FG::Union{JudiLing.SparseMatrixCSC, Matrix},
                   mat_type::Symbol)
Code by Yu-Ying Chuang.
"""
function total_distance(cue_obj::JudiLing.Cue_Matrix_Struct,
                        FG::Union{JudiLing.SparseMatrixCSC, Matrix},
                        mat_type::Symbol)

    if mat_type == :G
		FG = FG'
	end

    all_dist = Vector{Float64}(undef, length(cue_obj.gold_ind))
	for i in 1:length(all_dist)
	    gis = cue_obj.gold_ind[i]
	    dist1 = evaluate(Euclidean(), zeros(size(FG)[2]), FG[gis[1],:])

	    tot_dist = dist1
	    if length(gis)!=1
	        for j in 2:length(gis)
	            tmp_dist = evaluate(Euclidean(), FG[gis[(j-1)],:], FG[gis[j],:])
	            tot_dist += tmp_dist
	        end
	    end
        all_dist[i] = tot_dist
	end
	return(all_dist)
end


"""
    function uncertainty(SC::Union{JudiLing.SparseMatrixCSC, Matrix},
                         SChat::Union{JudiLing.SparseMatrixCSC, Matrix};
                         method::Union{String, Symbol} = "corr")
Sum of correlation/mse/cosine similarity of SChat with all vectors in SC and the ranks of this correlation/mse/cosine similarity.

Measure developed by Motoki Saito. Note: the current version of uncertainty is not completely tested against its original implementation in [pyldl](https://github.com/msaito8623/pyldl).

# Arguments
- SC::Union{JudiLing.SparseMatrixCSC, Matrix}: S or C matrix of the data of interest
- SChat::Union{JudiLing.SparseMatrixCSC, Matrix}: Shat or Chat matrix of the data of interest
- method::Union{String, Symbol} = "corr": Method to compute similarity

# Examples
```jldoctest
julia> Shat = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
julia> S = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
julia> JudiLingMeasures.uncertainty(S, Shat, method="corr") # default
4-element Vector{Float64}:
 5.3583589101779605
 5.45682093125373
 5.330660086961848
 6.675366532252984
julia> JudiLingMeasures.uncertainty(S, Shat, method="mse")
4-element Vector{Float64}:
 3.909090909090909
 5.44186046511628
 5.314285714285714
 5.375
julia> JudiLingMeasures.uncertainty(S, Shat, method="cosine")
4-element Vector{Float64}:
 5.4030832120059165
 4.610897476199016
 4.744838560168514
 3.526925997296189
```
"""
function uncertainty(SC::Union{JudiLing.SparseMatrixCSC, Matrix},
                     SChat::Union{JudiLing.SparseMatrixCSC, Matrix};
                     method::Union{String, Symbol} = "corr")
    if method == "corr"
        cor_sc = correlation_rowwise(SChat, SC)
    elseif method == "mse"
        cor_sc = mse_rowwise(SChat, SC)
    elseif method == "cosine"
        cor_sc = cosine_similarity(SChat, SC)
    end
    cor_sc = normalise_matrix_rowwise(cor_sc)
    ranks = mapreduce(permutedims, vcat, map(x -> ordinalrank(x).-1, eachrow(cor_sc)))
    vec(sum(cor_sc .* ranks, dims=2))
end

"""
    function functional_load(F::Union{JudiLing.SparseMatrixCSC, Matrix},
                             Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
                             cue_obj::JudiLing.Cue_Matrix_Struct;
                             cue_list::Union{Vector{String}, Missing}=missing,
                             method::Union{String, Symbol}="corr")
Correlation/MSE of rows in F of triphones in word w and semantic vector of w.

Measure developed by Motoki Saito. Note: the current version of Functional Load is not completely tested against its original implementation in [pyldl](https://github.com/msaito8623/pyldl).

# Arguments
- F::Union{JudiLing.SparseMatrixCSC, Matrix}: The comprehension matrix F
- Shat::Union{JudiLing.SparseMatrixCSC, Matrix}: The predicted semantic matrix of the data of interest
- cue_obj::JudiLing.Cue_Matrix_Struct: The cue object of the data of interest.
- cue_list::Union{Vector{String}, Missing}=missing: List of cues for which functional load should be computed. Each cue in the list corresponds to one word in Shat/cue_obj and cue and corresponding words have to be in the same order.
- method::Union{String, Symbol}="corr": If "corr", correlation between row in F and semantic vector in S is computed. If "mse", mean squared error is used.

# Example
```jldoctest
julia> using JudiLing, DataFrames
julia> dat = DataFrame("Word"=>["abc", "bcd", "cde"]);
julia> cue_obj = JudiLing.make_cue_matrix(dat, grams=3, target_col=:Word);
julia> n_features = size(cue_obj.C, 2);
julia> S = JudiLing.make_S_matrix(
    dat,
    ["Word"],
    [],
    ncol=n_features);
julia> F = JudiLing.make_transform_matrix(cue_obj.C, S);
julia> Shat = cue_obj.C * F;
julia> JudiLingMeasures.functional_load(F, Shat, cue_obj)
3-element Vector{Any}:
 [1.0, 1.0, 1.0]
 [0.9999999999999999, 1.0, 1.0]
 [0.9999999999999998, 0.9999999999999998, 1.0]
julia> JudiLingMeasures.functional_load(F, Shat, cue_obj, cue_list=["#ab", "#bc", "#cd"])
3-element Vector{Any}:
 1.0
 0.9999999999999999
 0.9999999999999998
julia> JudiLingMeasures.functional_load(F, Shat, cue_obj, cue_list=["#ab", "#bc", "#cd"], method="mse")
3-element Vector{Any}:
 13.929885322944285
  5.26127506129032
 10.371443574053322
julia> JudiLingMeasures.functional_load(F, Shat, cue_obj, method="mse")
3-element Vector{Any}:
 [13.929885322944285, 13.929885322944251, 13.929885322944255]
 [5.26127506129032, 5.261275061290302, 5.261275061290303]
 [10.371443574053322, 10.371443574053368, 10.371443574053368]
```
"""
function functional_load(F::Union{JudiLing.SparseMatrixCSC, Matrix},
                         Shat::Union{JudiLing.SparseMatrixCSC, Matrix},
                         cue_obj::JudiLing.Cue_Matrix_Struct;
                         cue_list::Union{Vector{String}, Missing}=missing,
                         method::Union{String, Symbol}="corr")
    if ismissing(cue_list)
         ngrams = cue_obj.gold_ind
    else
        ngrams = [[cue_obj.f2i[cue]] for cue in cue_list]
    end
    # if method == "corr"
    #     cor_fs = JudiLingMeasures.correlation_rowwise(F, Shat)
    # elseif method == "mse"
    #     cor_fs = JudiLingMeasures.mse_rowwise(F, Shat)
    # end
     functional_loads = []
     for (index, n) in enumerate(ngrams)
         #s = cor_fs[n, index]
         if method == "corr"
             s = JudiLingMeasures.correlation_rowwise(F[n,:], Shat[[index],:])
         elseif method == "mse"
             s = JudiLingMeasures.mse_rowwise(F[n,:], Shat[[index],:])
         end
         if !ismissing(cue_list)
             append!(functional_loads, s)
         else
             append!(functional_loads, [s])
        end

     end
     functional_loads
 end
