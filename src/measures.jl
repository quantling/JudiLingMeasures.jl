# L1NORM = SEMANTIC VECTOR LENGTH and L2NORM
"""
    L1Norm(Shat::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L1 Norm of Shat.
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
function L1Norm(Shat::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(l1_rowwise(Shat))
end

"""
    L2Norm(Shat::Union{JudiLing.SparseMatrixCSC, Matrix})
Compute the L2 Norm of Shat.
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
function L2Norm(Shat::Union{JudiLing.SparseMatrixCSC, Matrix})
    vec(l2_rowwise(Shat))
end

"""
    density(cor_s::Union{JudiLing.SparseMatrixCSC, Matrix};
            n::Int=8)
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
                 Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
Return the support in `Chat` for all target ngrams of each target word.
"""
function semantic_support_for_form(cue_obj::JudiLing.Cue_Matrix_Struct,
                      Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
    ngrams = cue_obj.gold_ind
    support = []
    for (index, n) in enumerate(ngrams)
        s = Chat[index, n]
        append!(support, sum(s))
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
"""
function target_path_sum(gpi)
    #JudiLing.get_total_support(gpi)
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
    path_sums./lengths
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
    path_sums./lengths
end

"""
    path_entropies_chat(res_learn,
                        Chat::Union{JudiLing.SparseMatrixCSC, Matrix})
"""
function path_entropies_chat(res_learn,
                             Chat::Union{JudiLing.SparseMatrixCSC, Matrix})

    entropies = []
    for i=1:size(res_learn)[1]
        sums = []
        for cand in res_learn[i]
            s = Chat[i, cand.ngrams_ind]
            append!(sums, sum(s))
        end
        append!(entropies, entropy(sums))
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
    weakest_links = []
    lengths = []
    for (i, n) in enumerate(ngrams)
        append!(lengths, length(n))
        l = Chat[i, n]
        append!(weakest_links, findmin(l)[1])
    end
    lengths./weakest_links
end
