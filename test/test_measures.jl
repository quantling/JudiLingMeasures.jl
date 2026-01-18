########################################
# test measures
########################################

if !Sys.iswindows()
  pandas = pyimport("pandas")
  np = pyimport("numpy")
  pm = pyimport("discriminative_lexicon_model.mapping")
  lmea = pyimport("discriminative_lexicon_model.measures")
end


# define some data to test with
ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]
ma5 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]; [4 2 -9 1]]

# define some data to test with
dat = DataFrame("Word"=>["abc", "bcd", "cde"])
val_dat = DataFrame("Word"=>["abc"])
cue_obj, cue_obj_val = JudiLing.make_combined_cue_matrix(
    dat,
    val_dat,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )
n_features = size(cue_obj.C, 2)
S, S_val = JudiLing.make_combined_S_matrix(
    dat,
    val_dat,
    ["Word"],
    [],
    ncol=n_features,
    add_noise=false)
G = JudiLing.make_transform_matrix(S, cue_obj.C)
Chat = S * G
Chat_val = S_val * G
F = JudiLing.make_transform_matrix(cue_obj.C, S)
Shat = cue_obj.C * F
Shat_val = cue_obj_val.C * F
A = cue_obj.A
max_t = JudiLing.cal_max_timestep(dat, :Word)

res_learn, gpi_learn, rpi_learn = JudiLing.learn_paths_rpi(
    dat,
    dat,
    cue_obj.C,
    S,
    F,
    Chat,
    A,
    cue_obj.i2f,
    cue_obj.f2i, # api changed in 0.3.1
    check_gold_path = true,
    gold_ind = cue_obj.gold_ind,
    Shat_val = Shat,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    keep_sep = false,
    target_col = :Word,
    verbose = true
)

res_learn_val, gpi_learn_val, rpi_learn_val = JudiLing.learn_paths_rpi(
    dat,
    val_dat,
    cue_obj.C,
    S_val,
    F,
    Chat_val,
    cue_obj_val.A,
    cue_obj_val.i2f,
    cue_obj_val.f2i, # api changed in 0.3.1
    check_gold_path = true,
    gold_ind = cue_obj_val.gold_ind,
    Shat_val = Shat_val,
    max_t = max_t,
    max_can = 10,
    grams = 3,
    threshold = 0.05,
    tokenized = false,
    keep_sep = false,
    target_col = :Word,
    verbose = true
)

results, cor_s_all, df, pred_df = JudiLingMeasures.make_measure_preparations(dat, S, Shat,
                                   res_learn, cue_obj, rpi_learn)

results_val, cor_s_all_val, df_val, pred_df_val = JudiLingMeasures.make_measure_preparations(val_dat, S, S_val, Shat_val,
                                  res_learn_val, cue_obj, cue_obj_val, rpi_learn_val)
# tests

@testset "Make measure preparations" begin
    @testset "Training data" begin
        @test cor_s_all == cor(Shat, S, dims=2)
        @test results == dat
    end

    @testset "Validation data" begin
       @test cor_s_all_val == cor(Shat_val, vcat(S_val, S), dims=2)
       @test size(cor_s_all_val) == (size(Shat_val, 1), size(S_val, 1) + size(S, 1))
       @test results_val == val_dat
   end
end

@testset "L1 Norm" begin
    @test JudiLingMeasures.L1Norm(ma1) == [6; 6; 6]
    @test JudiLingMeasures.L1Norm(zeros((1,1))) == [0]
    @test JudiLingMeasures.L1Norm(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.L1Norm([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; 6; 6])
    @test isapprox(JudiLingMeasures.L1Norm(Chat), map(sum, eachrow(abs.(Chat))))
end

@testset "L2 Norm" begin
    @test JudiLingMeasures.L2Norm(ma1) == [sqrt(14); sqrt(14); sqrt(14)]
    @test JudiLingMeasures.L2Norm(zeros((1,1))) == [0]
    @test JudiLingMeasures.L2Norm(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.L2Norm([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; sqrt(14); sqrt(14)])
end


cor_s = JudiLingMeasures.correlation_rowwise(ma2, ma3)
cor_s2 = JudiLingMeasures.correlation_rowwise(ma2, ma5)

@testset "Density" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.density(cor_s, n=2), vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)
        @test JudiLingMeasures.density(zeros((1,1)), n=1) == [0]
        @test JudiLingMeasures.density(ones((1,1)), n=1) == [1]
        @test isequal(JudiLingMeasures.density([[1 2 missing]; [-1 -2 -3]; [1 2 3]], n=2), [missing; -1.5; 2.5])
        @test_throws ArgumentError JudiLingMeasures.density(zeros((1,1))) == [0]
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.density(cor_s2, n=2), vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)
    end
end

@testset "ALC" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.ALC(cor_s), [0.18675475, -0.03090124999999999, -0.06819962499999999, 0.011247725000000014], rtol=1e-4)
        @test JudiLingMeasures.ALC(zeros((1,1))) == [0]
        @test JudiLingMeasures.ALC(ones((1,1))) == [1]
        @test isequal(JudiLingMeasures.ALC([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; -2.; 2.])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.ALC(cor_s2), [ 0.20685225658219636, -0.16126729728684563, -0.15910392072943358, -0.057005049620928595], rtol=1e-4)
    end
end

@testset "EDNN" begin
    @testset "Training data" begin
        @test JudiLingMeasures.EDNN(ma1, ma4) == [1., sqrt(4), 1.]
        @test JudiLingMeasures.EDNN(zeros((1,1)), zeros((1,1))) == [0]
        @test JudiLingMeasures.EDNN(ones((1,1)), zeros((1,1))) == [1]
        @test_throws MethodError JudiLingMeasures.EDNN([[1 2 missing]; [-1 -2 -3]; [1 2 3]], ma4)
    end
    @testset "Validation data" begin
        @test JudiLingMeasures.EDNN(ma1, ma4, ma4) == [1., sqrt(4), 1.]
    end
end

@testset "NNC" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.NNC(cor_s), [0.816497, 0.988623, 0.862538, 0.354787], rtol=1e-4)
        @test JudiLingMeasures.NNC(zeros((1,1))) == [0]
        @test JudiLingMeasures.NNC(ones((1,1))) == [1]
        @test isequal(JudiLingMeasures.NNC([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; -1; 3])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.NNC(cor_s2), [0.816497, 0.988623, 0.862538, 0.354787], rtol=1e-4)
    end
end

@testset "last_support" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.last_support(cue_obj, Chat), [Chat[1,cue_obj.gold_ind[1][end]], Chat[2,cue_obj.gold_ind[2][end]], Chat[3,cue_obj.gold_ind[3][end]]], rtol=1e-4)
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.last_support(cue_obj_val, Chat_val), [Chat_val[1,cue_obj_val.gold_ind[1][end]]], rtol=1e-4)
    end
end

@testset "path_counts" begin
    @testset "Training data" begin
        @test JudiLingMeasures.path_counts(df) == [1,1, 1]
        df_mock = DataFrame("utterance"=>[1,1],
                        "pred"=>["abc", "abd"])
        @test JudiLingMeasures.path_counts(df_mock) == [2]
        df_mock2 = DataFrame()
        @test_throws ArgumentError JudiLingMeasures.path_counts(df_mock2)
    end
    @testset "Validation data" begin
        @test JudiLingMeasures.path_counts(df_val) == [1]
    end
end

@testset "path_sum" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.path_sum(pred_df), [2.979, 2.979, 2.979], rtol=1e-3)
        pred_df_mock = DataFrame("timestep_support"=>[missing, [1,2,3], [0,0,0], [0,1,missing]])
        @test isequal(JudiLingMeasures.path_sum(pred_df_mock), [missing; 6; 0; missing])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.path_sum(pred_df_val), [2.979], rtol=1e-3)
    end
end

@testset "within_path_entropies" begin
    @testset "Training data" begin
        # Note: the result of this is different to other entropy measures as a) the values are scaled between 0 and 1 first, and b) log2 instead of log is used
        @test isapprox(JudiLingMeasures.within_path_entropies(pred_df), [1.584962500721156, 1.584962500721156, 1.584962500721156], rtol=1e-1)
        pred_df_mock = DataFrame("timestep_support"=>[missing, [0,1,missing]])
        @test isequal(JudiLingMeasures.within_path_entropies(pred_df_mock), [missing, missing])
        pred_df_mock2 = DataFrame("timestep_support"=>[[1,2,3], [1,1,1]])
        @test isapprox(JudiLingMeasures.within_path_entropies(pred_df_mock2), [JudiLingMeasures.entropy([1,2,3]),
                                                                               JudiLingMeasures.entropy([1,1,1])])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.within_path_entropies(pred_df_val), [1.584962500721156], rtol=1e-1)
    end
end

@testset "ALDC" begin
    @testset "Training data" begin
        @test JudiLingMeasures.ALDC(df) == [0, 0, 0]
        df_mock = DataFrame("utterance"=>[1,1],
                        "pred"=>["abc", "abd"],
                        "identifier"=>["abc", "abc"])
        @test JudiLingMeasures.ALDC(df_mock) == [0.5]
    end
    @testset "Validation data" begin
        @test JudiLingMeasures.ALDC(df_val) == [0]
    end
end

@testset "Mean word support" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.mean_word_support(res_learn, pred_df),
                            [sum(pred_df.timestep_support[1])/length(pred_df.timestep_support[1]), sum(pred_df.timestep_support[2])/length(pred_df.timestep_support[2]), sum(pred_df.timestep_support[3])/length(pred_df.timestep_support[3])], rtol=1e-4)
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.mean_word_support(res_learn_val, pred_df_val),
                            [sum(pred_df_val.timestep_support[1])/length(pred_df_val.timestep_support[1])], rtol=1e-4)
    end
end

@testset "TargetCorrelation" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.target_correlation(cor_s), [0.662266, 0.29554, -0.863868, 0.354787], rtol=1e-4)
        @test isapprox(JudiLingMeasures.target_correlation(zeros(1,1)), [0.], rtol=1e-4)
        @test isapprox(JudiLingMeasures.target_correlation(ones(1,1)), [1.], rtol=1e-4)
        @test isequal(JudiLingMeasures.target_correlation(Matrix{Missing}(missing, 1,1)), [missing])
        @test isapprox(JudiLingMeasures.target_correlation(cor_s_all), [1.0, 1.0, 1.0], rtol=1e-4)

        @test isapprox(JudiLingMeasures.target_correlation(ma2, ma3), [0.662266, 0.29554, -0.863868, 0.354787], rtol=1e-4)
        @test isnan(JudiLingMeasures.target_correlation(zeros(1,1), zeros(1,1))[1])
        @test isnan(JudiLingMeasures.target_correlation(ones(1,1), ones(1,1))[1])
        @test isapprox(JudiLingMeasures.target_correlation(Shat, S), [1.0, 1.0, 1.0], rtol=1e-4)
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.target_correlation(cor_s2), [0.662266, 0.29554, -0.863868, 0.354787], rtol=1e-4)
        @test isapprox(JudiLingMeasures.target_correlation(cor_s_all_val), [1.0], rtol=1e-4)

        @test isapprox(JudiLingMeasures.target_correlation(Shat_val, S_val), [1.0], rtol=1e-4)
    end
end

@testset "Rank" begin
    @testset "Training data" begin
        @test JudiLingMeasures.rank(cor_s) == [2,2,4,1]
        @test JudiLingMeasures.rank(zeros(1,1)) == [1]
        @test JudiLingMeasures.rank(ones(1,1)) == [1]
        @test_throws TypeError JudiLingMeasures.rank(Matrix{Missing}(missing, 1,1))
    end
    @testset "Validation data" begin
        @test JudiLingMeasures.rank(cor_s2) == [2,2,5,1]
    end
end

@testset "lwlr" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.lwlr(res_learn, pred_df), [3. /pred_df.weakest_support[1], 3. /pred_df.weakest_support[2], 3. /pred_df.weakest_support[3]], rtol=1e-4)
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.lwlr(res_learn_val, pred_df_val), [3. /pred_df_val.weakest_support[1]], rtol=1e-4)
    end
end

@testset "PathSumChat" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.path_sum_chat(res_learn, Chat),
                       [sum(Chat[1,[1,2,3]]), sum(Chat[2,[4,5,6]]), sum(Chat[3,[7,8,9]])])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.path_sum_chat(res_learn_val, Chat_val),
                       [sum(Chat[1,[1,2,3]])])
    end
end

@testset "C-Precision" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.c_precision(Chat, cue_obj.C), diag(JudiLing.eval_SC(Chat, cue_obj.C, R=true)[2]))
        cor_c = JudiLingMeasures.correlation_rowwise(Chat, cue_obj.C)
        @test isapprox(JudiLingMeasures.c_precision(Chat, cue_obj.C), JudiLingMeasures.target_correlation(cor_c))
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.c_precision(Chat_val, cue_obj_val.C), diag(JudiLing.eval_SC(Chat_val, cue_obj_val.C, R=true)[2]))
        cor_c_val = JudiLingMeasures.correlation_rowwise(Chat_val, cue_obj_val.C)
        @test isapprox(JudiLingMeasures.c_precision(Chat_val, cue_obj_val.C), JudiLingMeasures.target_correlation(cor_c_val))
        cor_c_val = JudiLingMeasures.correlation_rowwise(Chat_val, vcat(cue_obj_val.C, cue_obj.C))
        @test isapprox(JudiLingMeasures.c_precision(Chat_val, cue_obj_val.C), JudiLingMeasures.target_correlation(cor_c_val))
    end
end

@testset "Semantic Support For Form" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj, Chat), [sum(Chat[1,[1,2,3]]), sum(Chat[2,[4,5,6]]), sum(Chat[3,[7,8,9]])])
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj, Chat), JudiLingMeasures.path_sum_chat(res_learn, Chat))
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj, Chat, sum_supports=false), [Chat[1,[1,2,3]], Chat[2,[4,5,6]], Chat[3,[7,8,9]]] )
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj_val, Chat_val), [sum(Chat[1,[1,2,3]])])
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj_val, Chat_val), JudiLingMeasures.path_sum_chat(res_learn_val, Chat_val))
        @test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj_val, Chat_val, sum_supports=false), [Chat[1,[1,2,3]]] )
    end
end

@testset "SCPP" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.SCPP(df, dat), JudiLingMeasures.NNC(cor_s_all))
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.SCPP(df_val, val_dat), JudiLingMeasures.NNC(cor_s_all_val))
    end
end

@testset "MeanWordSupportChat" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.mean_word_support_chat(res_learn, Chat), [sum(Chat[1,[1,2,3]])/3, sum(Chat[2,[4,5,6]])/3, sum(Chat[3,[7,8,9]])/3])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.mean_word_support_chat(res_learn_val, Chat_val),
                                                               [sum(Chat[1,[1,2,3]])/3])
    end
end

@testset "lwlrChat" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.lwlr_chat(res_learn, Chat), [3. /findmin(Chat[1,[1,2,3]])[1], 3. /findmin(Chat[2,[4,5,6]])[1], 3. /findmin(Chat[3,[7,8,9]])[1]])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.lwlr_chat(res_learn_val, Chat_val), [3. /findmin(Chat[1,[1,2,3]])[1]])
    end
end

@testset "Path Entropies Chat" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.path_entropies_chat(res_learn, Chat), [JudiLingMeasures.entropy([sum(Chat[1,[1,2,3]])]),
                                                               JudiLingMeasures.entropy([sum(Chat[2,[4,5,6]])]),
                                                               JudiLingMeasures.entropy([sum(Chat[3,[7,8,9]])])])
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.path_entropies_chat(res_learn_val, Chat_val), [JudiLingMeasures.entropy([sum(Chat[1,[1,2,3]])])])
    end
end

@testset "Target Path Sum" begin
    @testset "Training data" begin
        @test isapprox(JudiLingMeasures.target_path_sum(gpi_learn), JudiLingMeasures.path_sum(pred_df))
    end
    @testset "Validation data" begin
        @test isapprox(JudiLingMeasures.target_path_sum(gpi_learn_val), JudiLingMeasures.path_sum(pred_df_val))
    end
end

@testset "Path Entropies SCP" begin
    @testset "Training data" begin
        @test JudiLingMeasures.path_entropies_scp(df) == vec([0. 0. 0.])
    end
    @testset "Validation data" begin
        @test JudiLingMeasures.path_entropies_scp(df_val) == vec([0.])
    end
end

@testset "Total Distance" begin
    @testset "Training data" begin
        ngrams = cue_obj.gold_ind
        distances = []
        for ngram in ngrams
            dist1 = Distances.Euclidean()(zeros(size(F,2), 1), F[ngram[1],:])
            dist2 = Distances.Euclidean()(F[ngram[1],:], F[ngram[2],:])
            dist3 = Distances.Euclidean()(F[ngram[2],:], F[ngram[3],:])
            append!(distances, [dist1+dist2+dist3])
        end
        @test isapprox(JudiLingMeasures.total_distance(cue_obj, F, :F), distances)

        ngrams = cue_obj.gold_ind
        distances = []
        for ngram in ngrams
            dist1 = Distances.Euclidean()(zeros(size(G,1), 1), G[:,ngram[1]])
            dist2 = Distances.Euclidean()(G[:,ngram[1]], G[:,ngram[2]])
            dist3 = Distances.Euclidean()(G[:,ngram[2]], G[:,ngram[3]])
            append!(distances, [dist1+dist2+dist3])
        end
        @test isapprox(JudiLingMeasures.total_distance(cue_obj, G, :G), distances)
    end
    @testset "Validation data" begin
        ngrams = cue_obj_val.gold_ind
        distances = []
        for ngram in ngrams
            dist1 = Distances.Euclidean()(zeros(size(F,2), 1), F[ngram[1],:])
            dist2 = Distances.Euclidean()(F[ngram[1],:], F[ngram[2],:])
            dist3 = Distances.Euclidean()(F[ngram[2],:], F[ngram[3],:])
            append!(distances, [dist1+dist2+dist3])
        end
        @test isapprox(JudiLingMeasures.total_distance(cue_obj_val, F, :F), distances)

        ngrams = cue_obj_val.gold_ind
        distances = []
        for ngram in ngrams
            dist1 = Distances.Euclidean()(zeros(size(G,1), 1), G[:,ngram[1]])
            dist2 = Distances.Euclidean()(G[:,ngram[1]], G[:,ngram[2]])
            dist3 = Distances.Euclidean()(G[:,ngram[2]], G[:,ngram[3]])
            append!(distances, [dist1+dist2+dist3])
        end
        @test isapprox(JudiLingMeasures.total_distance(cue_obj_val, G, :G), distances)
    end
end

@testset "Uncertainty" begin
    @testset "Training data" begin
        @testset "correlation" begin
        cor_c = JudiLingMeasures.correlation_rowwise(Chat, cue_obj.C)
        cor_s = JudiLingMeasures.correlation_rowwise(Shat, S)
        @test isapprox(JudiLingMeasures.uncertainty(cue_obj.C, Chat),
                       [sum(JudiLingMeasures.normalise_vector(cor_c[1,:]) .* (ordinalrank(cor_c[1,:]).-1)),
                        sum(JudiLingMeasures.normalise_vector(cor_c[2,:]) .* (ordinalrank(cor_c[2,:]).-1)),
                        sum(JudiLingMeasures.normalise_vector(cor_c[3,:]) .* (ordinalrank(cor_c[3,:]).-1))])
        @test isapprox(JudiLingMeasures.uncertainty(S, Shat),
                       [sum(JudiLingMeasures.normalise_vector(cor_s[1,:]) .* (ordinalrank(cor_s[1,:]).-1)),
                        sum(JudiLingMeasures.normalise_vector(cor_s[2,:]) .* (ordinalrank(cor_s[2,:]).-1)),
                        sum(JudiLingMeasures.normalise_vector(cor_s[3,:]) .* (ordinalrank(cor_s[3,:]).-1))])
        end
        @testset "mse" begin
            mse_c = JudiLingMeasures.mse_rowwise(Chat, cue_obj.C)
            @test isapprox(JudiLingMeasures.uncertainty(cue_obj.C, Chat, method="mse"),
                           [sum(JudiLingMeasures.normalise_vector(mse_c[1,:]) .* (ordinalrank(mse_c[1,:]).-1)),
                            sum(JudiLingMeasures.normalise_vector(mse_c[2,:]) .* (ordinalrank(mse_c[2,:]).-1)),
                            sum(JudiLingMeasures.normalise_vector(mse_c[3,:]) .* (ordinalrank(mse_c[3,:]).-1))])

        end
        @testset "cosine" begin
            cosine_c = JudiLingMeasures.cosine_similarity(Chat, cue_obj.C)
            @test isapprox(JudiLingMeasures.uncertainty(cue_obj.C, Chat, method="cosine"),
                           [sum(JudiLingMeasures.normalise_vector(cosine_c[1,:]) .* (ordinalrank(cosine_c[1,:]).-1)),
                            sum(JudiLingMeasures.normalise_vector(cosine_c[2,:]) .* (ordinalrank(cosine_c[2,:]).-1)),
                            sum(JudiLingMeasures.normalise_vector(cosine_c[3,:]) .* (ordinalrank(cosine_c[3,:]).-1))])
        end
    end
    @testset "Validation data" begin
        @testset "correlation" begin
        cor_c = JudiLingMeasures.correlation_rowwise(Chat_val, vcat(cue_obj_val.C, cue_obj.C))
        cor_s = JudiLingMeasures.correlation_rowwise(Shat_val, vcat(S_val, S))
        @test isapprox(JudiLingMeasures.uncertainty(cue_obj_val.C, Chat_val, cue_obj.C),
                       [sum(JudiLingMeasures.normalise_vector(cor_c[1,:]) .* (ordinalrank(cor_c[1,:]).-1))])
        @test isapprox(JudiLingMeasures.uncertainty(S_val, Shat_val, S),
                       [sum(JudiLingMeasures.normalise_vector(cor_s[1,:]) .* (ordinalrank(cor_s[1,:]).-1))])
        end
        @testset "mse" begin
            mse_c = JudiLingMeasures.mse_rowwise(Chat_val, vcat(cue_obj_val.C, cue_obj.C))
            @test isapprox(JudiLingMeasures.uncertainty(cue_obj_val.C, Chat_val, cue_obj.C, method="mse"),
                           [sum(JudiLingMeasures.normalise_vector(mse_c[1,:]) .* (ordinalrank(mse_c[1,:]).-1))], rtol=1e-4)

        end
        @testset "cosine" begin
            cosine_c = JudiLingMeasures.cosine_similarity(Chat_val, vcat(cue_obj_val.C, cue_obj.C))
            @test isapprox(JudiLingMeasures.uncertainty(cue_obj_val.C, Chat_val, cue_obj.C, method="cosine"),
                           [sum(JudiLingMeasures.normalise_vector(cosine_c[1,:]) .* (ordinalrank(cosine_c[1,:]).-1))])
        end
    end

    if !Sys.iswindows()
      @testset "Test against pyldl" begin
          infl = pandas.DataFrame(pydict(Dict("word"=>["walk","walked","walks"],
                                "lemma"=>["walk","walk","walk"],
                                "person"=>["1/2","1/2/3","3"],
                                "tense"=>["pres","past","pres"])))

           cmat = pm.gen_cmat(infl["word"])
           smat = pm.gen_smat_sim(infl, form="word", sep="/", dim_size=5, seed=10)
           chat = pm.gen_chat(smat=smat, cmat=cmat)
           shat = pm.gen_shat(cmat=cmat, smat=smat)

           cmat_jl = pyconvert(Matrix, cmat)
           chat_jl = pyconvert(Matrix, chat)
           smat_jl = pyconvert(Matrix, smat)
           shat_jl = pyconvert(Matrix, shat)

           words = ["walk", "walked", "walks"]
           expected = [pyconvert(Float64, lmea.uncertainty(word, chat, cmat)) for word in words]

           @test isapprox(JudiLingMeasures.uncertainty(cmat_jl, chat_jl, method="cosine"),
                          expected)

           expected = [pyconvert(Float64, lmea.uncertainty(word, shat, smat)) for word in words]

           @test isapprox(JudiLingMeasures.uncertainty(smat_jl, shat_jl, method="cosine"),
                          expected)

      end
    end
end

@testset "Functional Load" begin
    @testset "Training data" begin
        @testset "Test_within_JudiLingMeasures" begin
            @test isapprox(JudiLingMeasures.functional_load(F, Shat, cue_obj),
                           [cor(F, Shat, dims=2)[[1,2,3], 1],
                            cor(F, Shat, dims=2)[[4,5,6], 2],
                            cor(F, Shat, dims=2)[[7,8,9], 3]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat, cue_obj, cue_list=["#ab", "#bc", "#cd"]),
                           [cor(F, Shat, dims=2)[1, 1],
                            cor(F, Shat, dims=2)[4, 2],
                            cor(F, Shat, dims=2)[7, 3]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat, cue_obj, cue_list=["#ab", "#bc", "#cd"], method="mse"),
                           [JudiLingMeasures.mse_rowwise(F, Shat)[1, 1],
                            JudiLingMeasures.mse_rowwise(F, Shat)[4, 2],
                            JudiLingMeasures.mse_rowwise(F, Shat)[7, 3]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat, cue_obj, method="mse"),
                           [JudiLingMeasures.mse_rowwise(F, Shat)[[1,2,3], 1],
                            JudiLingMeasures.mse_rowwise(F, Shat)[[4,5,6], 2],
                            JudiLingMeasures.mse_rowwise(F, Shat)[[7,8,9], 3]])

        end
    end
    @testset "Validation data" begin
        @testset "Test_within_JudiLingMeasures" begin
            @test isapprox(JudiLingMeasures.functional_load(F, Shat_val, cue_obj_val),
                           [cor(F, Shat_val, dims=2)[[1,2,3], 1]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat_val, cue_obj_val, cue_list=["#ab"]),
                           [cor(F, Shat_val, dims=2)[1, 1]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat_val, cue_obj_val, cue_list=["#ab"], method="mse"),
                           [JudiLingMeasures.mse_rowwise(F, Shat_val)[1, 1]])
            @test isapprox(JudiLingMeasures.functional_load(F, Shat_val, cue_obj_val, method="mse"),
                           [JudiLingMeasures.mse_rowwise(F, Shat_val)[[1,2,3], 1]])

        end
    end

    if !Sys.iswindows()
      @testset "Test against pyldl" begin

           # defining all the stuff necessary for pyldl
           infl = pandas.DataFrame(pydict(Dict("word"=>["walk","walked","walks"],
                                "lemma"=>["walk","walk","walk"],
                                "person"=>["1/2","1/2/3","3"],
                                "tense"=>["pres","past","pres"])))
           cmat = pm.gen_cmat(infl["word"])
           smat = pm.gen_smat_sim(infl, form="word", sep="/", dim_size=5, seed=10)
           fmat = pm.gen_fmat(cmat, smat)
           chat = pm.gen_chat(smat=smat, cmat=cmat)
           shat = pm.gen_shat(cmat=cmat, smat=smat)

           # defining all the stuff necessary for JudiLingMeasures
           infl_jl = DataFrame("word"=>["walk","walked","walks"],
                                "lemma"=>["walk","walk","walk"],
                                "person"=>["1/2","1/2/3","3"],
                                "tense"=>["pres","past","pres"])
           cue_obj_jl = JudiLing.make_cue_matrix(infl_jl, target_col="word", grams=3)

           sfx = ["ed#", "#wa"]

           fmat_jl = pyconvert(Matrix, fmat)
           shat_jl = pyconvert(Matrix, shat)

           expected = pyconvert(Vector{Float64}, [lmea.functional_load("ed#", fmat, "walk", smat, "mse"),
                             lmea.functional_load("#wa", fmat, "walked", smat, "mse")])

           @test isapprox(JudiLingMeasures.functional_load(fmat_jl,
                                                  shat_jl,
                                                  cue_obj_jl,
                                                  cue_list=sfx,
                                                  method="mse"), expected, rtol=1e-3)

                                                  expected = pyconvert(Vector{Float64}, [lmea.functional_load("ed#", fmat, "walk", smat, "corr"),
                                                                    lmea.functional_load("#wa", fmat, "walked", smat, "corr")])

            @test isapprox(JudiLingMeasures.functional_load(fmat_jl,
                                                   shat_jl,
                                                   cue_obj_jl,
                                                   cue_list=sfx,
                                                   method="corr"), expected, rtol=1e-3)
       end
    end
end

@testset "All measures" begin
    @testset "Training data" begin
        # just make sure that this function runs without error
        all_measures =  JudiLingMeasures.compute_all_measures_train(dat, # the data of interest
                                                             cue_obj, # the cue_obj of the training data
                                                             Chat, # the Chat of the data of interest
                                                             S, # the S matrix of the data of interest
                                                             Shat, # the Shat matrix of the data of interest
                                                             F, # the F matrix
                                                             G,
                                                             res_learn_train=res_learn, # the output of learn_paths for the data of interest
                                                             gpi_learn_train=gpi_learn, # the gpi_learn object of the data of interest
                                                             rpi_learn_train=rpi_learn,# the rpi_learn object of the data of interest
                                                             sem_density_n=2)
        @test all_measures != 1
        @test !("ProductionUncertainty" in names(all_measures))
        all_measures =  JudiLingMeasures.compute_all_measures_train(dat, # the data of interest
                                                             cue_obj, # the cue_obj of the training data
                                                             Chat, # the Chat of the data of interest
                                                             S, # the S matrix of the data of interest
                                                             Shat, # the Shat matrix of the data of interest
                                                             F, # the F matrix
                                                             G,
                                                             res_learn_train=res_learn, # the output of learn_paths for the data of interest
                                                             gpi_learn_train=gpi_learn, # the gpi_learn object of the data of interest
                                                             rpi_learn_train=rpi_learn,# the rpi_learn object of the data of interest
                                                             sem_density_n=2,
                                                             calculate_production_uncertainty=true)
         @test all_measures != 1
         @test "ProductionUncertainty" in names(all_measures)
         all_measures =  JudiLingMeasures.compute_all_measures_train(dat, # the data of interest
                                                              cue_obj, # the cue_obj of the training data
                                                              Chat, # the Chat of the data of interest
                                                              S, # the S matrix of the data of interest
                                                              Shat, # the Shat matrix of the data of interest
                                                              F, # the F matrix
                                                              G,
                                                              res_learn_train=res_learn, # the output of learn_paths for the data of interest
                                                              gpi_learn_train=gpi_learn, # the gpi_learn object of the data of interest
                                                              rpi_learn_train=rpi_learn,# the rpi_learn object of the data of interest
                                                              sem_density_n=2,
                                                              low_cost_measures_only=true)
          @test all_measures != 1
          @test !("DistanceTravelledF" in names(all_measures))
          @test !("DistanceTravelledG" in names(all_measures))
          @test "WithinPathEntropies" in names(all_measures)
          @test !("ProductionUncertainty" in names(all_measures))
          all_measures =  JudiLingMeasures.compute_all_measures_train(dat, # the data of interest
                                                               cue_obj, # the cue_obj of the training data
                                                               Chat, # the Chat of the data of interest
                                                               S, # the S matrix of the data of interest
                                                               Shat, # the Shat matrix of the data of interest
                                                               F, # the F matrix
                                                               G,
                                                               sem_density_n=2)
           @test all_measures != 1
           @test "DistanceTravelledF" in names(all_measures)
           @test "DistanceTravelledG" in names(all_measures)
           @test !("WithinPathEntropies" in names(all_measures))

           all_measures =  JudiLingMeasures.compute_all_measures_train(dat, # the data of interest
                                                                cue_obj, # the cue_obj of the training data
                                                                Chat, # the Chat of the data of interest
                                                                S, # the S matrix of the data of interest
                                                                Shat, # the Shat matrix of the data of interest
                                                                sem_density_n=2)
            @test all_measures != 1
            @test !("DistanceTravelledF" in names(all_measures))
            @test !("DistanceTravelledG" in names(all_measures))
    end
    @testset "Validation data" begin
        # just make sure that this function runs without error
        all_measures =  JudiLingMeasures.compute_all_measures_val(val_dat, # the data of interest
                                                             cue_obj, # the cue_obj of the training data
                                                             cue_obj_val,
                                                             Chat_val, # the Chat of the data of interest
                                                             S, # the S matrix of the data of interest
                                                             S_val,
                                                             Shat_val, # the Shat matrix of the data of interest
                                                             F, # the F matrix
                                                             G,
                                                             res_learn_val=res_learn_val, # the output of learn_paths for the data of interest
                                                             gpi_learn_val=gpi_learn_val, # the gpi_learn object of the data of interest
                                                             rpi_learn_val=rpi_learn_val,# the rpi_learn object of the data of interest
                                                             sem_density_n=2)
        @test all_measures != 1
        @test !("ProductionUncertainty" in names(all_measures))
        all_measures =  JudiLingMeasures.compute_all_measures_val(val_dat, # the data of interest
                                                             cue_obj, # the cue_obj of the training data
                                                             cue_obj_val,
                                                             Chat_val, # the Chat of the data of interest
                                                             S, # the S matrix of the data of interest
                                                             S_val,
                                                             Shat_val, # the Shat matrix of the data of interest
                                                             F, # the F matrix
                                                             G,
                                                             res_learn_val=res_learn_val, # the output of learn_paths for the data of interest
                                                             gpi_learn_val=gpi_learn_val, # the gpi_learn object of the data of interest
                                                             rpi_learn_val=rpi_learn_val,# the rpi_learn object of the data of interest
                                                             sem_density_n=2,
                                                             calculate_production_uncertainty=true)
         @test all_measures != 1
         @test "ProductionUncertainty" in names(all_measures)
         all_measures =  JudiLingMeasures.compute_all_measures_val(val_dat, # the data of interest
                                                              cue_obj, # the cue_obj of the training data
                                                              cue_obj_val,
                                                              Chat_val, # the Chat of the data of interest
                                                              S, # the S matrix of the data of interest
                                                              S_val,
                                                              Shat_val, # the Shat matrix of the data of interest
                                                              F, # the F matrix
                                                              G,
                                                              res_learn_val=res_learn_val, # the output of learn_paths for the data of interest
                                                              gpi_learn_val=gpi_learn_val, # the gpi_learn object of the data of interest
                                                              rpi_learn_val=rpi_learn_val,# the rpi_learn object of the data of interest
                                                              sem_density_n=2,
                                                              low_cost_measures_only=true)
          @test all_measures != 1
          @test !("DistanceTravelledF" in names(all_measures))
          @test !("DistanceTravelledG" in names(all_measures))
          @test "WithinPathEntropies" in names(all_measures)
          @test !("ProductionUncertainty" in names(all_measures))
          all_measures =  JudiLingMeasures.compute_all_measures_val(val_dat, # the data of interest
                                                               cue_obj, # the cue_obj of the training data
                                                               cue_obj_val,
                                                               Chat_val, # the Chat of the data of interest
                                                               S, # the S matrix of the data of interest
                                                               S_val,
                                                               Shat_val, # the Shat matrix of the data of interest
                                                               F, # the F matrix
                                                               G,
                                                               sem_density_n=2)
           @test "DistanceTravelledF" in names(all_measures)
           @test "DistanceTravelledG" in names(all_measures)
           @test !("WithinPathEntropies" in names(all_measures))

           all_measures =  JudiLingMeasures.compute_all_measures_val(val_dat, # the data of interest
                                                                cue_obj, # the cue_obj of the training data
                                                                cue_obj_val,
                                                                Chat_val, # the Chat of the data of interest
                                                                S, # the S matrix of the data of interest
                                                                S_val,
                                                                Shat_val,
                                                                sem_density_n=2)
            @test all_measures != 1
            @test !("DistanceTravelledF" in names(all_measures))
            @test !("DistanceTravelledG" in names(all_measures))
    end
end
