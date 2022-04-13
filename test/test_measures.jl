########################################
# test measures
########################################

# define some data to test with
ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]

# define some data to test with
dat = DataFrame("Word"=>["abc", "bcd", "cde"])
cue_obj = JudiLing.make_cue_matrix(
    dat,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )
n_features = size(cue_obj.C, 2)
S = JudiLing.make_S_matrix(
    dat,
    ["Word"],
    [],
    ncol=n_features)
G = JudiLing.make_transform_matrix(S, cue_obj.C)
Chat = S * G
F = JudiLing.make_transform_matrix(cue_obj.C, S)
Shat = cue_obj.C * F
A = cue_obj.A
max_t = JudiLing.cal_max_timestep(dat, :Word)

res_learn, gpi_learn, rpi_learn = JudiLingMeasures.learn_paths_rpi(
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

results, cor_s_all, df, pred_df = JudiLingMeasures.make_measure_preparations(dat, S, Shat,
                                   res_learn, cue_obj, cue_obj, rpi_learn)


# tests

@testset "Make measure preparations" begin
    @test cor_s_all == cor(Shat, S, dims=2)
    @test results == dat
end

@testset "L1 Norm" begin
    @test JudiLingMeasures.L1Norm(ma1) == [6; 6; 6]
    @test JudiLingMeasures.L1Norm(zeros((1,1))) == [0]
    @test JudiLingMeasures.L1Norm(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.L1Norm([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; 6; 6])
end

@testset "L2 Norm" begin
    @test JudiLingMeasures.L2Norm(ma1) == [sqrt(14); sqrt(14); sqrt(14)]
    @test JudiLingMeasures.L2Norm(zeros((1,1))) == [0]
    @test JudiLingMeasures.L2Norm(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.L2Norm([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; sqrt(14); sqrt(14)])
end


cor_s = JudiLingMeasures.correlation_rowwise(ma2, ma3)

@testset "Density" begin
    @test isapprox(JudiLingMeasures.density(cor_s, n=2), vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)
    @test JudiLingMeasures.density(zeros((1,1)), n=1) == [0]
    @test JudiLingMeasures.density(ones((1,1)), n=1) == [1]
    @test isequal(JudiLingMeasures.density([[1 2 missing]; [-1 -2 -3]; [1 2 3]], n=2), [missing; -1.5; 2.5])
    @test_throws MethodError JudiLingMeasures.density(zeros((1,1))) == [0]
end

@testset "ALC" begin
    @test isapprox(JudiLingMeasures.ALC(cor_s), [0.18675475, -0.03090124999999999, -0.06819962499999999, 0.011247725000000014], rtol=1e-4)
    @test JudiLingMeasures.ALC(zeros((1,1))) == [0]
    @test JudiLingMeasures.ALC(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.ALC([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; -2.; 2.])
end

@testset "EDNN" begin
    @test JudiLingMeasures.EDNN(ma1, ma4) == [1., sqrt(4), 1.]
    @test JudiLingMeasures.EDNN(zeros((1,1)), zeros((1,1))) == [0]
    @test JudiLingMeasures.EDNN(ones((1,1)), zeros((1,1))) == [1]
    @test_throws MethodError JudiLingMeasures.EDNN([[1 2 missing]; [-1 -2 -3]; [1 2 3]], ma4)
end

@testset "NNC" begin
    @test isapprox(JudiLingMeasures.NNC(cor_s), [0.816497, 0.988623, 0.862538, 0.354787], rtol=1e-4)
    @test JudiLingMeasures.NNC(zeros((1,1))) == [0]
    @test JudiLingMeasures.NNC(ones((1,1))) == [1]
    @test isequal(JudiLingMeasures.NNC([[1 2 missing]; [-1 -2 -3]; [1 2 3]]), [missing; -1; 3])
end

@testset "last_support" begin
    @test isapprox(JudiLingMeasures.last_support(cue_obj, Chat), [0.99991, 0.999773, 0.999857], rtol=1e-4)
end

@testset "path_counts" begin
    @test JudiLingMeasures.path_counts(df) == [1,1, 1]
    df_mock = DataFrame("utterance"=>[1,1],
                    "pred"=>["abc", "abd"])
    @test JudiLingMeasures.path_counts(df_mock) == [2]
    df_mock2 = DataFrame()
    @test_throws ArgumentError JudiLingMeasures.path_counts(df_mock2)
end

@testset "path_sum" begin
    @test isapprox(JudiLingMeasures.path_sum(pred_df), [2.979, 2.979, 2.979], rtol=1e-3)
    pred_df_mock = DataFrame("timestep_support"=>[missing, [1,2,3], [0,0,0], [0,1,missing]])
    @test isequal(JudiLingMeasures.path_sum(pred_df_mock), [missing; 6; 0; missing])
end

@testset "within_path_entropies" begin
    # Note: the result of this is different to other entropy measures as a) the values are scaled between 0 and 1 first, and b) log2 instead of log is used
    @test isapprox(JudiLingMeasures.within_path_entropies(pred_df), [1.584962500721156, 1.584962500721156, 1.584962500721156], rtol=1e-1)
    pred_df_mock = DataFrame("timestep_support"=>[missing, [0,1,missing]])
    @test isequal(JudiLingMeasures.within_path_entropies(pred_df_mock), [missing, missing])
    pred_df_mock2 = DataFrame("timestep_support"=>[[1,2,3], [0,0,0]])
    @test isapprox(JudiLingMeasures.within_path_entropies(pred_df_mock2), [JudiLingMeasures.entropy([1,2,3]), 0.])
end

@testset "ALDC" begin
    @test JudiLingMeasures.ALDC(df) == [0, 0, 0]
    df_mock = DataFrame("utterance"=>[1,1],
                    "pred"=>["abc", "abd"],
                    "identifier"=>["abc", "abc"])
    @test JudiLingMeasures.ALDC(df_mock) == [0.5]
end

@testset "Mean word support" begin
    @test isapprox(JudiLingMeasures.mean_word_support(res_learn, pred_df), [0.993288, 0.993288, 0.993288], rtol=1e-4)
end

@test isapprox(JudiLingMeasures.target_correlation(cor_s), [0.662266, 0.29554, -0.863868, 0.354787], rtol=1e-4)

@test JudiLingMeasures.rank(cor_s) == [2,2,4,1]

@test isapprox(JudiLingMeasures.lwlr(res_learn, pred_df), [3. /0.993288, 3. /0.993152, 3. /0.993235], rtol=1e-4)

@test isapprox(JudiLingMeasures.path_sum_chat(res_learn, Chat), [sum(Chat[1,[1,2,3]]), sum(Chat[2,[4,5,6]]), sum(Chat[3,[7,8,9]])])

@test isapprox(JudiLingMeasures.L1Norm(Chat), map(sum, eachrow(abs.(Chat))))

@test isapprox(JudiLingMeasures.c_precision(Chat, cue_obj.C), diag(JudiLing.eval_SC(Chat, cue_obj.C, R=true)[2]))

@test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj, Chat), [sum(Chat[1,[1,2,3]]), sum(Chat[2,[4,5,6]]), sum(Chat[3,[7,8,9]])])

@test isapprox(JudiLingMeasures.semantic_support_for_form(cue_obj, Chat), JudiLingMeasures.path_sum_chat(res_learn, Chat))

@test isapprox(JudiLingMeasures.SCPP(df, dat), JudiLingMeasures.NNC(cor_s_all))

@test isapprox(JudiLingMeasures.mean_word_support_chat(res_learn, Chat), [sum(Chat[1,[1,2,3]])/3, sum(Chat[2,[4,5,6]])/3, sum(Chat[3,[7,8,9]])/3])

@test isapprox(JudiLingMeasures.lwlr_chat(res_learn, Chat), [3. /findmin(Chat[1,[1,2,3]])[1], 3. /findmin(Chat[2,[4,5,6]])[1], 3. /findmin(Chat[3,[7,8,9]])[1]])

@test isapprox(JudiLingMeasures.path_entropies_chat(res_learn, Chat), [JudiLingMeasures.entropy([sum(Chat[1,[1,2,3]])]),
                                                       JudiLingMeasures.entropy([sum(Chat[2,[4,5,6]])]),
                                                       JudiLingMeasures.entropy([sum(Chat[3,[7,8,9]])])])

@test isapprox(JudiLingMeasures.target_path_sum(gpi_learn), JudiLingMeasures.path_sum(pred_df))

@test JudiLingMeasures.path_entropies_scp(df) == vec([0. 0. 0.])
