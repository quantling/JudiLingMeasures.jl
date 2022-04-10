#############################
# Test helper functions
#############################

# define some data to test with
ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]

# tests
@test vec(JudiLingMeasures.l1_rowwise(ma1)) == [6; 6; 6]

@test vec(JudiLingMeasures.l2_rowwise(ma1)) == vec([sqrt(14); sqrt(14); sqrt(14)])

@test isapprox(JudiLingMeasures.correlation_rowwise(ma2, ma3),   [[0.662266   0.174078    0.816497  -0.905822];
                                         [-0.41762    0.29554    -0.990148   0.988623];
                                         [-0.308304   0.0368355  -0.863868   0.862538];
                                         [0.207514  -0.0909091  -0.426401   0.354787]], rtol=1e-4)

cor_s = JudiLingMeasures.correlation_rowwise(ma2, ma3)

@test isapprox(vec(JudiLingMeasures.sem_density_mean(cor_s, 2)), vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)

@test vec(JudiLingMeasures.mean_rowwise(ma1)) == [2.; -2; 2]

@test isapprox(JudiLingMeasures.euclidean_distance_array(ma1, ma4), [[1. sqrt(52) 1.];
                                                    [sqrt(45) sqrt(4) sqrt(53)];
                                                    [1. sqrt(52) 1.]])

eucl_sims = JudiLingMeasures.euclidean_distance_array(ma1, ma4)

@test JudiLingMeasures.get_nearest_neighbour_eucl(eucl_sims) == [1., sqrt(4), 1.]

@test vec(JudiLingMeasures.max_rowwise(ma1)) == [3., -1, 3]

df = DataFrame("test"=>[1, 2, 3])
@test JudiLingMeasures.count_rows(df.test) == 3

@test JudiLingMeasures.get_avg_levenshtein(["abc", "abc", "abc"], ["abd", "abc", "ebd"]) == 1.

@test ismissing(JudiLingMeasures.entropy([]))

#@test isapprox(entropy([0.1,0.2,0.3]), 0.91, rtol=1e-2)


dat = DataFrame("Word"=>["abc", "bcd", "cde"])
cue_obj = JudiLing.make_cue_matrix(
    dat,
    grams=3,
    target_col=:Word,
    tokenized=false,
    keep_sep=false
    )

@test isapprox(JudiLingMeasures.correlation_diagonal_rowwise(ma2, ma3), diag(cor_s))

@test isapprox(JudiLingMeasures.cosine_similarity(ma1, ma4), [[0.979958 -0.857143 0.963624]
                                             [-0.979958 0.857143 -0.963624]
                                             [0.979958 -0.857143 0.963624]], rtol=1e-4)

@test ismissing(JudiLingMeasures.safe_sum([]))

@test JudiLingMeasures.safe_sum([1,2,3]) == 6

@test ismissing(JudiLingMeasures.safe_length(missing))

@test JudiLingMeasures.safe_length("abc") == 3
