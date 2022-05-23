#############################
# Test helper functions
#############################

# define some data to test with
ma1 = [[1 2 3]; [-1 -2 -3]; [1 2 3]]
ma2 = [[1 2 1 1]; [1 -2 3 1]; [1 -2 3 3]; [0 0 1 2]]
ma3 = [[-1 2 1 1]; [1 2 3 1]; [1 2 0 1]; [0.5 -2 1.5 0]]
ma4 = [[1 2 2]; [1 -2 -3]; [0 2 3]]

cor_s = JudiLingMeasures.correlation_rowwise(ma2, ma3)

# tests
@testset "l1_rowwise" begin
    @test vec(JudiLingMeasures.l1_rowwise(ma1)) == [6; 6; 6]
    @test vec(JudiLingMeasures.l1_rowwise(zeros(1,1))) == [0]
    @test vec(JudiLingMeasures.l1_rowwise(ones(1,1))) == [1]
    @test isequal(vec(JudiLingMeasures.l1_rowwise([[1 2 missing]; [-1 -2 -3]; [1 2 3]])), [missing; 6; 6])
end

@testset "l2_rowwise" begin
    @test vec(JudiLingMeasures.l2_rowwise(ma1)) == vec([sqrt(14); sqrt(14); sqrt(14)])
end

@testset "correlation_rowwise" begin
    @test isapprox(JudiLingMeasures.correlation_rowwise(ma2, ma3),   [[0.662266   0.174078    0.816497  -0.905822];
                                             [-0.41762    0.29554    -0.990148   0.988623];
                                             [-0.308304   0.0368355  -0.863868   0.862538];
                                             [0.207514  -0.0909091  -0.426401   0.354787]], rtol=1e-4)
     @test isapprox(JudiLingMeasures.correlation_rowwise(ma2, ma3),  JudiLing.eval_SC(ma2, ma3, R=true)[2], rtol=1e-4)
end

@testset "sem_density_mean" begin
    @test isapprox(vec(JudiLingMeasures.sem_density_mean(cor_s, 2)),
                   vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)
end

@testset "mean_rowwise" begin
    @test vec(JudiLingMeasures.mean_rowwise(ma1)) == [2.; -2; 2]
end

@testset "euclidean_distance_array" begin
    @test isapprox(JudiLingMeasures.euclidean_distance_array(ma1, ma4), [[1. sqrt(52) 1.];
                                                        [sqrt(45) sqrt(4) sqrt(53)];
                                                        [1. sqrt(52) 1.]])
end

@testset "get_nearest_neighbour_eucl" begin
    eucl_sims = JudiLingMeasures.euclidean_distance_array(ma1, ma4)

    @test JudiLingMeasures.get_nearest_neighbour_eucl(eucl_sims) == [1., sqrt(4), 1.]
end

@testset "max_rowwise" begin
    @test vec(JudiLingMeasures.max_rowwise(ma1)) == [3., -1, 3]
end

@testset "count_rows" begin
    df = DataFrame("test"=>[1, 2, 3])
    @test JudiLingMeasures.count_rows(df.test) == 3
end

@testset "get_avg_levenshtein" begin
    @test JudiLingMeasures.get_avg_levenshtein(["abc", "abc", "abc"],
                                               ["abd", "abc", "ebd"]) == 1.
end

@testset "entropy" begin
    @test ismissing(JudiLingMeasures.entropy([]))
    @test isapprox(entropy([0.1,0.2,0.3]), 0.91, rtol=1e-2)
end

@testset "correlation_diagonal_rowwise" begin
    @test isapprox(JudiLingMeasures.correlation_diagonal_rowwise(ma2, ma3),
                   diag(cor_s))
end

@testset "cosine_similarity" begin
    @test isapprox(JudiLingMeasures.cosine_similarity(ma1, ma4),
                   [[0.979958 -0.857143 0.963624]
                    [-0.979958 0.857143 -0.963624]
                    [0.979958 -0.857143 0.963624]], rtol=1e-4)
end

@testset "safe_sum" begin
    @test ismissing(JudiLingMeasures.safe_sum([]))
    @test JudiLingMeasures.safe_sum([1,2,3]) == 6
end

@testset "safe_length" begin
    @test ismissing(JudiLingMeasures.safe_length(missing))
    @test JudiLingMeasures.safe_length("abc") == 3
end

@testset "mse_rowwise" begin
    @test isapprox(JudiLingMeasures.mse_rowwise([0.855642  0.160356  0.134059],
                                                [0.645707  0.258852  0.79831]),
                   [0.1650011857473333])

end

@testset "normalise_vector" begin
    @test isapprox(JudiLingMeasures.normalise_vector([1.,2.,3.]),
                   [0., 0.5, 1.])
end

@testset "normalise_matrix_rowwise" begin
    @test isapprox(JudiLingMeasures.normalise_matrix_rowwise(ma1),
                   [[0. 0.5 1.]; [1. 0.5 0.]; [0. 0.5 1.]])
end
