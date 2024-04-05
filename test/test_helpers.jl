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
    @test vec(JudiLingMeasures.l2_rowwise(zeros(1,1))) == [0]
    @test vec(JudiLingMeasures.l2_rowwise(ones(1,1))) == [1]
    @test isequal(vec(JudiLingMeasures.l2_rowwise([[1 2 missing]; [-1 -2 -3]; [1 2 3]])), [missing; sqrt(14); sqrt(14)])
end

@testset "correlation_rowwise" begin
    @test isapprox(JudiLingMeasures.correlation_rowwise(ma2, ma3),
                   [[0.662266   0.174078    0.816497  -0.905822];
                    [-0.41762    0.29554    -0.990148   0.988623];
                    [-0.308304   0.0368355  -0.863868   0.862538];
                    [0.207514  -0.0909091  -0.426401   0.354787]], rtol=1e-4)
     @test isapprox(JudiLingMeasures.correlation_rowwise(ma2, ma3),
                    JudiLing.eval_SC(ma2, ma3, R=true)[2], rtol=1e-4)
    @test isapprox(JudiLingMeasures.correlation_rowwise([1. 2. 3.], [5. 1. 19.]),
                   [0.7406128966515281])
   @test isequal(JudiLingMeasures.correlation_rowwise([1. 2. missing], [5. 1. 19.]), fill(missing, 1,1))
   @test ismissing(JudiLingMeasures.correlation_rowwise(Matrix(undef, 0,0), Matrix(undef, 0,0)))
end

@testset "sem_density_mean" begin
    @test isapprox(vec(JudiLingMeasures.sem_density_mean(cor_s, 2)),
                   vec([0.7393784999999999 0.6420815 0.44968675 0.2811505]), rtol=1e-4)
    cs = JudiLingMeasures.correlation_rowwise([1. 2. 3.], [5. 1. 19.])
    @test isapprox(vec(JudiLingMeasures.sem_density_mean(cs,1)),
                   vec([0.7406128966515281]))
    @test_throws ArgumentError JudiLingMeasures.sem_density_mean(cs,5)
    cs = JudiLingMeasures.correlation_rowwise(ma2, ma3)
    @test isapprox(vec(JudiLingMeasures.sem_density_mean(cs, 3)),
                   vec([0.550947 0.28884766666666667 0.19702316666666667 0.15713063333333335]), rtol=1e-4)
end

@testset "mean_rowwise" begin
    @test vec(JudiLingMeasures.mean_rowwise(ma1)) == [2.; -2; 2]
    @test vec(JudiLingMeasures.mean_rowwise([1. 2. 3.])) == [2.]
    @test ismissing(JudiLingMeasures.mean_rowwise(Matrix(undef, 0,0)))
    @test vec(JudiLingMeasures.mean_rowwise(fill(3., 1,1))) == [3.]
end

@testset "euclidean_distance_rowwise" begin
    @test isapprox(JudiLingMeasures.euclidean_distance_rowwise(ma1, ma4), [[1. sqrt(52) 1.];
                                                        [sqrt(45) sqrt(4) sqrt(53)];
                                                        [1. sqrt(52) 1.]])
    @test isapprox(JudiLingMeasures.euclidean_distance_rowwise([1. 2. 3.],
                                                               [5. 1. 19.]),
                   [16.522711641858304])
   @test isapprox(JudiLingMeasures.euclidean_distance_rowwise([1. 2.],
                                                              [5. 1.]),
                  JudiLingMeasures.l2_rowwise([1. 2.] .- [5. 1.]))
  @test isapprox(JudiLingMeasures.euclidean_distance_rowwise(fill(1., 1,1),
                                                             fill(5., 1,1)),
                 JudiLingMeasures.l2_rowwise(fill(-4., 1,1)))
end

@testset "get_nearest_neighbour_eucl" begin
    eucl_sims = JudiLingMeasures.euclidean_distance_rowwise(ma1, ma4)
    @test JudiLingMeasures.get_nearest_neighbour_eucl(eucl_sims) == [1., sqrt(4), 1.]
    eucl_sims = JudiLingMeasures.euclidean_distance_rowwise([1. 2. 3.],
                                                               [5. 1. 19.])
    @test JudiLingMeasures.get_nearest_neighbour_eucl(eucl_sims) == [16.522711641858304]
end

@testset "max_rowwise" begin
    @test vec(JudiLingMeasures.max_rowwise(ma1)) == [3., -1, 3]
    @test vec(JudiLingMeasures.max_rowwise(ma3)) == [2, 3, 2, 1.5]
    @test vec(JudiLingMeasures.max_rowwise(Matrix(undef, 0, 0))) == []
    @test isequal(vec(JudiLingMeasures.max_rowwise(fill(missing, 1,1))), [missing])
end

@testset "count_rows" begin
    df = DataFrame("test"=>[1, 2, 3])
    @test JudiLingMeasures.count_rows(df) == 3
    @test JudiLingMeasures.count_rows(DataFrame()) == 0
end

@testset "get_avg_levenshtein" begin
    @test JudiLingMeasures.get_avg_levenshtein(["abc", "abc", "abc"],
                                               ["abd", "abc", "ebd"]) == 1.
   @test JudiLingMeasures.get_avg_levenshtein(["", ""],
                                              ["", ""]) == 0.
  @test ismissing(JudiLingMeasures.get_avg_levenshtein([],
                                                       []))
@test ismissing(JudiLingMeasures.get_avg_levenshtein([missing],
                                                    [missing]))
end

@testset "entropy" begin
    @test ismissing(JudiLingMeasures.entropy([]))
    @test isapprox(JudiLingMeasures.entropy([0.1,0.2,0.3]), 1.4591479170272448)
    @test ismissing(JudiLingMeasures.entropy([0., 0.]))
    @test ismissing(JudiLingMeasures.entropy([1., missing]))
    @test isapprox(JudiLingMeasures.entropy([5. 9. 12. 13.]), 1.9196526847108202)
end

@testset "correlation_diagonal_rowwise" begin
    @test isapprox(JudiLingMeasures.correlation_diagonal_rowwise(ma2, ma3),
                   diag(cor_s))
   @test isapprox(JudiLingMeasures.correlation_diagonal_rowwise([1. 2. 3.], [5. 1. 19.]),
                 [0.7406128966515281])
     @test isapprox(JudiLingMeasures.correlation_diagonal_rowwise([[1. 2. 3.]
                                                                   [1. 2. 3.]],
                                                                  [[5. 1. 19.]
                                                                   [5. 1. 19.]]),
                   [0.7406128966515281, 0.7406128966515281])
   @test isequal(JudiLingMeasures.correlation_diagonal_rowwise([1.],
                                                                [1.]),
                 [NaN])
end

@testset "cosine_similarity" begin
    @test isapprox(JudiLingMeasures.cosine_similarity(ma1, ma4),
                   [[0.979958 -0.857143 0.963624]
                    [-0.979958 0.857143 -0.963624]
                    [0.979958 -0.857143 0.963624]], rtol=1e-4)
    @test isapprox(JudiLingMeasures.cosine_similarity([1. 2. 3.], [5. 1. 19.]),
                  [0.8694817556685039])
      @test isapprox(JudiLingMeasures.cosine_similarity([1. 2. 3.], [5. 1. 19.]),
                    JudiLingMeasures.cosine_similarity([5. 1. 19.], [1. 2. 3.]))
  @test isapprox(JudiLingMeasures.cosine_similarity([[1. 2. 3.]
                                                     [4. 2. 7]],
                                                    [[5. 1. 19.]
                                                     [18. 12. 6.]]),
                [[0.8694817556685039 0.7142857142857143]
                 [0.9485313083322907 0.7400128699009549]])
end

@testset "safe_sum" begin
    @test ismissing(JudiLingMeasures.safe_sum([]))
    @test JudiLingMeasures.safe_sum([1,2,3]) == 6
    @test JudiLingMeasures.safe_sum([1]) == 1
end

@testset "safe_length" begin
    @test ismissing(JudiLingMeasures.safe_length(missing))
    @test JudiLingMeasures.safe_length("abc") == 3
    @test JudiLingMeasures.safe_length("") == 0
end

@testset "safe_divide" begin
    @test ismissing(JudiLingMeasures.safe_divide(1, missing))
    @test ismissing(JudiLingMeasures.safe_divide(missing, 1))
    @test ismissing(JudiLingMeasures.safe_divide(1, 0))
    @test isapprox(JudiLingMeasures.safe_divide(1.,2.), 1. /2.)
end

@testset "mse_rowwise" begin
    @test isapprox(JudiLingMeasures.mse_rowwise([0.855642  0.160356  0.134059],
                                                [0.645707  0.258852  0.79831]),
                   [0.1650011857473333])
   @test isapprox(JudiLingMeasures.mse_rowwise([[0.855642  0.160356  0.134059]
                                                [0.855642  0.160356  0.134059]],
                                               [[0.645707  0.258852  0.79831]
                                                [0.645707  0.258852  0.79831]]),
                  fill(0.1650011857473333, 2,2))

     @test isapprox(JudiLingMeasures.mse_rowwise([1. 2. 3.],
                                                 [1. 5. 9.]),
                    [15.0])

    @test isapprox(JudiLingMeasures.mse_rowwise([[1. 2. 3.]
                                                 [5. 19. 2.]],
                                                [[1. 5. 9.]
                                                 [13. 2. 1.]]),
                   [[15.0  49.333333333333336]
                    [87.0 118.0]])

end

@testset "normalise_vector" begin
    @test isapprox(JudiLingMeasures.normalise_vector([1.,2.,3.]),
                   [0., 0.5, 1.])
   @test isapprox(JudiLingMeasures.normalise_vector([-1.,-2.,-3.]),
                  [1., 0.5, 0.])
  @test isequal(JudiLingMeasures.normalise_vector([1., 1.]),
                 [NaN, NaN])
    @test JudiLingMeasures.normalise_vector([]) == []
end

@testset "normalise_matrix_rowwise" begin
    @test isapprox(JudiLingMeasures.normalise_matrix_rowwise(ma1),
                   [[0. 0.5 1.]; [1. 0.5 0.]; [0. 0.5 1.]])
   @test isapprox(JudiLingMeasures.normalise_matrix_rowwise([[1. 2. 3.]
                                                             [-1. -2. -3.]]),
                  [[0. 0.5 1.]; [1. 0.5 0.]])

  @test isequal(JudiLingMeasures.normalise_matrix_rowwise([[1. 1. 1.]
                                                            [1. 1. 1.]]),
                 fill(NaN, 2, 3))
    @test JudiLingMeasures.normalise_matrix_rowwise(Matrix(undef, 0,0)) == Matrix(undef, 0,0)
end
