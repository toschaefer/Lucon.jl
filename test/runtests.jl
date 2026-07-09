using Lucon
using Test
using LinearAlgebra
using Logging
using Random

include("../src/BrockettLoss.jl")
import .BrockettLoss

# the optimizer reports on every iteration, which would drown the test output
silent(f) = with_logger(f, NullLogger())

# the Brockett criterion L(U) = tr(U'HUN) with N_nm = n δ_nm
BrockettLossValue(H, U) = real(tr(U' * H * U * Diagonal(1.0:size(H,1))))

dim = 10
rng = MersenneTwister(2023)
Hcomplex = Hermitian(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
Hreal    = Hermitian(rand(rng,dim,dim) .- 0.5)


@testset "Lucon.jl" begin

    # a hermitian matrix, diagonalized from a random unitary matrix
    @testset "unitary group" begin
        BL = BrockettLoss.LossFunctional(Hcomplex)
        (U, _) = qr(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
        (U, _) = silent() do
            BrockettLoss.optimize(BL, Matrix(U), GradNormBreak=1.0E-8)
        end
        Σdiff = Diagonal(eigen(Hcomplex).values) - U'*Hcomplex*U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7 # should be < 1.0E-7 if GradNormBreak=1.0E-8
    end

    # the identity is a real valued matrix, but on the unitary group both the ascent
    # direction and the rotation exp(μH) it generates are complex
    @testset "unitary group, starting from the identity" begin
        BL = BrockettLoss.LossFunctional(Hcomplex)
        (U, _) = silent() do
            BrockettLoss.optimize(BL, Matrix{ComplexF64}(I,dim,dim), GradNormBreak=1.0E-8)
        end
        @test norm(U'*U - I) < 1.0E-10
        Σdiff = Diagonal(eigen(Hcomplex).values) - U'*Hcomplex*U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7
    end

    # a real symmetric matrix has to stay on the orthogonal group
    @testset "orthogonal group" begin
        BL = BrockettLoss.LossFunctional(Hreal)
        (U, _) = silent() do
            BrockettLoss.optimize(BL, Matrix{Float64}(I,dim,dim), GradNormBreak=1.0E-8)
        end
        @test eltype(U) == Float64
        @test norm(U'*U - I) < 1.0E-10
        Σdiff = Diagonal(eigen(Hreal).values) - U'*Hreal*U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7
    end

    # MaxIter counts the rotations of U, and the loss belongs to the U that is returned
    @testset "returned loss and matrix are consistent" begin
        BL = BrockettLoss.LossFunctional(Hcomplex)
        U0 = Matrix{ComplexF64}(I,dim,dim)
        for MaxIter in (0, 1, 5)
            (U, Loss) = silent() do
                BrockettLoss.optimize(BL, copy(U0), MaxIter=MaxIter)
            end
            @test Loss ≈ BrockettLossValue(Hcomplex, U)
        end
        (U, _) = silent() do
            BrockettLoss.optimize(BL, copy(U0), MaxIter=0)
        end
        @test U == U0
    end

    # the line search reports a step size of zero rather than throwing
    @testset "polynomial of the line search without a positive root" begin
        @test Lucon.SmallestPositiveRoot([-2.0, -1.0, 1.0]) ≈ 2.0 # (μ+1)(μ-2)
        @test Lucon.SmallestPositiveRoot([1.0, 0.0, 1.0]) === nothing # μ²+1
        @test Lucon.SmallestPositiveRoot([-2.0, 1.0, 0.0]) ≈ 2.0 # vanishing leading coefficient
    end

end
