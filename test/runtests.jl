using Lucon
using Test
using LinearAlgebra
using Random

include("../src/BrockettLoss.jl")
import .BrockettLoss

# the Brockett criterion L(U) = tr(U'HUN) with N_nm = n δ_nm
BrockettLossValue(H, U) = real(tr(U' * H * U * Diagonal(1.0:size(H,1))))

dim = 10
rng = MersenneTwister(2023)
Hcomplex = Hermitian(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
Hreal    = Hermitian(rand(rng,dim,dim) .- 0.5)


@testset "Lucon.jl" begin

    # a hermitian matrix, diagonalized from a random unitary matrix
    @testset "unitary group" begin
        BL = BrockettLoss.BrockettCriterion(Hcomplex)
        (U, _) = qr(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
        Res = BrockettLoss.optimize(BL, Matrix(U), MaxGradientTolerance=1.0E-8)
        Σdiff = Diagonal(eigen(Hcomplex).values) - Res.U'*Hcomplex*Res.U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7 # should be < 1.0E-7 if MaxGradientTolerance=1.0E-8
    end

    # the identity is a real valued matrix, but on the unitary group both the ascent
    # direction and the rotation exp(μH) it generates are complex
    @testset "unitary group, starting from the identity" begin
        BL = BrockettLoss.BrockettCriterion(Hcomplex)
        Res = BrockettLoss.optimize(BL, Matrix{ComplexF64}(I,dim,dim), MaxGradientTolerance=1.0E-8)
        @test norm(Res.U'*Res.U - I) < 1.0E-10
        Σdiff = Diagonal(eigen(Hcomplex).values) - Res.U'*Hcomplex*Res.U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7
    end

    # a real symmetric matrix has to stay on the orthogonal group
    @testset "orthogonal group" begin
        BL = BrockettLoss.BrockettCriterion(Hreal)
        Res = BrockettLoss.optimize(BL, Matrix{Float64}(I,dim,dim), MaxGradientTolerance=1.0E-8)
        @test eltype(Res.U) == Float64
        @test norm(Res.U'*Res.U - I) < 1.0E-10
        Σdiff = Diagonal(eigen(Hreal).values) - Res.U'*Hreal*Res.U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7
    end

    # any callable is a loss functional, so optimize accepts a closure and do syntax
    @testset "the loss functional is an ordinary callable" begin
        N = Diagonal(1.0:dim)
        Res = Lucon.optimize(Matrix{ComplexF64}(I,dim,dim); UDegree=2, Maximize=true) do U, CalcLoss
            Γ = Hcomplex*U*N
            (Γ, CalcLoss ? real(dot(U, Γ)) : 0.0)
        end
        @test Lucon.Converged(Res)
        Σdiff = Diagonal(eigen(Hcomplex).values) - Res.U'*Hcomplex*Res.U
        @test (√real(Σdiff⋅Σdiff)) < 1.0E-7
    end

    # MaxIter counts the rotations of U, and the loss belongs to the U that is returned
    @testset "the returned result is consistent" begin
        BL = BrockettLoss.BrockettCriterion(Hcomplex)
        U0 = Matrix{ComplexF64}(I,dim,dim)
        for MaxIter in (0, 1, 5)
            Res = BrockettLoss.optimize(BL, copy(U0), MaxIter=MaxIter)
            @test Res.Loss ≈ BrockettLossValue(Hcomplex, Res.U)
            @test Res.Iterations == MaxIter
            @test Res.Status == :maxiter
            @test !Lucon.Converged(Res)
        end
        Res = BrockettLoss.optimize(BL, copy(U0), MaxIter=0)
        @test Res.U == U0

        Res = BrockettLoss.optimize(BL, copy(U0), MaxGradientTolerance=1.0E-8)
        @test Lucon.Converged(Res)
        @test Res.MaxGradient < 1.0E-8
    end

    # optimize prints nothing by itself and reports its progress through the callback
    @testset "callback" begin
        BL = BrockettLoss.BrockettCriterion(Hcomplex)
        U0 = Matrix{ComplexF64}(I,dim,dim)

        @test_logs BrockettLoss.optimize(BL, copy(U0), MaxIter=3) # asserts that nothing is logged

        Trace = Int[]
        BrockettLoss.optimize(BL, copy(U0), MaxIter=3,
                              Callback = State -> (push!(Trace, State.Iteration); false))
        @test Trace == 1:4 # the callback also sees the iterate that MaxIter breaks on

        # a callback which returns true stops the iteration, leaving U and Loss consistent
        Res = BrockettLoss.optimize(BL, copy(U0), Callback = State -> State.Iteration == 3)
        @test Res.Status == :callback
        @test Res.Loss ≈ BrockettLossValue(Hcomplex, Res.U)
    end

    # the line search reports a step size of zero rather than throwing
    @testset "polynomial of the line search without a positive root" begin
        @test Lucon.SmallestPositiveRoot([-2.0, -1.0, 1.0]) ≈ 2.0 # (μ+1)(μ-2)
        @test Lucon.SmallestPositiveRoot([1.0, 0.0, 1.0]) === nothing # μ²+1
        @test Lucon.SmallestPositiveRoot([-2.0, 1.0, 0.0]) ≈ 2.0 # vanishing leading coefficient
    end

end
