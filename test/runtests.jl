using Lucon
using Test
using Aqua
using LinearAlgebra
using Random

include("../examples/BrockettLoss.jl")
import .BrockettLoss

# the Brockett criterion L(U) = tr(U'HUN) with N_nm = n δ_nm
brockett_loss_value(H, U) = real(tr(U' * H * U * Diagonal(1.0:size(H,1))))

dim = 10
rng = MersenneTwister(2023)
H_complex = Hermitian(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
H_real    = Hermitian(rand(rng,dim,dim) .- 0.5)


@testset "Lucon.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Lucon)
    end

    # a hermitian matrix, diagonalized from a random unitary matrix
    @testset "unitary group" begin
        L = BrockettLoss.LossFunction(H_complex)
        (U, _) = qr(rand(rng,dim,dim) .- 0.5 + (rand(rng,dim,dim) .- 0.5)*im)
        res = BrockettLoss.optimize(L, Matrix(U), max_gradient_tolerance=1e-8)
        Σ_diff = Diagonal(eigen(H_complex).values) - res.U'*H_complex*res.U
        @test (√real(Σ_diff⋅Σ_diff)) < 1e-7 # should be < 1e-7 if max_gradient_tolerance=1e-8
    end

    # the identity is a real valued matrix, but on the unitary group both the ascent
    # direction and the rotation exp(μH) it generates are complex
    @testset "unitary group, starting from the identity" begin
        L = BrockettLoss.LossFunction(H_complex)
        res = BrockettLoss.optimize(L, Matrix{ComplexF64}(I,dim,dim), max_gradient_tolerance=1e-8)
        @test norm(res.U'*res.U - I) < 1e-10
        Σ_diff = Diagonal(eigen(H_complex).values) - res.U'*H_complex*res.U
        @test (√real(Σ_diff⋅Σ_diff)) < 1e-7
    end

    # a real symmetric matrix has to stay on the orthogonal group
    @testset "orthogonal group" begin
        L = BrockettLoss.LossFunction(H_real)
        res = BrockettLoss.optimize(L, Matrix{Float64}(I,dim,dim), max_gradient_tolerance=1e-8)
        @test eltype(res.U) == Float64
        @test norm(res.U'*res.U - I) < 1e-10
        Σ_diff = Diagonal(eigen(H_real).values) - res.U'*H_real*res.U
        @test (√real(Σ_diff⋅Σ_diff)) < 1e-7
    end

    # any callable is a loss functional, so optimize accepts a closure and do syntax
    @testset "the loss functional is an ordinary callable" begin
        N = Diagonal(1.0:dim)
        res = Lucon.optimize(Matrix{ComplexF64}(I,dim,dim); max_taylor_degree=2, maximize=true) do U, calc_loss
            Γ = H_complex*U*N
            (Γ, calc_loss ? real(dot(U, Γ)) : 0.0)
        end
        @test Lucon.converged(res)
        Σ_diff = Diagonal(eigen(H_complex).values) - res.U'*H_complex*res.U
        @test (√real(Σ_diff⋅Σ_diff)) < 1e-7
    end

    # max_iter counts the rotations of U, and the loss belongs to the U that is returned
    @testset "the returned result is consistent" begin
        L = BrockettLoss.LossFunction(H_complex)
        U0 = Matrix{ComplexF64}(I,dim,dim)
        for max_iter in (0, 1, 5)
            res = BrockettLoss.optimize(L, copy(U0), max_iter=max_iter)
            @test res.loss ≈ brockett_loss_value(H_complex, res.U)
            @test res.iterations == max_iter
            @test res.status == :max_iter
            @test !Lucon.converged(res)
        end
        res = BrockettLoss.optimize(L, copy(U0), max_iter=0)
        @test res.U == U0

        res = BrockettLoss.optimize(L, copy(U0), max_gradient_tolerance=1e-8)
        @test Lucon.converged(res)
        @test res.max_gradient < 1e-8
    end

    # optimize prints nothing by itself and reports its progress through the callback
    @testset "callback" begin
        L = BrockettLoss.LossFunction(H_complex)
        U0 = Matrix{ComplexF64}(I,dim,dim)

        @test_logs BrockettLoss.optimize(L, copy(U0), max_iter=3) # asserts that nothing is logged

        trace = Int[]
        BrockettLoss.optimize(L, copy(U0), max_iter=3,
                              callback = state -> (push!(trace, state.iteration); false))
        @test trace == 1:4 # the callback also sees the iterate that max_iter breaks on

        # a callback which returns true stops the iteration, leaving U and loss consistent
        res = BrockettLoss.optimize(L, copy(U0), callback = state -> state.iteration == 3)
        @test res.status == :callback
        @test res.loss ≈ brockett_loss_value(H_complex, res.U)
    end

    # the line search reports a step size of zero rather than throwing
    @testset "polynomial of the line search without a positive root" begin
        @test Lucon.smallest_positive_root([-2.0, -1.0, 1.0]) ≈ 2.0 # (μ+1)(μ-2)
        @test Lucon.smallest_positive_root([1.0, 0.0, 1.0]) === nothing # μ²+1
        @test Lucon.smallest_positive_root([-2.0, 1.0, 0.0]) ≈ 2.0 # vanishing leading coefficient
    end

end
