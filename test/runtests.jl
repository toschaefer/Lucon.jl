using Lucon
using Test
using LinearAlgebra

include("../src/BrockettLoss.jl")
import .BrockettLoss

@testset "Lucon.jl" begin
    
    # random hermitian matrix (to be diagonalized via BrockettLoss functional using Lucon)
    dim = 10
    H = Hermitian(rand(dim,dim).-0.5+(rand(dim,dim).-0.5)*im)
    BL = BrockettLoss.LossFunctional(H)
    # random initial unitary matrix
    R = rand(dim,dim).-0.5 + (rand(dim,dim).-0.5)*im
    (U, _) = qr(R)
    U = Matrix(U)
    (U, _) = BrockettLoss.optimize(
                    BL,
    	            U,
    	            GradNormBreak=1.0E-12,
    	            PolynomialLineSearchDegree=5
    	        )
    # build diagonal matrix with U obtained from Lucon and compare with exact diagonal matrix
    Σapprox = U'*H*U
    Σexact  = Diagonal(eigen(H).values)
    Σdiff   = Σexact - Σapprox
    @test (√real(Σdiff⋅Σdiff)) < 1.0E-6 # should be < 1.0E-6 if GradNormBreak=1.0E-12

end
