"""
BrockettLoss --- diagonalizing an hermitian matrix iteratively using the Brockett criteroin

This simple module serves as a template for the implementation of functionals to optimize with the Lucon.jl module.
Furthermore it is used to test the Lucon.jl module.

The functional considered here is given by
L(U) = real(trace(U'HUN))
where H is the hermitian matrix to be diagonalized and N is a diagonal matrix with distinct entries in ascending order.
The euclidean derivative simply reads
dL/dU' = HUN

The code basically follows the functional as proposed in
R.W. Brockett - Linear Algebra and its Applications, 146 (1991)
"""
module BrockettLoss

using LinearAlgebra

import Lucon


"""
The Brockett criterion, holding the hermitian matrix H to be diagonalized and the diagonal
matrix N. Both are stored with a concrete type, and N is built once rather than on every call.
"""
struct BrockettCriterion{T<:Number, A<:AbstractMatrix{T}}
    H::Hermitian{T,A}
    N::Diagonal{Float64,Vector{Float64}}
end

# the N matrix is a diagonal matrix with entries N_nn = n
BrockettCriterion(H::Hermitian) = BrockettCriterion(H, Diagonal([1.0*n for n=1:size(H,1)]))


"""
Calculate and return the Euclidean derivative of the loss functional and, if `CalcLoss` is
set, the loss itself. Making the functional callable is all that `Lucon.optimize` requires.
"""
function (B::BrockettCriterion)(U::AbstractMatrix, CalcLoss::Bool)
    Γ = B.H*U*B.N # Euclidean derivative has same type and dimension as U
    # the trace of U'Γ is the Frobenius product of U and Γ
    Loss = CalcLoss ? real(dot(U, Γ)) : 0.0
    return (Γ, Loss)
end


# wrapper for the optimization function from the Lucon module
function optimize(
    B::BrockettCriterion,
    U::AbstractMatrix;
    MaxGradientTolerance::Real = 1.0E-10,
    kwargs...
)
    return Lucon.optimize(B, U; UDegree=2, Maximize=true, MaxGradientTolerance, kwargs...)
end


end # module BrockettLoss
