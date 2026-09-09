"""
BrockettLoss --- diagonalizing an hermitian matrix iteratively using the Brockett criterion

This simple module serves as a template for the implementation of functionals to optimize with the Lucon.jl module.
Furthermore it is used to test the Lucon.jl module.

The functional considered here is given by
L(U) = real(trace(U'HUN))
where H is the hermitian matrix to be diagonalized and N is a diagonal matrix with distinct entries in ascending order.
The Euclidean derivative simply reads
dL/dU' = HUN

Since L(U) = Σ_n N_nn (U'HU)_nn, the ascending entries of N are what makes the maximum unique:
the largest weight has to meet the largest diagonal element, so L is maximal when U'HU is
diagonal with the eigenvalues in ascending order.

The code basically follows the functional as proposed in
R.W. Brockett - Linear Algebra and its Applications, 146 (1991)
"""
module BrockettLoss

using LinearAlgebra

import Lucon


"""
The loss function L(U) = tr(U'HUN), known as the Brockett criterion, holding the hermitian
matrix H to be diagonalized and the diagonal matrix N. Both are stored with a concrete type,
and N is built once rather than on every call.
"""
struct LossFunction{TH<:Hermitian, TN<:Diagonal}
    H::TH
    N::TN
end

# the N matrix is a diagonal matrix with entries N_nn = n, distinct and ascending
LossFunction(H::Hermitian) = LossFunction(H, Diagonal(float.(1:size(H,1))))


"""
Calculate and return the Euclidean derivative of the loss function `L` at `U` and, if
`CalcLoss` is set, the loss itself. It is the Euclidean and not the Riemannian gradient that
is asked for here, since `Lucon.optimize` forms the latter itself. This is the function Lucon
needs, and the line below hands it over by making `L` itself callable.
"""
function EuclideanGradient(L::LossFunction, U::AbstractMatrix, CalcLoss::Bool)
    Γ = L.H*U*L.N # Euclidean derivative has same type and dimension as U
    # the loss tr(U'HUN) = tr(U'Γ) is the Frobenius product of U and Γ, which dot evaluates
    # without ever forming the matrix product U'Γ
    Loss = CalcLoss ? real(dot(U, Γ)) : 0.0
    return (Γ, Loss)
end

# A method whose name is an argument makes instances of that argument's type callable:
# from here on L(U, CalcLoss) calls EuclideanGradient(L, U, CalcLoss), so that a loss
# function can be passed to Lucon.optimize wherever a function is expected.
(L::LossFunction)(U::AbstractMatrix, CalcLoss::Bool) = EuclideanGradient(L, U, CalcLoss)


"""
Maximize the loss function, i.e. diagonalize the hermitian matrix it holds.

`UDegree=2`, since L(U + tZ) is quadratic in t, and `Maximize=true` are properties of the
functional rather than of the call site, so they are fixed here instead of being left to the
caller. Every other keyword is passed on to `Lucon.optimize`.
"""
function optimize(L::LossFunction, U::AbstractMatrix; kwargs...)
    return Lucon.optimize(L, U; UDegree=2, Maximize=true, kwargs...)
end


end # module BrockettLoss
