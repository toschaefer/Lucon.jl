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


export optimize


"""
The loss functional contains the hermitian matrix to be diagonalized
"""
struct LossFunctional <: Lucon.LossFunctional
    H::Hermitian{<:Number}
end



# wrapper for the optimization function from the Lucon module
function optimize(
    L::LossFunctional,
    U::AbstractMatrix{T};
    MinIter::Integer = 0,
    MaxIter::Integer = typemax(Int),
    MaxGradientTolerance::Real = 1.0E-10,
    SolverAlgo::Symbol = :CGPR,
    PolynomialLineSearchDegree::Integer = 5
)::Tuple{AbstractMatrix{T},Float64} where T<:Number
    UDegree = 2
    sgn = +1.0 # maximization
    return Lucon.optimize(
        L,
	U,
	UDegree,
	sgn,
	MinIter=MinIter,
	MaxIter=MaxIter,
	MaxGradientTolerance=MaxGradientTolerance,
	SolverAlgo=SolverAlgo,
	PolynomialLineSearchDegree=PolynomialLineSearchDegree
    )
end



"""
Calculate and return the Euclidean derivative of the loss functional and the loss
function itself. This function overloads the empty function EuclideanDerivative
from the Lucon module.
"""
function Lucon.EuclideanDerivative(
    L::LossFunctional,
    U::AbstractMatrix{T},
    CalcLoss::Bool
)::Tuple{AbstractMatrix{T},Float64} where T<:Number
    Loss = 0.0 # the value of the Loss function
    H = L.H # alias
    dim = size(H,1)
    N = Diagonal([1.0*n for n=1:dim]) # the N matrix is a diagonal matrix with entries N_nn = n
    Γ = H*U*N # Euclidean derivative has same type and dimension as U
    # the trace of U'Γ is the Frobenius product of U and Γ
    CalcLoss && (Loss = real(dot(U, Γ)))
    return (Γ, Loss)
end



end # module BrockettLoss


