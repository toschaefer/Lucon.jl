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
    U::Matrix{T};
    MinIter=0,
    MaxIter=-1,
    GradNormBreak = 1.0E-10,
    SolverAlgo = "CG-PR",
    PolynomialLineSearchDegree = 5
)::Tuple{Matrix{T},Float64} where T<:Number
    UDegree = 2
    sgn = +1.0 # maximization
    return Lucon.optimize(
        L,
	U,
	UDegree,
	sgn,
	MinIter=MinIter,
	MaxIter=MaxIter,
	GradNormBreak=GradNormBreak,
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
    U::Matrix{T},
    CalcLoss::Bool
)::Tuple{Matrix{T},Float64} where T<:Number
    Γ = zero(similar(U)) # Euclidean derivative has same type and dimension as U
    Loss = 0.0 # the value of the Loss function
    H = L.H # alias
    dim = size(H,1)
    N = Diagonal([1.0*n for n=1:dim]) # the N matrix is a diagonal matrix with entries N_nn = n
    Γ = H*U*N
    (CalcLoss == true) && (Loss = real(tr(U'*Γ)))
    return (Γ, Loss)
end 



end # module BrockettLoss

