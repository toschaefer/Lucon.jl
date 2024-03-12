"""
Copyright 2023 Tobias Schäfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



"""
Lucon --- loss optimization under unitary constraint
Provides solver to find optimal unitary matrix to maximize/minimize a loss function which maps the unitary matrix onto a number.

The conjugate gradient method basically follows the lines of the publication,
T. Abrudan et al. / Signal Processing 89 (2009) 1704–1714 (dx.doi.org/10.1016/j.sigpro.2009.03.015)
"""

module Lucon

using LinearAlgebra
using Printf
using Logging

export optimize

abstract type LossFunctional end

# "empty" implementation of EuclideanDerivative method
function EuclideanDerivative(
    ::LossFunctional,
    ::Matrix{<:T}, 
    ::Bool
)::Tuple{Matrix{<:T}, Float64} where T<:Number
    error("EuclideanDerivative method not implemented for this functional type")
end


"""
Calculate the optimal unitary matrix U iteratively
"""
function optimize(
    L::LossFunctional,
    U::Matrix{T},
    UDegree::Integer,
    sgn::Float64;
    MinIter = 0,
    MaxIter = -1,
    GradNormBreak = 1.0E-8,
    SolverAlgo = "CG-PR",
    PolynomialLineSearchDegree = 5
)::Tuple{Matrix{T},Float64} where T<:Number

    # currently only the CG-PR (conjugate gradient Polak-Ribièrre algorithm is implemented)
    if SolverAlgo != "CG-PR"
        error("algorithm " * SolverAlgo * " currently not supported in Lucon")
    end

    G = Dict{String, Matrix{T}}()
    G["curr"] = zero(similar(U)) # will hold Riemannian derivative of current Iteration
    G["prev"] = zero(similar(U)) # will hold Riemannian derivative of previous Iteration

    # init ascent direction H with zeros
    H = zero(similar(G["curr"]))

    Loss = 0.0 # value of loss function in each Iteration

    sgn = sign(sgn) # renormalize sgn to +1.0 or -1.0 using the sign(...) function

    # some output
    @info "in Lucon.optimize"
    @info " #iter  grad. norm            loss-function"

    # the main iteration loop (break condition via GradientNorm or MaxIter)
    Iteration = 0
    while true 
        
	MaxIter >= 0 && Iteration >= MaxIter && break
        Iteration += 1

        # get Eucledean derivative Γ and Loss function
        (Γ, Loss) = EuclideanDerivative(L,U,true)

        # construct current Riemannian derivative G["current"]
        G["curr"] = Γ * U'
        G["curr"] = G["curr"] - G["curr"]'

        # calculate gradient norm via Frobenius norm
        GradientNorm = sqrt(real(G["curr"]⋅G["curr"]))

	output = @sprintf("%6d %11.3e %24.16e\n", Iteration, GradientNorm, Loss)
	@info output

	# check if convergence is reached
	GradientNorm < GradNormBreak && Iteration > MinIter && break

	# Calculate conjugate gradient Polak-Ribière-Polyak (CG-PR) update factor 
	if Iteration > 1
	    CGPR_Factor = real((G["curr"]⋅(G["curr"]-G["prev"]))) / real(G["prev"]⋅G["prev"])
	else
	    CGPR_Factor = 0.0
	end

	# update "prev"
        G["prev"] = copy(G["curr"])

	# update ascent direction
        H = G["curr"] + CGPR_Factor * H

	# check if set-back of the history of H (CGPR factor) is necessary
	if (0.5*real(H⋅G["curr"]) < 0.0) || ( ((Iteration-1)%size(U,1)==0) && (Iteration>2) )
	    H = G["curr"]
	    CGPR_Factor = 0.0
	end

	# find the optimal step size via polynomial line search 
	# (here we hard code a polynomial degree of 5)
	U = RotateUviaPolynomialLineSearch(L, Γ, U, H, UDegree, sgn, PolynomialLineSearchDegree) 

    end

    output = "reached break condition"
    @info output

    return (U,Loss)
end # optimize



"""
Perform a polynomial line search for the optimal step size μ for the conjugate-gradient algorithm.
The procedure is described in section 3.1 in T. Abrudan et al. / Signal Processing 89 (2009) 1704–1714 
"""
function RotateUviaPolynomialLineSearch(
    L::LossFunctional,
    Γ::Matrix{T},
    U::Matrix{T},
    H::Matrix{T}, 
    UDegree::Integer,
    sgn::Float64,
    PolynomialDegree::Integer
)::Matrix{T} where T<:Number

    # diagonalize the skew symmetric matrix H by
    # constructing the hermitian matrix H*im and diagonalize
    AuxEigenvals, V = eigen(Hermitian(H*1.0im))
    Eigenvals = -1.0im * AuxEigenvals # convert back to eigenvalues of H

    # sampling points of μ = 0*μstep, 1*μstep, 2*μstep, ...
    maxAbsEigenval = maximum(map(abs,AuxEigenvals))
    μstep = 2.0*pi/PolynomialDegree/UDegree/maxAbsEigenval

    # set up rotation matrix exp(sgn*μstep*H)
    Σ = Diagonal( map(exp, sgn*μstep*Eigenvals) )
    # calc trafo matrix for U
    isreal(U) ? R = real(V*Σ*V') : R = V*Σ*V'

    # for every μ>0 we calculate the derivative dLdμ = d/dμ L(exp(sgn*μ*H)U) 
    dLdμ = Array{Float64}(undef, PolynomialDegree)
    rotatedU = copy(U)
    for i = 1:PolynomialDegree
        rotatedU = R*rotatedU
	(rotatedΓ, _) = EuclideanDerivative(L,rotatedU,false)
	# finally construct dLdμ, see Eq. (14) in T. Abrudan et al., Signal Processing 89 (2009) 1704–1714
	dLdμ[i] = 2*sgn*real( tr( rotatedΓ*(H*rotatedU)' ) ) 
    end

    # we don't forget the case of μ = 0 and store it separatly 
    dLdμ0 = 2*sgn*real( tr( Γ*(H*U)' ) )

    # set up the coefficients for the polynomial by solving the linear System Ma=b for a
    b = dLdμ .- dLdμ0
    M = Matrix{Float64}(undef, PolynomialDegree, PolynomialDegree)
    for i = 1:PolynomialDegree
        for j = 1:PolynomialDegree
	    M[i,j] = (i*μstep)^j
        end
    end
    a = M\b # solve linear system

    # We aim for the smallest root of the polynomial p(x) = a₀ + a₁x¹ + a₂x² + ... 
    # To this end we construct the companion matrix cM and calculate its eigenvalues
    cM = zero(similar(M))
    for i = 1:PolynomialDegree-1
      cM[i,i+1] = 1.0
    end
    for j = 2:PolynomialDegree
      cM[PolynomialDegree,j] = -1.0*a[j-1]/a[PolynomialDegree]
    end
    cM[PolynomialDegree,1] = -1.0*dLdμ0/a[PolynomialDegree]
    (roots, _) = eigen(cM) # calculate eigenvalues of companion matrix, equivalent to the roots

    # the optimal μ corresponds to the smallest positive real root of the polynomial p(x) from above
    μOpt = minimum(filter(x->x>0.0, real(filter(isreal, roots))))

    # rotate U with optimal μ: rotatedU = exp(sgn*μOpt*H) U
    Σ = Diagonal( map(exp, sgn*μOpt*Eigenvals) )
    isreal(U) ? rotatedU = real(V*Σ*V')*U : rotatedU = V*Σ*V'*U

    return rotatedU
end

end # Lucon


