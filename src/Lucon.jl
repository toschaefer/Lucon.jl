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
    ::AbstractMatrix{T},
    ::Bool
)::Tuple{AbstractMatrix{T}, Float64} where T<:Number
    error("EuclideanDerivative method not implemented for this functional type")
end


"""
Calculate the optimal unitary matrix U iteratively.

Arguments:
* `L`: the loss functional, a sub-type of `Lucon.LossFunctional` which provides a method
  `Lucon.EuclideanDerivative`.
* `U`: the initial unitary matrix. Its element type selects the group the optimization runs
  on, the orthogonal group for a real and the unitary group for a complex element type.
* `UDegree`: the order q of the loss functional, i.e. the highest power of t appearing in the
  Taylor expansion of L(U + tZ). It sets the width T_μ = 2π/(q|ω_max|) of the line search
  window, see Eq. (15) in T. Abrudan et al.
* `sgn`: +1.0 to maximize and -1.0 to minimize the loss functional.

Keyword arguments:
* `MinIter`: no convergence is signalled before this number of iterations is reached.
* `MaxIter`: upper limit for the number of rotations of U, by default unlimited.
* `MaxGradientTolerance`: convergence threshold for the largest absolute element of the
  Riemannian gradient. Unlike the Frobenius norm, this maximum norm is independent of the
  size of the system, so that one and the same threshold converges a subsystem and a
  supersystem built from copies of it to the same accuracy.
* `SolverAlgo`: currently only the conjugate gradient Polak-Ribièrre algorithm, `:CGPR`.
* `PolynomialLineSearchDegree`: the order P of the polynomial used in the line search, 3 to 5.

Returns the optimal U together with the value of the loss functional at that U.
"""
function optimize(
    L::LossFunctional,
    U::AbstractMatrix{T},
    UDegree::Integer,
    sgn::Real;
    MinIter::Integer = 0,
    MaxIter::Integer = typemax(Int),
    MaxGradientTolerance::Real = 1.0E-8,
    SolverAlgo::Symbol = :CGPR,
    PolynomialLineSearchDegree::Integer = 5
)::Tuple{AbstractMatrix{T},Float64} where T<:Number

    # currently only the CG-PR (conjugate gradient Polak-Ribièrre algorithm is implemented)
    SolverAlgo === :CGPR || throw(ArgumentError("algorithm :$SolverAlgo currently not supported in Lucon"))
    UDegree >= 1 || throw(ArgumentError("UDegree must be a positive integer"))
    PolynomialLineSearchDegree >= 1 || throw(ArgumentError("PolynomialLineSearchDegree must be a positive integer"))
    MinIter >= 0 || throw(ArgumentError("MinIter must be non-negative"))
    MaxIter >= 0 || throw(ArgumentError("MaxIter must be non-negative"))

    # renormalize sgn to +1.0 or -1.0, a vanishing sgn leaves no direction to move along
    sgn = float(sign(sgn))
    iszero(sgn) && throw(ArgumentError("sgn must be positive (maximize) or negative (minimize)"))

    Gprev = zero(U) # will hold Riemannian derivative of previous Iteration

    # init ascent direction H with zeros
    H = zero(U)

    Loss = 0.0 # value of loss function in each Iteration

    # some output
    @info "in Lucon.optimize"
    @info " #iter   max|grad|            loss-function"

    # the main iteration loop (break condition via the gradient, MaxIter or the step size)
    Iteration = 0
    Message = "reached break condition: maximum number of iterations"
    while true

        Iteration += 1

        # get Eucledean derivative Γ and Loss function
        (Γ, Loss) = EuclideanDerivative(L,U,true)

        # construct current Riemannian derivative Gcurr, see Eq. (2)
        Gcurr = Γ * U'
        Gcurr = Gcurr - Gcurr'

        # the maximum norm of the gradient does not grow with the size of the system
        MaxGradient = maximum(abs, Gcurr)

        @info @sprintf("%6d %11.3e %24.16e", Iteration, MaxGradient, Loss)

        # check if convergence is reached
        if MaxGradient < MaxGradientTolerance && Iteration > MinIter
            Message = "reached break condition: largest gradient element below MaxGradientTolerance"
            break
        end

        # MaxIter counts the rotations of U, of which none has been performed yet
        Iteration > MaxIter && break

        # Calculate conjugate gradient Polak-Ribière-Polyak (CG-PR) update factor, see Eq. (10)
        if Iteration > 1
            CGPR_Factor = real(Gcurr⋅(Gcurr-Gprev)) / real(Gprev⋅Gprev)
        else
            CGPR_Factor = 0.0
        end

        # update "prev"
        Gprev = copy(Gcurr)

        # update ascent direction
        H = Gcurr + CGPR_Factor * H

        # check if set-back of the history of H (CGPR factor) is necessary
        if (0.5*real(H⋅Gcurr) < 0.0) || ( ((Iteration-1)%size(U,1)==0) && (Iteration>2) )
            H = copy(Gcurr)
        end

        # find the optimal step size via polynomial line search
        (U, μ) = RotateUviaPolynomialLineSearch(L, Gcurr, U, H, UDegree, sgn, PolynomialLineSearchDegree)

        # a vanishing step size leaves U unchanged and no further progress can be made
        if μ == 0.0
            @warn "Lucon.optimize: line search found no positive step size"
            Message = "reached break condition: line search found no step size"
            break
        end

    end

    @info Message

    return (U,Loss)
end # optimize



"""
Assemble the rotation matrix R = exp(x*H) from the eigenvectors V and the eigenvalues
-im*Λ of the skew-hermitian matrix H. The diagonal factor exp(-im*x*Λ) is applied as a
column scaling of V, which fuses into a single broadcast and leaves the expression free
of scalar indexing, so that it also runs on a GPU.
On the orthogonal group H is real and skew-symmetric, so that R is real up to roundoff.
"""
ScaleEigenvectors(V, Λ, x) = V .* transpose(cis.(-x .* Λ))
RotationMatrix(::Type{T}, V, Λ, x) where T<:Real    = real.(ScaleEigenvectors(V,Λ,x) * V')
RotationMatrix(::Type{T}, V, Λ, x) where T<:Complex = ScaleEigenvectors(V,Λ,x) * V'



"""
Smallest strictly positive real root of the polynomial p(μ) = c[1] + c[2]μ¹ + c[3]μ² + ...
obtained from the eigenvalues of the companion matrix of p. Returns `nothing` if p has no
such root, see step 8 of Table 1 in T. Abrudan et al.
"""
function SmallestPositiveRoot(c::Vector{Float64})::Union{Float64,Nothing}

    # negligible leading coefficients render the companion matrix ill-conditioned and
    # produce spurious roots of the order of 1/eps, so lower the degree of p instead
    scale = maximum(abs, c)
    lead = findlast(x -> abs(x) > eps(Float64)*scale, c)
    (lead === nothing || lead < 2) && return nothing

    degree = lead - 1
    cM = zeros(Float64, degree, degree)
    for i = 1:degree-1
        cM[i,i+1] = 1.0
    end
    for j = 1:degree
        cM[degree,j] = -c[j]/c[lead]
    end
    roots = eigvals(cM) # the eigenvalues of the companion matrix are the roots of p

    positiveRealRoots = [real(r) for r in roots if isreal(r) && real(r) > 0.0]
    isempty(positiveRealRoots) && return nothing

    return minimum(positiveRealRoots)
end



"""
Perform a polynomial line search for the optimal step size μ for the conjugate-gradient algorithm.
The procedure is described in section 3.1 in T. Abrudan et al. / Signal Processing 89 (2009) 1704–1714

Returns the rotated matrix exp(sgn*μ*H)U together with the step size μ. A step size of zero means
that the line search found no local optimum along the geodesic, in which case U is returned unchanged.
`G` is the Riemannian gradient at U, from which the derivative at μ=0 is read off directly.
"""
function RotateUviaPolynomialLineSearch(
    L::LossFunctional,
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
    H::AbstractMatrix{T},
    UDegree::Integer,
    sgn::Float64,
    PolynomialDegree::Integer
)::Tuple{AbstractMatrix{T},Float64} where T<:Number

    # diagonalize the skew symmetric matrix H by
    # constructing the hermitian matrix H*im and diagonalize
    AuxEigenvals, V = eigen(Hermitian(H*1.0im)) # eigenvalues of H are -im*AuxEigenvals

    # sampling points of μ = 0*μstep, 1*μstep, 2*μstep, ...
    maxAbsEigenval = maximum(abs, AuxEigenvals)
    iszero(maxAbsEigenval) && return (U, 0.0) # H vanishes only in a stationary point
    μstep = 2π / (PolynomialDegree * UDegree * maxAbsEigenval)

    # set up rotation matrix exp(sgn*μstep*H)
    R = RotationMatrix(T, V, AuxEigenvals, sgn*μstep)

    # for every μ>0 we calculate the derivative dLdμ = d/dμ L(exp(sgn*μ*H)U), see Eq. (14)
    # in T. Abrudan et al. The trace of Γ(HU)' is its Frobenius product, which spares us
    # from forming the matrix product itself.
    dLdμ = Vector{Float64}(undef, PolynomialDegree)
    rotatedU = copy(U)
    for i = 1:PolynomialDegree
        rotatedU = R*rotatedU
        (rotatedΓ, _) = EuclideanDerivative(L,rotatedU,false)
        dLdμ[i] = 2*sgn*real( dot(H*rotatedU, rotatedΓ) )
    end

    # at μ = 0 the derivative is the Frobenius product of H with the Riemannian gradient
    dLdμ0 = sgn*real( dot(G, H) )

    # set up the coefficients for the polynomial by solving the linear System Ma=b for a
    b = dLdμ .- dLdμ0
    M = Matrix{Float64}(undef, PolynomialDegree, PolynomialDegree)
    for i = 1:PolynomialDegree
        for j = 1:PolynomialDegree
            M[i,j] = (i*μstep)^j
        end
    end
    a = M\b # solve linear system

    # the optimal μ corresponds to the smallest positive real root of p(μ) = a₀ + a₁μ¹ + a₂μ² + ...
    μOpt = SmallestPositiveRoot(vcat(dLdμ0, a))
    μOpt === nothing && return (U, 0.0)

    # rotate U with optimal μ: rotatedU = exp(sgn*μOpt*H) U
    rotatedU = RotationMatrix(T, V, AuxEigenvals, sgn*μOpt) * U

    return (rotatedU, μOpt)
end

end # Lucon

