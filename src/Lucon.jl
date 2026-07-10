"""
Lucon --- loss optimization under unitary constraint
Provides solver to find optimal unitary matrix to maximize/minimize a loss function which maps the unitary matrix onto a number.

The conjugate gradient method basically follows the lines of the publication,
T. Abrudan et al. / Signal Processing 89 (2009) 1704–1714 (dx.doi.org/10.1016/j.sigpro.2009.03.015)
"""
module Lucon

using LinearAlgebra
using Printf


"""
The outcome of `optimize`.

* `U`: the optimal unitary matrix.
* `Loss`: the value of the loss functional at `U`.
* `MaxGradient`: the largest absolute element of the Riemannian gradient at `U`.
* `Iterations`: the number of rotations of `U` that were performed, the quantity bounded by `MaxIter`.
* `Status`: why the iteration stopped. One of `:converged`, `:maxiter`, `:callback`, or
  `:linesearch` if the line search found no positive step size.

Use `Lucon.Converged` to ask whether `MaxGradientTolerance` was reached.
"""
struct Result{M<:AbstractMatrix}
    U::M
    Loss::Float64
    MaxGradient::Float64
    Iterations::Int
    Status::Symbol
end

"""
Did `optimize` reach the requested `MaxGradientTolerance`?
"""
Converged(Res::Result) = Res.Status === :converged

function Base.show(io::IO, ::MIME"text/plain", Res::Result)
    println(io, "Lucon.Result")
    println(io, "  Status:      ", Res.Status)
    println(io, "  Iterations:  ", Res.Iterations)
    @printf(io, "  Loss:        %.16e\n", Res.Loss)
    @printf(io, "  max|grad|:   %.3e\n", Res.MaxGradient)
      print(io, "  U:           ", summary(Res.U))
end


"""
Callback for `optimize` which prints the iteration count, the largest absolute element of the
Riemannian gradient and the value of the loss functional, one line per iteration.
"""
struct PrintTrace
    io::IO
end
PrintTrace() = PrintTrace(stdout)

function (Trace::PrintTrace)(State)
    State.Iteration == 1 && println(Trace.io, " #iter   max|grad|            loss-function")
    @printf(Trace.io, "%6d %11.3e %24.16e\n", State.Iteration, State.MaxGradient, State.Loss)
    return false
end


"""
Calculate the optimal unitary matrix U iteratively.

Arguments:
* `Gradient`: a callable `Gradient(U, CalcLoss::Bool)` which returns the tuple `(Γ, Loss)`.
  Here Γ_ij = ∂L/∂conj(U_ij) is the Euclidean derivative of the loss functional L at U. The
  value of L is only read when `CalcLoss` is true, so computing it may be skipped otherwise.
  Any callable will do, in particular a closure or a struct carrying precomputed quantities.
* `U`: the initial unitary matrix. Its element type selects the group the optimization runs
  on, the orthogonal group for a real and the unitary group for a complex element type.

Keyword arguments:
* `UDegree`: the order q of the loss functional, i.e. the highest power of t appearing in the
  Taylor expansion of L(U + tZ). It sets the width T_μ = 2π/(q|ω_max|) of the line search
  window, see Eq. (15) in T. Abrudan et al.
* `Maximize`: maximize the loss functional instead of minimizing it.
* `MinIter`: no convergence is signalled before this number of iterations is reached.
* `MaxIter`: upper limit for the number of rotations of U, by default unlimited.
* `MaxGradientTolerance`: convergence threshold for the largest absolute element of the
  Riemannian gradient. Unlike the Frobenius norm, this maximum norm is independent of the
  size of the system, so that one and the same threshold converges a subsystem and a
  supersystem built from copies of it to the same accuracy.
* `SolverAlgo`: currently only the conjugate gradient Polak-Ribièrre algorithm, `:CGPR`.
* `PolynomialLineSearchDegree`: the order P of the polynomial used in the line search, 3 to 5.
* `Callback`: a function called once per iteration with the named tuple
  `(; Iteration, MaxGradient, Loss, U)`, before the break conditions are tested. Returning
  `true` from it stops the iteration. `optimize` prints nothing on its own; pass
  `Lucon.PrintTrace()` to obtain a convergence trace on `stdout`.

Returns a `Lucon.Result`. Since the loss functional is an ordinary callable, `optimize` may be
called with `do` syntax:

    Result = Lucon.optimize(U; UDegree=2, Maximize=true) do U, CalcLoss
        Γ = H*U*N
        (Γ, CalcLoss ? real(dot(U, Γ)) : 0.0)
    end
"""
function optimize(
    Gradient,
    U::AbstractMatrix{T};
    UDegree::Integer,
    Maximize::Bool = false,
    MinIter::Integer = 0,
    MaxIter::Integer = typemax(Int),
    MaxGradientTolerance::Real = 1.0E-8,
    SolverAlgo::Symbol = :CGPR,
    PolynomialLineSearchDegree::Integer = 5,
    Callback = nothing
)::Result where T<:Number

    # currently only the CG-PR (conjugate gradient Polak-Ribièrre algorithm is implemented)
    SolverAlgo === :CGPR || throw(ArgumentError("algorithm :$SolverAlgo currently not supported in Lucon"))
    UDegree >= 1 || throw(ArgumentError("UDegree must be a positive integer"))
    PolynomialLineSearchDegree >= 1 || throw(ArgumentError("PolynomialLineSearchDegree must be a positive integer"))
    MinIter >= 0 || throw(ArgumentError("MinIter must be non-negative"))
    MaxIter >= 0 || throw(ArgumentError("MaxIter must be non-negative"))

    sgn = Maximize ? +1.0 : -1.0

    Gprev = zero(U) # will hold Riemannian derivative of previous Iteration

    # init ascent direction H with zeros
    H = zero(U)

    Loss = 0.0        # value of loss function in each Iteration
    MaxGradient = 0.0 # largest absolute element of the Riemannian gradient

    # the main iteration loop (break condition via the gradient, MaxIter or the step size)
    Iteration = 0
    Status = :maxiter
    while true

        Iteration += 1

        # get Eucledean derivative Γ and Loss function
        (Γ, Loss) = Gradient(U, true)

        # construct current Riemannian derivative Gcurr, see Eq. (2)
        Gcurr = Γ * U'
        Gcurr = Gcurr - Gcurr'

        # the maximum norm of the gradient does not grow with the size of the system
        MaxGradient = maximum(abs, Gcurr)

        # a callback which returns true asks the iteration to stop
        if Callback !== nothing && Callback((; Iteration, MaxGradient, Loss, U)) === true
            Status = :callback
            break
        end

        # check if convergence is reached
        if MaxGradient < MaxGradientTolerance && Iteration > MinIter
            Status = :converged
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
        (U, μ) = RotateUviaPolynomialLineSearch(Gradient, Gcurr, U, H, UDegree, sgn, PolynomialLineSearchDegree)

        # a vanishing step size leaves U unchanged and no further progress can be made
        if iszero(μ)
            @warn "Lucon.optimize: line search found no positive step size"
            Status = :linesearch
            break
        end

    end

    @debug "Lucon.optimize stopped with status :$Status"

    # every break condition is tested before U is rotated, so one rotation less than iterations
    return Result(U, Loss, MaxGradient, Iteration - 1, Status)
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
    Gradient,
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
        (rotatedΓ, _) = Gradient(rotatedU, false)
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
