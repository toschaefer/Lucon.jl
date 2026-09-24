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
* `loss`: the value of the loss functional at `U`.
* `max_gradient`: the largest absolute element of the Riemannian gradient at `U`.
* `iterations`: the number of rotations of `U` that were performed, the quantity bounded by `max_iter`.
* `status`: why the iteration stopped. One of `:converged`, `:max_iter`, `:callback`, or
  `:line_search` if the line search found no positive step size.

`loss` and `max_gradient` are always `Float64`, whatever the element type of `U`.
Use `Lucon.converged` to ask whether `max_gradient_tolerance` was reached.
"""
struct Result{M<:AbstractMatrix}
    U::M
    loss::Float64
    max_gradient::Float64
    iterations::Int
    status::Symbol
end

"""
Did `optimize` reach the requested `max_gradient_tolerance`?
"""
converged(res::Result) = res.status === :converged

function Base.show(io::IO, ::MIME"text/plain", res::Result)
    println(io, "Lucon.Result")
    println(io, "  status:        ", res.status)
    println(io, "  iterations:    ", res.iterations)
    @printf(io, "  loss:          %.16e\n", res.loss)
    @printf(io, "  max_gradient:  %.3e\n", res.max_gradient)
      print(io, "  U:             ", summary(res.U))
end


"""
Callback for `optimize` which prints the iteration count, the largest absolute element of the
Riemannian gradient, the value of the loss functional and the wall clock time one iteration took,
one line per iteration. The first line carries no time, since the callback is called from within
the iteration it would measure.
"""
mutable struct PrintTrace
    io::IO
    previous_time::UInt64
end
PrintTrace(io::IO = stdout) = PrintTrace(io, zero(UInt64))

function (trace::PrintTrace)(state)
    now = time_ns()
    if state.iteration == 1
        println(trace.io, " #iter   max|grad|            loss-function        time [s]")
        @printf(trace.io, "%6d %11.3e %24.16e %15s\n",
                state.iteration, state.max_gradient, state.loss, "-")
    else
        @printf(trace.io, "%6d %11.3e %24.16e %15.2e\n",
                state.iteration, state.max_gradient, state.loss, (now - trace.previous_time)/1e9)
    end
    trace.previous_time = now
    return false
end


"""
Calculate the optimal unitary matrix U iteratively.

Arguments:
* `gradient`: a callable `gradient(U, calc_loss::Bool)` which returns the tuple `(Γ, loss)`.
  Here Γ_ij = ∂L/∂conj(U_ij) is the Euclidean derivative of the loss functional L at U. The
  value of L is only read when `calc_loss` is true, so computing it may be skipped otherwise.
  Any callable will do, in particular a closure or a struct carrying precomputed quantities.
* `U`: the initial unitary matrix. Its element type selects the group the optimization runs
  on, the orthogonal group for a real and the unitary group for a complex element type.

Keyword arguments:
* `max_taylor_degree`: the order q of the loss functional, i.e. the highest power of t
  appearing in the Taylor expansion of L(U + tZ). It sets the width T_μ = 2π/(q|ω_max|) of
  the line search window, see Eq. (15) in T. Abrudan et al.
* `maximize`: maximize the loss functional instead of minimizing it.
* `min_iter`: no convergence is signalled before this number of iterations is reached.
* `max_iter`: upper limit for the number of rotations of U, by default unlimited.
* `max_gradient_tolerance`: convergence threshold for the largest absolute element of the
  Riemannian gradient. Unlike the Frobenius norm, this maximum norm is independent of the
  size of the system, so that one and the same threshold converges a subsystem and a
  supersystem built from copies of it to the same accuracy.
* `solver_algo`: currently only the conjugate gradient Polak-Ribière algorithm, `:CGPR`.
* `line_search_samples`: the number P of points at which the line search samples the derivative
  of L inside the window, each costing one gradient evaluation. A polynomial of degree P is
  fitted through them, so P must be at least 3 to resolve one oscillation; 3 to 5 is reasonable.
* `callback`: a function called once per iteration with the named tuple
  `(; iteration, max_gradient, loss, U)`, before the break conditions are tested. Returning
  `true` from it stops the iteration. `optimize` prints nothing on its own; pass
  `Lucon.PrintTrace()` to obtain a convergence trace on `stdout`.

Returns a `Lucon.Result`. Since the loss functional is an ordinary callable, `optimize` may be
called with `do` syntax:

    result = Lucon.optimize(U; max_taylor_degree=2, maximize=true) do U, calc_loss
        Γ = H*U*N
        (Γ, calc_loss ? real(dot(U, Γ)) : 0.0)
    end
"""
function optimize(
    gradient,
    U::AbstractMatrix{T};
    max_taylor_degree::Integer,
    maximize::Bool = false,
    min_iter::Integer = 0,
    max_iter::Integer = typemax(Int),
    max_gradient_tolerance::Real = 1e-8,
    solver_algo::Symbol = :CGPR,
    line_search_samples::Integer = 5,
    callback = nothing
)::Result where T<:Number

    # currently only the CG-PR (conjugate gradient Polak-Ribière algorithm is implemented)
    solver_algo === :CGPR || throw(ArgumentError("algorithm :$solver_algo currently not supported in Lucon"))
    max_taylor_degree >= 1 || throw(ArgumentError("max_taylor_degree must be a positive integer"))
    line_search_samples >= 3 || throw(ArgumentError("line_search_samples must be at least 3"))
    min_iter >= 0 || throw(ArgumentError("min_iter must be non-negative"))
    max_iter >= 0 || throw(ArgumentError("max_iter must be non-negative"))

    sgn = maximize ? +1.0 : -1.0

    G_prev = zero(U) # will hold Riemannian derivative of previous iteration

    # the ascent direction, called H as in T. Abrudan et al.; not to be confused with the
    # hermitian matrix H of the Brockett example, which never enters this function
    H = zero(U)

    loss = 0.0         # value of loss function in each iteration
    max_gradient = 0.0 # largest absolute element of the Riemannian gradient

    # the main iteration loop (break condition via the gradient, max_iter or the step size)
    iteration = 0
    status = :max_iter
    while true

        iteration += 1

        # get Euclidean derivative Γ and loss function
        (Γ, loss) = gradient(U, true)

        # construct current Riemannian derivative G, see Eq. (2)
        G = Γ * U'
        G = G - G'

        # the maximum norm of the gradient does not grow with the size of the system
        max_gradient = maximum(abs, G)

        # a callback which returns true asks the iteration to stop
        if callback !== nothing && callback((; iteration, max_gradient, loss, U)) === true
            status = :callback
            break
        end

        # check if convergence is reached
        if max_gradient < max_gradient_tolerance && iteration > min_iter
            status = :converged
            break
        end

        # max_iter counts the rotations of U, of which none has been performed yet
        iteration > max_iter && break

        # Calculate conjugate gradient Polak-Ribière-Polyak (CG-PR) update factor, see Eq. (10)
        if iteration > 1
            cgpr_factor = real(G⋅(G-G_prev)) / real(G_prev⋅G_prev)
        else
            cgpr_factor = 0.0
        end

        # update "prev"
        G_prev = copy(G)

        # update ascent direction
        H = G + cgpr_factor * H

        # check if set-back of the history of H (CGPR factor) is necessary
        if (0.5*real(H⋅G) < 0.0) || ( ((iteration-1)%size(U,1)==0) && (iteration>2) )
            H = copy(G)
        end

        # find the optimal step size via polynomial line search
        (U, μ) = polynomial_line_search(gradient, G, U, H, max_taylor_degree, sgn, line_search_samples)

        # a vanishing step size leaves U unchanged and no further progress can be made
        if iszero(μ)
            status = :line_search
            break
        end

    end

    @debug "Lucon.optimize stopped with status :$status"

    # every break condition is tested before U is rotated, so one rotation less than iterations
    return Result(U, loss, max_gradient, iteration - 1, status)
end # optimize



"""
Apply the diagonal factor exp(-im*x*Λ) as a column scaling of the eigenvectors V, which
fuses into a single broadcast and leaves the expression free of scalar indexing, so that it
also runs on a GPU.
"""
scale_eigenvectors(V, Λ, x) = V .* transpose(cis.(-x .* Λ))

"""
Assemble the rotation matrix R = exp(x*H) from the eigenvectors V and the eigenvalues
-im*Λ of the skew-hermitian matrix H. On the orthogonal group H is real and skew-symmetric,
so that R is real up to roundoff.
"""
rotation_matrix(::Type{T}, V, Λ, x) where T<:Real    = real.(scale_eigenvectors(V,Λ,x) * V')
rotation_matrix(::Type{T}, V, Λ, x) where T<:Complex = scale_eigenvectors(V,Λ,x) * V'



"""
Smallest strictly positive real root of the polynomial p(μ) = c[1] + c[2]μ¹ + c[3]μ² + ...
obtained from the eigenvalues of the companion matrix of p. Returns `nothing` if p has no
such root, see step 8 of Table 1 in T. Abrudan et al.
"""
function smallest_positive_root(c::AbstractVector{<:Real})::Union{Float64,Nothing}

    # negligible leading coefficients render the companion matrix ill-conditioned and
    # produce spurious roots of the order of 1/eps, so lower the degree of p instead
    scale = maximum(abs, c)
    lead = findlast(x -> abs(x) > eps(Float64)*scale, c)
    (lead === nothing || lead < 2) && return nothing

    degree = lead - 1
    companion = zeros(Float64, degree, degree)
    for i = 1:degree-1
        companion[i,i+1] = 1.0
    end
    for j = 1:degree
        companion[degree,j] = -c[j]/c[lead]
    end
    roots = eigvals(companion) # the eigenvalues of the companion matrix are the roots of p

    positive_real_roots = [real(r) for r in roots if isreal(r) && real(r) > 0.0]
    isempty(positive_real_roots) && return nothing

    return minimum(positive_real_roots)
end



"""
Perform a polynomial line search for the optimal step size μ for the conjugate-gradient algorithm.
The procedure is described in section 3.1 in T. Abrudan et al. / Signal Processing 89 (2009) 1704–1714

Returns the rotated matrix exp(sgn*μ*H)U together with the step size μ. A step size of zero means
that the line search found no local optimum along the geodesic, in which case U is returned unchanged.
`G` is the Riemannian gradient at U, from which the derivative at μ=0 is read off directly.
"""
function polynomial_line_search(
    gradient,
    G::AbstractMatrix{T},
    U::AbstractMatrix{T},
    H::AbstractMatrix{T},
    max_taylor_degree::Integer,
    sgn::Float64,
    polynomial_degree::Integer
)::Tuple{AbstractMatrix{T},Float64} where T<:Number

    # diagonalize the skew symmetric matrix H by
    # constructing the hermitian matrix H*im and diagonalize
    Λ, V = eigen(Hermitian(H*im)) # eigenvalues of H are -im*Λ

    # sampling points of μ = 0*μ_step, 1*μ_step, 2*μ_step, ...
    max_abs_eigval = maximum(abs, Λ)
    iszero(max_abs_eigval) && return (U, 0.0) # H vanishes only in a stationary point
    μ_step = 2π / (polynomial_degree * max_taylor_degree * max_abs_eigval)

    # set up rotation matrix exp(sgn*μ_step*H)
    R = rotation_matrix(T, V, Λ, sgn*μ_step)

    # for every μ>0 we calculate the derivative dLdμ = d/dμ L(exp(sgn*μ*H)U), see Eq. (14)
    # in T. Abrudan et al. The trace of Γ(HU)' is its Frobenius product, which spares us
    # from forming the matrix product itself.
    dLdμ = Vector{Float64}(undef, polynomial_degree)
    U_rotated = copy(U)
    for i = 1:polynomial_degree
        U_rotated = R*U_rotated
        (Γ_rotated, _) = gradient(U_rotated, false)
        dLdμ[i] = 2*sgn*real( dot(H*U_rotated, Γ_rotated) )
    end

    # at μ = 0 the derivative is the Frobenius product of H with the Riemannian gradient
    dLdμ0 = sgn*real( dot(G, H) )

    # set up the coefficients for the polynomial by solving the linear system Ma=b for a
    b = dLdμ .- dLdμ0
    M = Matrix{Float64}(undef, polynomial_degree, polynomial_degree)
    for i = 1:polynomial_degree
        for j = 1:polynomial_degree
            M[i,j] = (i*μ_step)^j
        end
    end
    a = M\b # solve linear system

    # the optimal μ corresponds to the smallest positive real root of p(μ) = a₀ + a₁μ¹ + a₂μ² + ...
    μ_opt = smallest_positive_root(vcat(dLdμ0, a))
    μ_opt === nothing && return (U, 0.0)

    # rotate U with optimal μ: U_rotated = exp(sgn*μ_opt*H) U
    U_rotated = rotation_matrix(T, V, Λ, sgn*μ_opt) * U

    return (U_rotated, μ_opt)
end

end # Lucon
