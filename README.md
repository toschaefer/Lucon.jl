# Lucon

[![Build Status](https://github.com/toschaefer/Lucon.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/toschaefer/Lucon.jl/actions/workflows/CI.yml?query=branch%3Amain)

Lucon (**L**oss optimization under **U**nitary **CON**straint) optimizes loss functions mapping a unitary matrix onto a number. A conjugate-gradient algorithm is used following the work by [T. Abrudan et al., Signal Processing 89 (2009) 1704–1714](https://dx.doi.org/10.1016/j.sigpro.2009.03.015).  

The module presents potential applications in various fields. For instance, it can be employed for tasks such as orbital rotations (e.g., orbital localization) in quantum chemistry and materials science, as well as for various tasks in signal processing applications or machine learning algorithms. The main motivation for Lucon.jl is given by orbital localizations for calculations in materials physics and quantum chemistry. These will be referenced here shortly.

The code is designed in a way that users can implement arbitrary loss functionals with little effort for optimization with Lucon.jl. As a template the [BrockettLoss.jl](src/BrockettLoss.jl) functional can be used (see example below). 

To provide a very simple and illustrative example of the module's potential use cases, consider the following loss functional that can be used to diagonalize a hermitian matrix.
```math
L(U) = \text{tr}(U^\dagger H U N)
```
Here, $H$ is a hermitian matrix (to be diagonalized) and $N$ is a diagonal matrix with distinct entries in ascending order, $N_{nm} = n\delta_{nm}$. Lucon finds the optimal $U$ which maximizes the loss functional.  For this particular choice of $L(U)$ (also known as [Brockett criterion](https://doi.org/10.1016/0024-3795(91)90021-N)), the optimal unitary matrix is the one that diagonalizes $H$.

## Install

In the Julia REPL, simply run the following commands:
```julia
using Pkg
Pkg.add("Lucon")
```


## Usage

In order to use Lucon to optimize a loss functional $L(U)$ one has to provide a Julia function that calculates the Eucledean derivative $\Gamma_{ij} = \partial L / \partial u^*_{ij}$. For the above example (Brockett criterion) the Eucledean derivative simply reads $\Gamma = \partial L /\partial U^\dagger = H U N$.

The loss functional is any callable `Gradient(U, CalcLoss)` returning the tuple `(Γ, Loss)`. The value of the loss is only read when `CalcLoss` is `true`, so its computation may be skipped otherwise. Nothing has to be sub-typed and no method of Lucon has to be overloaded, which means that `optimize` can be called with `do` syntax:

```julia
import Lucon
using LinearAlgebra

# set up your hermitian matrix H and initial unitary U
N = Diagonal([1.0*n for n=1:size(H,1)]) # the N matrix is a diagonal matrix with entries N_nn = n

Result = Lucon.optimize(U; UDegree=2, Maximize=true) do U, CalcLoss
    Γ = H*U*N # Euclidean derivative has same type and dimension as U
    # the trace of U'Γ is the Frobenius product of U and Γ
    (Γ, CalcLoss ? real(dot(U, Γ)) : 0.0)
end
```

When the functional has to carry precomputed quantities, give them to a struct and make the struct callable. Store them with a concrete type and build them once, since the functional is evaluated once per iteration and once for every sampling point of the line search, and therefore dominates the run time:

```julia
struct BrockettCriterion{T<:Number, A<:AbstractMatrix{T}}
    H::Hermitian{T,A}
    N::Diagonal{Float64,Vector{Float64}}
end

BrockettCriterion(H::Hermitian) = BrockettCriterion(H, Diagonal([1.0*n for n=1:size(H,1)]))

function (B::BrockettCriterion)(U::AbstractMatrix, CalcLoss::Bool)
    Γ = B.H*U*B.N
    (Γ, CalcLoss ? real(dot(U, Γ)) : 0.0)
end

Result = Lucon.optimize(BrockettCriterion(H), U; UDegree=2, Maximize=true)
```
Accept an `AbstractMatrix` so that `U` may live on a GPU.
The full example and its usage can be found in the source file [BrockettLoss.jl](src/BrockettLoss.jl) and in the test file [runtests.jl](test/runtests.jl).  
Both can be used as a **template** to implement arbitrary loss functionals.

`optimize` returns a `Lucon.Result`:
```julia
julia> Result
Lucon.Result
  Status:      converged
  Iterations:  79
  Loss:        9.5330162221636101e+00
  max|grad|:   4.030e-09
  U:           6×6 Matrix{ComplexF64}

julia> Lucon.Converged(Result)
true
```
`Status` is one of `:converged`, `:maxiter`, `:callback`, or `:linesearch` if the line search found no positive step size.

The full signature reads
```julia
Result = Lucon.optimize(
    Gradient, 
    U; 
    UDegree, 
    Maximize=false, 
    MinIter=0, 
    MaxIter=typemax(Int), 
    MaxGradientTolerance=1.0e-8, 
    PolynomialLineSearchDegree=5,
    Callback=nothing
)
```
* `UDegree` is the order $q$ of the loss functional, i.e. the highest power of $t$ appearing in the Taylor expansion of $L(U + tZ)$. It sets the width $T_\mu = 2\pi/(q\,|\omega_\text{max}|)$ of the window the line search scans. It has no default because it is a property of the functional. For the Brockett criterion above $q=2$.
* `Maximize` maximizes $L(U)$ instead of minimizing it.

The element type of the initial `U` selects the group that is optimized over, the orthogonal group for a real and the unitary group for a complex element type. `MaxIter` limits the number of rotations of `U` and is unlimited by default.

Convergence is reached once the largest absolute element of the Riemannian gradient $G$ drops below `MaxGradientTolerance`. This maximum norm is used instead of the Frobenius norm because it does not grow with the size of the system: if a supersystem is built from $N$ non-interacting copies of a subsystem, then $\max_{ij}|G_{ij}|$ is unchanged while $\|G\|_F$ grows as $\sqrt{N}$. One and the same `MaxGradientTolerance` therefore converges subsystem and supersystem to the same accuracy per degree of freedom. 

## Output

`optimize` prints nothing. Progress is reported through `Callback`, a function which is called once per iteration with the named tuple `(; Iteration, MaxGradient, Loss, U)` and which stops the iteration when it returns `true`. To print a convergence trace, pass the ready-made `Lucon.PrintTrace`:
```julia
Result = Lucon.optimize(
    Gradient, 
    U; 
    UDegree=2, 
    Callback=Lucon.PrintTrace() # or Lucon.PrintTrace(stderr)
)
```
```
 #iter   max|grad|            loss-function
     1   2.136e+00  -1.2117919646339959e+00
     2   1.929e+00   3.4753409237499291e+00
     3   1.604e+00   6.7459345042215935e+00
```
The callback is equally the place to record a convergence history, to checkpoint `U`, or to stop on a criterion of your own:
```julia
History = Float64[]
RecordLoss(State) = (push!(History, State.Loss); State.Iteration ≥ 100)

Result = Lucon.optimize(
    Gradient, 
    U; 
    UDegree=2, 
    Callback=RecordLoss
)
```
A callback which stopped the iteration leaves `Result.Status == :callback`. The reason for which the iteration stopped is in addition emitted as a `@debug` message and can be made visible with `ENV["JULIA_DEBUG"] = "Lucon"`.

## How to cite?

Benjamin Wöckinger, Alexander Rumpf, Tobias Schäfer. *Convergence and Properties of Intrinsic Bond Orbitals in Solids*, [J. Chem. Theory Comput. 2025, 21, 20, 10515–10526](https://doi.org/10.1021/acs.jctc.5c00130)
