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

In order to use Lucon to optimize a loss functional $L(U)$ one has provide a Julia function that calculates the Eucledean derivative $\Gamma_{ij} = \partial L / \partial u^*_{ij}$. For the above example (Brockett criterion) the Eucledean derivative simply reads $\Gamma = \partial L /\partial U^\dagger = H U N$.  

To this end a sub-type of the abstract type ```Lucon.LossFunctional``` has to be defined which can hold all quantities necessary for the loss functional (for the example, it is only the hermitian matrix $H$).
The Eucledean derivative has to be provided by overloading the function ```Lucon.EuclideanDerivative```.

```julia
import Lucon

struct LossFunctional <: Lucon.LossFunctional
    H::Hermitian{<:Number}
end

function Lucon.EuclideanDerivative(
    L::LossFunctional,
    U::AbstractMatrix{T},
    CalcLoss::Bool
)::Tuple{AbstractMatrix{T},Float64} where T<:Number
    Loss = 0.0 # the value of the Loss functional
    dim = size(L.H,1)
    N = Diagonal([1.0*n for n=1:dim]) # the N matrix is a diagonal matrix with entries N_nn = n
    Γ = L.H*U*N # Euclidean derivative has same type and dimension as U
    # the trace of U'Γ is the Frobenius product of U and Γ
    (CalcLoss == true) && (Loss = real(dot(U, Γ)))
    return (Γ, Loss)
end
```
`EuclideanDerivative` is evaluated once per iteration and once for every sampling point of the line search, so it dominates the run time. Accept an `AbstractMatrix` so that `U` may live on a GPU.
The optimization can then be performed via
```julia
# set up your hermitian matrix H and initial unitary U
L = LossFunctional(H)
(U, Loss) = optimize(L,U)
```
The full example and its usage can be found in the source file [BrockettLoss.jl](src/BrockettLoss.jl) and in the test file [runtests.jl](test/runtests.jl).  
Both can be used as a **template** to implement arbitrary loss functionals.

The `optimize` function above is a thin wrapper around `Lucon.optimize`, which takes two further positional arguments:
```julia
(U, Loss) = Lucon.optimize(L, U, UDegree, sgn; MinIter=0, MaxIter=-1, GradNormBreak=1.0e-8, PolynomialLineSearchDegree=5)
```
* `UDegree` is the order $q$ of the loss functional, i.e. the highest power of $t$ appearing in the Taylor expansion of $L(U + tZ)$. It sets the width $T_\mu = 2\pi/(q\,|\omega_\text{max}|)$ of the window the line search scans. For the Brockett criterion above $q=2$.
* `sgn` is `+1.0` to maximize and `-1.0` to minimize $L(U)$.

The element type of the initial `U` selects the group that is optimized over, the orthogonal group for a real and the unitary group for a complex element type. Convergence is reached once the Frobenius norm of the Riemannian gradient drops below `GradNormBreak`, and `MaxIter` limits the number of rotations of `U`.

## How to cite?

Benjamin Wöckinger, Alexander Rumpf, Tobias Schäfer. *Convergence and Properties of Intrinsic Bond Orbitals in Solids*, [J. Chem. Theory Comput. 2025, 21, 20, 10515–10526](https://doi.org/10.1021/acs.jctc.5c00130)
