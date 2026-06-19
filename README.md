# AutoBessel.jl

Bessel function of the first kind $J_ν(z)$ for complex order and argument.

## Quickstart

```julia
using AutoBessel

besselj(0.5, 1.0)            # real order, real argument
besselj(1 + 2im, 3 - 1im)    # complex order and argument
besselj(2, 0)                # special value at z = 0
```

`besselj(ν, z)` picks a backend automatically from Bessels.jl, Arb, and a custom power series implementation.

## Context

- Bessels.jl : complex support is limited to Airy + order 0/1 Bessel functions, and generic Bessel routines target real args/orders. Source: https://github.com/JuliaMath/Bessels.jl
- Arb supports complex ν,z with automatic algorithm choice: acb_hypgeom_bessel_j, using both 0F1 and asymptotic forms. Best oracle. Source: https://arblib.org/acb_hypgeom.html
- Algorithm 912 explicitly targets cylindrical functions with complex order and complex argument. Likely best paper/software model for full domain. Source: https://dl.acm.org/doi/10.1145/1916461.1916471
- SciPy/MATLAB/AMOS: fast double precision, but order is real; argument may be complex. SciPy docs: jv = “real order and complex argument”, uses AMOS zbesj. Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html
- AMOS Algorithm 644: complex argument, nonnegative real order. Good reference/compat layer, not complex order. Source: https://www.netlib.org/toms-2014-06-10/644
- mpmath supports arbitrary complex order/argument, arbitrary precision, slower. Source: https://mpmath.org/doc/current/functions/bessel.html
