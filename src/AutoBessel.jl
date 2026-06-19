module AutoBessel

import Arblib
import Bessels
import Gamma

export besselj

const _LOG_FLOATMAX = log(floatmax(Float64))
const _DEFAULT_ARB_PREC = 192
const _DEFAULT_MAX_ARB_PREC = 3072

struct BesselJConvergenceError <: Exception
    msg::String
end

Base.showerror(io::IO, err::BesselJConvergenceError) = print(io, err.msg)

"""
    besselj(ν, z)

Compute Bessel function of the first kind ``J_ν(z)`` for real or complex order and argument.

Uses Bessels.jl for supported real-real calls, a fast
complex power series where it is well-conditioned, and Arblib's fast midpoint
wrapper for large or delicate inputs. Call `besselj_series`, `besselj_arb_fast`,
or `besselj_arb` directly to force a backend.
"""
function besselj(ν::Real, z::Real)
    if isfinite(ν) && isfinite(z) && (z >= 0 || isinteger(ν))
        return Bessels.besselj(ν, z)
    end
    return besselj(complex(float(ν)), complex(float(z)))
end

function besselj(ν, z)
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    _domain_at_zero(νc, zc) !== nothing && return _domain_at_zero(νc, zc)
    if _series_candidate(νc, zc)
        try
            return besselj_series(νc, zc)
        catch err
            err isa BesselJConvergenceError || rethrow()
        end
    end
    return besselj_arb_fast(νc, zc)
end

"""
    besselj_series(ν, z; tol=eps(Float64), maxterms=10000)

Evaluate ``J_ν(z)`` by the convergent ``0F1`` power series on the principal
branch. This is fastest for modest `abs(z)` and order not near negative
integers. Throws `BesselJConvergenceError` if convergence or scaling fails.
"""
function besselj_series(ν, z; tol::Real = eps(Float64), maxterms::Integer = 10_000, kwargs...)
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    special = _domain_at_zero(νc, zc)
    special !== nothing && return special

    reflected = _negative_integer_reflection(νc)
    reflected !== nothing && return reflected[2] * besselj_series(reflected[1], zc; tol, maxterms)

    _near_gamma_pole(νc) && throw(BesselJConvergenceError("series is ill-conditioned near Γ(ν+1) pole"))

    logpref = νc * log(zc / 2) - Gamma.loggamma(νc + 1)
    isfinite(real(logpref)) && isfinite(imag(logpref)) ||
        throw(BesselJConvergenceError("nonfinite Bessel J series prefactor"))
    real(logpref) <= _LOG_FLOATMAX ||
        throw(BesselJConvergenceError("Bessel J series prefactor overflows Float64"))

    w = -(zc * zc) / 4
    term = 1.0 + 0.0im
    sum = term
    c = 0.0 + 0.0im
    atol = max(float(tol), eps(Float64))

    for k in 1:maxterms
        denom = k * (νc + k)
        iszero(denom) && throw(BesselJConvergenceError("series denominator hit Γ pole"))
        term *= w / denom
        y = term - c
        t = sum + y
        c = (t - sum) - y
        sum = t
        if k > 4 && abs(term) <= atol * max(abs(sum), 1.0)
            ans = exp(logpref) * sum
            isfinite(real(ans)) && isfinite(imag(ans)) ||
                throw(BesselJConvergenceError("nonfinite Bessel J series result"))
            return ans
        end
    end

    throw(BesselJConvergenceError("Bessel J series did not converge in $maxterms terms"))
end

"""
    besselj_asymptotic(ν, z; maxterms=200)

Evaluate Arb's normalized large-argument asymptotic expansion in pure Julia.
Accurate when `abs(z)` is much larger than `abs(ν)` in suitable sectors, but
does not provide an error bound. Use `besselj` for production default behavior.
"""
function besselj_asymptotic(ν, z; maxterms::Integer = 200)
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    iszero(zc) && throw(DomainError((ν, z), "large-argument asymptotic expansion is undefined at z = 0"))

    a = νc + 0.5
    b = 2νc + 1
    if real(zc) > sqrt(eps(Float64)) * abs(zc)
        θ = zc - (2νc + 1) * (π / 4)
        A1 = exp(im * θ)
        A2 = exp(-im * θ)
        C = inv(sqrt(2π * zc))
    else
        iz = im * zc
        A1 = iz^(-a) * zc^νc * exp(iz)
        A2 = (-iz)^(-a) * zc^νc * exp(-iz)
        C = inv(sqrt(2π))
    end

    U2 = _hyperu_star_asymptotic(a, b, 2im * zc; maxterms)
    U1 = _hyperu_star_asymptotic(a, b, -2im * zc; maxterms)
    return C * (A1 * U1 + A2 * U2)
end

"""
    besselj_ball(ν, z; prec=192)

Return Arb's complex ball enclosure for principal-branch ``J_ν(z)``.
"""
function besselj_ball(ν, z; prec::Integer = _DEFAULT_ARB_PREC, kwargs...)
    res = Arblib.Acb()
    Arblib.hypgeom_bessel_j!(res, Arblib.Acb(ν), Arblib.Acb(z); prec)
    return res
end

"""
    besselj_arb_fast(ν, z; kwargs...)

Evaluate principal-branch ``J_ν(z)`` through Arblib's fast Float64 wrapper.
Falls back to checked ball arithmetic ([`besselj_arb`](@ref)) when the wrapper
throws or returns a non-finite result, since `fpwrap_bessel_j` defaults to
`error_on_failure = false` and signals failure by returning `NaN`/`Inf` rather
than throwing. Any `kwargs` (`prec`, `maxprec`, `rtol`, `atol`) are forwarded to
the fallback.
"""
function besselj_arb_fast(ν, z; kwargs...)
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    special = _domain_at_zero(νc, zc)
    special !== nothing && return special

    res = try
        Arblib.fpwrap_bessel_j(νc, zc)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end

    res !== nothing && isfinite(real(res)) && isfinite(imag(res)) && return res
    return besselj_arb(νc, zc; kwargs...)
end

"""
    besselj_arb(ν, z; prec=192, maxprec=3072, rtol=16eps(), atol=16eps())

Evaluate with Arb ball arithmetic and return the `ComplexF64` midpoint after
checking that the ball radius is small enough for Float64 use.
"""
function besselj_arb(
        ν,
        z;
        prec::Integer = _DEFAULT_ARB_PREC,
        maxprec::Integer = _DEFAULT_MAX_ARB_PREC,
        rtol::Real = 16eps(Float64),
        atol::Real = 16eps(Float64),
        kwargs...,
    )
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    (isnan(νc) || isnan(zc)) && return ComplexF64(NaN, NaN)
    special = _domain_at_zero(νc, zc)
    special !== nothing && return special

    p = max(64, Int(prec))
    while p <= maxprec
        ball = besselj_ball(νc, zc; prec = p)
        mid = ComplexF64(ball)
        err = _ball_radius64(ball)
        if isfinite(real(mid)) && isfinite(imag(mid))
            scale = max(abs(mid), 1.0)
            err <= max(float(atol), float(rtol) * scale) && return mid
        elseif err == 0 || _overflows_float64(ball)
            return mid
        end
        p *= 2
    end

    throw(BesselJConvergenceError("Arb Bessel J ball too wide for Float64 after $maxprec bits"))
end

function _domain_at_zero(ν::ComplexF64, z::ComplexF64)
    !iszero(z) && return nothing
    iszero(ν) && return 1.0 + 0.0im
    iszero(imag(ν)) && real(ν) > 0 && return 0.0 + 0.0im
    _negative_integer_reflection(ν) !== nothing && return 0.0 + 0.0im
    throw(DomainError((ν, z), "J_ν(0) is finite only for ν = 0 or real(ν) > 0 with imag(ν) = 0"))
end

function _negative_integer_reflection(ν::ComplexF64)
    iszero(imag(ν)) || return nothing
    isinteger(real(ν)) || return nothing
    n = round(Int, real(ν))
    n < 0 || return nothing
    return complex(float(-n)), (iseven(n) ? 1.0 : -1.0)
end

function _near_gamma_pole(ν::ComplexF64)
    abs(imag(ν)) <= 1.0e-7 || return false
    n = round(real(ν))
    return real(ν) <= -1 && abs(real(ν) - n) <= 1.0e-7
end

function _series_candidate(ν::ComplexF64, z::ComplexF64)
    isfinite(real(ν)) && isfinite(imag(ν)) && isfinite(real(z)) && isfinite(imag(z)) || return false
    abs(z) <= 12 || return false
    abs(ν) <= 200 || return false
    return !_near_gamma_pole(ν)
end

function _hyperu_star_asymptotic(a, b, z; maxterms::Integer = 200)
    term = 1.0 + 0.0im
    sum = term
    best_sum = sum
    best = abs(term)
    c = 0.0 + 0.0im
    c1 = a - b + 1

    for k in 1:maxterms
        term *= -((a + k - 1) * (c1 + k - 1)) / (k * z)
        y = term - c
        t = sum + y
        c = (t - sum) - y
        sum = t

        at = abs(term)
        if at < best
            best = at
            best_sum = sum
        elseif k > 2
            return best_sum
        end
        at <= eps(Float64) * max(abs(sum), 1.0) && return sum
    end

    return best_sum
end

function _ball_radius64(ball)
    rr = Float64(Arblib.radius(Arblib.Arb, Arblib.realref(ball)))
    ri = Float64(Arblib.radius(Arblib.Arb, Arblib.imagref(ball)))
    return hypot(rr, ri)
end

function _overflows_float64(ball)
    lb = Float64(Arblib.abs_lbound(Arblib.Arf, ball))
    return lb > floatmax(Float64)
end

end
