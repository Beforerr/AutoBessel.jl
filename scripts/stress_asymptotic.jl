# Run with : julia --project=. scripts/stress_asymptotic.jl --first-seed 1 --seeds 20 --samples 1000 --prec 768 --rtol 1e-10

using Arblib
using AutoBessel
using Printf
using Random

const DEFAULT_SEEDS = 20
const DEFAULT_SAMPLES = 2_000
const DEFAULT_PREC = 1024
const DEFAULT_RTOL = 1.0e-10

function parseargs(args)
    opts = Dict(
        "first-seed" => "1",
        "seeds" => string(DEFAULT_SEEDS),
        "samples" => string(DEFAULT_SAMPLES),
        "prec" => string(DEFAULT_PREC),
        "rtol" => string(DEFAULT_RTOL),
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            i == length(args) && error("missing value for $arg")
            opts[key] = args[i + 1]
            i += 2
        else
            error("unknown argument $arg")
        end
    end
    return (
        first_seed = parse(Int, opts["first-seed"]),
        seeds = parse(Int, opts["seeds"]),
        samples = parse(Int, opts["samples"]),
        prec = parse(Int, opts["prec"]),
        rtol = parse(Float64, opts["rtol"]),
    )
end

relerr(a, b) = abs(a - b) / max(abs(b), 1.0)

function arb_ref(ν, z, prec)
    return AutoBessel.besselj_arb(ν, z; prec, maxprec = max(prec, 4096), rtol = 1.0e-13, atol = 1.0e-13)
end

function asymptotic_probe_candidate(ν, z)
    νc = ComplexF64(ν)
    zc = ComplexF64(z)
    isfinite(real(νc)) && isfinite(imag(νc)) && isfinite(real(zc)) && isfinite(imag(zc)) || return false
    az = abs(zc)
    aν = abs(νc)
    az >= 48 || return false
    real(zc) >= 0 || return false
    abs(imag(νc)) <= 2 || return false
    abs(imag(zc)) < 650 || return false
    return (aν <= 8 && az >= 6 * max(aν, 1.0)) || az >= 24 * max(aν, 1.0)
end

function loguniform(rng, lo, hi)
    return exp(log(lo) + rand(rng) * (log(hi) - log(lo)))
end

function signedloguniform(rng, lo, hi)
    return (rand(rng, Bool) ? 1 : -1) * loguniform(rng, lo, hi)
end

function jitter(rng, x; scale = 1.0e-12)
    return x + scale * max(abs(x), 1.0) * randn(rng)
end

function randcomplex(rng, r; axis = nothing)
    if axis === :real
        θ = rand(rng, (0.0, π)) + 1.0e-14randn(rng)
    elseif axis === :imag
        θ = rand(rng, (π / 2, -π / 2)) + 1.0e-14randn(rng)
    elseif axis === :cut
        θ = rand(rng, (π, -π)) + 1.0e-14randn(rng)
    else
        θ = 2π * rand(rng)
    end
    return ComplexF64(r * cis(θ))
end

function random_case(rng)
    family = rand(rng, 1:14)

    if family == 1
        # Intended asymptotic region, small/moderate order.
        aν = loguniform(rng, 1.0e-3, 8)
        ratio = loguniform(rng, 6, 120)
        az = max(48, ratio * max(aν, 1))
    elseif family == 2
        # Intended asymptotic region, larger order.
        aν = loguniform(rng, 8, 240)
        ratio = loguniform(rng, 24, 160)
        az = max(48, ratio * aν)
    elseif family == 3
        # Boundary just below/above current ratio gates.
        aν = loguniform(rng, 0.05, 160)
        center = aν <= 8 ? 6.0 : 24.0
        ratio = center * exp(0.04randn(rng))
        az = max(48, ratio * max(aν, 1))
    elseif family == 4
        # Near axes where branch/prefactor choices are fragile.
        aν = loguniform(rng, 0.1, 180)
        ratio = loguniform(rng, 5, 100)
        az = max(48, ratio * max(aν, 1))
    elseif family == 5
        # Transition-ish region: should normally avoid asymptotic auto path.
        aν = loguniform(rng, 1, 160)
        az = loguniform(rng, 0.25, 4.0) * max(aν, 1)
    elseif family == 6
        # Small/moderate z: series/Arb fallback validation.
        aν = loguniform(rng, 1.0e-6, 80)
        az = loguniform(rng, 1.0e-12, 24)
    elseif family == 7
        # Large imaginary order.
        aν = loguniform(rng, 8, 260)
        az = loguniform(rng, 0.25, 80) * max(aν, 1)
    elseif family == 8
        # Negative real branch side.
        aν = loguniform(rng, 1.0e-6, 120)
        az = loguniform(rng, 1.0e-8, 800)
    elseif family == 9
        # Very large magnitude but still within Float64 exp range.
        aν = loguniform(rng, 1.0e-3, 260)
        az = loguniform(rng, 48, 2_000)
    elseif family == 10
        # Near negative-integer Γ poles.
        n = rand(rng, 1:80)
        ν = ComplexF64(-n + signedloguniform(rng, 1.0e-15, 1.0e-6) + signedloguniform(rng, 1.0e-15, 1.0e-6) * im)
        z = randcomplex(rng, loguniform(rng, 1.0e-8, 120), axis = rand(rng, (nothing, :real, :imag, :cut)))
        return ν, z
    elseif family == 11
        # Exact and near integer/half-integer orders.
        base = rand(rng, -80:80) + rand(rng, (0.0, 0.5))
        ν = ComplexF64(base + signedloguniform(rng, 1.0e-15, 1.0e-7) + signedloguniform(rng, 1.0e-15, 1.0e-7) * im)
        z = randcomplex(rng, loguniform(rng, 1.0e-6, 500), axis = rand(rng, (nothing, :real, :imag, :cut)))
        return ν, z
    elseif family == 12
        # Around |imag(z)| cutoff used to avoid exponential overflow.
        aν = loguniform(rng, 1.0e-3, 80)
        iz = jitter(rng, rand(rng, (-700.0, 700.0)); scale = 1.0e-3)
        rz = signedloguniform(rng, 1.0e-9, 2_000)
        ν = randcomplex(rng, aν)
        return ComplexF64(ν + 0.2randn(rng) + 0.2randn(rng) * im), ComplexF64(rz + iz * im)
    elseif family == 13
        # Large arbitrary plane for fallback coverage.
        ν = ComplexF64(signedloguniform(rng, 1.0e-6, 300) + signedloguniform(rng, 1.0e-6, 300) * im)
        z = ComplexF64(signedloguniform(rng, 1.0e-10, 2_000) + signedloguniform(rng, 1.0e-10, 2_000) * im)
        return ν, z
    else
        # Hand-biased old failures: imaginary z and order-to-argument ratio near 6.
        aν = loguniform(rng, 30, 120)
        az = aν * loguniform(rng, 5.5, 7.0)
        ν = randcomplex(rng, aν) + ComplexF64(0.2randn(rng), 0.2randn(rng))
        return ComplexF64(ν), randcomplex(rng, az, axis = :imag)
    end

    θν = 2π * rand(rng)
    ν = ComplexF64(aν * cis(θν) + 0.2randn(rng) + 0.2randn(rng) * im)

    if family == 4
        z = randcomplex(rng, az, axis = rand(rng, (:real, :imag, :cut)))
    elseif family == 8
        z = randcomplex(rng, az, axis = :cut)
    else
        z = randcomplex(rng, az)
    end
    return ν, z
end

function update_worst!(state, err, ν, z, got, ref)
    if !isfinite(err) || err > state.err
        state.err = err
        state.ν = ν
        state.z = z
        state.got = got
        state.ref = ref
    end
    return state
end

mutable struct Worst
    err::Float64
    ν::ComplexF64
    z::ComplexF64
    got::ComplexF64
    ref::ComplexF64
end

Worst() = Worst(-Inf, NaN + NaN * im, NaN + NaN * im, NaN + NaN * im, NaN + NaN * im)

function print_worst(label, w)
    println(label)
    println("  err = ", @sprintf("%.3e", w.err))
    println("  ν   = ", w.ν)
    println("  z   = ", w.z)
    println("  got = ", w.got)
    return println("  ref = ", w.ref)
end

function run_seed(seed, samples, prec, rtol)
    rng = Xoshiro(seed)
    total = 0
    checked = 0
    skipped_nonfinite_ref = 0
    ref_errors = 0
    candidate = 0
    asymp_finite_bad = 0
    asymp_nonfinite = 0
    asymp_errors = 0
    auto_bad = 0
    auto_nonfinite = 0
    auto_errors = 0
    worst_asymp = Worst()
    worst_auto = Worst()

    for _ in 1:samples
        ν, z = random_case(rng)
        total += 1
        ref = try
            arb_ref(ν, z, prec)
        catch
            ref_errors += 1
            continue
        end
        if !(isfinite(real(ref)) && isfinite(imag(ref)))
            skipped_nonfinite_ref += 1
            continue
        end
        checked += 1

        auto = try
            AutoBessel.besselj(ν, z)
        catch
            auto_errors += 1
            update_worst!(worst_auto, Inf, ν, z, NaN + NaN * im, ref)
            nothing
        end
        if auto === nothing
        elseif isfinite(real(auto)) && isfinite(imag(auto))
            auto_err = relerr(auto, ref)
            auto_bad += auto_err > rtol
            update_worst!(worst_auto, auto_err, ν, z, auto, ref)
        else
            auto_nonfinite += 1
            update_worst!(worst_auto, Inf, ν, z, auto, ref)
        end

        if asymptotic_probe_candidate(ν, z)
            candidate += 1
            asymp = try
                AutoBessel.besselj_asymptotic(ν, z)
            catch
                asymp_errors += 1
                update_worst!(worst_asymp, Inf, ν, z, NaN + NaN * im, ref)
                nothing
            end
            if asymp === nothing
            elseif isfinite(real(asymp)) && isfinite(imag(asymp))
                asymp_err = relerr(asymp, ref)
                asymp_finite_bad += asymp_err > rtol
                update_worst!(worst_asymp, asymp_err, ν, z, asymp, ref)
            else
                asymp_nonfinite += 1
                update_worst!(worst_asymp, Inf, ν, z, asymp, ref)
            end
        end
    end

    return (
        seed = seed,
        total = total,
        checked = checked,
        skipped_nonfinite_ref = skipped_nonfinite_ref,
        ref_errors = ref_errors,
        candidate = candidate,
        asymp_finite_bad = asymp_finite_bad,
        asymp_nonfinite = asymp_nonfinite,
        asymp_errors = asymp_errors,
        auto_bad = auto_bad,
        auto_nonfinite = auto_nonfinite,
        auto_errors = auto_errors,
        worst_asymp = worst_asymp,
        worst_auto = worst_auto,
    )
end

function main(args = ARGS)
    opts = parseargs(args)
    last_seed = opts.first_seed + opts.seeds - 1
    println("stress_asymptotic first_seed=$(opts.first_seed) last_seed=$last_seed samples=$(opts.samples) prec=$(opts.prec) rtol=$(opts.rtol)")

    totals = Dict(
        :total => 0,
        :checked => 0,
        :skipped_nonfinite_ref => 0,
        :ref_errors => 0,
        :candidate => 0,
        :asymp_finite_bad => 0,
        :asymp_nonfinite => 0,
        :asymp_errors => 0,
        :auto_bad => 0,
        :auto_nonfinite => 0,
        :auto_errors => 0,
    )
    global_worst_asymp = Worst()
    global_worst_auto = Worst()

    for seed in opts.first_seed:last_seed
        result = run_seed(seed, opts.samples, opts.prec, opts.rtol)
        println(
            "seed=", seed,
            " total=", result.total,
            " checked=", result.checked,
            " candidate=", result.candidate,
            " asymp_bad=", result.asymp_finite_bad,
            " asymp_nonfinite=", result.asymp_nonfinite,
            " asymp_errors=", result.asymp_errors,
            " auto_bad=", result.auto_bad,
            " auto_nonfinite=", result.auto_nonfinite,
            " auto_errors=", result.auto_errors,
            " worst_asymp=", @sprintf("%.2e", result.worst_asymp.err),
            " worst_auto=", @sprintf("%.2e", result.worst_auto.err),
        )

        for key in keys(totals)
            totals[key] += getproperty(result, key)
        end
        update_worst!(
            global_worst_asymp,
            result.worst_asymp.err,
            result.worst_asymp.ν,
            result.worst_asymp.z,
            result.worst_asymp.got,
            result.worst_asymp.ref,
        )
        update_worst!(
            global_worst_auto,
            result.worst_auto.err,
            result.worst_auto.ν,
            result.worst_auto.z,
            result.worst_auto.got,
            result.worst_auto.ref,
        )
    end

    println("summary")
    for key in (
            :total,
            :checked,
            :skipped_nonfinite_ref,
            :ref_errors,
            :candidate,
            :asymp_finite_bad,
            :asymp_nonfinite,
            :asymp_errors,
            :auto_bad,
            :auto_nonfinite,
            :auto_errors,
        )
        println("  ", key, " = ", totals[key])
    end
    print_worst("worst asymptotic candidate", global_worst_asymp)
    print_worst("worst auto", global_worst_auto)

    return if totals[:auto_bad] > 0 ||
            totals[:auto_nonfinite] > 0 ||
            totals[:auto_errors] > 0 ||
            totals[:asymp_finite_bad] > 0 ||
            totals[:asymp_errors] > 0
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
