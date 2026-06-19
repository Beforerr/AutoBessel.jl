using Arblib
using Bessels
using AutoBessel
using Test

arb_ref(ν, z; prec = 512) = ComplexF64(AutoBessel.besselj_ball(ν, z; prec))
relerr(a, b) = abs(a - b) / max(abs(b), 1.0)

@testset "real compatibility" begin
    for (ν, z) in ((0, 1.25), (1, 2.5), (2.3, 4.5), (-3, 7.0))
        @test AutoBessel.besselj(ν, z) ≈ Bessels.besselj(ν, z) rtol = 2.0e-14 atol = 2.0e-14
    end
end

@testset "series path against Arb" begin
    cases = (
        (0.0 + 0.0im, 0.25 + 0.1im),
        (1.2 + 0.3im, 3.4 + 0.2im),
        (2.5 - 0.75im, -0.7 + 0.4im),
        (10.0 + 4.0im, 5.0 - 3.0im),
        (0.25 + 8.0im, 8.0 + 1.0im),
    )
    for (ν, z) in cases
        got = AutoBessel.besselj_series(ν, z)
        ref = arb_ref(ν, z)
        @test relerr(got, ref) < 2.0e-12
    end
end

@testset "auto path against Arb" begin
    cases = (
        (1.2 + 0.3im, 3.4 + 0.2im),
        (-2.4 + 0.1im, 0.8 - 0.6im),
        (3.5 - 2.0im, 40.0 + 30.0im),
        (20.0 + 10.0im, -25.0 + 2.0im),
        (1.5 + 0.0im, -3.0 + 0.0im),
        (-4.0 + 0.0im, 2.0 + 3.0im),
    )
    for (ν, z) in cases
        got = AutoBessel.besselj(ν, z)
        fast = AutoBessel.besselj_arb_fast(ν, z)
        ref = arb_ref(ν, z; prec = 768)
        @test relerr(got, ref) < 4.0e-12
        @test relerr(fast, ref) < 4.0e-12
    end
end

@testset "hard cases against Arb" begin
    cases = (
        (0.8 - 0.666im, -12.0 + 0.0im, 5.0e-12),
        (30.0 + 20.0im, 28.0 + 5.0im, 5.0e-12),
        (3.5 - 2.0im, 40.0 + 30.0im, 5.0e-12),
        (-3.0 + 1.0e-9im, 1.5 - 0.2im, 5.0e-11),
        (50.3 - 86.4im, -0.1 + 0.0im, 5.0e-11),
        (20.0 + 10.0im, -25.0 + 2.0im, 5.0e-12),
    )
    for (ν, z, rtol) in cases
        ref = arb_ref(ν, z; prec = 1024)
        @test relerr(AutoBessel.besselj(ν, z), ref) < rtol
        @test relerr(AutoBessel.besselj_arb_fast(ν, z), ref) < rtol
    end
end

@testset "pure Julia large-argument asymptotic" begin
    cases = (
        (3.5 - 2.0im, 40.0 + 30.0im, 5.0e-12),
        (2.3 + 0.4im, 80.0 + 0.0im, 5.0e-12),
        (1.2 + 0.2im, -80.0 + 40.0im, 5.0e-12),
        (2.3 - 3.264im, 0.0 + 128.0im, 5.0e-11),
        (4.0 + 1.0im, -128.0 - 64.0im, 5.0e-11),
    )
    for (ν, z, rtol) in cases
        ref = arb_ref(ν, z; prec = 1024)
        @test relerr(AutoBessel.besselj_asymptotic(ν, z), ref) < rtol
        @test relerr(AutoBessel.besselj(ν, z), ref) < rtol
    end
end

@testset "principal branch cut" begin
    ν = 0.75 + 0.2im
    x = 3.0
    jx = AutoBessel.besselj_series(ν, x + 0im)
    upper = AutoBessel.besselj_series(ν, -x + 0.0im)
    lower = AutoBessel.besselj_series(ν, complex(-x, -0.0))
    @test upper ≈ exp(im * π * ν) * jx rtol = 2.0e-13 atol = 2.0e-13
    @test lower ≈ exp(-im * π * ν) * jx rtol = 2.0e-13 atol = 2.0e-13
    @test abs(upper - lower) > 0.1
end

@testset "zero argument" begin
    @test AutoBessel.besselj(0, 0) == 1
    @test AutoBessel.besselj(2.5 + 0im, 0 + 0im) == 0
    @test AutoBessel.besselj(-2 + 0im, 0 + 0im) == 0
    @test_throws DomainError AutoBessel.besselj(1 + im, 0)
end

@testset "arb_fast non-finite fallback" begin
    z700 = 0.0 + 700.0im
    ref = arb_ref(0.5 + 0im, z700; prec = 1024)
    @test relerr(AutoBessel.besselj_arb_fast(0.5 + 0im, z700), ref) < 4.0e-12

    z750 = 0.0 + 750.0im
    fp = Arblib.fpwrap_bessel_j(ComplexF64(0.5), ComplexF64(z750))
    @test !(isfinite(real(fp)) && isfinite(imag(fp)))
    r = AutoBessel.besselj_arb_fast(0.5 + 0im, z750)
    @test isinf(abs(r))
    @test AutoBessel.besselj(0.5 + 0im, z750) == r
    @test isinf(abs(AutoBessel.besselj_arb(0.5 + 0im, z750)))
end

@testset "nan propagation" begin
    @test isnan(AutoBessel.besselj(2.3, NaN))
    @test isnan(AutoBessel.besselj(NaN, 2.0))
    @test isnan(AutoBessel.besselj_arb_fast(2.3 + 0im, NaN + 0im))
    @test isnan(AutoBessel.besselj_arb(2.3 + 0im, NaN + 0im))
end

@testset "input types" begin
    @test AutoBessel.besselj_series(1.2f0, 3.4f0) isa ComplexF64
    @test AutoBessel.besselj_arb(big"1.2", big"3.4") isa ComplexF64

    setprecision(256) do
        high = AutoBessel.besselj_ball(big"1.2" + big"0.3" * im, big"3.4" + big"0.2" * im; prec = 256)
        rounded = AutoBessel.besselj_ball(1.2 + 0.3im, 3.4 + 0.2im; prec = 256)
        @test ComplexF64(high) ≈ ComplexF64(rounded) rtol = 1.0e-15
    end
end
