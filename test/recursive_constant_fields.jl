using Enzyme, LLVM, Test

# A `Const` array beside active data in a struct that a recursive function passes on.
#
# `sum(Broadcast.instantiate(Broadcast.broadcasted(f, ds, y)))`, with active `ds` and
# `Const` data `y`, passes the `Broadcasted` by reference to Base's `_mapreduce`, which
# reduces 16 or more elements with the pairwise `mapreduce_impl`, a function that calls
# itself with the same struct. The struct's slot holds `y` beside active data, so Enzyme
# needs a shadow for `y` there. The constant-field proof gives the field the array itself
# as its shadow once it shows that the field only holds constant data that is only read,
# but the recursive call made it give the whole slot up. Static Reverse then made a
# zero-filled copy of `y` on every call (Julia 1.10 and 1.13), which runtime activity
# does not, and an Enzyme without the mixed-activity handler's copy raises
# `EnzymeRuntimeActivityError` instead. See `constant_field_places` in
# src/compiler/constant_fields.jl.

struct RCFLoc
    μ::Float64
    σ::Float64
end
rcf_lp(d::RCFLoc, y) = -0.5 * ((y - d.μ) / d.σ)^2 - log(d.σ)
rcf_wlp(d::RCFLoc, y, w) = w * rcf_lp(d, y)

# Active values materialized into an array, reduced lazily beside `Const` data.
function rcf_lazy(θ, X, y)
    mu = X * θ[1:(end - 1)]
    d = RCFLoc.(mu, exp(θ[end]))
    return sum(Broadcast.instantiate(Broadcast.broadcasted(rcf_lp, d, y)))
end

# The same with the constructor fused into the broadcast.
function rcf_fused(θ, X, y)
    mu = X * θ[1:(end - 1)]
    bc = Broadcast.broadcasted(rcf_lp, Broadcast.broadcasted(RCFLoc, mu, exp(θ[end])), y)
    return sum(Broadcast.instantiate(bc))
end

# Two `Const` arrays in one broadcast.
function rcf_weighted(θ, X, y, w)
    mu = X * θ[1:(end - 1)]
    d = RCFLoc.(mu, exp(θ[end]))
    return sum(Broadcast.instantiate(Broadcast.broadcasted(rcf_wlp, d, y, w)))
end

function rcf_closed(θ, X, y, w = ones(length(y)))
    mu = X * θ[1:(end - 1)]
    s = exp(θ[end])
    z = (y .- mu) ./ s
    return vcat(X' * (w .* z ./ s), sum(w .* (z .^ 2 .- 1)))
end

function rcf_grad(mode, f, θ, args...)
    g = zero(θ)
    autodiff(mode, f, Active, Duplicated(θ, g), map(Const, args)...)
    return g
end

# Bytes a warm gradient allocates.
function rcf_alloc(mode, f, θ, args...)
    rcf_grad(mode, f, θ, args...)
    best = typemax(Int)
    for _ in 1:3
        best = min(best, @allocated(rcf_grad(mode, f, θ, args...)))
    end
    return best
end

@testset "Const data in a lazy broadcast that a recursive reduction passes on" begin
    # 1 and 15 reduce without `mapreduce_impl`, 16 and 40 in its sequential loop, and
    # 3000 recurse past its block size of 1024.
    @testset "n = $n" for n in (1, 3, 15, 16, 40, 3000)
        X = [sin(i + 2j) for i in 1:n, j in 1:2]
        y = [cos(3i) for i in 1:n]
        w = [1.0 + 0.5 * sin(i) for i in 1:n]
        θ = [0.1, 0.2, -0.3]
        X0, y0, w0 = copy(X), copy(y), copy(w)
        for (f, args, ref) in (
                (rcf_lazy, (X, y), rcf_closed(θ, X, y)),
                (rcf_fused, (X, y), rcf_closed(θ, X, y)),
                (rcf_weighted, (X, y, w), rcf_closed(θ, X, y, w)),
            )
            @testset "$f" begin
                g = rcf_grad(Reverse, f, θ, args...)
                @test g ≈ ref rtol = 1.0e-10
                @test rcf_grad(set_runtime_activity(Reverse), f, θ, args...) ≈ ref rtol = 1.0e-10
                @test autodiff(Forward, f, Duplicated(θ, [0.0, 0.0, 1.0]), map(Const, args)...)[1] ≈ ref[3] rtol = 1.0e-10
                @test X == X0 && y == y0 && w == w0
                # No zero copy of the `Const` data: static Reverse allocates what
                # runtime activity does.
                static = rcf_alloc(Reverse, f, θ, args...)
                runtime = rcf_alloc(set_runtime_activity(Reverse), f, θ, args...)
                @test static < runtime + sizeof(y)
            end
        end
    end
end

@testset "constant_field_places through recursive calls" begin
    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module, """
            define void @f(i64 %r1, i64 %r2, i64 %b, i64 %c, i64 %d, i64 %e) {
              ret void
            }
            """
        )
        f = functions(mod)["f"]
        r1, r2, b, c, d, e = parameters(f)
        Field = Enzyme.Compiler.ConstantField
        places(cf, base) = Set(Enzyme.Compiler.constant_field_places(cf, base))

        # `b` points 8 bytes into the slot `r1`, and passes itself on unchanged, as a
        # recursive call does: that adds no place.
        cf = Enzyme.Compiler.ConstantFields(f, Set{LLVM.Value}())
        cf.edges[b] = Field[(r1, 8), (b, 0)]
        @test places(cf, b) == Set(Field[(r1, 8)])
        @test !(b in cf.escaped)

        # A cycle that moves the offset could reach unboundedly many places.
        cf = Enzyme.Compiler.ConstantFields(f, Set{LLVM.Value}())
        cf.edges[c] = Field[(r1, 0), (c, 16)]
        places(cf, c)
        @test c in cf.escaped

        # Two functions that call each other, each with a call site of its own: both
        # reach both slots, whichever is asked first.
        for first in (d, e)
            cf = Enzyme.Compiler.ConstantFields(f, Set{LLVM.Value}())
            cf.edges[d] = Field[(r1, 0), (e, 0)]
            cf.edges[e] = Field[(d, 0), (r2, 0)]
            places(cf, first)
            @test places(cf, d) == Set(Field[(r1, 0), (r2, 0)])
            @test places(cf, e) == Set(Field[(r1, 0), (r2, 0)])
            @test isempty(cf.escaped)
        end
    end
end
