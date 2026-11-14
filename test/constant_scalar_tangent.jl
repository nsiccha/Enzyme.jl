using Enzyme, Test
const CST_Rules = Enzyme.EnzymeRules
Base.@noinline function cst_partials!(x, data, scale, initial, seed, buffer)
    if axes(x) == axes(data)
        value = 0.0
        for i in eachindex(x)
            z = scale * x[i] + data[i]
            value += sin(z)
            buffer[i] += cos(z)
        end
        total = length(x) > 0 ? initial + value : initial
        for i in eachindex(parent(buffer))
            parent(buffer)[i] = iszero(parent(buffer)[i]) ? zero(scale * parent(buffer)[i]) : scale * parent(buffer)[i]
        end
    else
        value = 0.0
        for i in eachindex(x)
            z = scale * x[i] + data[mod1(i, length(data))]
            value += sin(z)
            buffer[i] += cos(z)
        end
        total = length(x) > 0 ? initial + value : initial
        for i in eachindex(parent(buffer))
            parent(buffer)[i] = iszero(parent(buffer)[i]) ? zero(scale * parent(buffer)[i]) : scale * parent(buffer)[i]
        end
    end
    return total, seed
end

mutable struct CSTBuffer
    tape::Vector{Float64}
    empty::Vector{Float64}
end
CSTBuffer() = (empty = Float64[]; CSTBuffer(empty, empty))
@inline cst_take!(slot, n, ::Val{false}) = zeros(n)
@inline function cst_take!(slot, n, ::Val{true})
    tape = slot.tape
    length(tape) == n || return zeros(n)
    slot.tape = slot.empty
    return fill!(tape, 0.0)
end
@inline cst_give!(slot, tape, ::Val{true}) = (slot.tape = tape; nothing)
@inline cst_give!(slot, tape, ::Val{false}) = nothing
struct CSTSineSum{Reuse} end
function (::CSTSineSum)(x, data, s, initial, slot)
    total = initial
    for i in eachindex(x)
        total += sin(s * x[i] + data[mod1(i, length(data))])
    end
    return total
end
struct CSTGathered{Initial, F}
    f::F
end
function (g::CSTGathered{Initial})(q, idx, data, s, initial, slot) where {Initial}
    offset = Initial === nothing ? initial : Initial
    return g.f(Base.SubArray(q, (idx,)), data, s, offset, slot)
end
function CST_Rules.augmented_primal(
        config::CST_Rules.RevConfig, f::Const{CSTSineSum{Reuse}},
        ::Type{Active{Float64}}, x::Duplicated, data::Const, s::Const, initial::Const,
        slot::Enzyme.Annotation
    ) where {Reuse}
    n = length(parent(x.val))
    tape = cst_take!(slot.val, n + 1, Val(Reuse))
    buffer = Base.SubArray(view(tape, 1:n), parentindices(x.val))
    value, partial = cst_partials!(x.val, data.val, s.val, initial.val, 1.0, buffer)
    tape[end] = partial
    return CST_Rules.AugmentedReturn(CST_Rules.needs_primal(config) ? value : nothing, nothing, tape)
end
function CST_Rules.reverse(
        config::CST_Rules.RevConfig, f::Const{CSTSineSum{Reuse}}, dy::Active, tape,
        x::Duplicated, data::Const, s::Const, initial::Const, slot::Enzyme.Annotation
    ) where {Reuse}
    for i in eachindex(parent(x.dval))
        parent(x.dval)[i] += dy.val * tape[i]
    end
    cst_give!(slot.val, tape, Val(Reuse))
    return (nothing, nothing, nothing, nothing, nothing)
end
function cst_gradient(q, f, idx, data, scale, initial, slot)
    dq = Enzyme.make_zero(q)
    autodiff(
        Reverse, Const(f), Active, Duplicated(q, dq), Const(idx),
        Const(data), Const(scale), Const(initial), Duplicated(slot, Enzyme.make_zero(slot))
    )
    return dq
end
# The two paths in cst_partials! leave the unused initial-value select in
# the helper's aggregate return. The outer forward pass still builds its shadow
# while differentiating a reverse rule that reuses a mutable buffer.
@testset "Constant scalar tangents in a reused reverse buffer" begin
    for reuse in (false, true), initial in (0.0, 1.25, nothing)
        @testset "reuse=$reuse initial=$initial" begin
            idx = [mod1(i, 3) for i in 1:12]
            data = [sin(i / 3) / 2 for i in 1:12]
            direction = [0.2, -0.7, 1.1]
            slot = CSTBuffer()
            f = CSTGathered{initial, CSTSineSum{reuse}}(CSTSineSum{reuse}())
            retained = nothing
            for point in ([0.1, -0.4, 0.3], [-0.8, 0.6, 1.2], [0.1, -0.4, 0.3])
                expected = zeros(3)
                for (i, j) in enumerate(idx)
                    expected[j] -= 1.2^2 * sin(1.2 * point[j] + data[i]) * direction[j]
                end
                result = only(
                    autodiff(
                        Forward, Const(cst_gradient), Duplicated,
                        Duplicated(point, direction), Const(f), Const(idx), Const(data),
                        Const(1.2), Const(2.5), Duplicated(slot, Enzyme.make_zero(slot))
                    )
                )
                @test result ≈ expected rtol = 3.0e-10 atol = 3.0e-10
                if reuse
                    @test length(slot.tape) == length(point) + 1
                    if retained !== nothing
                        @test slot.tape === retained
                    end
                    retained = slot.tape
                end
            end
            point = [0.4, -0.2, 0.5]
            expected = zeros(3)
            for (i, j) in enumerate(idx)
                expected[j] -= 1.2^2 * sin(1.2 * point[j] + data[i]) * direction[j]
            end
            result = only(
                autodiff(
                    Forward, Const(cst_gradient), BatchDuplicated,
                    BatchDuplicated(point, (direction, 2 .* direction)), Const(f),
                    Const(idx), Const(data), Const(1.2), Const(2.5),
                    BatchDuplicated(slot, (Enzyme.make_zero(slot), Enzyme.make_zero(slot)))
                )
            )
            @test result[1] ≈ expected rtol = 3.0e-10 atol = 3.0e-10
            @test result[2] ≈ 2 .* expected rtol = 3.0e-10 atol = 3.0e-10
        end
    end
end
