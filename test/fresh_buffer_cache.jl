using Enzyme, Test

@noinline function fresh_buffer_distance(x, m)
    total = zero(m)
    for i in eachindex(x)
        total += (x[i] - m)^2
    end
    return total
end

function fresh_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    scratch = zeros(length(x))
    for i in eachindex(x)
        scratch[i] = x[i] + m
    end
    return distance + sum(scratch)
end

function overwritten_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    for i in eachindex(x)
        x[i] += 2.0
    end
    return distance + sum(x)
end

@testset "Fresh buffer writes preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3])
        m = 0.2
        expected = 2sum(m .- x)
        @test autodiff(Reverse, fresh_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected + length(x)
        # A genuine overwrite must retain the old values for the reverse pass.
        @test autodiff(Reverse, overwritten_buffer_loss, Active, Const(copy(x)), Active(m))[1][2] ≈ expected
    end

    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(
        io, fresh_buffer_loss, Active,
        Tuple{Const{Vector{Float64}}, Active{Float64}};
        dump_module = true, second_stage = false, debuginfo = :none
    )
    ir = String(take!(io))
    # The split callee rereads x in reverse, so its augmented forward pass has
    # no tape to return. The later allocation's buffer cannot alias x.
    @test occursin(r"define[^\n]*void @augmented_julia_fresh_buffer_distance_", ir)
end
