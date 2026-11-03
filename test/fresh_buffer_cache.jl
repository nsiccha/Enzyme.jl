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

fresh_buffer_ir(f, argtypes) = sprint() do io
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, argtypes;
        dump_module = true, second_stage = false, debuginfo = :none
    )
end

# On Julia 1.10 a split callee rereads x in reverse when nothing after the call
# can overwrite x, so its augmented forward pass has no tape to return.
const tape_free_distance = r"define[^\n]*void @augmented_julia_fresh_buffer_distance_"

# `vec` of a fresh matrix after the call: Julia 1.10 fills a dims box and calls
# jl_reshape_array, which writes only the header it creates.
function reshaped_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    scratch = zeros(length(x), 1)
    for i in eachindex(x)
        scratch[i, 1] = x[i] + m
    end
    return distance + sum(vec(scratch))
end

# The same with x carried by a structured Const operand, which reaches the
# compiled function as a field of an aggregate argument.
function structured_buffer_loss(data, m)
    distance = fresh_buffer_distance(data.x, m)
    scratch = zeros(length(data.x), 1)
    for i in eachindex(data.x)
        scratch[i, 1] = data.x[i] + m
    end
    return distance + sum(vec(scratch))
end

# A broadcast after the call that allocates its output on each of its branches
# and merges the headers before writing through them.
function broadcast_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    a = zeros(length(x), 1)
    b = ones(length(x), 1)
    for i in eachindex(x)
        a[i, 1] = x[i] + m
    end
    return distance + sum(vec(a) .- vec(b))
end

# A genuine overwrite of the structured operand after the call.
function structured_overwrite_loss(data, m)
    distance = fresh_buffer_distance(data.x, m)
    for i in eachindex(data.x)
        data.x[i] += 2.0
    end
    return distance + sum(vec(reshape(data.x, :, 1)))
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
    # On Julia 1.10 the split callee rereads x in reverse, so its augmented
    # forward pass has no tape to return. Later Julia versions merge fresh
    # Memory with a shared empty singleton before loading its data pointer;
    # that control-dependent case still conservatively retains the tape.
    if VERSION < v"1.11-"
        @test occursin(r"define[^\n]*void @augmented_julia_fresh_buffer_distance_", ir)
    else
        @test_broken occursin(r"define[^\n]*void @augmented_julia_fresh_buffer_distance_", ir)
    end
end

@testset "Fresh boxes and reshapes after the call preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3])
        m = 0.2
        expected = 2sum(m .- x) + length(x)
        @test autodiff(Reverse, reshaped_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected
        data = (; x)
        @test autodiff(Reverse, structured_buffer_loss, Active, Const(data), Active(m))[1][2] ≈ expected
        @test autodiff(Reverse, broadcast_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected
        # A genuine overwrite must retain the old values for the reverse pass.
        overwritten = (; x = copy(x))
        @test autodiff(Reverse, structured_overwrite_loss, Active, Const(overwritten), Active(m))[1][2] ≈
            expected - length(x)
        @test overwritten.x == x .+ 2.0
    end

    reshaped = fresh_buffer_ir(reshaped_buffer_loss, Tuple{Const{Vector{Float64}}, Active{Float64}})
    merged = fresh_buffer_ir(broadcast_buffer_loss, Tuple{Const{Vector{Float64}}, Active{Float64}})
    structured = fresh_buffer_ir(
        structured_buffer_loss,
        Tuple{Const{@NamedTuple{x::Vector{Float64}}}, Active{Float64}}
    )
    overwrite = fresh_buffer_ir(
        structured_overwrite_loss,
        Tuple{Const{@NamedTuple{x::Vector{Float64}}}, Active{Float64}}
    )
    @test !occursin(tape_free_distance, overwrite)
    if VERSION < v"1.11-"
        @test occursin(tape_free_distance, reshaped)
        @test occursin(tape_free_distance, structured)
        @test occursin(tape_free_distance, merged)
    else
        # Later Julia versions merge fresh Memory with a shared empty
        # singleton before loading its data pointer (testset above).
        @test_broken occursin(tape_free_distance, reshaped)
        @test_broken occursin(tape_free_distance, structured)
        @test_broken occursin(tape_free_distance, merged)
    end
end
