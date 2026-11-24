using Enzyme, LinearAlgebra, Test

data = (; y = zeros(2), A = zeros(2, 2))
field_loss(q) = -(dot(q, q) + sum(abs2, data.y - data.A * q)) / 2
typed_field_loss(q) = field_loss(q)::Float64
field_sum(x) = x * sum(data.y)
field_gradient(f, x) = only(Enzyme.gradient(Reverse, f, x))

# The first getter changes which record the second getter reads.
struct RebindingFields
    y::Vector{Float64}
    next::@NamedTuple{y::Vector{Float64}, A::Matrix{Float64}}
end
function Base.getproperty(r::RebindingFields, name::Symbol)
    if name === :y
        global data = getfield(r, :next)
    end
    return getfield(r, name)
end

# Even an immutable NamedTuple can have a more-specific Julia getter.
struct FieldTag end
const TaggedFields = @NamedTuple{tag::FieldTag, y::Vector{Float64}, alternate::Vector{Float64}}
function Base.getproperty(r::TaggedFields, name::Symbol)
    return getfield(r, name === :y ? :alternate : name)
end

@noinline write_first!(a, x) = (a[1] = x; nothing)
@noinline function store_and_read(a, x)
    Base.inferencebarrier(write_first!)(a, x)
    return Base.inferencebarrier(first)(a)^2
end
store_loss(x) = store_and_read(data.y, x)

# One getter precedes the loop. Its sole use in the IR runs twice and needs
# the same shadow both times.
@noinline times_first!(a, x) = (a[1] *= x)
function loop_field_loss(x)
    a = data.y
    result = zero(x)
    for i in 1:2
        result = Base.inferencebarrier(times_first!)(a, x)
    end
    return result
end

# An active receiver returned by an opaque call; both fields point to q.
@noinline aliased_fields(q) = (; y = q, A = q)
function active_fields_loss(q)
    r = Base.inferencebarrier(aliased_fields)(q)
    return dot(r.y, q) + sum(r.A)
end

@testset "Field identity, mutation and binding order" begin
    q = [1.0, -0.5]
    matrix = [2.0 3.0; 1.0 -2.0]
    for (y, A) in (([0.7, -0.1], matrix), (2q, 2matrix), (q, matrix))
        global data = (; y, A)
        ys, As = copy(y), copy(A)
        @test field_gradient(field_loss, q) ≈ A' * (y - A * q) - q
        @test data.y === y && data.A === A
        @test y == ys && A == As
    end

    storage = [2.0, -1.0]
    global data = (; y = storage, A = reshape(storage, 2, 1))
    x = [0.25]
    @test field_gradient(field_loss, x) ≈ data.A' * (storage - data.A * x) - x
    @test Base.mightalias(data.y, data.A) && storage == [2.0, -1.0]

    next = (; y = fill(10.0, 2), A = matrix)
    old_y = [-2.0, 3.0]
    global data = RebindingFields(old_y, next)
    @test field_gradient(field_loss, q) ≈ matrix' * (old_y - matrix * q) - q
    @test data === next

    @test field_gradient(store_loss, 2.0) == 4.0
    @test next.y == [2.0, 10.0]
    @test field_gradient(loop_field_loss, 3.0) == 12.0
    @test next.y == [18.0, 10.0]
    @test field_gradient(active_fields_loss, q) ≈ 2q .+ 1

    global data = (; tag = FieldTag(), y = [1.0], alternate = [3.0, 4.0])
    @test field_gradient(field_sum, 2.0) == 7.0
end

@testset "Captured fields between split reverse passes" begin
    q = [0.25, -0.5]
    y = [2.0, 1.0]
    A = [2.0 1.0; 3.0 -1.0]
    global data = (; y, A)
    args = Duplicated(q, zero(q))
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(typed_field_loss)}, Active, typeof(args)
    )
    tape, value, _ = fwd(Const(typed_field_loss), args)
    @test value ≈ -(dot(q, q) + sum(abs2, y - A * q)) / 2
    global data = (; y = zero(y), A = zero(A))
    rev(Const(typed_field_loss), args, 1.0, tape)
    @test args.dval ≈ A' * (y - A * q) - q
end

function field_loss_allocation(n)
    global data = (; y = fill(2.0, n), A = Matrix{Float64}(I, n, n))
    q = fill(0.5, n)
    @test field_gradient(field_loss, q) == ones(n)
    return @allocated field_gradient(field_loss, q)
end

function field_sum_allocation(n)
    global data = (; y = fill(2.0, n), A = zeros(1, 1))
    @test field_gradient(field_sum, 3.0) == 2n
    return @allocated field_gradient(field_sum, 3.0)
end

@testset "Read-only dynamic field projections" begin
    small = field_loss_allocation(64)
    large = field_loss_allocation(512)
    # A shadow matrix alone would add >2 MB. The residuals and gradient
    # are vectors, so their storage grows linearly.
    @test large - small < 128 * 1024

    small = field_sum_allocation(128)
    large = field_sum_allocation(8192)
    # A scalar gradient needs no vector. A shadow of the field would add 63 KiB.
    @test large - small < 16 * 1024
end
