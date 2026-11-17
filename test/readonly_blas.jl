using Enzyme, LinearAlgebra, Test

@testset "Readonly captures passed to BLAS" begin
    for T in (Float32, Float64)
        A = T[1 2; 3 4; -2 0.5]
        b = T[0.7, -1.2]
        saved_A, saved_b = copy(A), copy(b)

        # A raw pointer to the captured matrix is a read-only gemv operand.
        # The leading unused coordinate also exercises a nonzero view offset.
        u = T[0.4; b]
        f = x -> sum(abs2, A * view(x, 2:length(x)))
        @test only(gradient(Reverse, f, u)) ≈ T[0; 2 .* (A' * (A * b))]
        @test A == saved_A
        @test u == T[0.4; b]

        # Both the input vector and a matrix transpose can be captured.
        v = T[0.2, -0.5, 0.8]
        ft = x -> sum(abs2, A' * x)
        @test only(gradient(Reverse, ft, v)) ≈ 2 .* (A * (A' * v))
        fv = x -> sum(abs2, x * b)
        @test only(gradient(Reverse, fv, A)) ≈ 2 .* ((A * b) * b')
        @test b == saved_b

        # gemm's captured input is also read-only, though the call writes C.
        B = T[0.2 -0.3; 0.7 0.4]
        fm = x -> sum(abs2, A * x)
        @test only(gradient(Reverse, fm, B)) ≈ 2 .* (A' * (A * B))
        @test A == saved_A

        # Reading the source of axpy is valid when its destination is fresh.
        fa = x -> begin
            local output = zeros(T, length(b))
            BLAS.axpy!(x, b, output)
            sum(abs2, output)
        end
        @test only(gradient(Reverse, fa, T(0.5))) ≈ sum(abs2, b)
        @test b == saved_b

        # The same BLAS calls must still reject captured output storage.
        y = T[0.1, 0.2, 0.3]
        saved_y = copy(y)
        fw = x -> sum(mul!(y, A, x))
        @test_throws Enzyme.Compiler.EnzymeMutabilityException gradient(Reverse, fw, b)
        @test y == saved_y

        # One value in both a reading and a writing position is still a write.
        falias = x -> sum(BLAS.axpy!(x, b, b))
        @test_throws Enzyme.Compiler.EnzymeMutabilityException gradient(Reverse, falias, T(0.5))
        @test b == saved_b

        fs = x -> sum(BLAS.scal!(length(b), x, b, 1))
        @test_throws Enzyme.Compiler.EnzymeMutabilityException gradient(Reverse, fs, T(0.5))
        @test b == saved_b
    end
end
