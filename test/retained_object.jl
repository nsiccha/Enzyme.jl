using Enzyme, Test
using Enzyme: LLVM

function retained_section_object(asm::String)
    LLVM.@dispose ctx = LLVM.Context() mod = LLVM.Module("retained_sections") begin
        LLVM.triple!(mod, LLVM.triple(Enzyme.Compiler.JIT.get_jit()))
        LLVM.inline_asm!(mod, asm)
        return Enzyme.Compiler.emit_retained_object(mod)
    end
end

@testset "Retained objects distinguish loaded bytes from ELF metadata" begin
    if Enzyme.Compiler.retention_supported()
        data = zeros(UInt8, 16)
        GC.@preserve data begin
            address = UInt(pointer(data))
            # A real mapped address makes these independent of address-space layout.
            metadata = retained_section_object(
                """
                .section .retained_metadata,"",@progbits
                .quad $address
                .text
                ret
                """
            )
            @test Enzyme.Compiler.address_in_object(metadata) === nothing

            # Section names are arbitrary: the flags, not a name whitelist, decide
            # whether bytes can become part of the linked code or data.
            loaded = retained_section_object(
                """
                .section .retained_data,"a",@progbits
                .byte 0x55
                .quad $address
                """
            )
            @test Enzyme.Compiler.address_in_object(loaded) == address

            code = retained_section_object(
                """
                .text
                movabsq \$$address, %rax
                ret
                """
            )
            @test Enzyme.Compiler.address_in_object(code) == address

            # NOBITS reserves storage but has no bytes at its file offset. Here that
            # offset belongs to metadata containing the address.
            bss = retained_section_object(
                """
                .section .retained_bss,"aw",@nobits
                .zero 16
                .section .retained_metadata,"",@progbits
                .quad $address
                """
            )
            @test Enzyme.Compiler.address_in_object(bss) === nothing

            # Two adjacent sections are separate loader allocations. A word must
            # fit entirely inside one section, even if their file bytes abut.
            low = join([Int((address >> (8 * i)) & 0xff) for i in 0:3], ",")
            high = join([Int((address >> (8 * i)) & 0xff) for i in 4:7], ",")
            boundary = retained_section_object(
                """
                .text
                .byte $low
                .section .retained_tail,"a",@progbits
                .byte $high
                """
            )
            @test Enzyme.Compiler.address_in_object(boundary) === nothing
        end
    end
end
