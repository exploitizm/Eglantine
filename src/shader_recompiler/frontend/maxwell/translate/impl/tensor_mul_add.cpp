// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"

namespace Shader::Maxwell {
namespace {
// HMMA.16816/1688.F16 written out in IR (after heydemoura/EglantinePS5 35b485dc2a, "native HMMA
// lowering"): D(16x8) = A(16xK) * B(Kx8) + C with f16 fragments, K = 8 or 16. Lane t = 4g + q holds
// rows g and g + 8, columns 2q and 2q + 1 of C and D; its A registers hold rows g/g + 8, k = 2q..2q+1
// (+8 for the second block of 8); its B registers hold k = 2q..2q+1 (+8) of column g. Lane t needs
// A rows g and g + 8, held by lanes 4g..4g+3, and B columns 2q and 2q + 1, held by lanes 8q..8q+3
// and 8q+4..8q+7: 32 shuffles for K = 16, as the TensorMulAdd helper does, with the same products,
// order and f16 rounding after each block of 8. The helper is a function with a loop that RADV
// inlines at every call: Ocarina of Time's network passes call it 464-768 times and took minutes to
// compile on a first boot. Straight-line, the same passes compiled in seconds on the PS5 but ran
// at half the speed (the movie's frames took 73 ms, 37 ms with the helper): about three times the
// code of the helper's loops. So EDEN_NX2_HMMA_NATIVE=1 turns it on; the helper is the default.
IR::F32 RoundToHalf(IR::IREmitter& ir, const IR::F32& value) {
    // As the helper: round the f32 significand to f16 precision, ties to even, explicitly (an
    // f16 round trip may be removed by drivers); half subnormals lie on the 2^-24 grid.
    const auto imm = [&](u32 v) { return ir.Imm32(v); };
    const IR::U32 bits{ir.BitCast<IR::U32, IR::F32>(value)};
    const IR::U32 magnitude{ir.BitwiseAnd(bits, imm(0x7fffffffU))};
    const IR::U32 tie{ir.BitwiseAnd(ir.ShiftRightLogical(magnitude, imm(13)), imm(1))};
    const IR::U32 rounded{
        ir.BitwiseAnd(ir.IAdd(ir.IAdd(magnitude, imm(0xfff)), tie), imm(~0x1fffU))};
    const IR::U32 normal{IR::U32{ir.Select(ir.IGreaterThanEqual(rounded, imm(0x47800000), false),
                                           imm(0x7f800000), rounded)}};
    const IR::FpControl exact{.no_contraction = true};
    const IR::F32 scaled{
        ir.FPMul(ir.BitCast<IR::F32, IR::U32>(magnitude), ir.Imm32(16777216.0f), exact)};
    const IR::U32 subnormal{ir.BitCast<IR::U32, IR::F32>(
        IR::F32{ir.FPMul(IR::F32{ir.FPRoundEven(scaled)}, ir.Imm32(0x1p-24f), exact)})};
    const IR::U32 finite{IR::U32{ir.Select(ir.IGreaterThanEqual(magnitude, imm(0x38800000), false),
                                           normal, subnormal)}};
    const IR::U32 result{IR::U32{ir.Select(ir.IGreaterThanEqual(magnitude, imm(0x7f800000), false),
                                           magnitude, finite)}};
    return ir.BitCast<IR::F32, IR::U32>(
        ir.BitwiseOr(result, ir.BitwiseAnd(bits, imm(0x80000000U))));
}

void NativeHmma(TranslatorVisitor& v, IR::IREmitter& ir, u32 a, u32 b, u32 c, u32 dest,
                bool short_depth) {
    const IR::U32 lane{ir.LaneId()};
    const IR::U32 row_lanes{ir.BitwiseAnd(lane, ir.Imm32(0x1c))};
    const IR::U32 column_lanes{ir.ShiftLeftLogical(ir.BitwiseAnd(lane, ir.Imm32(3)), ir.Imm32(3))};
    const IR::U32 clamp{ir.Imm32(31)};
    const IR::U32 segment{ir.Imm32(0)};
    const auto unpack = [&](const IR::U32& packed, u32 half) {
        return IR::F32{ir.CompositeExtract(ir.UnpackHalf2x16(packed), half)};
    };
    const IR::FpControl exact{.no_contraction = true};
    IR::F32 acc[2][2];
    for (u32 row = 0; row < 2; ++row) {
        for (u32 col = 0; col < 2; ++col) {
            acc[row][col] = c == 255 ? ir.Imm32(0.0f) : unpack(v.X(IR::Reg{c + row}), col);
        }
    }
    const u32 blocks = short_depth ? 1U : 2U;
    for (u32 block = 0; block < blocks; ++block) {
        const IR::U32 a_rows[2]{v.X(IR::Reg{a + block * 2}), v.X(IR::Reg{a + block * 2 + 1})};
        const IR::U32 b_column{v.X(IR::Reg{b + block})};
        for (u32 k = 0; k < 4; ++k) {
            const IR::U32 a_lane{ir.BitwiseOr(row_lanes, ir.Imm32(k))};
            IR::U32 a_values[2];
            for (u32 row = 0; row < 2; ++row) {
                a_values[row] = ir.ShuffleIndex(a_rows[row], a_lane, clamp, segment);
            }
            IR::U32 b_values[2];
            for (u32 col = 0; col < 2; ++col) {
                const IR::U32 b_lane{IR::U32{ir.IAdd(column_lanes, ir.Imm32(col * 4 + k))}};
                b_values[col] = ir.ShuffleIndex(b_column, b_lane, clamp, segment);
            }
            for (u32 row = 0; row < 2; ++row) {
                for (u32 col = 0; col < 2; ++col) {
                    for (u32 half = 0; half < 2; ++half) {
                        acc[row][col] = IR::F32{ir.FPFma(unpack(a_values[row], half),
                                                         unpack(b_values[col], half),
                                                         acc[row][col], exact)};
                    }
                }
            }
        }
        for (u32 row = 0; row < 2; ++row) {
            for (u32 col = 0; col < 2; ++col) {
                acc[row][col] = RoundToHalf(ir, acc[row][col]);
            }
        }
    }
    for (u32 row = 0; row < 2; ++row) {
        v.X(IR::Reg{dest + row}, ir.PackHalf2x16(ir.CompositeConstruct(acc[row][0], acc[row][1])));
    }
}
} // Anonymous namespace

void TranslatorVisitor::TENSOR_MUL_ADD(u64 insn) {
    union {
        u64 raw;
        BitField<0, 8, IR::Reg> dest_reg;
        BitField<8, 8, IR::Reg> matrix_a;
        BitField<20, 8, IR::Reg> matrix_b;
        BitField<39, 8, IR::Reg> matrix_c;
    } const tensor{insn};

    const IR::Reg ar = tensor.matrix_a;
    const IR::Reg br = tensor.matrix_b;
    const IR::Reg cr = tensor.matrix_c;
    const IR::Reg dr = tensor.dest_reg;

    const u64 ai = static_cast<u64>(ar);
    const u64 bi = static_cast<u64>(br);
    const u64 ci = static_cast<u64>(cr);
    const u64 di = static_cast<u64>(dr);
    const bool has_c = ci != 255;
    static const bool disabled{std::getenv("EDEN_NX2_NO_TENSOR") != nullptr};
    if (disabled || ((insn >> 48) & 1) == 0) {
        throw NotImplementedException("Unsupported matrix multiply encoding");
    }
    const bool short_depth = (insn >> 49) & 1;
    static const bool native_hmma = [] {
        const char* value = std::getenv("EDEN_NX2_HMMA_NATIVE");
        return value != nullptr && value[0] == '1';
    }();
    if (native_hmma) {
        NativeHmma(*this, ir, static_cast<u32>(ai), static_cast<u32>(bi), static_cast<u32>(ci),
                   static_cast<u32>(di), short_depth);
        return;
    }

    const IR::U32 a0 = X(ar);
    const IR::U32 a1 = X(IR::Reg{ai + 1});
    const IR::U32 a2 = short_depth ? ir.Imm32(0u) : X(IR::Reg{ai + 2});
    const IR::U32 a3 = short_depth ? ir.Imm32(0u) : X(IR::Reg{ai + 3});
    const IR::U32 b0 = X(br);
    const IR::U32 b1 = short_depth ? ir.Imm32(0u) : X(IR::Reg{bi + 1});
    const IR::U32 c0 = has_c ? X(cr) : ir.Imm32(0u);
    const IR::U32 c1 = has_c ? X(IR::Reg{ci + 1}) : ir.Imm32(0u);

    const IR::Value result{ir.TensorMulAdd(
                                  ir.CompositeConstruct(a0, a1, a2, a3),
                                  ir.CompositeConstruct(b0, b1), ir.CompositeConstruct(c0, c1),
                                  short_depth ? 8U : 16U)};
    X(dr, IR::U32{ir.CompositeExtract(result, 0)});
    X(IR::Reg{di + 1}, IR::U32{ir.CompositeExtract(result, 1)});
}

} // namespace Shader::Maxwell
