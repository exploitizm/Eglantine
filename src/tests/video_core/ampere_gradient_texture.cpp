// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

Instruction Gradient(u32 mask, u32 dest, u32 high, u32 dimension = 1,
                     u32 predicate = 7, u32 binding = 2, bool flag = false) {
    return {{(8U << 24) | (dest << 16) | (predicate << 12) | 0xb6cU,
             16U | (0x123U << 8) | (binding << 22) | (dimension << 29),
             0x001e0000U | (mask << 8) | high | (u32{flag} << 26), 0x000e3000}};
}
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

TEST_CASE("Ampere TXD packs every nonempty channel mask without extra result writes",
          "[video_core][ampere]") {
    for (u32 mask = 1; mask < 16; ++mask) {
        const u32 count = std::popcount(mask);
        for (const u32 dimension : {1U, 5U}) {
            for (const u32 predicate : {2U, 7U, 10U}) {
                for (const bool split : {false, true}) {
                    for (const bool flag : {false, true}) {
                        const u32 dest = count == 1 ? 13 : 12;
                        const u32 high = count <= 2 ? 255 : split ? 32 : dest + 2;
                        const std::array code{Gradient(mask, dest, high, dimension, predicate, 2, flag), Exit};
                        const auto lowered = LowerToMaxwell(code);
                        REQUIRE(static_cast<bool>(lowered));
                        REQUIRE(lowered.required_texture_buffer == 2);
                        REQUIRE(lowered.instructions.size() == (split && count > 2 ? count + 2 : 2));
                        const u64 txd = lowered.instructions[0];
                        REQUIRE((txd >> 56) == 0xde);
                        REQUIRE(((txd >> 8) & 255) == 8);
                        REQUIRE(((txd >> 20) & 255) == 16);
                        REQUIRE(((txd >> 28) & 7) == (dimension == 5 ? 3 : 2));
                        REQUIRE(((txd >> 31) & 15) == mask);
                        REQUIRE(((txd >> 36) & 0x1fff) == 0x123);
                        REQUIRE(((txd >> 49) & 1) == flag);
                        // Model canonical TXD's dense selected-component writes, followed
                        // by the emitted MOVs. Distinct channel tags expose bad packing;
                        // sentinels expose writes beyond a partial destination pair.
                        std::array<int, 256> registers;
                        registers.fill(-1);
                        u32 out = txd & 255;
                        for (u32 component = 0; component < 4; ++component)
                            if (mask & (1U << component)) registers[out++] = 100 + component;
                        for (size_t i = 0; i + 1 < lowered.instructions.size(); ++i) {
                            const u64 insn = lowered.instructions[i];
                            REQUIRE(((insn >> 16) & 15) == predicate);
                            if (i == 0) continue;
                            REQUIRE((insn >> 48) == 0x5c98);
                            registers[insn & 255] = registers[(insn >> 20) & 255];
                        }
                        std::array<int, 248> expected;
                        expected.fill(-1);
                        u32 packed = 0;
                        for (u32 component = 0; component < 4; ++component) {
                            if (!(mask & (1U << component))) continue;
                            const u32 target = packed < 2 ? dest + packed : high + packed - 2;
                            expected[target] = 100 + component;
                            ++packed;
                        }
                        for (u32 reg = 0; reg < expected.size(); ++reg)
                            REQUIRE(registers[reg] == expected[reg]);
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere TXD retains descriptor binding and offset boundaries", "[video_core][ampere]") {
    for (const u32 binding : {0U, 17U}) {
        auto sample = Gradient(5, 12, 255, 1, 7, binding);
        sample.words[1] = (sample.words[1] & ~(0x3fffU << 8)) | (0x1fffU << 8);
        const std::array code{sample, Exit};
        const auto lowered = LowerToMaxwell(code);
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.required_texture_buffer == binding);
        REQUIRE(((lowered.instructions[0] >> 36) & 0x1fff) == 0x1fff);
    }
    const std::array conflict{Gradient(3, 12, 255), Gradient(15, 12, 14, 1, 7, 3), Exit};
    const auto lowered = LowerToMaxwell(conflict);
    REQUIRE_FALSE(static_cast<bool>(lowered));
    REQUIRE(lowered.error == "native texture descriptor buffer conflict");
}

TEST_CASE("Ampere TXD rejects malformed masks, destinations and modes", "[video_core][ampere]") {
    for (const auto sample : {
             Gradient(0, 12, 255), Gradient(1, 13, 32), Gradient(3, 12, 32),
             Gradient(3, 13, 255), Gradient(15, 13, 32), Gradient(15, 12, 246),
             Gradient(1, 246, 255), Gradient(1, 13, 255, 0), Gradient(1, 13, 255, 2),
             Gradient(1, 13, 255, 1, 7, 18)}) {
        const std::array code{sample, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    for (const u32 bit : {12U, 16U, 21U, 27U}) {
        auto sample = Gradient(5, 12, 255);
        sample.words[2] ^= 1U << bit;
        const std::array code{sample, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    for (const u32 field : {24U, 32U}) {
        auto sample = Gradient(5, 12, 255);
        sample.words[field / 32] = (sample.words[field / 32] & ~(255U << (field % 32))) |
                                  ((field == 24 ? 246U : 247U) << (field % 32));
        const std::array code{sample, Exit};
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
    }
    auto sample = Gradient(5, 12, 255);
    sample.words[1] |= 1U << 21; // Descriptor offset 0x2000 is outside Maxwell's field.
    const std::array code{sample, Exit};
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(code)));
}
} // namespace
