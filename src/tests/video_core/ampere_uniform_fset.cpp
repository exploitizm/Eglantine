// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

constexpr std::array<Instruction, 2> UniformComparison{{
    {{0x03047c0a, 5, 0x0b807000, 0x000e3000}},
    {{0x0000794d, 0, 0x03800000, 0}},
}};

TEST_CASE("Ampere uniform FSET preserves comparison, predicate and boolean modes",
          "[video_core][ampere]") {
    for (const u32 source : {0U, 21U, 22U, 27U, 63U}) {
        for (const u32 guard : {2U, 7U, 10U}) {
            for (u32 compare = 0; compare < 16; ++compare) {
                for (u32 boolean = 0; boolean < 3; ++boolean) {
                    for (const u32 pc : {0U, 7U, 15U}) {
                        auto code = UniformComparison;
                        code[0].words[0] = 0x03040c0aU | (guard << 12);
                        code[0].words[1] = source;
                        code[0].words[2] = 0x08000000U | (pc << 23) |
                                          (boolean << 10) | (compare << 12) | (1U << 16);
                        const auto lowered = LowerToMaxwell(code);
                        REQUIRE(static_cast<bool>(lowered));
                        REQUIRE(lowered.instructions.size() == 2);
                        const u32 mapped = source < 22 ? source + 216 : source < 28 ? source + 218 : 255;
                        REQUIRE(lowered.instructions[0] ==
                                (0x5800000000000000ULL | (u64{guard} << 16) | 4 |
                                 (3ULL << 8) | (u64{mapped} << 20) | (u64{pc} << 39) |
                                 (u64{boolean} << 45) | (u64{compare} << 48) |
                                 (1ULL << 52) | (1ULL << 55)));
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere uniform FSET shares register modifier lowering", "[video_core][ampere]") {
    for (u32 modifiers = 0; modifiers < 16; ++modifiers) {
        for (const bool ftz : {false, true}) {
            auto uniform = UniformComparison;
            uniform[0].words[1] |= ((modifiers >> 2) & 3U) << 30;
            uniform[0].words[2] |= (modifiers & 3U) << 8;
            uniform[0].words[2] |= u32{ftz} << 16;
            const auto lowered = LowerToMaxwell(uniform);
            REQUIRE(static_cast<bool>(lowered));
            // UR5 maps to R221. The ordinary register form must lower identically,
            // including absolute/negate materialization and the FTZ setting.
            auto reg = uniform;
            reg[0].words[0] = 0x0304720a;
            reg[0].words[1] = (reg[0].words[1] & ~255U) | 221U;
            reg[0].words[2] &= ~(1U << 27);
            const auto expected = LowerToMaxwell(reg);
            REQUIRE(static_cast<bool>(expected));
            REQUIRE(lowered.instructions == expected.instructions);
        }
    }
}

TEST_CASE("Ampere uniform FSET rejects unknown modes and register aliases",
          "[video_core][ampere]") {
    for (const u32 source : {64U, 255U}) { // UR28-UR63 lower through scratch registers
        auto invalid = UniformComparison;
        invalid[0].words[1] = source;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    for (const u32 bit : {0U, 7U, 17U, 22U}) {
        auto invalid = UniformComparison;
        invalid[0].words[2] |= 1U << bit;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto invalid = UniformComparison;
    invalid[0].words[2] |= 3U << 10;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    for (const u32 field : {16U, 24U}) {
        invalid = UniformComparison;
        invalid[0].words[0] = (invalid[0].words[0] & ~(255U << field)) | (216U << field);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    // Detect uniform use before lowering earlier instructions, so they cannot
    // overwrite the emulated uniform bank through ordinary high GPR accesses.
    const std::array<Instruction, 3> aliases{{
        {{0x00047312, 216, 0x00301000, 0x000e3000}},
        UniformComparison[0], UniformComparison[1],
    }};
    const auto lowered = LowerToMaxwell(aliases);
    REQUIRE_FALSE(static_cast<bool>(lowered));
    REQUIRE(lowered.error_offset == 0);
    REQUIRE(lowered.error == "native register shadowing mode");
}
} // namespace
