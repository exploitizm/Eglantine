// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

constexpr std::array<Instruction, 2> Conversion{{
    {{0x00047312, 2, 0x00301000, 0x000e3000}},
    {{0x0000794d, 0, 0x03800000, 0}},
}};

TEST_CASE("Ampere I2F64 preserves signedness, rounding and predicates", "[video_core][ampere]") {
    for (const bool is_signed : {false, true}) {
        for (u32 rounding = 0; rounding < 4; ++rounding) {
            for (const u32 predicate : {2U, 7U, 10U}) {
                for (const u32 source : {0U, 2U, 244U}) {
                    auto code = Conversion;
                    code[0].words[0] = 0x00040312U | (predicate << 12);
                    code[0].words[1] = source;
                    code[0].words[2] |= (u32{is_signed} << 10) | (rounding << 14);
                    const auto lowered = LowerToMaxwell(code);
                    REQUIRE(static_cast<bool>(lowered));
                    REQUIRE(lowered.instructions.size() == 2);
                    // Canonical Maxwell I2F: F32 destination, 64-bit source pair,
                    // signedness at bit 13 and rounding at bits 39-40.
                    REQUIRE(lowered.instructions[0] ==
                            (0x5cb8000000000000ULL | (u64{predicate} << 16) | 4 |
                             (2ULL << 8) | (3ULL << 10) | (u64{is_signed} << 13) |
                             (u64{source} << 20) | (u64{rounding} << 39)));
                }
            }
        }
    }
}

TEST_CASE("Ampere I2F64 retains pair and modifier rejection", "[video_core][ampere]") {
    for (const bool is_signed : {false, true}) {
        auto code = Conversion;
        code[0].words[2] |= u32{is_signed} << 10;
        for (const u32 source : {1U, 3U, 245U, 246U, 254U, 255U, 0x10002U}) {
            auto invalid = code;
            invalid[0].words[1] = source;
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
        }
        for (const u32 bit : {8U, 9U, 11U, 12U, 16U, 22U, 27U}) {
            auto invalid = code;
            invalid[0].words[2] ^= 1U << bit;
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
        }
        auto invalid = code;
        invalid[0].words[0] = 0x00f67312;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
}
} // namespace
