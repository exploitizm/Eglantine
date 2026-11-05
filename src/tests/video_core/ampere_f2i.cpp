// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;

TEST_CASE("Ampere F2I round up preserves conversion mode", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};

    for (const bool is_signed : {false, true}) {
        const Instruction convert{{0x00067305, 0x00000009,
                                   is_signed ? 0x0021b100U : 0x0020b000U, 0}};
        const auto lowered = LowerToMaxwell(std::array{convert, exit});
        REQUIRE(static_cast<bool>(lowered));
        const auto instruction = std::ranges::find_if(lowered.instructions, [](u64 word) {
            return (word & 0xfff0000000000000ULL) == 0x5cb0000000000000ULL;
        });
        REQUIRE(instruction != lowered.instructions.end());
        REQUIRE((*instruction & 0xff) == 6);
        REQUIRE(((*instruction >> 8) & 3) == 2);
        REQUIRE(((*instruction >> 10) & 3) == 2);
        REQUIRE(((*instruction >> 12) & 1) == static_cast<u64>(is_signed));
        REQUIRE(((*instruction >> 39) & 3) == 2);
        REQUIRE(((*instruction >> 44) & 1) == static_cast<u64>(is_signed));
    }

    const Instruction unsupported{{0x00067305, 0x00000009, 0x0020b001, 0}};
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(std::array{unsupported, exit})));
}

TEST_CASE("Ampere F2I signed byte retains signed truncation and FTZ", "[video_core][ampere]") {
    using Shader::Ampere::Instruction;
    using Shader::Ampere::LowerToMaxwell;
    const Instruction convert{{0x00097305, 0x00000005, 0x0021e100, 0}};
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const auto lowered = LowerToMaxwell(std::array{convert, exit});
    REQUIRE(static_cast<bool>(lowered));
    const auto conversion = std::ranges::find_if(lowered.instructions, [](u64 word) {
        return (word >> 48) == 0x5cb0;
    });
    REQUIRE(conversion != lowered.instructions.end());
    const u64 f2i = *conversion;
    REQUIRE((f2i & 0xff) == 9);
    REQUIRE(((f2i >> 8) & 3) == 2);
    REQUIRE(((f2i >> 10) & 3) == 2);
    REQUIRE(((f2i >> 12) & 1) == 1);
    REQUIRE(((f2i >> 39) & 3) == 3);
    REQUIRE(((f2i >> 44) & 1) == 1);

}

TEST_CASE("Ampere U16 F2I applies constant float modifiers before conversion",
          "[video_core][ampere]") {
    // Captured from Pokopia's rejected fragment shader; nvdisasm SM87:
    // F2I.FTZ.U16.NTZ R6, |c[0x3][0xd0]|.
    for (const u32 predicate : {2U, 7U, 10U}) {
        const std::array<Instruction, 2> plain{{
            {{0x00060b05U | (predicate << 12), 0x00c03400, 0x00212800, 0x000e2200}},
            {{0x0000794d, 0, 0x03800000, 0}},
        }};
        const auto baseline = LowerToMaxwell(plain);
        REQUIRE(static_cast<bool>(baseline));
        const u64 guard = u64{predicate} << 16;
        REQUIRE(baseline.instructions.front() ==
                (0x4c98000000000000ULL | guard | 252 | (15ULL << 39) |
                 (0x34ULL << 20) | (3ULL << 34)));
        for (const u32 modifier : {0U, 0x40000000U, 0x80000000U, 0xc0000000U}) {
            auto code = plain;
            code[0].words[1] |= modifier;
            const auto lowered = LowerToMaxwell(code);
            REQUIRE(static_cast<bool>(lowered));
            REQUIRE(lowered.instructions.front() == baseline.instructions.front());
            size_t prefix = 1;
            if (modifier & 0x40000000U) {
                REQUIRE(lowered.instructions[prefix++] ==
                        (0x0400000000000000ULL | guard | 252 | (252ULL << 8) |
                         (0x7fffffffULL << 20)));
            }
            if (modifier & 0x80000000U) {
                REQUIRE(lowered.instructions[prefix++] ==
                        (0x0400000000000000ULL | guard | 252 | (252ULL << 8) |
                         (0x80000000ULL << 20) | (2ULL << 53)));
            }
            // Existing U16 FTZ/NTZ conversion, saturation, and execution guard
            // must remain identical after the source modifier instructions.
            REQUIRE(lowered.instructions.size() == baseline.instructions.size() + prefix - 1);
            REQUIRE(std::equal(baseline.instructions.begin() + 1, baseline.instructions.end(),
                               lowered.instructions.begin() + prefix));
            auto invalid = code;
            invalid[0].words[1] |= 1U << 6; // Unaligned constant-buffer address.
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
            invalid = code;
            invalid[0].words[2] |= 0x08000000U; // Uniform mode on constant source.
            REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
        }
    }
}
