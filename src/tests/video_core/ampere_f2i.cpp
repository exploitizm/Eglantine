// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

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

TEST_CASE("Ampere register bindless textures preserve scalar and vector operand packing",
          "[video_core][ampere]") {
    const std::array<Instruction, 4> samples{{
        {{0x16167361, 0x78000018, 0x041e0718, 0x108f4400}},
        {{0x00007361, 0x38000027, 0x041e0726, 0x002f4400}},
        {{0x0c0d7361, 0xa800001e, 0x041e41ff, 0x001f4400}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    const std::array<std::array<u32, 5>, 3> sources{{
        {22, 23, 24, 25, 255}, {0, 1, 39, 255, 255}, {12, 13, 14, 30, 31},
    }};
    for (u32 i = 0; i < 3; ++i) {
        const std::array<Instruction, 2> code{samples[i], samples[3]};
        const auto lowered = LowerToMaxwell(code);
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.required_texture_buffer == ~0U);
        const u32 coordinates = i == 1 ? 2 : 3;
        const u32 inputs = coordinates + 1 + (i == 2 ? 1 : 0);
        for (u32 j = 0; j < inputs; ++j) {
            const u32 target = j < coordinates ? 250 + j : 246 + j - coordinates;
            REQUIRE(lowered.instructions[j] ==
                    (0x5c98000000070000ULL | target | (15ULL << 39) | (u64{sources[i][j]} << 20)));
        }
        const u64 texture = lowered.instructions[inputs];
        REQUIRE(((texture >> 20) & 255) == 246);
        REQUIRE(((texture >> 28) & 7) == (i == 0 ? 6U : i == 1 ? 2U : 3U));
        REQUIRE(((texture >> 50) & 1) == (i == 2 ? 1U : 0U));
        REQUIRE((texture >> 56) == 0xde);
    }
}

TEST_CASE("Ampere register bindless textures reject unsupported modifiers and scratch spans",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x16167361, 0x78000018, 0x041e0718, 0x108f4400}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    for (const u32 bit : {40U, 58U, 59U, 77U, 80U, 81U, 91U}) {
        auto invalid = code;
        invalid[0].words[bit / 32] ^= 1U << (bit % 32);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    for (const u32 word : {0xf5167361U, 0x16f57361U}) {
        auto invalid = code;
        invalid[0].words[0] = word;
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto invalid = code;
    invalid[0].words[1] = 0x780000f5;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
}

TEST_CASE("Ampere half explicit texel fetches retain LOD and pack RGB into split registers",
          "[video_core][ampere]") {
    for (const u32 guard : {1U, 7U, 9U}) {
        const std::array<Instruction, 2> code{{
            {{0x04040b66U | (guard << 12), 0x33c0440e, 0x059e8705, 0x002f4200}},
            {{0x0000794d, 0, 0x03800000, 0}},
        }};
        const auto lowered = LowerToMaxwell(code);
        REQUIRE(static_cast<bool>(lowered));
        REQUIRE(lowered.required_texture_buffer == 15);
        REQUIRE(lowered.instructions.size() == 11);
        const u64 pred = u64{guard} << 16;
        REQUIRE(lowered.instructions[3] ==
                (0xdc00000000000000ULL | pred | 250 | (250ULL << 8) | (246ULL << 20) |
                 (2ULL << 28) | (7ULL << 31) | (0x44ULL << 36) | (1ULL << 49) |
                 (7ULL << 51) | (1ULL << 55)));
        for (u32 pair = 0; pair < 2; ++pair) {
            const u32 low = pair == 0 ? 250 : 252, high = pair == 0 ? 251 : 255;
            const u32 offset = 4 + pair * 3;
            REQUIRE(lowered.instructions[offset] ==
                    (0x5ca8000000000000ULL | pred | 246 | (1ULL << 8) | (2ULL << 10) |
                     (u64{low} << 20)));
            REQUIRE(lowered.instructions[offset + 1] ==
                    (0x5ca8000000000000ULL | pred | 247 | (1ULL << 8) | (2ULL << 10) |
                     (u64{high} << 20)));
            REQUIRE(lowered.instructions[offset + 2] ==
                    (0x5c18000000000000ULL | pred | (4 + pair) | (247ULL << 8) |
                     (246ULL << 20) | (16ULL << 39)));
        }
    }
}

TEST_CASE("Ampere half texture results keep the first component in the low lane",
          "[video_core][ampere][half_texture]") {
    const std::array<Instruction, 2> samples{{
        {{0x04047b66, 0x33c0440e, 0x059e8705, 0x002f4200}},
        {{0x06007b60, 0x33c0440f, 0x059e8702, 0x101fc200}},
    }};
    for (const auto& sample : samples) {
        for (const bool split : {false, true}) {
            for (const bool rgba : {false, true}) {
                const bool texel_fetch = sample.Bits(0, 12) == 0xb66;
                if (texel_fetch && (!split || rgba)) {
                    continue;
                }
                auto instruction = sample;
                if (!texel_fetch) {
                    instruction.words[2] = (instruction.words[2] & ~0xfffU) |
                                           (split ? 2U : 255U) | (rgba ? 0xf00U : 0x700U);
                }
                const std::array<Instruction, 2> code{{
                    instruction,
                    {{0x0000794d, 0, 0x03800000, 0}},
                }};
                const auto lowered = LowerToMaxwell(code);
                REQUIRE(static_cast<bool>(lowered));
                std::array<u32, 256> converted_components{};
                converted_components[250] = 0x3c00;
                converted_components[251] = 0x3800;
                converted_components[252] = 0x3400;
                converted_components[253] = 0x3a00;
                std::array<u32, 256> registers{};
                u32 packed_registers{};
                for (const u64 word : lowered.instructions) {
                    const u32 dest = word & 255;
                    if ((word >> 48) == 0x5ca8) {
                        registers[dest] = converted_components[(word >> 20) & 255];
                    } else if ((word >> 48) == 0x5c18) {
                        registers[dest] = (registers[(word >> 8) & 255] << ((word >> 39) & 31)) +
                                          registers[(word >> 20) & 255];
                        ++packed_registers;
                    }
                }
                REQUIRE(packed_registers == 2);
                REQUIRE(registers[instruction.Bits(16, 8)] == 0x38003c00);
                const u32 second = split ? instruction.Bits(64, 8) : instruction.Bits(16, 8) + 1;
                REQUIRE(registers[second] == (rgba ? 0x3a003400U : 0x00003400U));
            }
        }
    }
}

TEST_CASE("Ampere half explicit texel fetches keep unsupported formats and register spans rejected",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x04047b66, 0x33c0440e, 0x059e8705, 0x002f4200}},
        {{0x0000794d, 0, 0x03800000, 0}},
    }};
    for (const u32 bit : {59U, 60U, 62U, 76U, 77U, 78U, 80U, 81U, 91U}) {
        auto invalid = code;
        invalid[0].words[bit / 32] ^= 1U << (bit % 32);
        REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    }
    auto invalid = code;
    invalid[0].words[0] = 0xf5047b66;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
    invalid = code;
    invalid[0].words[1] = 0x33c044f6;
    REQUIRE_FALSE(static_cast<bool>(LowerToMaxwell(invalid)));
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
} // namespace
