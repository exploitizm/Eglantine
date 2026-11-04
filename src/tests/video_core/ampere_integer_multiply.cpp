// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <iomanip>
#include <random>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};

struct IntegerMachine {
    std::array<u32, 256> registers{};
    std::array<bool, 7> predicates{};
    std::array<std::array<u32, 64>, 18> constants{};
    bool carry{};

    u32 Read(u32 index) const {
        return index == 255 ? 0 : registers[index];
    }

    void Run(std::span<const u64> code) {
        for (const u64 word : code) {
            const auto bits = [&](u32 offset, u32 count) {
                return static_cast<u32>((word >> offset) & ((u64{1} << count) - 1));
            };
            const u32 guard = bits(16, 3);
            if (!((guard == 7 || predicates[guard]) != bool(bits(19, 1))))
                continue;
            const u32 dest = bits(0, 8);
            const u32 a = Read(bits(8, 8)), b = Read(bits(20, 8)), c = Read(bits(39, 8));
            const u64 opcode = word & 0xfff8000000000000ULL;
            u32 result{};
            if ((word & 0xffc0000000000000ULL) == 0x5b00000000000000ULL || (word >> 56) == 0x36) {
                const u32 x = (a >> (bits(53, 1) * 16)) & 0xffff;
                const u32 y =
                    (word >> 56) == 0x36 ? bits(20, 16) : (b >> (bits(35, 1) * 16)) & 0xffff;
                const u32 mode = bits(50, 3);
                REQUIRE(mode <= 2);
                result = x * y + (mode == 0 ? c : mode == 1 ? c & 0xffff : c >> 16);
            } else if ((word & 0xfff0000000000000ULL) == 0x5b60000000000000ULL) {
                REQUIRE(bits(49, 3) == 5); // ISETP.NE.U32.AND
                REQUIRE(bits(45, 2) == 0);
                REQUIRE(bits(48, 1) == 0);
                REQUIRE(bits(43, 1) == 0);
                REQUIRE(bits(39, 4) == 7);
                const bool comparison = a != b;
                if (bits(3, 3) != 7) predicates[bits(3, 3)] = comparison;
                if (bits(0, 3) != 7) predicates[bits(0, 3)] = !comparison;
                continue;
            } else if (opcode == 0x5c10000000000000ULL) {
                const u32 x = bits(49, 1) ? 0U - a : a;
                const u32 y = bits(48, 1) ? 0U - b : b;
                const u64 sum = u64{x} + y + (bits(43, 1) && carry);
                result = static_cast<u32>(sum);
                if (bits(47, 1))
                    carry = (sum >> 32) != 0;
            } else if (opcode == 0x5c18000000000000ULL) {
                result = (a << bits(39, 5)) + b;
            } else if (opcode == 0x3828000000000000ULL) {
                const u32 shift = bits(20, 19);
                REQUIRE(shift < 32);
                result =
                    bits(48, 1) ? static_cast<u32>(std::bit_cast<s32>(a) >> shift) : a >> shift;
            } else if (opcode == 0x5c40000000000000ULL) {
                REQUIRE(bits(41, 2) == 0);
                result = a & b;
            } else if (opcode == 0x3800000000000000ULL) {
                const u32 offset = bits(20, 8), width = bits(28, 8);
                REQUIRE(width == 8);
                REQUIRE(offset <= 24);
                result = (a >> offset) & 0xff;
                if (bits(48, 1) && (result & 0x80))
                    result |= 0xffffff00;
            } else if (opcode == 0x5ca0000000000000ULL) {
                const u32 select = bits(39, 3);
                const bool take_a = (select == 7 || predicates[select]) != bool(bits(42, 1));
                result = take_a ? a : b;
            } else if ((word >> 56) == 0x04) {
                REQUIRE(bits(53, 2) == 2); // LOP32I.XOR
                result = a ^ bits(20, 32);
            } else if (opcode == 0x5c98000000000000ULL) {
                result = b;
            } else if ((word >> 56) == 0x01) {
                result = bits(20, 32);
            } else if (opcode == 0x4c98000000000000ULL) {
                const u32 bank = bits(34, 5), offset = bits(20, 14);
                REQUIRE(bank < constants.size());
                REQUIRE(offset < constants[bank].size());
                result = constants[bank][offset];
            } else if ((word >> 56) == 0xef) {
                REQUIRE(bits(48, 3) == 5);
                const u32 bank = bits(36, 5), offset = bits(20, 16) + a;
                REQUIRE(bank < constants.size());
                REQUIRE(offset % 4 == 0);
                REQUIRE(offset / 4 + 1 < constants[bank].size());
                result = constants[bank][offset / 4];
                registers[dest + 1] = constants[bank][offset / 4 + 1];
            } else if ((word >> 56) == 0xe3) {
                break;
            } else {
                FAIL("Unexpected emitted instruction " << std::hex << word);
            }
            if (dest != 255)
                registers[dest] = result;
        }
    }
};

u64 Product(u32 a, u32 b, bool is_signed) {
    if (!is_signed)
        return u64{a} * b;
    return static_cast<u64>(s64{std::bit_cast<s32>(a)} * std::bit_cast<s32>(b));
}

IntegerMachine SeedMachine() {
    IntegerMachine machine;
    for (u32 i = 0; i < machine.registers.size(); ++i)
        machine.registers[i] = 0xdead0000U + i;
    machine.predicates = {false, true, false, true, false, true, true};
    return machine;
}

void CheckUnchanged(const IntegerMachine& before, const IntegerMachine& after, u32 dest,
                    u32 count) {
    REQUIRE(after.predicates == before.predicates);
    for (u32 i = 0; i < 246; ++i) {
        if (i < dest || i >= dest + count)
            REQUIRE(after.registers[i] == before.registers[i]);
    }
}
} // namespace

TEST_CASE("Ampere high multiply respects operand forms signs aliases and predicates",
          "[video_core][ampere][integer_multiply]") {
    std::mt19937 random{0x949a921};
    for (const u32 opcode : {0x227U, 0x827U, 0xa27U}) {
        for (const bool is_signed : {false, true}) {
            for (const u32 dest : {0U, 2U, 4U, 6U}) {
                for (const u32 guard : {0U, 5U, 6U, 7U, 8U, 13U, 14U}) {
                    for (u32 test = 0; test < 24; ++test) {
                        auto machine = SeedMachine();
                        machine.registers[2] =
                            test < 4 ? std::array{0U, 1U, ~0U, 0x80000000U}[test] : random();
                        machine.registers[4] =
                            test < 4 ? std::array{~0U, 2U, ~0U, 0x80000000U}[test] : random();
                        machine.registers[6] = random();
                        machine.constants[2][4] = machine.registers[4];
                        const Instruction instruction{
                            {opcode | (guard << 12) | (dest << 16) | (2U << 24),
                             opcode == 0x227   ? 4U
                             : opcode == 0x827 ? machine.registers[4]
                                               : (2U << 22) | (16U << 6),
                             0x078e0006U | (u32{is_signed} << 9), 0}};
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        INFO(lowered.error);
                        REQUIRE(lowered);
                        const auto before = machine;
                        const bool active =
                            ((guard & 7) == 7 || before.predicates[guard & 7]) != bool(guard & 8);
                        const u32 expected =
                            active ? static_cast<u32>(
                                         Product(before.Read(2), before.Read(4), is_signed) >> 32) +
                                         before.Read(6)
                                   : before.Read(dest);
                        machine.Run(lowered.instructions);
                        REQUIRE(machine.Read(dest) == expected);
                        CheckUnchanged(before, machine, dest, 1);
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere wide multiply preserves the full sum for all operand forms and aliases",
          "[video_core][ampere][integer_multiply]") {
    std::mt19937 random{0x825625};
    for (const u32 opcode : {0x225U, 0x625U, 0x825U}) {
        for (const bool is_signed : {false, true}) {
            for (const u32 dest : {0U, 2U, 4U, 6U}) {
                for (const u32 guard : {0U, 5U, 6U, 7U, 8U, 13U, 14U}) {
                    for (u32 test = 0; test < 24; ++test) {
                        auto machine = SeedMachine();
                        machine.registers[2] =
                            test < 4 ? std::array{0U, 1U, ~0U, 0x80000000U}[test] : random();
                        machine.registers[4] =
                            test < 4 ? std::array{~0U, 2U, ~0U, 0x80000000U}[test] : random();
                        machine.registers[6] = test < 4 ? ~0U : random();
                        machine.registers[7] = random();
                        machine.constants[2][4] = machine.registers[6];
                        machine.constants[2][5] = machine.registers[7];
                        const Instruction instruction{
                            {opcode | (guard << 12) | (dest << 16) | (2U << 24),
                             opcode == 0x225   ? 4U
                             : opcode == 0x825 ? machine.registers[4]
                                               : (2U << 22) | (16U << 6),
                             0x078e0000U | (opcode == 0x625 ? 4U : 6U) | (u32{is_signed} << 9), 0}};
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        INFO(lowered.error);
                        REQUIRE(lowered);
                        const auto before = machine;
                        const bool active =
                            ((guard & 7) == 7 || before.predicates[guard & 7]) != bool(guard & 8);
                        const u64 expected =
                            active ? Product(before.Read(2), before.Read(4), is_signed) +
                                         before.Read(6) + (u64{before.Read(7)} << 32)
                                   : before.Read(dest) | (u64{before.Read(dest + 1)} << 32);
                        machine.Run(lowered.instructions);
                        REQUIRE((machine.Read(dest) | (u64{machine.Read(dest + 1)} << 32)) ==
                                expected);
                        CheckUnchanged(before, machine, dest, 2);
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere IDP4 accumulates signed and unsigned bytes without scratch or alias corruption",
          "[video_core][ampere][integer_multiply]") {
    for (const bool signed_a : {false, true}) {
        for (const bool signed_b : {false, true}) {
            for (const u32 dest : {0U, 2U, 4U, 6U}) {
                for (const u32 guard : {0U, 5U, 6U, 7U, 8U, 13U, 14U}) {
                    for (const u32 input : {0x01010101U, 0x80ff7f01U, ~0U, 0U}) {
                        auto machine = SeedMachine();
                        machine.registers[2] = input;
                        machine.registers[4] = 0xff7f8001U;
                        machine.registers[6] = 7;
                        const auto before = machine;
                        const Instruction instruction{
                            {0x226U | (guard << 12) | (dest << 16) | (2U << 24), 4,
                             6U | (u32{signed_a} << 9) | (u32{signed_b} << 10), 0}};
                        const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                        INFO(lowered.error);
                        REQUIRE(lowered);
                        u32 expected = before.Read(6);
                        for (u32 i = 0; i < 4; ++i) {
                            const auto byte = [i](u32 value, bool is_signed) -> s32 {
                                const u32 part = (value >> (8 * i)) & 255;
                                return is_signed && part >= 128 ? static_cast<s32>(part) - 256
                                                                : part;
                            };
                            expected += static_cast<u32>(byte(before.Read(2), signed_a) *
                                                         byte(before.Read(4), signed_b));
                        }
                        const bool active =
                            ((guard & 7) == 7 || before.predicates[guard & 7]) != bool(guard & 8);
                        machine.Run(lowered.instructions);
                        REQUIRE(machine.Read(dest) == (active ? expected : before.Read(dest)));
                        CheckUnchanged(before, machine, dest, 1);
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere runtime traps are not silently erased",
          "[video_core][ampere][integer_multiply]") {
    const auto trap = LowerToMaxwell(std::array{Instruction{{0x795c, 4, 0x00300000, 0}}, Exit});
    REQUIRE_FALSE(trap);
    REQUIRE(trap.error == "native runtime breakpoint/trap");
    for (const u32 guard : {0U, 7U, 8U}) {
        for (const u32 mode : {0U, 1U, 2U, 0xffffffffU}) {
            const auto lowered =
                LowerToMaxwell(std::array{Instruction{{0x95cU | (guard << 12), 0, mode, 0}}, Exit});
            REQUIRE_FALSE(lowered);
            REQUIRE(lowered.error == "native runtime breakpoint/trap");
        }
    }
    REQUIRE(LowerToMaxwell(std::array{Instruction{{0xf95c, 0, 0, 0}}, Exit}));
}

TEST_CASE("Ampere integer multiplication rejects unsupported modifiers",
          "[video_core][ampere][integer_multiply]") {
    for (const u32 opcode : {0x225U, 0x625U, 0x825U, 0x227U, 0x827U, 0xa27U}) {
        const Instruction valid{
            {opcode | 0x02007000U, opcode == 0x625 || opcode == 0xa27 ? 0U : 4U, 0x078e0006, 0}};
        REQUIRE(LowerToMaxwell(std::array{valid, Exit}));
        for (const u32 bit : {8U, 10U, 13U, 15U, 26U}) {
            auto invalid = valid;
            invalid.words[2] ^= 1U << bit;
            REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        }
    }
    for (const u32 bit : {8U, 11U, 12U, 13U, 15U}) {
        const Instruction invalid{{0x02007226, 4, 6U | (1U << bit), 0}};
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
    }
}


TEST_CASE("Ampere zero-product IMAD X preserves complemented addends and carry predicates",
          "[video_core][ampere][integer_multiply]") {
    for (const bool complement : {false, true}) {
        for (const bool invert_carry : {false, true}) {
            for (u32 carry = 0; carry < 8; ++carry) {
                for (const u32 source : {2U, 255U}) {
                    for (const u32 dest : {2U, 4U, 6U}) {
                        for (const u32 guard : {0U, 1U, 7U, 8U, 9U, 15U}) {
                            for (const u32 value : {0U, 1U, 0x7fffffffU, 0x80000000U, ~0U}) {
                                auto machine = SeedMachine();
                                machine.registers[2] = value;
                                const auto before = machine;
                                const Instruction instruction{{
                                    0x224U | (guard << 12) | (dest << 16) | (255U << 24),
                                    255,
                                    0x000e0400U | source | (u32{complement} << 11) |
                                        (carry << 23) | (u32{invert_carry} << 26), 0}};
                                const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                                INFO(lowered.error);
                                REQUIRE(lowered);
                                const bool active =
                                    ((guard & 7) == 7 || before.predicates[guard & 7]) != bool(guard & 8);
                                const bool carry_value =
                                    (carry == 7 || before.predicates[carry]) != invert_carry;
                                const u32 addend = complement ? ~before.Read(source) : before.Read(source);
                                const u32 expected = active ? addend + u32{carry_value} : before.Read(dest);
                                machine.Run(lowered.instructions);
                                REQUIRE(machine.Read(dest) == expected);
                                CheckUnchanged(before, machine, dest, 1);
                            }
                        }
                    }
                }
            }
        }
    }
}


TEST_CASE("Ampere unsigned wide IMAD reports full-sum carry after aliased guarded writes",
          "[video_core][ampere][integer_multiply]") {
    std::mt19937 random{0x225ca77};
    constexpr std::array<std::array<u32, 4>, 8> boundaries{{
        {0, 0, 0, 0}, {1, 1, ~0U, ~0U}, {~0U, ~0U, ~0U, ~0U},
        {~0U, ~0U, 0, ~0U}, {~0U, ~0U, ~0U, 0},
        {0x80000000U, 2, 0, ~0U}, {~0U, 1, 1, 0}, {1, 1, 0, 0},
    }};
    for (const u32 opcode : {0x225U, 0x625U, 0x825U}) {
        for (u32 carry_output = 0; carry_output < 7; ++carry_output) {
            for (const u32 dest : {0U, 2U, 4U, 6U}) {
                for (const u32 guard : {0U, 1U, 6U, 7U, 8U, 9U, 14U}) {
                    for (const bool initial_carry : {false, true}) {
                        for (u32 test = 0; test < 12; ++test) {
                            auto machine = SeedMachine();
                            machine.carry = initial_carry;
                            const auto values = test < boundaries.size() ? boundaries[test] :
                                std::array<u32, 4>{static_cast<u32>(random()), static_cast<u32>(random()),
                                                   static_cast<u32>(random()), static_cast<u32>(random())};
                            machine.registers[2] = values[0];
                            machine.registers[4] = values[1];
                            machine.registers[6] = values[2];
                            machine.registers[7] = values[3];
                            machine.constants[2][4] = values[2];
                            machine.constants[2][5] = values[3];
                            const Instruction instruction{{
                                opcode | (guard << 12) | (dest << 16) | (2U << 24),
                                opcode == 0x225 ? 4U : opcode == 0x825 ? values[1] : (2U << 22) | (16U << 6),
                                0x07800000U | (carry_output << 17) | (opcode == 0x625 ? 4U : 6U), 0}};
                            const auto lowered = LowerToMaxwell(std::array{instruction, Exit});
                            INFO(lowered.error);
                            REQUIRE(lowered);
                            const auto before = machine;
                            const bool active = ((guard & 7) == 7 || before.predicates[guard & 7]) != bool(guard & 8);
                            const unsigned __int128 sum =
                                static_cast<unsigned __int128>(values[0]) * values[1] +
                                values[2] + (static_cast<unsigned __int128>(values[3]) << 32);
                            const u64 expected = active ? static_cast<u64>(sum) :
                                before.Read(dest) | (u64{before.Read(dest + 1)} << 32);
                            auto expected_state = before;
                            if (active) expected_state.predicates[carry_output] = (sum >> 64) != 0;
                            machine.Run(lowered.instructions);
                            REQUIRE((machine.Read(dest) | (u64{machine.Read(dest + 1)} << 32)) == expected);
                            CheckUnchanged(expected_state, machine, dest, 2);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Ampere wide carry output retains signed and register mode restrictions",
          "[video_core][ampere][integer_multiply]") {
    for (const u32 opcode : {0x225U, 0x625U, 0x825U}) {
        const Instruction valid{{opcode | 0x02007000U,
            opcode == 0x625 ? (2U << 22) | (16U << 6) : 4U,
            0x07800000U | (opcode == 0x625 ? 4U : 6U), 0}};
        REQUIRE(LowerToMaxwell(std::array{valid, Exit}));
        auto invalid = valid;
        invalid.words[2] |= 1U << 9; // Signed carry output remains unsupported.
        REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        for (const u32 bit : {8U, 10U, 13U, 15U, 26U}) {
            invalid = valid;
            invalid.words[2] ^= 1U << bit;
            REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        }
        for (const u32 dest : {3U, 246U, 255U}) {
            invalid = valid;
            invalid.words[0] = (invalid.words[0] & ~0xff0000U) | (dest << 16);
            REQUIRE_FALSE(LowerToMaxwell(std::array{invalid, Exit}));
        }
    }
}
