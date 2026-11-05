// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <span>
#include <unordered_map>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
namespace IR = Shader::IR;
using Shader::Ampere::Instruction;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
constexpr Instruction Fetch{{0x02017b66, 0x33c04401, 0x059e8705, 0x00114400}};
constexpr std::array<Instruction, 2> Gathers{{
    {{0x06027b63, 0x33c04407, 0x041e8f03, 0x000fc200}},
    {{0x06027b63, 0x33c04407, 0x041e8f01, 0x000fc200}},
}};

class TextureEnvironment final : public Shader::Environment {
public:
    u32 binding{15};
    std::span<const u64> code;

    TextureEnvironment() { stage = Shader::Stage::Compute; }

    u64 ReadInstruction(u32 address) override {
        for (u32 index = 0; index < code.size(); ++index) {
            if (address == 0x50 + index * 8 + ((index + 1) / 3) * 8) {
                return code[index];
            }
        }
        FAIL("Read outside logged texture fixture at " << address);
        return 0;
    }
    u32 ReadCbufValue(u32 bank, u32 offset) override { return (bank << 16) | offset; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return binding; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 4096; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    bool HasNativeInstructions() const override { return true; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

u32 HalfFixture(u32 bits) {
    switch (bits) {
    case 0x00000000: return 0x0000;
    case 0x80000000: return 0x8000;
    case 0x3f800000: return 0x3c00;
    case 0xbf000000: return 0xb800;
    case 0x3e800000: return 0x3400;
    case 0x3f400000: return 0x3a00;
    case 0x3f801000: return 0x3c00;
    case 0x3f801001: return 0x3c01;
    default:
        FAIL("Unexpected half fixture " << bits);
        return 0;
    }
}

struct TextureTranslation {
    using Vector = std::array<u32, 4>;
    TextureEnvironment env;
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block block{pool};
    Shader::Maxwell::TranslatorVisitor visitor{env, block};
    std::array<u32, IR::NUM_REGS> registers{};
    Vector sample{0x3f800000, 0xbf000000, 0x3e800000, 0x3f400000};
    Vector coordinates{};
    u32 level{};
    u32 textures{};

    void Translate(const Shader::Ampere::LoweredProgram& lowered, u32 guard) {
        REQUIRE(lowered.required_texture_buffer == env.TextureBoundBuffer());
        for (const u64 word : lowered.instructions) {
            if ((word >> 48) == 0xe300) {
                continue;
            }
            REQUIRE(((word >> 16) & 15) == guard);
            if ((word >> 48) == 0x5c98) {
                visitor.MOV_reg(word);
            } else if ((word >> 48) == 0x4c98) {
                visitor.MOV_cbuf(word);
            } else if ((word >> 48) == 0x5ca8) {
                visitor.F2F_reg(word);
            } else if ((word >> 48) == 0x5c18) {
                visitor.ISCADD_reg(word);
            } else if ((word >> 56) == 0xd2) {
                visitor.TLDS(word);
            } else if ((word >> 56) == 0xc8) {
                visitor.TLD4(word);
            } else if ((word >> 56) == 0xde) {
                visitor.TEX_b(word);
            } else if ((word >> 60) == 0xc) {
                visitor.TEX(word);
            } else {
                FAIL("Unexpected lowered texture instruction " << word);
            }
        }
    }

    void Run() {
        std::unordered_map<const IR::Inst*, Vector> values;
        const auto read = [&](IR::Value value) -> Vector {
            if (!value.IsImmediate()) {
                return values.at(value.InstRecursive());
            }
            return {value.Type() == IR::Type::F32 ? std::bit_cast<u32>(value.F32()) : value.U32()};
        };
        for (const IR::Inst& inst : block) {
            const auto arg = [&](size_t i) { return read(inst.Arg(i))[0]; };
            Vector result{};
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetRegister:
                result[0] = registers.at(static_cast<size_t>(inst.Arg(0).Reg()));
                break;
            case IR::Opcode::SetRegister:
                registers.at(static_cast<size_t>(inst.Arg(0).Reg())) = arg(1);
                break;
            case IR::Opcode::BitCastU32F32:
            case IR::Opcode::BitCastF32U32:
                result[0] = arg(0);
                break;
            case IR::Opcode::CompositeConstructU32x2:
            case IR::Opcode::CompositeConstructF32x2:
            case IR::Opcode::CompositeConstructF16x2:
                result = {arg(0), arg(1)};
                break;
            case IR::Opcode::BoundImageFetch:
            case IR::Opcode::BoundImageGather:
                REQUIRE(arg(0) == 0x110);
                coordinates = read(inst.Arg(1));
                if (inst.GetOpcode() == IR::Opcode::BoundImageFetch) {
                    level = arg(3);
                    REQUIRE(inst.Flags<IR::TextureInstInfo>().relaxed_precision == 1);
                }
                result = sample;
                ++textures;
                break;
            case IR::Opcode::CompositeExtractF32x4:
                result[0] = read(inst.Arg(0)).at(arg(1));
                break;
            case IR::Opcode::ConvertF16F32:
                if (inst.Arg(0).IsImmediate()) {
                    REQUIRE(inst.Arg(0).F32() == 0.0f);
                } else {
                    REQUIRE(inst.Flags<IR::FpControl>().rounding == IR::FpRounding::RN);
                }
                result[0] = HalfFixture(arg(0));
                break;
            case IR::Opcode::PackFloat2x16: {
                const auto pair{read(inst.Arg(0))};
                result[0] = pair[0] | (pair[1] << 16);
                break;
            }
            case IR::Opcode::PackHalf2x16: {
                const auto pair{read(inst.Arg(0))};
                result[0] = HalfFixture(pair[0]) | (HalfFixture(pair[1]) << 16);
                break;
            }
            case IR::Opcode::ShiftLeftLogical32:
                result[0] = arg(0) << arg(1);
                break;
            case IR::Opcode::IAdd32:
                result[0] = arg(0) + arg(1);
                break;
            default:
                FAIL("Unexpected translated texture opcode " << IR::NameOf(inst.GetOpcode()));
            }
            values.emplace(&inst, result);
        }
    }
};
}

TEST_CASE("Welcome Tour array depth samples preserve alternate descriptor banks",
          "[video_core][ampere][logged_textures]") {
    const Instruction captured{{0x0a0d7b60, 0xb1012400, 0x041e41ff, 0x00114400}};
    for (const u32 first_binding : {2U, 15U}) {
        auto first{captured};
        first.words[1] = (first.words[1] & ~(31U << 22)) | (first_binding << 22);
        const auto lowered{Shader::Ampere::LowerToMaxwell(std::array{first, captured, captured, Exit})};
        INFO(lowered.error);
        REQUIRE(lowered);
        TextureTranslation translated;
        translated.env.binding = first_binding;
        translated.Translate(lowered, 7);
        IR::Program program;
        program.blocks = {&translated.block};
        program.post_order_blocks = {&translated.block};
        Shader::Optimization::SsaRewritePass(program);
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        Shader::Optimization::TexturePass(translated.env, program, host);
        REQUIRE(program.info.texture_descriptors.size() == 2);
        std::array<u32, 3> bindings{};
        size_t count{};
        for (const auto& inst : translated.block) {
            if (inst.GetOpcode() != IR::Opcode::ImageSampleDrefImplicitLod) {
                continue;
            }
            const auto flags{inst.Flags<IR::TextureInstInfo>()};
            REQUIRE(flags.is_depth == 1);
            REQUIRE(flags.type == Shader::TextureType::ColorArray2D);
            const auto& descriptor{program.info.texture_descriptors.at(flags.descriptor_index)};
            CHECK(descriptor.cbuf_offset == 0x490);
            CHECK(descriptor.handle_program == ~0U);
            REQUIRE(count < bindings.size());
            bindings[count++] = descriptor.cbuf_index;
        }
        CHECK(count == 3);
        CHECK(bindings == std::array<u32, 3>{first_binding, 4, 4});
    }
}

TEST_CASE("Welcome Tour logged half fetch writes odd scalar destinations",
          "[video_core][ampere][logged_textures]") {
    for (const u32 guard : {1U, 7U, 9U}) {
        auto instruction{Fetch};
        instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
        const auto lowered{Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit})};
        INFO(lowered.error);
        REQUIRE(lowered);
        TextureTranslation translated;
        translated.registers.fill(0xdeadbeef);
        translated.registers[1] = 3;
        translated.registers[2] = 10;
        translated.registers[3] = 20;
        const auto before{translated.registers};
        translated.Translate(lowered, guard);
        translated.Run();
        CHECK(translated.textures == 1);
        CHECK(translated.coordinates == TextureTranslation::Vector{10, 20});
        CHECK(translated.level == 3);
        CHECK(translated.registers[1] == 0xb8003c00);
        CHECK(translated.registers[5] == 0x00003400);
        for (u32 reg = 0; reg < 246; ++reg) {
            if (reg != 1 && reg != 5) {
                REQUIRE(translated.registers[reg] == before[reg]);
            }
        }
    }
}

TEST_CASE("Welcome Tour logged half gathers round and pack both destination orders",
          "[video_core][ampere][logged_textures]") {
    for (const auto& capture : Gathers) {
        for (const u32 guard : {1U, 7U, 9U}) {
            auto instruction{capture};
            instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
            const auto lowered{Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit})};
            INFO(lowered.error);
            REQUIRE(lowered);
            TextureTranslation translated;
            translated.registers.fill(0xdeadbeef);
            translated.registers[6] = 0x3e000000;
            translated.registers[7] = 0x3f400000;
            translated.sample = {0x3f801000, 0x3f801001, 0x80000000, 0x3f400000};
            const auto before{translated.registers};
            translated.Translate(lowered, guard);
            translated.Run();
            CHECK(translated.textures == 1);
            CHECK(translated.coordinates == TextureTranslation::Vector{0x3e000000, 0x3f400000});
            CHECK(translated.registers[2] == 0x3c013c00);
            CHECK(translated.registers[instruction.Bits(64, 8)] == 0x3a008000);
            for (u32 reg = 0; reg < 246; ++reg) {
                if (reg != 2 && reg != instruction.Bits(64, 8)) {
                    REQUIRE(translated.registers[reg] == before[reg]);
                }
            }
        }
    }
}

TEST_CASE("Logged half textures retain unsupported-mode rejections",
          "[video_core][ampere][logged_textures]") {
    for (const u32 bit : {76U, 77U, 78U, 80U, 91U}) {
        auto invalid{Gathers[0]};
        invalid.words[bit / 32] ^= 1U << (bit % 32);
        CHECK_FALSE(Shader::Ampere::LowerToMaxwell(std::array{invalid, Exit}));
    }
    auto invalid{Fetch};
    invalid.words[2] &= ~(1U << 15);
    CHECK_FALSE(Shader::Ampere::LowerToMaxwell(std::array{invalid, Exit}));
}

TEST_CASE("Welcome Tour logged half textures reach production SPIRV emission",
          "[video_core][ampere][logged_textures]") {
    for (const auto instruction : {Fetch, Gathers[0], Gathers[1]}) {
        const auto lowered{Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit})};
        INFO(lowered.error);
        REQUIRE(lowered);
        auto code{lowered.instructions};
        for (const u32 source : {instruction.Bits(16, 8), instruction.Bits(64, 8)}) {
            code.insert(code.end() - 1, 0xef58000000070000ULL | source | (255ULL << 8) |
                                            (u64{source * 4} << 20) | (4ULL << 48));
        }
        TextureEnvironment env;
        env.code = code;
        env.binding = lowered.required_texture_buffer;
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::Maxwell::Flow::CFG cfg{env, flow_pool, Shader::Maxwell::Location{0x50}};
        Shader::ObjectPool<IR::Inst> inst_pool;
        Shader::ObjectPool<IR::Block> block_pool;
        Shader::HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        auto program{Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host)};
        u32 stores{};
        for (const auto* block : program.blocks) {
            for (const auto& inst : *block) {
                stores += inst.GetOpcode() == IR::Opcode::WriteSharedU32;
            }
        }
        REQUIRE(stores == 2);
        Shader::Profile profile{};
        profile.supported_spirv = 0x00010400;
        const auto spirv{Shader::Backend::SPIRV::EmitSPIRV(profile, program)};
        REQUIRE(spirv.size() > 5);
        CHECK(spirv.front() == 0x07230203);
    }
}

TEST_CASE("Half gathers reject reserved second coordinates before scratch copies",
          "[video_core][ampere][logged_textures]") {
    for (u32 source = 246; source < 255; ++source) {
        auto instruction{Gathers[0]};
        instruction.words[1] = (instruction.words[1] & ~255U) | source;
        CAPTURE(source);
        CHECK_FALSE(Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit}));
    }
    for (const u32 source : {0U, 1U, 215U, 245U, 255U}) {
        auto instruction{Gathers[0]};
        instruction.words[1] = (instruction.words[1] & ~255U) | source;
        const auto lowered{Shader::Ampere::LowerToMaxwell(std::array{instruction, Exit})};
        REQUIRE(lowered);
        TextureTranslation translated;
        translated.registers[6] = 0x3e000000;
        if (source != 255) {
            translated.registers[source] = 0x3f400000;
        }
        translated.Translate(lowered, 7);
        translated.Run();
        CHECK(translated.coordinates ==
              TextureTranslation::Vector{0x3e000000, source == 255 ? 0U : 0x3f400000U});
        CHECK(translated.registers[2] == 0xb8003c00);
        CHECK(translated.registers[3] == 0x3a003400);
    }
    constexpr Instruction uniform_move{{0x00007882, 0, 0, 0}};
    for (const u32 high_uniforms : {0U, 4U, 8U}) {
        const u32 limit = 216 - high_uniforms;
        for (const u32 source : {limit - 1, limit, 245U, 250U, 254U, 255U}) {
            auto instruction{Gathers[0]};
            instruction.words[1] = (instruction.words[1] & ~255U) | source;
            const auto lowered{Shader::Ampere::LowerToMaxwellWith(
                std::array{uniform_move, instruction, Exit}, high_uniforms, false, false)};
            CAPTURE(high_uniforms, source);
            CHECK(static_cast<bool>(lowered) == (source < limit || source == 255));
        }
    }
}
