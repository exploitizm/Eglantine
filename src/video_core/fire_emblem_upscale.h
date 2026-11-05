// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <string_view>
#include <utility>
#include "common/common_types.h"
#include "video_core/textures/texture.h"
#include "video_core/textures/texture_view.h"

namespace VideoCommon::FireEmblemUpscale {

// Present boundaries for bounded network discovery, independent of shader success.
inline std::atomic<u64> DlssTraceFrame{};

constexpr u64 ProgramId = 0x04007EF01F8AE000ULL;
// NativeProgramKey of the captured DLSS final output pass, not its lowered shader hash.
constexpr u64 OutputProgramKey = 0x14b4222bbe23abeeULL;
constexpr u64 PrepassProgramKey = 0x4cbf6eff32525aecULL;
constexpr u64 StarFoxProgramId = 0x0400E88025B1A000ULL;
constexpr u64 StarFoxPrepassProgramKey = 0xd686ca0b9fad9e6dULL;
constexpr u64 StarFoxOutputProgramKey = 0x7d3764ff101bec27ULL;
constexpr u64 OcarinaProgramId = 0x0400486024DEC000ULL;
constexpr u64 OcarinaPrepassProgramKey = 0x2f5791cea2a85397ULL;
constexpr u64 OcarinaOutputProgramKey = 0x6093c9143ba0904dULL;
constexpr u64 OcarinaGameplayProgramKey = 0xf48db7481be856b9ULL;
constexpr u32 OcarinaGameplayInstructionCount = 1410;
constexpr std::array<u64, 13> OcarinaNetworkKeys{
    OcarinaPrepassProgramKey, 0x313d724f3e214203ULL, 0xdf2c83cf2ad70790ULL,
    0x1fdb95ea2565f49cULL, 0x0be2f024aea21a6cULL, 0x0618cc4a3ac8f33aULL,
    0x0618cc4a3ac8f33aULL, 0xe27336d38f1cc8f4ULL, 0x1fdb95ea2565f49cULL,
    0xecebb02dec6faa17ULL, 0x313d724f3e214203ULL, 0x2a366b38546a4d5eULL,
    OcarinaOutputProgramKey};

constexpr bool OcarinaStepMatches(u32 step, u64 key) {
    return step < OcarinaNetworkKeys.size() && OcarinaNetworkKeys[step] == key;
}

constexpr bool OcarinaNetworkContains(u64 key) {
    for (const u64 program : OcarinaNetworkKeys) {
        if (key == program) {
            return true;
        }
    }
    return false;
}

constexpr bool SuppressOcarinaOutput(bool spatial_applied, u32 skipped) {
    // An incomplete native evaluation must never consume unwritten intermediates.
    return spatial_applied || skipped != 0;
}

// Legacy names retained for the captured-instruction tests.
constexpr u32 PrepassColorOffset = 0x278;
constexpr u32 JitterOffset = 0x1b0;
constexpr u32 OutputColorInputOffset = 0x328;
constexpr u32 OutputColorOffset = 0x348;

struct Passes {
    u64 prepass;
    u64 output;
    bool spatial_only;
    u32 prepass_color_offset;
    std::optional<u32> jitter_offset;
    u32 output_color_input_offset;
    u32 output_color_offset;
    Tegra::Texture::TextureFormat input_format;
    Tegra::Texture::TextureFormat output_format;
    bool input_srgb;
    bool skip_network;
    bool standalone{};
};

constexpr std::optional<Passes> CapturedPasses(u64 program_id, u64 native_key = 0) {
    if (program_id == ProgramId) {
        return Passes{PrepassProgramKey, OutputProgramKey, false, 0x278, 0x1b0, 0x328, 0x348,
                      Tegra::Texture::TextureFormat::B10G11R11,
                      Tegra::Texture::TextureFormat::B10G11R11, false, false};
    }
    if (program_id == StarFoxProgramId) {
        return Passes{StarFoxPrepassProgramKey, StarFoxOutputProgramKey, true, 0x278, 0x1b0,
                      0x328, 0x348, Tegra::Texture::TextureFormat::A8B8G8R8,
                      Tegra::Texture::TextureFormat::R16G16B16A16, true, false};
    }
    if (program_id == OcarinaProgramId && native_key == OcarinaGameplayProgramKey) {
        return Passes{OcarinaGameplayProgramKey, OcarinaGameplayProgramKey, true, 0xc0,
                      std::nullopt, 0xc0, 0xd0, Tegra::Texture::TextureFormat::E5B9G9R9,
                      Tegra::Texture::TextureFormat::B10G11R11, false, true, true};
    }
    if (program_id == OcarinaProgramId) {
        // Movie capture: original converted RGB, not the chroma plane or network intermediate.
        return Passes{OcarinaPrepassProgramKey, OcarinaOutputProgramKey, true, 0x58,
                      std::nullopt, 0xa8, 0xc8, Tegra::Texture::TextureFormat::A8B8G8R8,
                      Tegra::Texture::TextureFormat::A8B8G8R8, false, true};
    }
    return std::nullopt;
}

struct JitterSample {
    GPUVAddr color{};
    u32 width{}, height{};
    std::array<f32, 2> offset{};
};

class FrameJitter {
public:
    void Capture(GPUVAddr color, u32 width, u32 height, std::array<f32, 2> offset) {
        sample.reset();
        if (color == 0 || width == 0 || height == 0 ||
            !std::isfinite(offset[0]) || !std::isfinite(offset[1]) ||
            std::abs(offset[0]) > 0.5f || std::abs(offset[1]) > 0.5f) {
            return;
        }
        sample = JitterSample{color, width, height, offset};
    }

    // Consume on every output attempt, even if its resources cannot be used.
    std::optional<JitterSample> Take() {
        return std::exchange(sample, std::nullopt);
    }

private:
    std::optional<JitterSample> sample;
};

inline bool UseReplacement(u64 program_id, u64 native_key, const char* option) {
    const auto passes = CapturedPasses(program_id, native_key);
    return passes && native_key == passes->output &&
           (option == nullptr || (std::string_view{option} == "fsr" || std::string_view{option} == "dlss"));
}

inline bool PlainImage(const Tegra::Texture::TICEntry& image) {
    using namespace Tegra::Texture;
    return (image.texture_type == TextureType::Texture2D ||
            image.texture_type == TextureType::Texture2DNoMipmap) && image.IsBlockLinear() &&
           image.Depth() == 1 && image.BaseLayer() == 0 &&
           image.max_mip_level == 0 && image.Address() != 0;
}

inline bool StarFoxColor(const Tegra::Texture::TICEntry& image,
                         Tegra::Texture::TextureFormat format,
                         Tegra::Texture::ComponentType component, bool srgb,
                         Tegra::Texture::SwizzleSource alpha = Tegra::Texture::SwizzleSource::A) {
    using namespace Tegra::Texture;
    return PlainImage(image) && image.format == format && (image.srgb_conversion != 0) == srgb &&
           image.r_type == component && image.g_type == component &&
           image.b_type == component && image.a_type == component &&
           image.x_source == SwizzleSource::R && image.y_source == SwizzleSource::G &&
           image.z_source == SwizzleSource::B && image.w_source == alpha &&
           image.res_min_mip_level == 0 && image.res_max_mip_level == 0 &&
           image.msaa_mode == MsaaMode::Msaa1x1 && image.block_width == 0 &&
           image.block_depth == 0 && image.is_sparse == 0;
}

inline bool ValidInput(const Tegra::Texture::TICEntry& input, bool spatial_only = false) {
    using namespace Tegra::Texture;
    if (spatial_only) {
        return StarFoxColor(input, TextureFormat::A8B8G8R8, ComponentType::UNORM, true);
    }
    return input.format == TextureFormat::B10G11R11 && input.IsBlockLinear();
}

inline bool ValidPair(const Tegra::Texture::TICEntry& input,
                      const Tegra::Texture::TICEntry& output, bool spatial_only = false) {
    using namespace Tegra::Texture;
    if (spatial_only) {
        return ValidInput(input, true) &&
               StarFoxColor(output, TextureFormat::R16G16B16A16, ComponentType::FLOAT, false) &&
               input.Address() != output.Address() && input.Width() == output.Width() &&
               input.Height() == output.Height();
    }
    return PlainImage(input) && PlainImage(output) &&
           input.format == TextureFormat::B10G11R11 && output.format == TextureFormat::B10G11R11 &&
           input.Address() != output.Address() && output.Width() > input.Width() &&
           output.Height() > input.Height();
}

inline bool ValidInput(const Tegra::Texture::TICEntry& input, const Passes& passes) {
    if (passes.standalone) {
        return StarFoxColor(VideoCommon::TextureViewHeader(input, true), passes.input_format,
                            Tegra::Texture::ComponentType::FLOAT, false,
                            Tegra::Texture::SwizzleSource::OneFloat) &&
               input.Width() >= 64 && input.Height() >= 64;
    }
    if (passes.skip_network) {
        return StarFoxColor(VideoCommon::TextureViewHeader(input, true), passes.input_format,
                            Tegra::Texture::ComponentType::UNORM,
                            passes.input_srgb) && input.Width() == 1920 && input.Height() == 1080;
    }
    return input.format == passes.input_format && ValidInput(input, passes.spatial_only);
}

inline bool ValidPair(const Tegra::Texture::TICEntry& input,
                      const Tegra::Texture::TICEntry& output, const Passes& passes) {
    if (passes.standalone) {
        if (!ValidInput(input, passes) ||
            !StarFoxColor(VideoCommon::TextureViewHeader(output, true), passes.output_format,
                          Tegra::Texture::ComponentType::FLOAT, false,
                          Tegra::Texture::SwizzleSource::OneFloat) ||
            input.Address() == output.Address() || output.Width() != 2560 ||
            output.Height() != 1440 || input.Width() >= output.Width() ||
            input.Height() >= output.Height() || input.Width() * 4 < output.Width() ||
            input.Height() * 4 < output.Height()) {
            return false;
        }
        // The captured gameplay source changes with dynamic resolution; allow its aspect
        // rounding while keeping the captured output and strictly bounded upscale ratio.
        const u64 from = u64{input.Width()} * output.Height();
        const u64 to = u64{input.Height()} * output.Width();
        return (from > to ? from - to : to - from) <= to / 32;
    }
    if (passes.skip_network) {
        return ValidInput(input, passes) &&
               StarFoxColor(VideoCommon::TextureViewHeader(output, true), passes.output_format,
                            Tegra::Texture::ComponentType::UNORM,
                            false) &&
               input.Address() != output.Address() && output.Width() == 2560 &&
               output.Height() == 1440;
    }
    return input.format == passes.input_format && output.format == passes.output_format &&
           ValidPair(input, output, passes.spatial_only);
}

// A successful substitute dispatch, not a shader rejection. Never cache it as a failure.
struct DispatchReplaced {};

} // namespace VideoCommon::FireEmblemUpscale
