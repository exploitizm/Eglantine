// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <optional>
#include <utility>
#include "video_core/surface.h"

namespace VideoCommon {
// Fermi2D has no E5B9G9R9 render-target format. A scoped override preserves the actual
// sampled TIC format when reusing the spatial upscale interface.
inline thread_local std::optional<VideoCore::Surface::PixelFormat> BlitSourceFormat;

class ScopedBlitSourceFormat {
public:
    explicit ScopedBlitSourceFormat(VideoCore::Surface::PixelFormat format)
        : previous{std::exchange(BlitSourceFormat, format)} {}
    ~ScopedBlitSourceFormat() {
        BlitSourceFormat = previous;
    }
    ScopedBlitSourceFormat(const ScopedBlitSourceFormat&) = delete;
    ScopedBlitSourceFormat& operator=(const ScopedBlitSourceFormat&) = delete;

private:
    std::optional<VideoCore::Surface::PixelFormat> previous;
};
} // namespace VideoCommon
