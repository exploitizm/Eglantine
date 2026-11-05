// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "video_core/textures/texture.h"

namespace VideoCommon {
using Tegra::Texture::TICEntry;

// Native descriptors address the selected view directly. The legacy NVN
// layer-base bookkeeping is not an additional array prefix for this ABI.
inline TICEntry TextureViewHeader(TICEntry entry, bool native) {
    if (native) {
        entry.layer_base_0_2.Assign(0);
        entry.layer_base_3_7.Assign(0);
        entry.layer_base_8_10.Assign(0);
    }
    return entry;
}

} // namespace VideoCommon
