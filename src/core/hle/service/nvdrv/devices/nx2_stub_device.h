// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "core/hle/service/nvdrv/devices/nvdevice.h"
#include "core/hle/service/nvdrv/devices/nx2_gpu_mapping_table.h"

namespace Service::Nvidia::Devices {

// Set once a Switch 2 NvRm has identified itself (0xD0-byte GetCharacteristics). Devices then
// answer ioctls they don't know with a logged, input-echoing success instead of NotImplemented.
inline std::atomic<bool> g_nx2_driver{false};

inline NvResult Nx2UnknownIoctl(const char* device, Ioctl command, std::span<const u8> input,
                                std::span<u8> output) {
    LOG_WARNING(Service_NVDRV, "(STUBBED) Switch 2 {} ioctl={:08X} group={:01X} command={:02X} in={:#x}",
                device, command.raw, command.group, command.cmd, input.size());
    std::fill(output.begin(), output.end(), u8{0});
    std::copy_n(input.begin(), std::min(input.size(), output.size()), output.begin());
    return NvResult::Success;
}

// GPU mappings a Switch 2 process made through nx2:gpu: CreateGpuAccessibleMemory (sub 101) maps
// the memory at GPU address == CPU address, and NormalMapGpuMemory4K (211) maps a memory handle at
// a chosen GPU address (also in sparse regions). Every nvhost-as-gpu address space applies each
// mapping, including ones made before the address space existed.
class Nx2GpuMemoryRegistry {
public:
    using Listener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;
    using UnmapListener = std::function<void(u64 gpu_address, u64 size)>;
    using ReleaseListener = std::function<void(u64 cpu_address, u64 size)>;
    /// A sparse region is created (true) or destroyed (false).
    using SparseListener = std::function<void(u64 gpu_address, u64 size, bool create)>;
    /// A block of a sparse region gets memory [cpu_address, +size).
    using BackListener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;
    using ReserveListener = std::function<void(u64 gpu_address, u64 size)>;
    /// A tile of a sparse region shows process memory [cpu_address, +size), or reads zero again
    /// (cpu_address 0).
    using TileListener = std::function<void(u64 gpu_address, u64 cpu_address, u64 size)>;

    static Nx2GpuMemoryRegistry& Get() {
        static Nx2GpuMemoryRegistry registry;
        return registry;
    }

    // The listeners (nvhost_as_gpu) queue work for the GPU thread, which waits for room in a full
    // queue, and the GPU thread itself calls BackSparse. So every call works out the listener calls
    // a change needs under `mutex` and makes them after it is released: holding it across them
    // stopped Mario Kart World at its first load screen (a CPU core waiting for the queue under
    // it, the GPU thread waiting for it to back a sparse block). Calls from the game's side hold
    // `ordered` until their listener calls are made, so the address spaces see the changes in the
    // order they were made; BackSparse, which the GPU thread makes, does not wait for it.

    void Register(u64 address, u64 size) {
        MapAt(address, address, size);
    }

    /// The process address one mapping gives all of GPU range [gpu_address, +size), as the game
    /// last set it. The address spaces get each change later, from the GPU thread, so a submit
    /// made right after the game maps its command memory cannot read the words through them yet.
    std::optional<u64> TranslateRange(u64 gpu_address, u64 size) {
        std::scoped_lock lk{mutex};
        const auto first = mappings.Translate(gpu_address);
        if (!first || size == 0) {
            return first;
        }
        const auto last = mappings.Translate(gpu_address + size - 1);
        if (!last || *last != *first + size - 1) {
            return std::nullopt;
        }
        return first;
    }

    void Unregister(u64 address, u64 size) {
        Ordered([&](Calls& todo) {
            if (!Nx2GpuMappingTable::Valid(address, size)) {
                return;
            }
            pending_releases.try_emplace(std::pair{address, size});
            Apply(mappings.Unmap(address, size), todo);
        });
    }

    void MapAt(u64 gpu_address, u64 cpu_address, u64 size) {
        Ordered([&](Calls& todo) {
            const auto update = mappings.Map(gpu_address, cpu_address, size);
            if (!update.success) {
                LOG_ERROR(Service_NVDRV, "Conflicting or invalid nx2 GPU mapping {:#x}+{:#x}",
                          gpu_address, size);
                return;
            }
            Apply(update, todo);
        });
    }

    std::optional<u64> AcquireNative(u64 address, u64 size) {
        std::optional<u64> lease;
        Ordered([&](Calls& todo) {
            if (!Nx2GpuMappingTable::Valid(address, size) || SparseOverlaps(address, size)) {
                return;
            }
            const auto update = mappings.Acquire(address, size);
            if (!update.success) {
                return;
            }
            Apply(update, todo);
            lease = update.lease;
        });
        return lease;
    }

    void ReleaseNative(u64 lease) {
        Ordered([&](Calls& todo) { Apply(mappings.Release(lease), todo); });
    }

    // NormalUnmapGpuMemory4K, or destroying the memory: the GPU range stops reaching it.
    void Unmap(u64 gpu_address, u64 size) {
        Ordered([&](Calls& todo) { Apply(mappings.Unmap(gpu_address, size), todo); });
    }

    // Backing memory [cpu_address, +size) is about to be freed (its GPU mappings are gone): each
    // address space drops its SMMU mapping of it, in GPU-thread order.
    void ReleaseBacking(u64 cpu_address, u64 size, std::function<void()> retired = {}) {
        Ordered([&](Calls& todo) {
            auto& callbacks = pending_releases[{cpu_address, size}];
            if (retired) {
                callbacks.push_back(std::move(retired));
            }
            DrainReleases(todo);
        });
    }

    // Sparse regions (nx2:gpu 214 with the sparse flag): the GPU range reaches device memory of its
    // own from the start, and each block gets memory when the GPU first writes it (BackSparse).
    // The GPU mapping never changes, so the texture cache doesn't rebuild the images over it as
    // blocks get memory; unbacked blocks read zero.
    bool CreateSparse(u64 gpu_address, u64 size) {
        bool created = false;
        Ordered([&](Calls& todo) {
            if (!Nx2GpuMappingTable::Valid(gpu_address, size) ||
                mappings.HasNativeOverlap(gpu_address, size) || SparseOverlaps(gpu_address, size)) {
                LOG_ERROR(Service_NVDRV, "Conflicting or invalid nx2 sparse mapping {:#x}+{:#x}",
                          gpu_address, size);
                return;
            }
            sparse_regions.emplace(gpu_address, SparseRegion{size, {}});
            for (auto& [owner, listener] : listeners) {
                if (listener.sparse) {
                    todo.push_back([sparse = listener.sparse, gpu_address, size] {
                        sparse(gpu_address, size, true);
                    });
                    MapOverlays(listener, gpu_address, size, todo);
                }
            }
            created = true;
        });
        return created;
    }

    bool DestroySparse(u64 gpu_address, u64 size) {
        bool destroyed = false;
        Ordered([&](Calls& todo) {
            const auto it = sparse_regions.find(gpu_address);
            if (it == sparse_regions.end() || it->second.size != size) {
                LOG_ERROR(Service_NVDRV, "Unknown nx2 sparse region {:#x}+{:#x}", gpu_address, size);
                return;
            }
            sparse_regions.erase(it);
            for (auto& [owner, listener] : listeners) {
                if (listener.sparse) {
                    todo.push_back([sparse = listener.sparse, gpu_address, size] {
                        sparse(gpu_address, size, false);
                    });
                }
            }
            Apply(mappings.Unmap(gpu_address, size), todo);
            destroyed = true;
        });
        return destroyed;
    }

    // Usually from the GPU thread (BackSparseWrite): not ordered, see above.
    void BackSparse(u64 gpu_address, u64 cpu_address, u64 size) {
        std::shared_lock running{calls};
        Calls todo;
        {
            std::scoped_lock lk{mutex};
            auto it = sparse_regions.upper_bound(gpu_address);
            if (!Nx2GpuMappingTable::Valid(gpu_address, size) ||
                !Nx2GpuMappingTable::Valid(cpu_address, size) || it == sparse_regions.begin()) {
                LOG_ERROR(Service_NVDRV, "Invalid nx2 sparse backing {:#x}+{:#x}", gpu_address, size);
                return;
            }
            --it;
            auto& [start, region] = *it;
            if (gpu_address - start >= region.size || size > region.size - (gpu_address - start)) {
                LOG_ERROR(Service_NVDRV, "nx2 sparse backing {:#x}+{:#x} outside its region",
                          gpu_address, size);
                return;
            }
            region.blocks.insert_or_assign(gpu_address, SparseBacking{cpu_address, size});
            for (auto& [owner, listener] : listeners) {
                if (listener.back) {
                    todo.push_back([back = listener.back, gpu_address, cpu_address, size] {
                        back(gpu_address, cpu_address, size);
                    });
                }
            }
        }
        Run(todo);
    }

    // nx2:gpu 216 (sparse map requests): the tile [gpu_address, +size) of a sparse region now
    // shows the process memory at cpu_address, the game's own memory (sub 101), or with
    // cpu_address 0 it is unmapped and reads zero. The GPU mapping stays as it is; only the
    // region's device pages change, in GPU-thread order.
    void MapSparseTile(u64 gpu_address, u64 cpu_address, u64 size) {
        Ordered([&](Calls& todo) {
            auto it = sparse_regions.upper_bound(gpu_address);
            if (!Nx2GpuMappingTable::Valid(gpu_address, size) ||
                (cpu_address != 0 && !Nx2GpuMappingTable::Valid(cpu_address, size)) ||
                it == sparse_regions.begin()) return;
            --it;
            auto& [start, region] = *it;
            if (gpu_address - start >= region.size || size > region.size - (gpu_address - start))
                return;
            if (cpu_address != 0) region.tiles.Map(gpu_address, cpu_address, size);
            else region.tiles.Unmap(gpu_address, size);
            // Do not replay a retired backing into a newly opened address space. Keep only
            // pieces outside this remap, then replay the tile overlays below.
            const u64 end = gpu_address + size;
            std::map<u64, SparseBacking> kept;
            for (auto block = region.blocks.begin(); block != region.blocks.end();) {
                const u64 first = block->first;
                const auto backing = block->second;
                const u64 last = first + backing.size;
                if (last <= gpu_address || first >= end) { ++block; continue; }
                block = region.blocks.erase(block);
                if (first < gpu_address) kept.emplace(first, SparseBacking{backing.cpu, gpu_address - first});
                if (last > end) kept.emplace(end, SparseBacking{backing.cpu + end - first, last - end});
            }
            region.blocks.merge(kept);
            for (auto& [owner, listener] : listeners) {
                if (listener.tile) {
                    todo.push_back([tile = listener.tile, gpu_address, cpu_address, size] {
                        tile(gpu_address, cpu_address, size);
                    });
                }
            }
        });
    }

    void AddListener(const void* owner, Listener listener, UnmapListener unmap_listener,
                     ReleaseListener release_listener = {}, SparseListener sparse_listener = {},
                     BackListener back_listener = {}, ReserveListener reserve_listener = {},
                     TileListener tile_listener = {}) {
        Ordered([&](Calls& todo) {
            const auto& added =
                listeners
                    .insert_or_assign(owner, Listeners{std::move(listener), std::move(unmap_listener),
                                                       std::move(release_listener),
                                                       std::move(sparse_listener),
                                                       std::move(back_listener),
                                                       std::move(reserve_listener), std::move(tile_listener)})
                    .first->second;
            if (added.sparse) {
                for (const auto& [gpu_address, region] : sparse_regions) {
                    todo.push_back([sparse = added.sparse, gpu = gpu_address, size = region.size] {
                        sparse(gpu, size, true);
                    });
                    if (added.back) {
                        for (const auto& [block, backing] : region.blocks) {
                            todo.push_back([back = added.back, gpu = block, cpu = backing.cpu,
                                            bytes = backing.size] {
                                back(gpu, cpu, bytes);
                            });
                        }
                    }
                }
            }
            if (added.tile) {
                for (const auto& [gpu, region] : sparse_regions) {
                    for (const auto& tile : region.tiles.Effective()) {
                        todo.push_back([call = added.tile, tile] {
                            call(tile.gpu, tile.cpu, tile.size);
                        });
                    }
                }
            }
            for (const auto& m : mappings.Effective()) {
                todo.push_back([map = added.map, m] { map(m.gpu, m.cpu, m.size); });
            }
        });
    }

    void RemoveListener(const void* owner) {
        {
            std::scoped_lock lk{mutex};
            listeners.erase(owner);
        }
        // Calls that took the owner's listener before it went: let them finish.
        std::unique_lock done{calls};
    }

    // A new emulation session starts: forget the previous game's mappings and device ranges (its
    // address spaces removed their listeners when they closed).
    void Reset() {
        {
            std::scoped_lock lk{device_mutex};
            device_ranges.clear();
        }
        std::scoped_lock order{ordered};
        std::scoped_lock lk{mutex};
        mappings.Reset();
        pending_releases.clear();
        sparse_regions.clear();
    }

    // Device (SMMU) address of a CPU range an address space mapped; used to present images.
    void RecordDeviceRange(u64 cpu_address, u64 size, u64 device_address) {
        std::scoped_lock lk{device_mutex};
        device_ranges[cpu_address] = {size, device_address};
    }

    std::optional<u64> DeviceAddressOf(u64 cpu_address) {
        std::scoped_lock lk{device_mutex};
        auto it = device_ranges.upper_bound(cpu_address);
        if (it == device_ranges.begin()) {
            return std::nullopt;
        }
        --it;
        const auto& [start, range] = *it;
        if (cpu_address - start >= range.first) {
            return std::nullopt;
        }
        return range.second + (cpu_address - start);
    }

private:
    /// Listener calls a change needs, made once `mutex` is released.
    using Calls = std::vector<std::function<void()>>;

    // Separate lock: listeners record ranges (RecordDeviceRange) while other calls hold `mutex`.
    std::mutex device_mutex;
    std::map<u64, std::pair<u64, u64>> device_ranges; // cpu start -> (size, device address)

    struct Listeners {
        Listener map;
        UnmapListener unmap;
        ReleaseListener release;
        SparseListener sparse;
        BackListener back;
        ReserveListener reserve;
        TileListener tile;
    };

    struct SparseBacking {
        u64 cpu;
        u64 size;
    };

    struct SparseRegion {
        u64 size;
        std::map<u64, SparseBacking> blocks;
        Nx2GpuMappingTable tiles;
    };

    // A change from the game's side: `change` updates the state under `mutex` and adds the listener
    // calls it needs, which are made after `mutex`, in the order of the changes (`ordered`).
    template <class Change>
    void Ordered(Change&& change) {
        std::scoped_lock order{ordered};
        std::shared_lock running{calls};
        Calls todo;
        {
            std::scoped_lock lk{mutex};
            change(todo);
        }
        Run(todo);
    }

    static void Run(Calls& todo) {
        for (auto& call : todo) {
            call();
        }
    }

    bool SparseOverlaps(u64 gpu_address, u64 size) const {
        return std::ranges::any_of(sparse_regions, [&](const auto& entry) {
            const auto& [start, region] = entry;
            return gpu_address < start + region.size && start < gpu_address + size;
        });
    }

    // Needs `mutex`.
    void MapOverlays(const Listeners& listener, u64 gpu_address, u64 size, Calls& todo) {
        for (const auto& m : mappings.Effective()) {
            const u64 first = (std::max)(m.gpu, gpu_address);
            const u64 last = (std::min)(m.gpu + m.size, gpu_address + size);
            if (first < last) {
                todo.push_back([map = listener.map, first, cpu = m.cpu + (first - m.gpu), last] {
                    map(first, cpu, last - first);
                });
            }
        }
    }

    // Needs `mutex`.
    void Uncover(const Listeners& listener, u64 gpu_address, u64 size, Calls& todo) {
        const auto unmap = [&](u64 address, u64 bytes) {
            todo.push_back([call = listener.unmap, address, bytes] { call(address, bytes); });
        };
        if (!listener.reserve) {
            unmap(gpu_address, size);
            return;
        }
        const u64 end = gpu_address + size;
        u64 address = gpu_address;
        for (const auto& [start, region] : sparse_regions) {
            const u64 first = (std::max)(start, address);
            const u64 last = (std::min)(start + region.size, end);
            if (first >= last) {
                continue;
            }
            if (address < first) {
                unmap(address, first - address);
            }
            todo.push_back([reserve = listener.reserve, first, last] { reserve(first, last - first); });
            address = last;
        }
        if (address < end) {
            unmap(address, end - address);
        }
    }

    // Needs `mutex`.
    void Apply(const Nx2GpuMappingTable::Update& update, Calls& todo) {
        for (const auto& change : update.changes) {
            const auto& range = change.range;
            if (change.map && change.native) {
                pending_releases.try_emplace({range.cpu, range.size});
            }
            for (auto& [owner, listener] : listeners) {
                if (change.map) {
                    todo.push_back([map = listener.map, range] { map(range.gpu, range.cpu, range.size); });
                } else {
                    Uncover(listener, range.gpu, range.size, todo);
                }
            }
        }
        DrainReleases(todo);
    }

    // Needs `mutex`.
    void DrainReleases(Calls& todo) {
        std::erase_if(pending_releases, [&](auto& pending) {
            auto& [range, retired] = pending;
            const auto [cpu, size] = range;
            if (mappings.UsesBacking(cpu, size)) {
                return false;
            }
            {
                std::scoped_lock lk{device_mutex};
                std::erase_if(device_ranges, [&](const auto& entry) {
                    const auto& [start, backing] = entry;
                    return start < cpu + size && cpu < start + backing.first &&
                           !mappings.UsesBacking(start, backing.first);
                });
            }
            for (auto& [owner, listener] : listeners) {
                if (listener.release) {
                    todo.push_back([release = listener.release, address = cpu, bytes = size] {
                        release(address, bytes);
                    });
                }
            }
            for (auto& callback : retired) {
                todo.push_back(std::move(callback));
            }
            return true;
        });
    }

    std::mutex ordered; ///< Held by calls from the game's side until their listener calls are made
    std::shared_mutex calls; ///< Shared while listener calls run; RemoveListener waits for them
    std::mutex mutex;
    Nx2GpuMappingTable mappings;
    std::map<std::pair<u64, u64>, std::vector<std::function<void()>>> pending_releases;
    std::map<u64, SparseRegion> sparse_regions; ///< By GPU address
    std::map<const void*, Listeners> listeners;
};

// Device node that Switch 2 nvgpu drivers open but Switch 1 never had (e.g. /dev/nvhost-tsg-gpu).
// Every ioctl succeeds with a zeroed reply and is logged, so the driver's use of it shows up.
class nx2_stub_device final : public nvdevice {
public:
    explicit nx2_stub_device(Core::System& system_, std::string name_)
        : nvdevice{system_}, name{std::move(name_)} {}

    NvResult Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<u8> output) override {
        return Stub(fd, command, input, output, {});
    }
    NvResult Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input,
                    std::span<const u8> inline_input, std::span<u8> output) override {
        return Stub(fd, command, input, output, {});
    }
    NvResult Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                    std::span<u8> inline_output) override {
        return Stub(fd, command, input, output, inline_output);
    }

    void OnOpen(NvCore::SessionId session_id, DeviceFD fd) override {
        LOG_WARNING(Service_NVDRV, "(STUBBED) {} opened as fd {}", name, fd);
    }
    void OnClose(DeviceFD fd) override {}

private:
    NvResult Stub(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                  std::span<u8> inline_output) {
        LOG_WARNING(Service_NVDRV, "(STUBBED) {} fd={} ioctl={:08X} group={:01X} command={:02X} in={:#x}",
                    name, fd, command.raw, command.group, command.cmd, input.size());
        // Echo the input (in/out ioctls keep their fields), zero the rest.
        std::fill(output.begin(), output.end(), u8{0});
        std::copy_n(input.begin(), std::min(input.size(), output.size()), output.begin());
        std::fill(inline_output.begin(), inline_output.end(), u8{0});
        if (name == "/dev/nvhost-tsg-gpu") {
            // nvgpu TSG: subcontext create hands out a veid; bind/unbind/delete succeed; the SDK
            // probes the rest and must see them unsupported.
            if (command.raw == 0xc0105409 && output.size() >= 12) {
                const u32 veid = static_cast<u32>(std::countr_one(veid_mask));
                veid_mask |= u64{1} << (veid & 63);
                std::memcpy(output.data() + 8, &veid, sizeof(veid));
                return NvResult::Success;
            }
            if (command.raw == 0xc004540a && input.size() >= 4) {
                u32 veid{};
                std::memcpy(&veid, input.data(), sizeof(veid));
                veid_mask &= ~(u64{1} << (veid & 63));
                return NvResult::Success;
            }
            if (command.raw != 0xc0045401 && command.raw != 0xc0045402 &&
                command.raw != 0xc018540b) {
                return NvResult::NotImplemented;
            }
        }
        return NvResult::Success;
    }

    std::string name;
    u64 veid_mask{};
};

} // namespace Service::Nvidia::Devices
