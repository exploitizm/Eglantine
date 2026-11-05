// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>

#include "common/string_util.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"

namespace Service::Audio {

// Switch 2 virtual output devices (nn::audio::ListVirtualOutputDevices and friends), reached through
// the object auddmg:u command 0 returns. Command layout from Lichii (zortos293/Lichii, the same
// service in nx2_audio_device_manager.h). Ocarina of Time opens it ~12 s into boot; without it the
// stub session's empty reply left the sdk a null proxy and sead::MainThread called through it.
// Eden plays everything through one stereo 48 kHz output, so there is one virtual device. Streams
// are not routed through it here (no sysmix): attach and detach only succeed.
// - 0(u32): u32 count, the device names in the out buffer (0x100 bytes each).
// - 100(u32): the default device's name in the out buffer.
// - 300(16-byte stream format, name buffer): whether the format is supported (result only).
// - 401(u32, name buffer): the device's 0x228-byte property block in the out buffer.
// - 500(16 bytes, name buffer): an event signalled when the device's state changes.
// - 700(u32 stream id, u32 device functionality, name buffer) / 800(u32 stream id): attach / detach.
class INx2AudioVirtualDeviceManager final : public ServiceFramework<INx2AudioVirtualDeviceManager> {
public:
    explicit INx2AudioVirtualDeviceManager(Core::System& system_)
        : ServiceFramework{system_, "nx2:auddmg:u:manager"},
          service_context{system_, "nx2:auddmg:u:manager"},
          state_event{service_context.CreateEvent("nx2:auddmg:u:StateEvent")} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &INx2AudioVirtualDeviceManager::ListVirtualOutputDevices, "ListVirtualOutputDevices"},
            {100, &INx2AudioVirtualDeviceManager::GetDefaultVirtualOutputDevice, "GetDefaultVirtualOutputDevice"},
            {300, &INx2AudioVirtualDeviceManager::IsStreamFormatSupported, "IsVirtualOutputDeviceStreamFormatSupported"},
            {401, &INx2AudioVirtualDeviceManager::GetVirtualOutputDeviceProperties, "GetVirtualOutputDeviceProperties"},
            {500, &INx2AudioVirtualDeviceManager::AcquireStateChangeEvent, "AcquireVirtualOutputDeviceStateEvent (guessed)"},
            {700, &INx2AudioVirtualDeviceManager::AttachStream, "AttachStream"},
            {800, &INx2AudioVirtualDeviceManager::DetachStream, "DetachStream"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

    ~INx2AudioVirtualDeviceManager() override {
        service_context.CloseEvent(state_event);
    }

private:
    static constexpr std::size_t NameSize = 0x100;
    static constexpr std::string_view DeviceName = "AudioVirtualOutput0";
    static constexpr std::string_view PhysicalDeviceName = "AudioTvOutput";

    static void CopyName(u8* out, std::string_view name) {
        std::memcpy(out, name.data(), std::min(name.size(), NameSize - 1));
    }

    void ListVirtualOutputDevices(HLERequestContext& ctx) {
        std::array<u8, NameSize> names{};
        CopyName(names.data(), DeviceName);
        const u32 count = ctx.GetWriteBufferSize() >= NameSize ? 1 : 0;
        if (count) {
            ctx.WriteBuffer(names);
        }
        LOG_DEBUG(Service_Audio, "called, {} device", count);
        IPC::ResponseBuilder rb{ctx, 3};
        rb.Push(ResultSuccess);
        rb.Push(count);
    }

    void GetDefaultVirtualOutputDevice(HLERequestContext& ctx) {
        std::array<u8, NameSize> name{};
        CopyName(name.data(), DeviceName);
        ctx.WriteBuffer(name);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    // Stereo or 5.1 PCM16 / float at 48 kHz, like the host output (functionality 1: playback).
    void IsStreamFormatSupported(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto type = rp.Pop<u32>();
        const auto channels = rp.Pop<u32>();
        const auto rate = rp.Pop<u32>();
        const auto functionality = rp.Pop<u32>();
        const bool supported = functionality == 1 && rate == 48000 &&
                               (channels == 1 || channels == 2 || channels == 6);
        LOG_DEBUG(Service_Audio, "format support: sample type {}, {} channels, {} Hz, functionality {}: {}",
                  type, channels, rate, functionality, supported);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(supported ? ResultSuccess : ResultNotSupported);
    }

    // Property block: name (0x000), u32 type (0x100, 3), u32 (0x104, 2), u32 channel count
    // (0x108), u32 sample rate (0x10c), physical output name (0x110), its id (0x210), u64 latency
    // in ns (0x218), u32 device id (0x220, never 0: the sdk's streams refuse it).
    void GetVirtualOutputDeviceProperties(HLERequestContext& ctx) {
        std::array<u8, 0x228> block{};
        const auto name = ctx.ReadBuffer(0);
        std::memcpy(block.data(), name.data(), std::min(name.size(), NameSize - 1));
        const auto put32 = [&](std::size_t offset, u32 value) {
            std::memcpy(block.data() + offset, &value, sizeof(value));
        };
        put32(0x100, 3);
        put32(0x104, 2);
        put32(0x108, 2);     // stereo
        put32(0x10c, 48000); // Hz
        CopyName(block.data() + 0x110, PhysicalDeviceName);
        const u64 latency_ns = 10'000'000;
        std::memcpy(block.data() + 0x218, &latency_ns, sizeof(latency_ns));
        put32(0x220, 1);
        ctx.WriteBuffer(block);
        LOG_DEBUG(Service_Audio, "called for {}", Common::StringFromBuffer(name));
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    void AcquireStateChangeEvent(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 1};
        rb.Push(ResultSuccess);
        rb.PushCopyObjects(ctx, state_event->GetReadableEvent());
    }

    void AttachStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto stream_id = rp.Pop<u32>();
        const auto functionality = rp.Pop<u32>();
        LOG_WARNING(Service_Audio, "(STUBBED) attach stream {} to {} (functionality {})", stream_id,
                    Common::StringFromBuffer(ctx.ReadBuffer(0)), functionality);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(functionality == 1 ? ResultSuccess : ResultNotSupported);
    }

    void DetachStream(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const auto stream_id = rp.Pop<u32>();
        LOG_WARNING(Service_Audio, "(STUBBED) detach stream {}", stream_id);
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
    }

    KernelHelpers::ServiceContext service_context;
    Kernel::KEvent* state_event;
};

// Switch 2 "auddmg:u": command 0 (16 bytes and the pid) returns the virtual device manager.
class INx2AudioDeviceManagerService final : public ServiceFramework<INx2AudioDeviceManagerService> {
public:
    explicit INx2AudioDeviceManagerService(Core::System& system_)
        : ServiceFramework{system_, "nx2:auddmg:u"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &INx2AudioDeviceManagerService::OpenVirtualDeviceManager, "OpenVirtualDeviceManager (guessed)"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    void OpenVirtualDeviceManager(HLERequestContext& ctx) {
        LOG_DEBUG(Service_Audio, "called");
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<INx2AudioVirtualDeviceManager>(ctx, system);
    }
};

} // namespace Service::Audio
