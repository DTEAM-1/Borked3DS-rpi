// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include "common/assert.h"
#include "common/bit_set.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "video_core/pica/regs_shader.h"
#include "video_core/pica/shader_setup.h"

namespace Pica {

ShaderSetup::ShaderSetup() = default;

ShaderSetup::~ShaderSetup() = default;

void ShaderSetup::WriteUniformBoolReg(u32 value) {
    const auto bits = BitSet32(value);
    for (u32 i = 0; i < uniforms.b.size(); ++i) {
        uniforms.b[i] = bits[i];
    }
}

void ShaderSetup::WriteUniformIntReg(u32 index, const Common::Vec4<u8> values) {
    ASSERT(index < uniforms.i.size());
    uniforms.i[index] = values;
}

std::optional<u32> ShaderSetup::WriteUniformFloatReg(ShaderRegs& config, u32 value) {
    auto& uniform_setup = config.uniform_setup;
    const bool is_float32 = uniform_setup.IsFloat32();
    if (!uniform_queue.Push(value, is_float32)) {
        return std::nullopt;
    }

    const auto uniform = uniform_queue.Get(is_float32);
    if (uniform_setup.index >= uniforms.f.size()) {
        LOG_ERROR(HW_GPU, "Invalid float uniform index {}", uniform_setup.index.Value());
        return std::nullopt;
    }

    const u32 index = uniform_setup.index.Value();
    uniforms.f[index] = uniform;
    uniform_setup.index.Assign(index + 1);
    return index;
}

namespace {
std::atomic<u64> g_v396_hash_calls{0};
std::atomic<u64> g_v396_hash_ns{0};

template <typename F>
void V396TimedHash(F&& f) {
    if (!V396ShaderStatsEnabled()) {
        f();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    g_v396_hash_calls.fetch_add(1, std::memory_order_relaxed);
    g_v396_hash_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
        std::memory_order_relaxed);
}
} // namespace

bool V396ShaderStatsEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_V396_SHADER_STATS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

void V396ShaderStatsTake(u64& hash_calls, u64& hash_ns) {
    hash_calls = g_v396_hash_calls.exchange(0, std::memory_order_relaxed);
    hash_ns = g_v396_hash_ns.exchange(0, std::memory_order_relaxed);
}

u64 ShaderSetup::GetProgramCodeHash() {
    if (program_code_hash_dirty) {
        V396TimedHash([&] {
            program_code_hash = Common::ComputeHash64(&program_code, sizeof(program_code));
        });
        program_code_hash_dirty = false;
    }
    return program_code_hash;
}

u64 ShaderSetup::GetSwizzleDataHash() {
    if (swizzle_data_hash_dirty) {
        V396TimedHash([&] {
            swizzle_data_hash = Common::ComputeHash64(&swizzle_data, sizeof(swizzle_data));
        });
        swizzle_data_hash_dirty = false;
    }
    return swizzle_data_hash;
}

} // namespace Pica
