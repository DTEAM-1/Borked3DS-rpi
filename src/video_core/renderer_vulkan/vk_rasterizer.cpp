// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "common/literals.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

std::atomic<u64> g_vk_draw_counter{0};
std::atomic<u64> g_vk_accel_draw_counter{0};

namespace {

using TriangleTopology = Pica::PipelineRegs::TriangleTopology;
using VideoCore::SurfaceType;

using namespace Common::Literals;
using namespace Pica::Shader::Generator;

constexpr u64 STREAM_BUFFER_SIZE = 64_MiB;
constexpr u64 UNIFORM_BUFFER_SIZE = 4_MiB;
constexpr u64 TEXTURE_BUFFER_SIZE = 2_MiB;

constexpr vk::BufferUsageFlags BUFFER_USAGE =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer;

struct DrawParams {
    u32 vertex_count;
    s32 vertex_offset;
    u32 binding_count;
    std::array<u32, 16> bindings;
    bool is_indexed;
};

[[nodiscard]] u64 TextureBufferSize(const Instance& instance) {
    // Use the smallest texel size from the texel views which corresponds to eR32G32Sfloat.
    const u64 max_size = instance.MaxTexelBufferElements() * 8;
    return std::min(max_size, TEXTURE_BUFFER_SIZE);
}

[[nodiscard]] bool IsValidImageView(const vk::ImageView view) {
    return static_cast<bool>(view);
}

[[nodiscard]] bool IsEnvEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsDrawTraceEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_TRACE_DRAW");
}

[[nodiscard]] bool IsForceQuietDisplayEnabled() {
    // v115-D-MUX rollback: present tracing is now quiet, but AccelerateDisplay still floods TRACE_DRAW.
    // This suppresses display-path TRACE_DRAW while keeping PICA/backend TRACE_DRAW and
    // TRACE_ACCEL_STAGE visible for the current GenerateVertexShader gate.
    return IsEnvEnabled("BORKED3DS_V3DV_FORCE_QUIET_DISPLAY");
}

[[nodiscard]] bool IsStrictCompatEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_STRICT_COMPAT");
}

[[nodiscard]] bool IsForceAccelStageTraceEnabled() {
    // v100/v114 diagnostic:
    // Keep forced stage tracing available. v100 still stopped after the PICA pre_call before
    // the raw-enter marker, so v114 adds an even earlier no-argument entry-only probe.
    // This does not execute any extra Vulkan work by itself.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE");
}

[[nodiscard]] bool IsAccelEntryOnlyProbeEnabled() {
    // v114 diagnostic:
    // Keep entry-only mode available for fallback diagnostics, but normal v114 testing must
    // leave it disabled so the silent call-boundary probe can isolate the crash.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE");
}

[[nodiscard]] bool IsAccelSilentEntryReturnEnabled() {
    // v114 diagnostic fallback:
    // v109 proved the silent pica_core -> RasterizerVulkan::AccelerateDrawBatch call boundary
    // survives until hotkey exit. Keep this as a rollback switch, but normal v114 tests must
    // disable it so the first backend marker can be re-enabled.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_ACCEL_SILENT_ENTRY_RETURN");
}

[[nodiscard]] bool IsAccelRawEnterReturnEnabled() {
    // v114 diagnostic fallback:
    // v110 proved raw_enter_noargs is safe. Keep this as a rollback switch, but normal v114
    // testing disables it so raw_enter_simple can be reached next.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_ACCEL_RAW_ENTER_RETURN");
}

[[nodiscard]] bool IsAccelRawEnterSimpleReturnEnabled() {
    // v114 diagnostic:
    // Emit raw_enter_noargs and raw_enter_simple, then return true before stage=1.
    // This verifies the minimal backend metadata path (accel_id, indexed, env flags) without
    // entering stage logging, shader setup, SPIR-V, pipeline, descriptors, Draw(), or vkCmdDraw.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN");
}

[[nodiscard]] bool IsV114ShaderMultiplexEntrySafeEnabled() {
    // v114-C/v114-C2 corrective probe:
    // A/B passed, but the first C attempt cut the log immediately after the safe micro-HW
    // candidate while starting a Render.Vulkan line, before any completed TRACE_ACCEL_STAGE
    // marker. Keep the shader probe selected, but suppress early backend entry logs and avoid
    // probe-helper evaluation before SetupVertexShader. This isolates whether C is really
    // blocked by GLSL::GenerateVertexShader() or by the entry logging path.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_ENTRY_SAFE");
}

[[nodiscard]] bool IsV114ShaderMultiplexSilentStagesEnabled() {
    // v114-C2 corrective probe:
    // The entry-safe retry proved the rebuilt v114-C2 marker reaches the first safe micro-HW
    // candidate, but the log still cut at the first backend Render.Vulkan line. Keep the
    // generate-guarded shader probe selected, but silence the stage 1..6 TRACE_ACCEL_STAGE
    // logs. Those stages were already proven by v113/v114-A/v114-B; this retest isolates
    // SetupVertexShader + GLSL::GenerateVertexShader() without another noisy entry/stage log.
    return IsV114ShaderMultiplexEntrySafeEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_SILENT_STAGES");
}

[[nodiscard]] bool IsV114ShaderMultiplexFileTraceEnabled() {
    // v114-C3 corrective probe:
    // v114-C/C2 can cut the normal log when the backend shader probe should start.
    // Keep the generate-guarded shader path, but disable TRACE_ACCEL_STAGE logging in the
    // backend and write tiny breadcrumbs to a sidecar file instead.
    return IsV114ShaderMultiplexSilentStagesEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE");
}

void V114ShaderMultiplexFileTraceRaw(const char* message) {
    if (!IsV114ShaderMultiplexFileTraceEnabled()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_shader_probe.log", "a");
    if (fp == nullptr) {
        return;
    }
    std::fputs(message, fp);
    std::fputc('\n', fp);
    std::fclose(fp);
}

void V114ShaderMultiplexFileTraceReset() {
    if (!IsV114ShaderMultiplexFileTraceEnabled()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_shader_probe.log", "w");
    if (fp == nullptr) {
        return;
    }
    std::fputs("v115d_mux file_trace_reset\n", fp);
    std::fputs("v115d_a7x shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7y shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z15 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z16 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z17 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z18 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z23 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z23b shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z24 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z25 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26b shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26c shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26d shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26e shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26f shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26g shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26h shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26i shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26j shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26k shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26l shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp2 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3b shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3c shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3d shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3e shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3f shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3g shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3i shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3j shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3k shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3l shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z26mp3m shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z27 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z28 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z29 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z29b shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z29c shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z30 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z31b2 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z31c2 shader_file_trace_reset\n", fp);
    std::fputs("v115d_a7z31c3 shader_file_trace_reset\n", fp);
    std::fclose(fp);
}

void V114ShaderMultiplexFileTraceNumber(const char* label, u64 value) {
    if (!IsV114ShaderMultiplexFileTraceEnabled()) {
        return;
    }
    char buffer[256] = {};
    std::snprintf(buffer, sizeof(buffer), "%s=%llu", label,
                  static_cast<unsigned long long>(value));
    V114ShaderMultiplexFileTraceRaw(buffer);
}

[[nodiscard]] bool IsAccelStageTraceEnabled() {
    return IsDrawTraceEnabled() || IsEnvEnabled("BORKED3DS_V3DV_TRACE_ACCEL_STAGE") ||
           IsForceAccelStageTraceEnabled();
}

[[nodiscard]] bool IsTrivialVertexShaderProbeEnabled() {
    // v96 diagnostic only:
    // v95 reached TRACE_DRAW_PICA pre_call with stop_after=7, but the log did not show
    // stage=7/post_call. v96 used a trivial vertex shader to prove the generic VS bind
    // path is stable. Keep this switch as an explicit fallback, but v114 should normally
    // leave it disabled.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_USE_TRIVIAL_VERTEX_SHADER_PROBE");
}

[[nodiscard]] bool IsProgrammableVertexShaderGenerateProbeEnabled() {
    // v114 diagnostic fallback:
    // Non-guarded GLSL generation is kept as an explicit rollback/compare switch. Normal v114
    // tests should prefer the guarded probe first.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderConfigProbeEnabled() {
    // v114-A:
    // Validate SetupVertexShader config/load_flags + trivial VS bind on the rebuilt direct
    // handoff route, with ACCEL_STAGE_STOP_AFTER=7. No GLSL/SPIR-V/module work.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_CONFIG_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderBeforeGenerateOnlyProbeEnabled() {
    // v114-B:
    // Enter SetupVertexShader, build PicaVSConfig/load_flags, log before_generate_call, then
    // return before calling GLSL::GenerateVertexShader().
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_BEFORE_GENERATE_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderGenerateGuardedProbeEnabled() {
    // v114-C:
    // Run GLSL::GenerateVertexShader() in a guarded probe, then bind the trivial VS and return
    // at stage=7. No SPIR-V, no VkShaderModule, no pipeline/descriptors/draw.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderSpirvOnlyProbeEnabled() {
    // v114-D:
    // Generate GLSL, compile it to SPIR-V with CompileGLSLtoSPIRV(), then bind the trivial VS
    // and return at stage=7. No VkShaderModule, no pipeline/descriptors/draw.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_SPIRV_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderModuleOnlyProbeEnabled() {
    // v114-E:
    // Generate GLSL, compile GLSL->SPIR-V, create and immediately destroy a VkShaderModule,
    // then bind the trivial VS and return at stage=7. No pipeline/descriptors/draw.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_SHADER_MODULE_ONLY");
}

[[nodiscard]] bool IsPipelineBindProbeOnlyEnabled() {
    // v115-A rollback switch:
    // Reach BindPipeline(), then return before vertex/index buffer binding and before
    // vkCmdDraw/vkCmdDrawIndexed. Keep it available, but normal v115-D-MUX rollback testing should use
    // BORKED3DS_V3DV_PROBE_DESCRIPTOR_BIND_ONLY=1 instead.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_PROBE_PIPELINE_BIND_ONLY");
}

[[nodiscard]] bool IsDescriptorBindProbeOnlyEnabled() {
    // v115-D-MUX rollback rollback switch:
    // Replays the previous descriptor/vertex-bind probe. The v115-D-MUX rollback log reached
    // after_vertex_buffer_bind_record, but did not reach the explicit stage9 return marker.
    // Keep it available, but normal v115-D-MUX testing should use
    // BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ONLY=1 instead.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_PROBE_DESCRIPTOR_BIND_ONLY");
}

[[nodiscard]] bool IsFirstVkCmdDrawProbeOnlyEnabled() {
    // v115-D-MUX:
    // First guarded micro draw. Reuse the validated path through SetupIndexArray(), BindPipeline()
    // and bindVertexBuffers(), then record exactly one tiny vkCmdDraw/vkCmdDrawIndexed before
    // returning true. This is still limited by SAFE_PICA_HW_DRAW_BUDGET/MAX_VERTICES in pica_core.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ONLY");
}

[[nodiscard]] bool IsFirstVkCmdDrawZeroCountProbeOnlyEnabled() {
    // v115-D-MUX:
    // Same first micro-draw path as v115-C2, but record the final Vulkan draw with
    // vertex/index count forced to zero. This isolates whether the crash is caused by
    // executing/fetching the first real indexed vertices, while keeping shader, pipeline,
    // descriptors, index setup, and vertex-buffer binding unchanged.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_ONLY");
}

[[nodiscard]] bool IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled() {
    // v115-D-MUX rollback flag:
    // Keeps the already validated v115-C15 path available under its old env name.
    // Equivalent to v115-D-A: real bindVertexBuffers() + vkCmdDraw(0, 1, 0, 0).
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_REAL_VERTEX_BIND_ULTRA_QUIET_ONLY");
}

[[nodiscard]] bool IsV115DAMuxRealVertexBindDrawZeroEnabled() {
    // v115-D-A: real vertex bind + vkCmdDraw(0).
    // Validates real vertex-buffer offsets without consuming vertices.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT");
}

[[nodiscard]] bool IsV115DBMuxRealVertexBindDraw3Enabled() {
    // v115-D-B: real vertex bind + vkCmdDraw(3).
    // First true non-indexed vertex fetch using the same guarded micro-batch path.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_B_REAL_VERTEX_BIND_DRAWCMD_3");
}

[[nodiscard]] bool IsV115DCMuxRealVertexBindDraw6Enabled() {
    // v115-D-C: real vertex bind + vkCmdDraw(6).
    // Full non-indexed execution of the first 6-vertex micro-batch.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_C_REAL_VERTEX_BIND_DRAWCMD_6");
}

[[nodiscard]] bool IsV115DDMuxRealVertexBindDrawIndexedZeroEnabled() {
    // v115-D-D: indexed setup + vkCmdDrawIndexed(0).
    // Validates the indexed command path without consuming indices.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_D_INDEXED_SETUP_DRAWINDEXED_ZEROCOUNT");
}

[[nodiscard]] bool IsV115DEMuxRealVertexBindDrawIndexed3Enabled() {
    // v115-D-E: indexed setup + vkCmdDrawIndexed(3).
    // First true indexed index/vertex fetch, still limited to one safe micro-batch.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_E_INDEXED_SETUP_DRAWINDEXED_3");
}

[[nodiscard]] bool IsV115DMuxAnyDrawCommandProbeEnabled() {
    return IsV115DAMuxRealVertexBindDrawZeroEnabled() ||
           IsV115DBMuxRealVertexBindDraw3Enabled() ||
           IsV115DCMuxRealVertexBindDraw6Enabled() ||
           IsV115DDMuxRealVertexBindDrawIndexedZeroEnabled() ||
           IsV115DEMuxRealVertexBindDrawIndexed3Enabled();
}

[[nodiscard]] AttribLoadFlags MakeAccelAttribLoadFlag(Pica::PipelineRegs::VertexAttributeFormat format) {
    switch (format) {
    case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
    case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
        return AttribLoadFlags::Sint;
    case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
        return AttribLoadFlags::Uint;
    default:
        return AttribLoadFlags::Float;
    }
}

[[nodiscard]] bool IsStrictAccelInternalDryRunEnabled() {
    // v114: plan de travail 1 does not advance beyond function entry. v100 reached the
    // PICA pre_call but no raw-enter marker appeared. The recommended v114 emulators.cfg keeps
    // BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=1 for continuity but enables
    // BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE=1 and disables the GLSL generation probe. This
    // proves the call boundary before touching regs, shader setup, pipeline, descriptors, Draw(),
    // vkCmdDraw, or vkCmdDrawIndexed.
    // inside GLSL::GenerateVertexShader(). Still no SPIR-V, shader module, geometry shader
    // setup, pipeline bind, descriptors, Draw(), or vkCmdDraw.
    return IsStrictCompatEnabled() &&
           !IsEnvEnabled("BORKED3DS_V3DV_EXECUTE_ACCEL_INTERNAL_CMDS") &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_ACCEL_INTERNAL_DRY_RUN");
}

[[nodiscard]] bool IsSoftwareSkipAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_SOFTWARE_SKIP");
}

[[nodiscard]] bool IsSoftwareTexturesAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_SOFTWARE_TEXTURES");
}

[[nodiscard]] bool IsSoftwareClearProbeEnabled() {
    // v82: the v82 descriptorless clear bridge proved that the Pi5/V3DV render target
    // and final present path are alive, but it also creates the green moving rectangles
    // seen in Sonic. Do not enable that visible diagnostic by default anymore.
    // Use BORKED3DS_V3DV_ENABLE_SOFTWARE_CLEAR_PROBE=1 only when intentionally testing
    // the fake tile-clear probe, and keep BORKED3DS_V3DV_DISABLE_SOFTWARE_CLEAR_PROBE=1
    // in normal gameplay tests.
    return IsEnvEnabled("BORKED3DS_V3DV_ENABLE_SOFTWARE_CLEAR_PROBE") &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SOFTWARE_CLEAR_PROBE");
}

[[nodiscard]] bool IsFullSoftwareClearProbeEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_FULL_SOFTWARE_CLEAR_PROBE");
}

[[nodiscard]] bool IsStrictSoftwareNoopGuardDisabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SOFTWARE_NOOP_GUARD");
}

[[nodiscard]] bool IsStrictSoftwareRealDrawAllowed() {
    // v85: broad emergency opt-in. It allows every strict software vkCmdDraw() and should
    // stay off for normal Pi5/V3DV tests. The safer v85 path below opens only untextured,
    // no-depth software draws first.
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_REAL_SOFTWARE_DRAWS");
}

[[nodiscard]] bool IsStrictSafeUntexturedSoftwareDrawAllowed() {
    // v86: keep real software draws disabled while the PICA HW candidate is dry-run probed, but that still enters
    // SyncTextureUnits(), shader setup and the software vkCmdDraw path. For the next pass,
    // keep real software draws disabled unless a new explicit v85 diagnostic switch is set.
    //
    // This keeps the normal v86 test focused on controlled PICA/HW acceleration while still
    // preserving a manual escape hatch for comparing against v84.
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_SAFE_UNTEXTURED_SOFTWARE_DRAWS") &&
           IsEnvEnabled("BORKED3DS_V3DV_ALLOW_V114_REAL_SOFTWARE_DRAWS") &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_UNTEXTURED_SOFTWARE_DRAWS");
}

[[nodiscard]] u32 GetEnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return fallback;
    }

    constexpr unsigned long max_u32 = 0xFFFFFFFFul;
    return parsed > max_u32 ? 0xFFFFFFFFu : static_cast<u32>(parsed);
}

[[nodiscard]] bool IsV115DA7XTraceExpected() {
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT") &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) == 7;
}

void V115DA7XShaderTraceRaw(const char* message) {
    if (!IsV115DA7XTraceExpected()) {
        return;
    }
    V114ShaderMultiplexFileTraceRaw(message);
}

void V115DA7XShaderTraceNumber(const char* label, u64 value) {
    if (!IsV115DA7XTraceExpected()) {
        return;
    }
    V114ShaderMultiplexFileTraceNumber(label, value);
}

[[nodiscard]] bool IsV115DA7YTraceExpected() {
    // v115-D-A7Y: same activation boundary as A7X, but with extra breadcrumbs
    // strictly around PicaVSConfig construction and GLSL::GenerateVertexShader().
    return IsV115DA7XTraceExpected();
}

void V115DA7YShaderTraceRaw(const char* message) {
    if (!IsV115DA7YTraceExpected()) {
        return;
    }
    V114ShaderMultiplexFileTraceRaw(message);
}

void V115DA7YShaderTraceNumber(const char* label, u64 value) {
    if (!IsV115DA7YTraceExpected()) {
        return;
    }
    V114ShaderMultiplexFileTraceNumber(label, value);
}


[[nodiscard]] bool IsV115DA7ZTraceExpected() {
    // v115-D-A7Z: direct sidecar trace for the exact A7 stage-7 boundary.
    // Unlike A7X/A7Y, this does not depend on the shared v115d_mux sidecar helper,
    // so it can still leave breadcrumbs if the normal log or mux sidecar is cut off.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT") &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) == 7;
}

void V115DA7ZShaderTraceRaw(const char* message) {
    if (!IsV115DA7ZTraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z_shader_probe.log", "a");
    if (fp == nullptr) {
        return;
    }
    std::fputs(message, fp);
    std::fputc('\n', fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7ZShaderTraceReset() {
    if (!IsV115DA7ZTraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z_shader_probe.log", "w");
    if (fp == nullptr) {
        return;
    }
    std::fputs("v115d_a7z direct_file_trace_reset\n", fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7ZShaderTraceNumber(const char* label, u64 value) {
    if (!IsV115DA7ZTraceExpected()) {
        return;
    }
    char buffer[256] = {};
    std::snprintf(buffer, sizeof(buffer), "%s=%llu", label,
                  static_cast<unsigned long long>(value));
    V115DA7ZShaderTraceRaw(buffer);
}

[[nodiscard]] bool IsV115DA7Z2TraceExpected() {
    // v115-D-A7Z2: even earlier breadcrumbs for the same D-A stage-7 generate-guarded wall.
    // This trace intentionally does not depend on TRACE_DRAW or on the shared mux sidecar.
    // It is active only for the narrow failing configuration so normal Vulkan execution is not
    // affected outside this diagnostic path.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT") &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) == 7;
}

void V115DA7Z2ShaderTraceRaw(const char* message) {
    if (!IsV115DA7Z2TraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z2_shader_probe.log", "a");
    if (fp == nullptr) {
        return;
    }
    std::fputs(message, fp);
    std::fputc('\n', fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7Z2ShaderTraceReset() {
    if (!IsV115DA7Z2TraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z2_shader_probe.log", "w");
    if (fp == nullptr) {
        return;
    }
    std::fputs("v115d_a7z2 direct_file_trace_reset\n", fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7Z2ShaderTraceNumber(const char* label, u64 value) {
    if (!IsV115DA7Z2TraceExpected()) {
        return;
    }
    char buffer[256] = {};
    std::snprintf(buffer, sizeof(buffer), "%s=%llu", label,
                  static_cast<unsigned long long>(value));
    V115DA7Z2ShaderTraceRaw(buffer);
}

[[nodiscard]] bool IsV115DA7Z3TraceExpected() {
    // v115-D-A7Z3: last vk_rasterizer-side trace before entering GLSL::GenerateVertexShader().
    // A7Z/A7Z2 proved that stage 6, SetupVertexShader(), PicaVSConfig construction and the
    // generate call boundary are reached. A7Z3 dumps the exact config/vertex-layout state that
    // is handed to the GLSL generator, then leaves a final flushed marker immediately before
    // the crashing call. The next source file to patch is the generator itself.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT") &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) == 7;
}

[[nodiscard]] bool IsV115DA7Z3SkipGenerateWithTrivialVSEnabled() {
    // Optional safety switch, off by default. It proves the crash is caused by entering
    // GLSL::GenerateVertexShader() by dumping the same A7Z3 state and returning with the
    // already-safe trivial VS bind instead of calling the generator.
    return IsV115DA7Z3TraceExpected() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z3_SKIP_GENERATE_WITH_TRIVIAL_VS");
}

void V115DA7Z3ShaderTraceRaw(const char* message) {
    if (!IsV115DA7Z3TraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z3_shader_probe.log", "a");
    if (fp == nullptr) {
        return;
    }
    std::fputs(message, fp);
    std::fputc('\n', fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7Z3ShaderTraceReset() {
    if (!IsV115DA7Z3TraceExpected()) {
        return;
    }
    std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_a7z3_shader_probe.log", "w");
    if (fp == nullptr) {
        return;
    }
    std::fputs("v115d_a7z3 direct_file_trace_reset\n", fp);
    std::fflush(fp);
    std::fclose(fp);
}

void V115DA7Z3ShaderTraceNumber(const char* label, u64 value) {
    if (!IsV115DA7Z3TraceExpected()) {
        return;
    }
    char buffer[256] = {};
    std::snprintf(buffer, sizeof(buffer), "%s=%llu", label,
                  static_cast<unsigned long long>(value));
    V115DA7Z3ShaderTraceRaw(buffer);
}

void V115DA7Z3ShaderTraceBool(const char* label, bool value) {
    V115DA7Z3ShaderTraceNumber(label, static_cast<u64>(value));
}


[[nodiscard]] bool IsV115DA7Z5DescriptorReturnBeforeVertexBindEnabled() {
    // v115-D-A7Z5: descriptor-bind-only clean reached pipeline bind and DrawParams, then cut
    // while entering the vertex-buffer bind recording zone. This opt-in returns immediately
    // after descriptor pipeline + DrawParams, before bindVertexBuffers(), so we can separate
    // descriptor/pipeline validity from vertex/index binding command recording.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z5_DESCRIPTOR_RETURN_BEFORE_VERTEX_BIND");
}

[[nodiscard]] bool IsV115DA7Z5DescriptorVerboseRecordTraceEnabled() {
    // Extra breadcrumbs around the narrow vertex-bind scheduler.Record section. Keep this
    // separate from TRACE_DRAW so it can run in quiet sidecar-only tests.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z5_DESCRIPTOR_VERTEX_BIND_TRACE") ||
           IsV115DA7Z5DescriptorReturnBeforeVertexBindEnabled();
}

[[nodiscard]] bool IsV115DA7Z6DescriptorReturnAfterVertexBindRecordEnabled() {
    // v115-D-A7Z6: A7Z5 proved scheduler.Record for the vertex/index binding is reached
    // and that the outer code reaches after_scheduler_record_vertex_bind. The log then cut
    // before the descriptor-bind-only return marker. This switch returns immediately after
    // the vertex-bind record has been queued, before the older stage9 descriptor return
    // trace/log cluster. It isolates whether the remaining cut is in the post-record trace/log
    // tail rather than in the vertex/index bind record itself.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z6_DESCRIPTOR_RETURN_AFTER_VERTEX_BIND_RECORD");
}


[[nodiscard]] bool IsV115DA7Z7DescriptorMinimalVertexBindRecordEnabled() {
    // v115-D-A7Z7: A7Z6 did not reach its post-record marker and the newest sidecar
    // cuts immediately after the first pre-record vertex_bind_count breadcrumb. This
    // opt-in skips the noisy pre-record parameter breadcrumbs and queues only the
    // minimal bindVertexBuffers record, then returns true immediately. It separates
    // a logging/trace-tail problem from a real Vulkan vertex-buffer bind problem.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z7_DESCRIPTOR_MINIMAL_VERTEX_BIND_RECORD");
}

[[nodiscard]] bool IsV115DA7Z8DescriptorReturnAfterDrawParamsEnabled() {
    // v115-D-A7Z8: the A7Z7 sidecar cuts immediately after params_binding_count=3,
    // before params_vertex_count and before the A7Z7 minimal branch can run. This opt-in
    // returns immediately after DrawParams are built, before all parameter-count traces
    // and before scheduler.Record. It verifies whether the crash is caused by the
    // post-DrawParams trace cluster rather than by DrawParams itself.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z8_DESCRIPTOR_RETURN_AFTER_DRAWPARAMS");
}

[[nodiscard]] bool IsV115DA7Z8DescriptorMinimalVertexBindEarlyEnabled() {
    // v115-D-A7Z8: if returning after DrawParams is safe, this second opt-in queues a
    // minimal bindVertexBuffers record before the noisy params_* trace cluster. It keeps
    // the test multiplexed so the next step can be tried from emulators.cfg without
    // another rebuild.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z8_DESCRIPTOR_MINIMAL_VERTEX_BIND_EARLY");
}

[[nodiscard]] bool IsV115DA7Z9DescriptorReturnAfterPipelineBindRawEnabled() {
    // v115-D-A7Z9: A7Z8 now cuts after the raw after_descriptor_pipeline_bind
    // breadcrumb but before descriptor_pipeline_bind_ready is written. Return immediately
    // after BindPipeline() and the raw breadcrumb, before the numeric ready trace and
    // before DrawParams. This isolates whether the next crash is caused by the
    // pipeline-ready numeric trace/tail rather than BindPipeline itself.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z9_DESCRIPTOR_RETURN_AFTER_PIPELINE_BIND_RAW");
}

[[nodiscard]] bool IsV115DA7Z9DescriptorSkipPipelineReadyNumberEnabled() {
    // Secondary A7Z9 mux step: keep going past BindPipeline but skip the numeric
    // descriptor_pipeline_bind_ready trace that appears to be the next cut point.
    // This allows the already-present A7Z8 return-after-DrawParams test to be retried
    // without another rebuild.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z9_DESCRIPTOR_SKIP_PIPELINE_READY_NUMBER");
}

[[nodiscard]] bool IsV115DA7Z10DescriptorReturnAfterDrawParamsRawEnabled() {
    // v115-D-A7Z10: the A7Z9 skip-pipeline-ready test reaches
    // v115d_mux after_draw_params_build but does not reach the older A7Z8 return
    // branch. Return immediately after the raw after_draw_params_build breadcrumb,
    // before any older A7Z8 logic, params_* traces, vertex/index bind, or draw.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z10_DESCRIPTOR_RETURN_AFTER_DRAWPARAMS_RAW");
}

[[nodiscard]] bool IsV115DA7Z10DescriptorMinimalVertexBindEarlyRawEnabled() {
    // v115-D-A7Z10 second mux step: after the raw DrawParams return is proven safe,
    // record a minimal vkCmdBindVertexBuffers immediately after DrawParams, before
    // params_* traces and before the older A7Z8/A7Z5 trace clusters. This keeps the
    // next step multiplexed from emulators.cfg without requiring another rebuild.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z10_DESCRIPTOR_MINIMAL_VERTEX_BIND_EARLY_RAW");
}

[[nodiscard]] bool IsV115DA7Z11DAReturnBeforeOffsetsEnabled() {
    // v115-D-A7Z11: D-A draw0 reached the real-vertex-bind mux path and wrote
    // real_vertex_bind_mux_binding_count=3, then cut before after_record. Return
    // immediately after that breadcrumb, before offset conversion and scheduler.Record.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z11_DA_RETURN_BEFORE_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z11DAReturnAfterOffsetsEnabled() {
    // v115-D-A7Z11 second mux step: if returning before offsets is safe, build the
    // vk::DeviceSize offsets array and return before queuing scheduler.Record.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z11_DA_RETURN_AFTER_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z11DABindOnlyRecordEnabled() {
    // v115-D-A7Z11 third mux step: queue only vkCmdBindVertexBuffers in the D-A path,
    // then return before any vkCmdDraw. This compares D-A's real path with the descriptor
    // A7Z10 minimal bind that already passed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z11_DA_BIND_ONLY_RECORD");
}

[[nodiscard]] bool IsV115DA7Z11DADraw0RecordRawEnabled() {
    // v115-D-A7Z11 fourth mux step: queue a raw D-A bind + vkCmdDraw(0) with a reduced
    // post-record trace tail. This is only meant for D-A after the earlier A7Z11 gates pass.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z11_DA_DRAW0_RECORD_RAW");
}

[[nodiscard]] bool IsV115DA7Z12MuxReturnAfterPipelineBindEnabled() {
    // v115-D-A7Z12: D-D/indexed0 now reaches selected_step=4, final_indexed=1,
    // final_count=0, and pipeline_ready=true, then cuts before the normal
    // real_vertex_bind_mux_before_record breadcrumb. Return immediately after the
    // pipeline bind to isolate the indexed mux tail without touching offsets or
    // scheduler.Record.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z12_MUX_RETURN_AFTER_PIPELINE_BIND");
}

[[nodiscard]] bool IsV115DA7Z12MuxReturnBeforeOffsetsEnabled() {
    // Secondary A7Z12 step: after the post-pipeline return is safe, allow the
    // lightweight before_record/binding_count breadcrumbs, then return before
    // converting binding_offsets to vk::DeviceSize.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z12_MUX_RETURN_BEFORE_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z12MuxReturnAfterOffsetsEnabled() {
    // Secondary A7Z12 step: build real_offsets and return before any additional
    // command buffer record. This checks whether the indexed path survives offset
    // conversion separately from bind/draw recording.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z12_MUX_RETURN_AFTER_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z12MuxBindOnlyRecordEnabled() {
    // Secondary A7Z12 step: queue only vkCmdBindVertexBuffers through the real mux
    // path. SetupIndexArray() is still executed earlier for indexed source draws, but
    // this step avoids vkCmdDraw/vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z12_MUX_BIND_ONLY_RECORD");
}

[[nodiscard]] bool IsV115DA7Z12MuxDrawRawEnabled() {
    // Final A7Z12 raw draw step: unlike the older A7Z11_DA_DRAW0 switch, this uses
    // final_indexed and final_count. With D-D it records vkCmdDrawIndexed(0); with
    // D-E it records vkCmdDrawIndexed(3); with D-B/D-C it records draw(3/6).
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z12_MUX_DRAW_RAW");
}

[[nodiscard]] bool IsV115DA7Z13MuxTraceAfterBindingCountEnabled() {
    // v115-D-A7Z13: D-D drawIndexed0 raw cuts after binding_count=3 and before
    // the normal after-offsets breadcrumb. This snapshot records the active A7Z12
    // mux flags immediately after binding_count so a bad emulators.cfg line can be
    // separated from a crash inside offset conversion.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z13_MUX_TRACE_AFTER_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z13MuxReturnAfterBindingCountEnabled() {
    // First A7Z13 gate: return immediately after real_vertex_bind_mux_binding_count,
    // before checking A7Z11/A7Z12 return-before-offsets flags and before converting
    // binding_offsets. Use this with A7Z12_MUX_DRAW_RAW=1 to prove whether the raw
    // draw flag itself is safe before offsets are touched.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z13_MUX_RETURN_AFTER_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z13MuxManualOffsetsEnabled() {
    // Second A7Z13 gate: replace the compact std::transform offset conversion with
    // a very explicit bounded loop. This makes the D-D indexed0 raw path easier to
    // bisect on V3DV and gives a clean breadcrumb before and after the conversion.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z13_MUX_MANUAL_OFFSETS_BUILD");
}

[[nodiscard]] bool IsV115DA7Z13MuxReturnAfterManualOffsetsEnabled() {
    // Third A7Z13 gate: after the manual offsets loop succeeds, return before any
    // scheduler.Record/bind/draw. This confirms manual offset conversion alone is safe.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z13_MUX_RETURN_AFTER_MANUAL_OFFSETS");
}


[[nodiscard]] bool IsV115DA7Z14MuxReturnBeforeBindingCountEnabled() {
    // v115-D-A7Z14: A7Z13 proved the indexed D-D raw path can now cut before
    // real_vertex_bind_mux_binding_count=3. Return immediately after the
    // real_vertex_bind_mux_before_record breadcrumb, before reading/logging
    // binding_count, before offset conversion, and before scheduler.Record.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z14_MUX_RETURN_BEFORE_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z15MuxUltraCleanReturnBeforeBindingCountEnabled() {
    // v115-D-A7Z15: D-E reaches real_vertex_bind_mux_before_record but the
    // A7Z14 breadcrumb block can cut before PICA confirms a clean return.
    // This gate is intentionally shorter than A7Z14: emit only begin/true,
    // then return before binding_count, offsets, scheduler.Record, and
    // vkCmdDrawIndexed(3).
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z15_MUX_ULTRA_CLEAN_RETURN_BEFORE_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z16MuxUltraCleanReturnAfterBindingCountEnabled() {
    // v115-D-A7Z16: D-E is now proven safe after pipeline bind and safe with the
    // A7Z15 ultra-clean return before binding_count. This gate is the matching
    // ultra-clean checkpoint immediately after real_vertex_bind_mux_binding_count,
    // before any A7Z13 flag dump, offsets, scheduler.Record, or vkCmdDrawIndexed(3).
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z16_MUX_ULTRA_CLEAN_RETURN_AFTER_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z17MuxUltraCleanReturnBeforePipelineBindEnabled() {
    // v115-D-A7Z17: on the A7Z16 build, D-E can select final_count=3 but may cut
    // between final_vertex_offset and BindPipeline(). Return immediately before
    // pipeline_cache.BindPipeline(), after the mux selection breadcrumbs, to prove
    // the D-E mux selection itself returns cleanly before touching pipeline bind.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z17_MUX_ULTRA_CLEAN_RETURN_BEFORE_PIPELINE_BIND");
}

[[nodiscard]] bool IsV115DA7Z18MuxUltraCleanReturnFalseBeforePipelineBindEnabled() {
    // v115-D-A7Z18: A7Z17 proves D-E can reach the pre-pipeline-bind checkpoint and
    // emit the return-true marker, but PICA may still fail to confirm the caller-side
    // return. This variant returns false at the exact same checkpoint. If PICA logs
    // the after-AccelerateDrawBatch result=0, the issue is the successful-accel true
    // handoff path. If it still cuts, the issue is around the call/return boundary.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z18_MUX_ULTRA_CLEAN_RETURN_FALSE_BEFORE_PIPELINE_BIND");
}

[[nodiscard]] bool IsV115DA7Z23MuxReturnFalseAfterPipelineBindEnabled() {
    // v115-D-A7Z23: A7Z21 proved PICA can return cleanly when the backend returns
    // false before BindPipeline(). A7Z12 then proved D-E can reach pipeline_ready=true,
    // but PICA did not confirm the after-call marker when the backend returned true.
    // This checkpoint executes the same post-BindPipeline location as A7Z12, but
    // returns false instead of true, to isolate whether the caller-side failure is
    // caused by the successful AccelerateDrawBatch=true handoff or by pipeline bind
    // itself.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z23_MUX_RETURN_FALSE_AFTER_PIPELINE_BIND");
}

[[nodiscard]] bool IsV115DA7Z23BReturnFalseBeforePipelineBindEnabled() {
    // v115-D-A7Z23B: diagnostic rollback for the A7Z23 recheck. It uses a separate
    // switch to return false immediately before BindPipeline(), after final_vertex_offset
    // has already been logged. This lets us verify the A7Z23 source/config alignment
    // without touching pipeline_cache.BindPipeline().
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z23B_RETURN_FALSE_BEFORE_BIND_PIPELINE");
}

[[nodiscard]] bool IsV115DA7Z24MuxReturnFalseAfterBindingCountEnabled() {
    // v115-D-A7Z24: A7Z23 proved D-E can pass BindPipeline() and return false cleanly
    // to PICA. This next false-return checkpoint moves one safe step farther: read/log
    // real_vertex_bind_mux_binding_count, then return false before A7Z13 flag dumps,
    // offset conversion, scheduler.Record, or vkCmdDrawIndexed(3). This isolates
    // whether binding_count itself is safe separately from the successful true handoff.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z24_MUX_RETURN_FALSE_AFTER_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z25MuxReturnFalseBeforeBindingCountEnabled() {
    // v115-D-A7Z25: A7Z24 did not reach the binding_count breadcrumb, while the A7Z23
    // recheck on the same build still returns false cleanly after BindPipeline(). This
    // checkpoint moves just one micro-step farther than A7Z23: it executes the gate checks
    // after pipeline bind, then returns false immediately before the before_record /
    // binding_count section. No binding_count read, offset conversion, scheduler.Record,
    // or vkCmdDrawIndexed(3) is reached.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z25_MUX_RETURN_FALSE_BEFORE_BINDING_COUNT");
}

[[nodiscard]] bool IsV115DA7Z26MuxReturnFalseAfterBeforeRecordEnabled() {
    // v115-D-A7Z26 canonical gate.
    //
    // Important: do not OR this with the A7Z26F realignment alias. A7Z26F is a separate
    // earlier checkpoint used only to prove source/emulators.cfg alignment after selected_step.
    // Keeping it separate lets the next run advance from:
    //
    //   selected_step -> A7Z26F return false
    //
    // to:
    //
    //   selected_step -> final_indexed -> A7Z26E return false before final_count
    //
    // without rebuilding again for a different branch.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26_MUX_RETURN_FALSE_AFTER_BEFORE_RECORD") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26_FORCE");
}

[[nodiscard]] bool IsV115DA7Z26FReturnFalseAfterSelectedStepEnabled() {
    // v115-D-A7Z26F: early realignment checkpoint immediately after selected_step.
    // This intentionally has its own switch so it does not mask later A7Z26/E checkpoints.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26F_RETURN_FALSE_AFTER_SELECTED_STEP");
}

[[nodiscard]] bool IsV115DA7Z26GReturnFalseAfterFinalCountEnabled() {
    // v115-D-A7Z26G: the A7Z26E split-flags run proved selected_step -> final_indexed
    // is safe and returns false before final_count. Advance one micro-step: allow
    // final_count to be emitted, then return false before final_vertex_offset,
    // BindPipeline, before_record, binding_count, offsets, scheduler.Record,
    // vkCmdBindVertexBuffers, and vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26G_RETURN_FALSE_AFTER_FINAL_COUNT") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26G_RETURN_FALSE_BEFORE_FINAL_VERTEX_OFFSET");
}

[[nodiscard]] bool IsV115DA7Z26HReturnFalseAfterFinalVertexOffsetEnabled() {
    // v115-D-A7Z26H: A7Z26G proved final_count=0 is safe and returns false before
    // final_vertex_offset. Advance one micro-step: allow final_vertex_offset to be
    // emitted, then return false before BindPipeline, before_record, binding_count,
    // offsets, scheduler.Record, vkCmdBindVertexBuffers, and vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26H_RETURN_FALSE_AFTER_FINAL_VERTEX_OFFSET") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26H_RETURN_FALSE_BEFORE_BIND_PIPELINE");
}

[[nodiscard]] bool IsV115DA7Z26IReturnFalseAfterInternalBindingCountEnabled() {
    // v115-D-A7Z26I: the A7Z26 retry with A7Z23B trace-align cut after the early
    // internal binding_count breadcrumb and before internal_vertex_buffer_count /
    // internal_after_binding_count_valid / selected_step. This checkpoint avoids
    // before_record, BindPipeline re-entry, offsets, scheduler.Record, vertex-buffer
    // binding, and vkCmdDrawIndexed. It returns false immediately after the already
    // observed internal binding_count value to verify that the backend can unwind
    // cleanly from this earlier boundary.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26I_RETURN_FALSE_AFTER_INTERNAL_BINDING_COUNT") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26I_RETURN_FALSE_BEFORE_INTERNAL_VERTEX_BUFFER_COUNT");
}

[[nodiscard]] bool IsV115DA7Z26JReturnFalseAfterInternalFlagCacheEnabled() {
    // v115-D-A7Z26J: A7Z26I proved the new flag is read in both outer and internal scopes,
    // but the trace cut immediately after the internal flag-cache block and before
    // internal_after_stage10 / internal_binding_count. This gate returns false from that
    // exact boundary, before the stage-10 limiter lambda, before any internal binding_count
    // computation, before selected_step, BindPipeline, before_record, offsets,
    // scheduler.Record, vkCmdBindVertexBuffers, or vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26J_RETURN_FALSE_AFTER_INTERNAL_FLAGS") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26J_RETURN_FALSE_BEFORE_INTERNAL_STAGE10");
}

[[nodiscard]] bool IsV115DA7Z26KReturnFalseAfterInternalStage10Enabled() {
    // v115-D-A7Z26K: A7Z26J proved the backend can return false cleanly immediately
    // after the internal flag-cache block. Advance one micro-step: allow the stage-10
    // limiter path and internal_after_stage10 breadcrumb, then return false before the
    // vertex-count check, internal_binding_count, selected_step, BindPipeline,
    // before_record, offsets, scheduler.Record, vkCmdBindVertexBuffers, or
    // vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26K_RETURN_FALSE_AFTER_INTERNAL_STAGE10") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26K_RETURN_FALSE_BEFORE_INTERNAL_VERTEX_COUNT");
}

[[nodiscard]] bool IsV115DA7Z26LReturnFalseAfterInternalStage10SingleMarkerEnabled() {
    // v115-D-A7Z26L: A7Z26K reached the generic internal_after_stage10 breadcrumb but
    // cut before its second return marker. This variant uses a single dedicated marker
    // immediately after stage10, then returns false before the vertex-count check. It
    // avoids the two-breadcrumb tail and isolates whether the failure is caused by the
    // extra trace/log tail rather than the stage10 boundary itself.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z26L_RETURN_FALSE_AFTER_INTERNAL_STAGE10_SINGLE_MARKER") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z26L_RETURN_FALSE_BEFORE_INTERNAL_VERTEX_COUNT");
}

[[nodiscard]] bool IsV115DA7Z27MuxReturnFalseBeforeBindingCountNumberEnabled() {
    // v115-D-A7Z27: A7Z26 proved D-D can emit real_vertex_bind_mux_before_record and
    // return false cleanly. This checkpoint is now placed immediately after the validated
    // before_record breadcrumb and before A7Z26/A7Z15/A7Z14, so it is impossible to miss
    // when the env flag is active. It emits only raw breadcrumbs and returns false before
    // V114ShaderMultiplexFileTraceNumber(...binding_count...), offset conversion,
    // scheduler.Record, vkCmdBindVertexBuffers, or vkCmdDrawIndexed.
    //
    // Keep short aliases because the full env name is long and easy to mistype in
    // emulators.cfg during the v115-D bisect.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z27_MUX_RETURN_FALSE_BEFORE_BINDING_COUNT_NUMBER") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z27_SAFE_RETURN_FALSE") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z27_RETURN_FALSE");
}

[[nodiscard]] bool IsV115DA7Z28MuxSkipBindingCountNumberReturnFalseEnabled() {
    // v115-D-E-A7Z28: A7Z27 proved D-E can reach the point immediately before the
    // binding_count numeric breadcrumb and return false cleanly. A7Z24 still cuts when
    // the numeric trace is attempted. This checkpoint deliberately skips the
    // V114ShaderMultiplexFileTraceNumber(...binding_count...) call, emits only raw
    // breadcrumbs, then returns false before A7Z16/A7Z24/A7Z13 gates, offset conversion,
    // scheduler.Record, or vkCmdDrawIndexed(3). It isolates the numeric trace as the
    // suspect while keeping the Vulkan path state unchanged.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z28_MUX_SKIP_BINDING_COUNT_NUMBER_RETURN_FALSE");
}

[[nodiscard]] bool IsV115DA7Z29MuxSkipBindingCountNumberReturnFalseBeforeOffsetsEnabled() {
    // v115-D-E-A7Z29: A7Z28 proved the binding_count numeric breadcrumb is the local
    // hazard, not the logical transition after before_record. Keep skipping that number
    // breadcrumb, continue through the safe post-binding-count gates, and return false
    // immediately before offset conversion. This confirms the path can advance from
    // before_record to the pre-offset boundary without the numeric trace.
    return IsEnvEnabled(
        "BORKED3DS_V3DV_A7Z29_MUX_SKIP_BINDING_COUNT_NUMBER_RETURN_FALSE_BEFORE_OFFSETS");
}


[[nodiscard]] bool IsV115DA7Z29BMuxSkipBindingCountNumberReturnFalseBeforeOffsetsSafeEnabled() {
    // v115-D-E-A7Z29B: A7Z29 was too intrusive when it tried to continue after the
    // binding_count numeric breadcrumb skip. A7Z28 rechecked cleanly on that build, so
    // the build is valid and only the A7Z29 continuation must be cleaned up. This
    // variant keeps the A7Z28 numeric skip, uses raw breadcrumbs only, avoids logging
    // binding_count in console warnings, and returns false immediately before offset
    // conversion. It is the safe retry of the intended A7Z29 pre-offset boundary.
    return IsEnvEnabled(
        "BORKED3DS_V3DV_A7Z29B_MUX_SKIP_BINDING_COUNT_NUMBER_RETURN_FALSE_BEFORE_OFFSETS_SAFE");
}


[[nodiscard]] bool IsV115DA7Z29CMuxImmediateReturnFalseBeforeOffsetsEnabled() {
    // v115-D-E-A7Z29C: A7Z29B still cut after real_vertex_bind_mux_before_record before
    // its own breadcrumbs appeared. This makes the probe impossible to miss: use a short
    // env alias, check it immediately after the validated before_record breadcrumb, skip
    // the dangerous binding_count numeric trace, emit raw-only breadcrumbs, and return
    // false before any offset conversion, scheduler.Record, or vkCmdDrawIndexed(3).
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z29C_SAFE_RETURN_FALSE") ||
           IsEnvEnabled("BORKED3DS_V3DV_A7Z29C_MUX_RETURN_FALSE_BEFORE_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z30MuxManualOffsetsReturnFalseAfterOffsetsEnabled() {
    // v115-D-E-A7Z30: A7Z28 is the last stable checkpoint. Do not reuse the A7Z29
    // continuation probes. Instead, keep the A7Z28 binding_count numeric trace skip,
    // advance directly into a bounded manual offset conversion, emit raw-only sidecar
    // breadcrumbs, and return false immediately after offsets. This validates offset
    // construction before scheduler.Record, bindVertexBuffers, or vkCmdDrawIndexed(3).
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z30_MANUAL_OFFSETS_RETURN_FALSE_AFTER_OFFSETS");
}

[[nodiscard]] bool IsV115DA7Z31B2EmptyRecordReturnFalseEnabled() {
    // v115-D-E-A7Z31B2: restart from the stable A7Z30 base with a short env flag and
    // a direct branch immediately after before_record. It keeps the binding_count
    // numeric trace skipped, rebuilds offsets manually, records an empty scheduler
    // lambda only, then returns false before bindVertexBuffers or vkCmdDrawIndexed.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z31B2_EMPTY_RECORD");
}

[[nodiscard]] bool IsV115DA7Z31C2BindVertexBuffersOnlyReturnFalseEnabled() {
    // v115-D-E-A7Z31C2: restart strictly from the A7Z31B2 stable empty-record block,
    // then add only vkCmdBindVertexBuffers. No vkCmdDraw/vkCmdDrawIndexed yet.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z31C2_BIND_VERTEX_BUFFERS_ONLY");
}

[[nodiscard]] bool IsV115DA7Z31C3BindVertexBuffer0OnlyReturnFalseEnabled() {
    // v115-D-E-A7Z31C3: safer bind probe after A7Z31B2. Bind only vertex buffer slot 0
    // with a single captured VkBuffer handle and one captured offset. This avoids capturing
    // the rasterizer object or passing the full V3DV binding_count=3 path while still
    // proving whether vkCmdBindVertexBuffers itself can be recorded after the empty record.
    return IsEnvEnabled("BORKED3DS_V3DV_A7Z31C3_BIND_VERTEX_BUFFER0_ONLY");
}

[[nodiscard]] u32 GetAccelStageStopAfter() {
    // 0 means no stage-limit stop. Use this only to bisect a crash inside
    // AccelerateDrawBatch, for example:
    //   BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=6
    return GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0);
}

[[nodiscard]] bool ShouldStopAfterAccelStage(u32 stage) {
    const u32 stop_after = GetAccelStageStopAfter();
    return stop_after != 0 && stage >= stop_after;
}

[[nodiscard]] bool IsAccelEntryPreflightExpected() {
    // v114 raw-enter-simple handoff test: v110 proved raw_enter_noargs survives.
    // v114 keeps the guarded GLSL target available for later, but the normal test uses
    // BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN=0 with RAW_ENTER_RETURN=0.
    // This emits raw_enter_noargs + raw_enter_simple and performs no stage=1, no shader
    // generation, no PICA SPIR-V, no VkShaderModule creation, no pipeline state,
    // no descriptors, no Draw(), no vkCmdDraw, and no vkCmdDrawIndexed.
    return IsStrictCompatEnabled() && GetAccelStageStopAfter() != 0;
}

[[nodiscard]] u32 GetStrictSafeUntexturedSoftwareDrawBudget() {
    // Keep this bounded so a bad untextured path cannot flood V3DV with commands.
    // 256 is enough to prove whether the loading screen can receive real color writes.
    return GetEnvU32("BORKED3DS_V3DV_SAFE_UNTEXTURED_DRAW_BUDGET", 256);
}

[[nodiscard]] u32 GetSoftwareClearTileBudget() {
    // v82: the visible tile clear is opt-in only. Keep the diagnostic small when enabled.
    // 0 disables the tile clear completely; increase only for diagnosis.
    return GetEnvU32("BORKED3DS_V3DV_SOFTWARE_CLEAR_TILE_BUDGET", 16);
}

[[nodiscard]] u32 GetSoftwareClearTilePeriod() {
    // v82: submit one tile-clear every N software draws. This prevents the stable bridge
    // from turning into a command-stream stress test on V3DV while preserving visible
    // movement on screen.
    return std::max<u32>(1, GetEnvU32("BORKED3DS_V3DV_SOFTWARE_CLEAR_TILE_PERIOD", 4));
}

[[nodiscard]] bool IsNullSoftwareDrawProbeEnabled() {
    // v82: v79 proved even the first tiny untextured fixed-null software draw can
    // make Sonic close almost immediately on Pi5/V3DV. Keep the descriptorless
    // tile-clear bridge as the default stable path. Re-enter the real software
    // shader/pipeline/vkCmdDraw path only with an explicit opt-in.
    return IsEnvEnabled("BORKED3DS_V3DV_ENABLE_NULL_SOFTWARE_DRAW_PROBE");
}

[[nodiscard]] bool IsTexturedNullSoftwareDrawProbeAllowed() {
    // v82: v78 proved that the first controlled *textured* null-descriptor software draw
    // is still crash-prone on Pi5/V3DV. Keep textured real draws blocked by default and
    // only allow them via explicit opt-in once the untextured path is proven safe.
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_TEXTURED_NULL_SOFTWARE_DRAW_PROBE");
}

[[nodiscard]] bool ShouldAttemptNullSoftwareDrawProbe(u64 clear_index, u32 vertex_count,
                                                      u32 enabled_textures,
                                                      bool textures_disabled, bool depth_active,
                                                      u64& eligible_probe_index,
                                                      bool& textured_probe) {
    eligible_probe_index = 0;
    textured_probe = false;

    if (!IsNullSoftwareDrawProbeEnabled() || IsFullSoftwareClearProbeEnabled()) {
        return false;
    }

    // v82: keep the visible/stable tile-clear bridge as the default. The real
    // software path is now fully opt-in because v79 showed that even an untextured
    // null-descriptor draw can close the emulator. When explicitly enabled, only
    // tiny untextured/no-depth batches are considered first.
    if (vertex_count == 0 || vertex_count > 6 || depth_active) {
        return false;
    }

    const bool has_textures = enabled_textures != 0 && !textures_disabled;
    if (has_textures) {
        textured_probe = true;
        if (!IsTexturedNullSoftwareDrawProbeAllowed()) {
            return false;
        }

        static std::atomic<u64> textured_probe_counter{0};
        eligible_probe_index = ++textured_probe_counter;
        return eligible_probe_index == 1;
    }

    static std::atomic<u64> untextured_probe_counter{0};
    eligible_probe_index = ++untextured_probe_counter;

    // v82 opt-in: do only one early untextured probe. If that survives, a later
    // pass can make this sparse instead of one-shot.
    return eligible_probe_index == 1;
}

[[nodiscard]] bool IsStartupSoftwareQuarantineDisabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SOFTWARE_QUARANTINE");
}

[[nodiscard]] bool IsStartupSoftwareQuarantineForcedOff() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_STARTUP_SOFTWARE_DRAWS");
}

[[nodiscard]] bool IsPresentDebugClearDisabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_DISABLE_PRESENT_DEBUG_CLEAR");
}

[[nodiscard]] bool IsAcceleratedDisplayAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_ACCELERATED_DISPLAY");
}

[[nodiscard]] bool IsForcedNonAcceleratedDisplay() {
    return IsEnvEnabled("BORKED3DS_V3DV_FORCE_NON_ACCELERATED_DISPLAY");
}

[[nodiscard]] bool IsPresentImageClearAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_PRESENT_IMAGE_CLEAR");
}

[[nodiscard]] bool IsOwnedPresentTextureDebugDisabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_DISABLE_OWNED_PRESENT_TEXTURE_CLEAR");
}

[[nodiscard]] bool IsOwnedPresentTextureDebugEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_ENABLE_OWNED_PRESENT_TEXTURE_CLEAR");
}

[[nodiscard]] bool IsDuplicatePresentReuseDisabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_DISABLE_DUPLICATE_PRESENT_REUSE");
}

[[nodiscard]] bool IsStrictCompatFragileTextureFormat(u32 format) {
    // Pi5/V3DV strict mode: these small alpha/intensity/compressed formats are the exact
    // family that currently crashes during GetTextureSurface()/Surface creation in the
    // software fallback path. Bind a safe null texture before any surface is requested.
    //
    // Known PICA texture format values used by this fork/logs:
    //   4=IA8, 5=I8, 6=A8, 7=IA4, 8=I4, 9=A4/IA4-class in logs, 10=ETC1, 11=ETC1A4.
    // The log-visible crash is format 9 reported as IA4 by vk_texture_runtime.cpp.
    switch (format) {
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
        return true;
    default:
        return false;
    }
}

std::atomic<u64> g_vk_software_bypass_counter{0};
std::atomic<u64> g_vk_textured_software_bypass_counter{0};
std::atomic<u64> g_vk_large_textured_software_allow_counter{0};
std::atomic<u64> g_vk_batch42_textured_software_skip_counter{0};
std::atomic<u64> g_vk_nonindexed96_textured_software_skip_counter{0};
std::atomic<u64> g_vk_nonindexed36_textured_software_skip_counter{0};
std::atomic<u64> g_vk_indexed6_textured_late_startup_skip_counter{0};
std::atomic<u64> g_vk_indexed6_format0or1or9_followup_skip_counter{0};
std::atomic<u64> g_vk_indexed6_untextured_late_startup_skip_counter{0};
std::atomic<u64> g_vk_indexed6_generic_late_startup_skip_counter{0};
std::atomic<u64> g_vk_indexed24_untextured_poststartup_skip_counter{0};
std::atomic<u64> g_vk_indexed18_textured_followup_skip_counter{0};
std::atomic<u64> g_vk_indexed12_untextured_postindex24_skip_counter{0};
std::atomic<u64> g_vk_highdraw_indexed18_textured_skip_counter{0};
std::atomic<u64> g_vk_late_textured_pair_skip_counter{0};
std::atomic<u64> g_vk_non_bypassed_software_trace_counter{0};
std::atomic<u64> g_vk_medium_textured_software_skip_counter{0};
std::atomic<u64> g_vk_startup_textured_software_skip_counter{0};
std::atomic<u64> g_vk_strict_software_quarantine_counter{0};
std::atomic<u64> g_vk_strict_software_debug_clear_counter{0};
std::atomic<u64> g_vk_strict_safe_untextured_real_draw_counter{0};
std::atomic<u64> g_vk_strict_present_debug_clear_counter{0};
std::atomic<u64> g_vk_strict_owned_present_clear_counter{0};

struct StrictPresentDisplayCache {
    bool valid = false;
    PAddr framebuffer_addr = 0;
    u32 width = 0;
    u32 height = 0;
    u32 stride = 0;
    VideoCore::PixelFormat pixel_format{};
    Common::Rectangle<f32> texcoords{};
    vk::ImageView image_view{};
    u64 generation = 0;
};

StrictPresentDisplayCache g_vk_strict_present_display_cache{};

void RememberStrictPresentDisplay(PAddr framebuffer_addr, u32 width, u32 height, u32 stride,
                                  VideoCore::PixelFormat pixel_format,
                                  const Common::Rectangle<f32>& texcoords,
                                  vk::ImageView image_view) {
    if (!IsStrictCompatEnabled() || !static_cast<bool>(image_view)) {
        return;
    }

    g_vk_strict_present_display_cache.valid = true;
    g_vk_strict_present_display_cache.framebuffer_addr = framebuffer_addr;
    g_vk_strict_present_display_cache.width = width;
    g_vk_strict_present_display_cache.height = height;
    g_vk_strict_present_display_cache.stride = stride;
    g_vk_strict_present_display_cache.pixel_format = pixel_format;
    g_vk_strict_present_display_cache.texcoords = texcoords;
    g_vk_strict_present_display_cache.image_view = image_view;
    ++g_vk_strict_present_display_cache.generation;
}


[[nodiscard]] bool IsStrictPresentDisplayDuplicate(PAddr framebuffer_addr, u32 width, u32 height,
                                                   u32 stride,
                                                   VideoCore::PixelFormat pixel_format) {
    if (!IsStrictCompatEnabled() || IsDuplicatePresentReuseDisabled()) {
        return false;
    }

    const auto& cached = g_vk_strict_present_display_cache;
    if (!cached.valid || !static_cast<bool>(cached.image_view)) {
        return false;
    }

    return cached.framebuffer_addr == framebuffer_addr && cached.width == width &&
           cached.height == height && cached.stride == stride &&
           cached.pixel_format == pixel_format;
}

[[nodiscard]] bool IsDuplicateExternalReuseAllowed() {
    const char* value = std::getenv("BORKED3DS_V3DV_ALLOW_DUPLICATE_EXTERNAL_PRESENT_REUSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsDuplicateOwnedReuseAllowed() {
    const char* value = std::getenv("BORKED3DS_V3DV_ALLOW_DUPLICATE_OWNED_PRESENT_REUSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool TryReuseStrictPresentDisplay(PAddr framebuffer_addr, u32 width, u32 height,
                                                u32 stride, VideoCore::PixelFormat pixel_format,
                                                ScreenInfo& screen_info) {
    if (!IsStrictCompatEnabled() || IsDuplicatePresentReuseDisabled()) {
        return false;
    }

    const auto& cached = g_vk_strict_present_display_cache;
    if (!cached.valid || !static_cast<bool>(cached.image_view)) {
        return false;
    }

    if (cached.framebuffer_addr != framebuffer_addr || cached.width != width ||
        cached.height != height || cached.stride != stride || cached.pixel_format != pixel_format) {
        return false;
    }

    // v82: v67 reused the external cached view, v69/v70 reused the renderer-owned
    // per-slot view, and both variants still died immediately after the duplicate
    // top/right-eye AccelerateDisplay call. In strict Pi5/V3DV mode, the safest
    // next step is to stop presenting any duplicate right-eye view by default.
    // Renderer_DisableRightEyeRender is true in the logs, so sacrificing this
    // duplicate view should let PrepareRendertarget continue to the bottom screen
    // and then to DrawSingleScreen without binding a suspicious duplicate image.
    // Set BORKED3DS_V3DV_ALLOW_DUPLICATE_EXTERNAL_PRESENT_REUSE=1 or
    // BORKED3DS_V3DV_ALLOW_DUPLICATE_OWNED_PRESENT_REUSE=1 only for diagnosis.
    const bool allow_external_duplicate_reuse = IsDuplicateExternalReuseAllowed();
    const bool allow_owned_duplicate_reuse = IsDuplicateOwnedReuseAllowed();

    const vk::ImageView owned_view = screen_info.texture.image_view;
    const bool owned_valid = static_cast<bool>(owned_view);

    const char* duplicate_mode = "mono_skip_invalid_view";
    if (allow_external_duplicate_reuse) {
        screen_info.texcoords = cached.texcoords;
        screen_info.image_view = cached.image_view;
        duplicate_mode = "external_reuse";
    } else if (allow_owned_duplicate_reuse && owned_valid) {
        screen_info.texcoords = Common::Rectangle<f32>{0.0f, 0.0f, 1.0f, 1.0f};
        screen_info.image_view = owned_view;
        duplicate_mode = "owned_slot_fallback";
    } else {
        screen_info.texcoords = Common::Rectangle<f32>{0.0f, 0.0f, 1.0f, 1.0f};
        screen_info.image_view = vk::ImageView{};
    }

    if (IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v82 duplicate AccelerateDisplay {} addr=0x{:08x} width={} height={} stride={} pixel_format={} generation={} view_valid={} owned_valid={}; duplicate view suppressed by default to avoid Pi5/V3DV right-eye crash",
                    duplicate_mode, framebuffer_addr, width, height, stride,
                    static_cast<u32>(pixel_format), cached.generation,
                    static_cast<u32>(static_cast<bool>(screen_info.image_view)),
                    static_cast<u32>(owned_valid));
    }

    return true;
}


void RecordStrictOwnedPresentTextureClear(Scheduler& scheduler, RenderManager& renderpass_cache,
                                          vk::Image image, u64 clear_index,
                                          PAddr framebuffer_addr, u32 width, u32 height,
                                          u32 stride, VideoCore::PixelFormat pixel_format) {
    if (!static_cast<bool>(image)) {
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 owned-present texture clear skipped invalid_image clear_index={} addr=0x{:08x}",
                        clear_index, framebuffer_addr);
        }
        return;
    }

    renderpass_cache.EndRendering();

    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    scheduler.Record([image, range, clear_index](vk::CommandBuffer cmdbuf) {
        const float phase = static_cast<float>(clear_index % 6);
        const std::array<float, 4> color{
            phase == 0.0f || phase == 3.0f ? 0.95f : 0.05f,
            phase == 1.0f || phase == 3.0f ? 0.85f : 0.05f,
            phase == 2.0f || phase == 4.0f ? 0.95f : 0.10f,
            1.0f,
        };

        const vk::ImageMemoryBarrier pre_barrier{
            .srcAccessMask = {},
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearColorValue{color}, range);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });

    if (IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v82 owned-present texture clear submitted clear_index={} addr=0x{:08x} width={} height={} stride={} pixel_format={}",
                    clear_index, framebuffer_addr, width, height, stride,
                    static_cast<u32>(pixel_format));
    }
}

void RecordStrictPresentDebugClear(Scheduler& scheduler, RenderManager& renderpass_cache,
                                   Surface& surface, u64 clear_index, PAddr framebuffer_addr,
                                   u32 width, u32 height, u32 stride,
                                   VideoCore::PixelFormat pixel_format) {
    const vk::Image image = surface.Image();
    const vk::ImageAspectFlags aspect = surface.Aspect();
    if (!static_cast<bool>(image) || !static_cast<bool>(aspect & vk::ImageAspectFlagBits::eColor)) {
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 present-path debug clear skipped invalid_surface clear_index={} addr=0x{:08x} image_valid={}",
                        clear_index, framebuffer_addr, static_cast<u32>(static_cast<bool>(image)));
        }
        return;
    }

    renderpass_cache.EndRendering();

    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    scheduler.Record([image, range, clear_index](vk::CommandBuffer cmdbuf) {
        const float phase = static_cast<float>(clear_index % 6);
        const std::array<float, 4> color{
            phase == 0.0f || phase == 3.0f ? 0.95f : 0.05f,
            phase == 1.0f || phase == 3.0f ? 0.80f : 0.05f,
            phase == 2.0f || phase == 4.0f ? 0.95f : 0.10f,
            1.0f,
        };

        // v82: keep the present-source image in eGeneral. V3DV is more stable here than
        // bouncing an already-presentable/cache-owned image through eTransferDstOptimal on
        // every AccelerateDisplay call. The post barrier is enough for the present fragment
        // shader to sample the cleared image later in the same command stream.
        cmdbuf.clearColorImage(image, vk::ImageLayout::eGeneral, vk::ClearColorValue{color},
                               range);

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });

    if (IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v82 present-path debug clear submitted clear_index={} addr=0x{:08x} width={} height={} stride={} pixel_format={}",
                    clear_index, framebuffer_addr, width, height, stride,
                    static_cast<u32>(pixel_format));
    }
}

[[nodiscard]] bool ArePrimaryTexturesDisabled(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        if (textures[i].enabled) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool HasPrimaryTexturesEnabled(const Pica::RegsInternal& regs) {
    return !ArePrimaryTexturesDisabled(regs);
}

[[nodiscard]] u32 CountEnabledPrimaryTextures(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    u32 enabled = 0;
    for (u32 i = 0; i < 3; ++i) {
        enabled += textures[i].enabled ? 1u : 0u;
    }
    return enabled;
}

[[nodiscard]] bool HasSinglePrimaryTexture0Format8(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    return textures[0].enabled && !textures[1].enabled && !textures[2].enabled &&
           static_cast<u32>(textures[0].format) == 8u;
}

[[nodiscard]] bool HasSinglePrimaryTexture0Format0(const Pica::RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    return textures[0].enabled && !textures[1].enabled && !textures[2].enabled &&
           static_cast<u32>(textures[0].format) == 0u;
}



[[nodiscard]] bool HasActiveDepthState(const Pica::RegsInternal& regs) {
    return regs.framebuffer.output_merger.depth_test_enable != 0 ||
           regs.framebuffer.output_merger.depth_write_enable != 0;
}

// v50: the old strict-compat bypasses were useful for isolating crashes, but they now keep the
// framebuffer black by throwing away the software draws exposed by pica_core.cpp. Keep the helpers
// available behind an explicit opt-in variable, but default to drawing.
[[nodiscard]] bool CanUseSoftwareSkipWorkaround() {
    return IsStrictCompatEnabled() && IsSoftwareSkipAllowed();
}

[[nodiscard]] bool ShouldBypassFragileSoftwareDraw(const Pica::RegsInternal& regs,
                                                   std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size == 0 || vertex_batch_size > 24) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 24) {
        return false;
    }
    if (!ArePrimaryTexturesDisabled(regs)) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldBypassFragileTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                           std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size == 0 || vertex_batch_size > 96) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 96) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) > 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldAttemptTinyTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                         std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 6) {
        return false;
    }
    if (regs.pipeline.num_vertices != 6) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldAttemptMediumTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                           std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size < 12 || vertex_batch_size > 96 || vertex_batch_size == 42) {
        return false;
    }
    if (regs.pipeline.num_vertices < 12 || regs.pipeline.num_vertices > 96 ||
        regs.pipeline.num_vertices == 42) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipStartupTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                         std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size == 0 || vertex_batch_size > 320) {
        return false;
    }
    if (regs.pipeline.num_vertices == 0 || regs.pipeline.num_vertices > 320) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldAttemptLargeTexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                          std::size_t vertex_batch_size) {
    if (!IsStrictCompatEnabled()) {
        return false;
    }
    if (vertex_batch_size != 42) {
        return false;
    }
    if (regs.pipeline.num_vertices != 42) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs)) {
        return false;
    }
    if (CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering()) {
        return false;
    }
    if (HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipBatch42TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                         std::size_t vertex_batch_size) {
    return CanUseSoftwareSkipWorkaround() && ShouldAttemptLargeTexturedSoftwareDraw(regs, vertex_batch_size);
}

[[nodiscard]] bool ShouldSkipNonIndexed96TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                              std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size != 96 || regs.pipeline.num_vertices != 96) {
        return false;
    }
    if (!HasPrimaryTexturesEnabled(regs) || CountEnabledPrimaryTextures(regs) != 1) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering() || HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipNonIndexed36TexturedSoftwareDraw(const Pica::RegsInternal& regs,
                                                              std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size != 36 || regs.pipeline.num_vertices != 36) {
        return false;
    }
    if (!HasSinglePrimaryTexture0Format8(regs)) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering() || HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool ShouldSkipIndexed6TexturedLateStartupSoftwareDraw(
    const Pica::RegsInternal& regs, std::size_t vertex_batch_size) {
    if (!CanUseSoftwareSkipWorkaround()) {
        return false;
    }
    if (vertex_batch_size != 6 || regs.pipeline.num_vertices != 6) {
        return false;
    }
    if (!HasSinglePrimaryTexture0Format0(regs)) {
        return false;
    }
    if (regs.framebuffer.IsShadowRendering() || HasActiveDepthState(regs)) {
        return false;
    }
    return true;
}

} // Anonymous namespace

RasterizerVulkan::RasterizerVulkan(Memory::MemorySystem& memory, Pica::PicaCore& pica,
                                   VideoCore::CustomTexManager& custom_tex_manager,
                                   VideoCore::RendererBase& renderer,
                                   Frontend::EmuWindow& emu_window, const Instance& instance,
                                   Scheduler& scheduler, RenderManager& renderpass_cache,
                                   DescriptorUpdateQueue& update_queue_, u32 image_count)
    : RasterizerAccelerated{memory, pica}, instance{instance}, scheduler{scheduler},
      renderpass_cache{renderpass_cache}, update_queue{update_queue_},
      pipeline_cache{instance, scheduler, renderpass_cache, update_queue},
      runtime{instance, scheduler, renderpass_cache, update_queue, image_count},
      res_cache{memory, custom_tex_manager, runtime, regs, renderer},
      stream_buffer{instance, scheduler, BUFFER_USAGE, STREAM_BUFFER_SIZE},
      uniform_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformBuffer,
                     UNIFORM_BUFFER_SIZE},
      texture_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                     TextureBufferSize(instance)},
      texture_lf_buffer{instance, scheduler, vk::BufferUsageFlagBits::eUniformTexelBuffer,
                        TextureBufferSize(instance)},
      async_shaders{Settings::values.async_shader_compilation.GetValue()} {

    vertex_buffers.fill(stream_buffer.Handle());

    uniform_buffer_alignment = instance.UniformMinAlignment();
    uniform_size_aligned_vs_pica =
        Common::AlignUp<u32>(sizeof(VSPicaUniformData), uniform_buffer_alignment);
    uniform_size_aligned_vs = Common::AlignUp<u32>(sizeof(VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs = Common::AlignUp<u32>(sizeof(FSUniformData), uniform_buffer_alignment);

    MakeSoftwareVertexLayout();
    pipeline_info.vertex_layout = software_layout;

    const vk::Device device = instance.GetDevice();
    texture_lf_view = device.createBufferViewUnique({
        .buffer = texture_lf_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rg_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });
    texture_rgba_view = device.createBufferViewUnique({
        .buffer = texture_buffer.Handle(),
        .format = vk::Format::eR32G32B32A32Sfloat,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    });

    scheduler.RegisterOnSubmit([&renderpass_cache] { renderpass_cache.EndRendering(); });

    const auto buffer_set = pipeline_cache.Acquire(DescriptorHeapType::Buffer);
    update_queue.AddBuffer(buffer_set, 0, uniform_buffer.Handle(), 0, sizeof(VSPicaUniformData));
    update_queue.AddBuffer(buffer_set, 1, uniform_buffer.Handle(), 0, sizeof(VSUniformData));
    update_queue.AddBuffer(buffer_set, 2, uniform_buffer.Handle(), 0, sizeof(FSUniformData));
    update_queue.AddTexelBuffer(buffer_set, 3, *texture_lf_view);
    update_queue.AddTexelBuffer(buffer_set, 4, *texture_rg_view);
    update_queue.AddTexelBuffer(buffer_set, 5, *texture_rgba_view);

    const auto texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
    Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
    Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);
    for (u32 i = 0; i < 3; i++) {
        update_queue.AddImageSampler(texture_set, i, 0, null_surface.ImageView(),
                                     null_sampler.Handle());
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);
    update_queue.AddStorageImage(utility_set, 0, null_surface.StorageView());
    update_queue.AddImageSampler(utility_set, 1, 0, null_surface.ImageView(),
                                 null_sampler.Handle());
    update_queue.Flush();

    SyncEntireState();

    if (IsV115DA7ZTraceExpected()) {
        V115DA7ZShaderTraceReset();
        V115DA7ZShaderTraceRaw("v115d_a7z rasterizer_constructor_marker");
        V115DA7ZShaderTraceNumber("v115d_a7z constructor_d_a_draw0",
                                  static_cast<u64>(IsV115DAMuxRealVertexBindDrawZeroEnabled()));
        V115DA7ZShaderTraceNumber("v115d_a7z constructor_generate_guarded_probe",
                                  static_cast<u64>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));
        V115DA7ZShaderTraceNumber("v115d_a7z constructor_stage_stop_after",
                                  GetAccelStageStopAfter());
    }

    if (IsV115DA7Z2TraceExpected()) {
        V115DA7Z2ShaderTraceReset();
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 rasterizer_constructor_marker");
        V115DA7Z2ShaderTraceNumber("v115d_a7z2 constructor_d_a_draw0",
                                   static_cast<u64>(IsV115DAMuxRealVertexBindDrawZeroEnabled()));
        V115DA7Z2ShaderTraceNumber("v115d_a7z2 constructor_generate_guarded_probe",
                                   static_cast<u64>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));
        V115DA7Z2ShaderTraceNumber("v115d_a7z2 constructor_stage_stop_after",
                                   GetAccelStageStopAfter());
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v115d_a7z2 RasterizerVulkan constructor marker d_a_draw0={} generate_guarded={} stage_stop_after={}",
                    static_cast<u32>(IsV115DAMuxRealVertexBindDrawZeroEnabled()),
                    static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                    GetAccelStageStopAfter());
    }

    if (IsV115DA7Z3TraceExpected()) {
        V115DA7Z3ShaderTraceReset();
        V115DA7Z3ShaderTraceRaw("v115d_a7z3 rasterizer_constructor_marker");
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 constructor_d_a_draw0",
                                   static_cast<u64>(IsV115DAMuxRealVertexBindDrawZeroEnabled()));
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 constructor_generate_guarded_probe",
                                   static_cast<u64>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 constructor_stage_stop_after",
                                   GetAccelStageStopAfter());
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 constructor_skip_generate_with_trivial_vs",
                                   static_cast<u64>(IsV115DA7Z3SkipGenerateWithTrivialVSEnabled()));
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v115d_a7z3 RasterizerVulkan constructor marker d_a_draw0={} generate_guarded={} stage_stop_after={} skip_generate={}",
                    static_cast<u32>(IsV115DAMuxRealVertexBindDrawZeroEnabled()),
                    static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                    GetAccelStageStopAfter(),
                    static_cast<u32>(IsV115DA7Z3SkipGenerateWithTrivialVSEnabled()));
    }

    if (IsV114ShaderMultiplexFileTraceEnabled()) {
        V114ShaderMultiplexFileTraceReset();
        V114ShaderMultiplexFileTraceRaw("v115d_mux rasterizer_constructor_descriptor_bind_marker");
        V115DA7YShaderTraceRaw("v115d_a7y rasterizer_constructor_marker");
        V115DA7YShaderTraceNumber("v115d_a7y constructor_generate_guarded_probe",
                                  static_cast<u64>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));
        V115DA7YShaderTraceNumber("v115d_a7y constructor_stage_stop_after",
                                  GetAccelStageStopAfter());
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_generate_guarded_probe",
                                           static_cast<u64>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_stage_stop_after",
                                           GetAccelStageStopAfter());
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_pipeline_bind_probe",
                                           static_cast<u64>(IsPipelineBindProbeOnlyEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_descriptor_bind_probe",
                                           static_cast<u64>(IsDescriptorBindProbeOnlyEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_first_vkcmd_draw_probe",
                                           static_cast<u64>(IsFirstVkCmdDrawProbeOnlyEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_first_vkcmd_draw_zero_count_probe",
                                           static_cast<u64>(IsFirstVkCmdDrawZeroCountProbeOnlyEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_d_a_draw0_probe",
                                           static_cast<u64>(IsV115DAMuxRealVertexBindDrawZeroEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_d_b_draw3_probe",
                                           static_cast<u64>(IsV115DBMuxRealVertexBindDraw3Enabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_d_c_draw6_probe",
                                           static_cast<u64>(IsV115DCMuxRealVertexBindDraw6Enabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_d_d_drawindexed0_probe",
                                           static_cast<u64>(IsV115DDMuxRealVertexBindDrawIndexedZeroEnabled()));
        V114ShaderMultiplexFileTraceNumber("v115d_mux constructor_d_e_drawindexed3_probe",
                                           static_cast<u64>(IsV115DEMuxRealVertexBindDrawIndexed3Enabled()));
    }

    if (IsDrawTraceEnabled()) {
        if (IsV115DA7YTraceExpected()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_a7y RasterizerVulkan constructor marker d_a_draw0={} generate_guarded={} stage_stop_after={}",
                        static_cast<u32>(IsV115DAMuxRealVertexBindDrawZeroEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                        GetAccelStageStopAfter());
        }
        if (IsV115DA7ZTraceExpected()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_a7z RasterizerVulkan constructor marker d_a_draw0={} generate_guarded={} stage_stop_after={}",
                        static_cast<u32>(IsV115DAMuxRealVertexBindDrawZeroEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                        GetAccelStageStopAfter());
        }
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v115d_mux RasterizerVulkan constructor draw-command-mux marker strict_compat={} allow_software_textures={} quarantine_disabled={} pipeline_bind_probe={} descriptor_bind_probe={} first_vkcmd_draw_probe={} first_vkcmd_draw_zero_count_probe={} d_a_draw0={} d_b_draw3={} d_c_draw6={} d_d_drawindexed0={} d_e_drawindexed3={}",
                    static_cast<u32>(IsStrictCompatEnabled()),
                    static_cast<u32>(IsSoftwareTexturesAllowed()),
                    static_cast<u32>(IsStartupSoftwareQuarantineDisabled()),
                    static_cast<u32>(IsPipelineBindProbeOnlyEnabled()),
                    static_cast<u32>(IsDescriptorBindProbeOnlyEnabled()),
                    static_cast<u32>(IsFirstVkCmdDrawProbeOnlyEnabled()),
                    static_cast<u32>(IsFirstVkCmdDrawZeroCountProbeOnlyEnabled()),
                    static_cast<u32>(IsV115DAMuxRealVertexBindDrawZeroEnabled()),
                    static_cast<u32>(IsV115DBMuxRealVertexBindDraw3Enabled()),
                    static_cast<u32>(IsV115DCMuxRealVertexBindDraw6Enabled()),
                    static_cast<u32>(IsV115DDMuxRealVertexBindDrawIndexedZeroEnabled()),
                    static_cast<u32>(IsV115DEMuxRealVertexBindDrawIndexed3Enabled()));
    }
}

RasterizerVulkan::~RasterizerVulkan() = default;

void RasterizerVulkan::TickFrame() {
    res_cache.TickFrame();
}

void RasterizerVulkan::LoadDiskResources(const std::atomic_bool& stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    pipeline_cache.LoadDiskCache();
}

void RasterizerVulkan::SyncFixedState() {
    SyncCullMode();
    SyncBlendEnabled();
    SyncBlendFuncs();
    SyncBlendColor();
    SyncLogicOp();
    SyncStencilTest();
    SyncDepthTest();
    SyncColorWriteMask();
    SyncStencilWriteMask();
    SyncDepthWriteMask();
}

void RasterizerVulkan::SetupVertexArray() {
    const auto [vs_input_index_min, vs_input_index_max, vs_input_size] = vertex_info;
    auto [array_ptr, array_offset, invalidate] = stream_buffer.Map(vs_input_size, 16);

    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    const PAddr base_address = vertex_attributes.GetPhysicalBaseAddress();
    const u32 stride_alignment = instance.GetMinVertexStrideAlignment();

    VertexLayout& layout = pipeline_info.vertex_layout;
    layout.binding_count = 0;
    layout.attribute_count = 16;
    enable_attributes.fill(false);

    u32 buffer_offset = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }

        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; comp++) {
            const u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index >= 12) {
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
                continue;
            }

            const u32 size = vertex_attributes.GetNumElements(attribute_index);
            if (size == 0) {
                continue;
            }

            offset =
                Common::AlignUp(offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

            const u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
            const auto format = vertex_attributes.GetFormat(attribute_index);

            VertexAttribute& attribute = layout.attributes[input_reg];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(input_reg);
            attribute.offset.Assign(offset);
            attribute.type.Assign(format);
            attribute.size.Assign(size);

            enable_attributes[input_reg] = true;
            offset += vertex_attributes.GetStride(attribute_index);
        }

        const PAddr data_addr =
            base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        u32 data_size = loader.byte_count * vertex_num;
        res_cache.FlushRegion(data_addr, data_size);

        const MemoryRef src_ref = memory.GetPhysicalRef(data_addr);
        if (src_ref.GetSize() < data_size) {
            LOG_ERROR(Render_Vulkan,
                      "Vertex buffer size {} exceeds available space {} at address {:#016X}",
                      data_size, src_ref.GetSize(), data_addr);
        }

        const u8* src_ptr = src_ref.GetPtr();
        u8* dst_ptr = array_ptr + buffer_offset;

        const u32 aligned_stride =
            Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);
        if (aligned_stride == loader.byte_count) {
            std::memcpy(dst_ptr, src_ptr, data_size);
        } else {
            for (std::size_t vertex = 0; vertex < vertex_num; vertex++) {
                std::memcpy(dst_ptr + vertex * aligned_stride, src_ptr + vertex * loader.byte_count,
                            loader.byte_count);
            }
        }

        VertexBinding& binding = layout.bindings[layout.binding_count];
        binding.binding.Assign(layout.binding_count);
        binding.fixed.Assign(0);
        binding.stride.Assign(aligned_stride);

        binding_offsets[layout.binding_count++] = static_cast<u32>(array_offset + buffer_offset);
        buffer_offset += Common::AlignUp(aligned_stride * vertex_num, 4);
    }

    stream_buffer.Commit(buffer_offset);
    SetupFixedAttribs();
}

void RasterizerVulkan::SetupFixedAttribs() {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    VertexLayout& layout = pipeline_info.vertex_layout;

    auto [fixed_ptr, fixed_offset, _] = stream_buffer.Map(16 * sizeof(Common::Vec4f), 0);
    binding_offsets[layout.binding_count] = static_cast<u32>(fixed_offset);

    static const Common::Vec4f default_attrib{0.f, 0.f, 0.f, 1.f};
    std::memcpy(fixed_ptr, default_attrib.AsArray(), sizeof(Common::Vec4f));

    u32 offset = sizeof(Common::Vec4f);
    for (std::size_t i = 0; i < 16; i++) {
        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = pica.input_default_attributes[i];
                const std::array data = {attr.x.ToFloat32(), attr.y.ToFloat32(), attr.z.ToFloat32(),
                                         attr.w.ToFloat32()};

                const u32 data_size = sizeof(float) * static_cast<u32>(data.size());
                std::memcpy(fixed_ptr + offset, data.data(), data_size);

                VertexAttribute& attribute = layout.attributes[reg];
                attribute.binding.Assign(layout.binding_count);
                attribute.location.Assign(reg);
                attribute.offset.Assign(offset);
                attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
                attribute.size.Assign(4);

                offset += data_size;
                enable_attributes[reg] = true;
            }
        }
    }

    for (u32 i = 0; i < 16; i++) {
        if (!enable_attributes[i]) {
            VertexAttribute& attribute = layout.attributes[i];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(i);
            attribute.offset.Assign(0);
            attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
            attribute.size.Assign(4);
        }
    }

    VertexBinding& binding = layout.bindings[layout.binding_count];
    binding.binding.Assign(layout.binding_count++);
    binding.fixed.Assign(1);
    binding.stride.Assign(offset);

    stream_buffer.Commit(offset);
}

bool RasterizerVulkan::SetupVertexShader() {
    BORKED3DS_PROFILE("Vulkan", "Vertex Shader Setup");

    const bool v114_file_trace = IsV114ShaderMultiplexFileTraceEnabled();
    const bool trace_accel = IsAccelStageTraceEnabled() && !v114_file_trace;
    const bool trivial_vs_probe = IsTrivialVertexShaderProbeEnabled();
    const bool programmable_config_probe = IsProgrammableVertexShaderConfigProbeEnabled();
    const bool programmable_before_generate_probe =
        IsProgrammableVertexShaderBeforeGenerateOnlyProbeEnabled();
    const bool programmable_generate_guarded_probe =
        IsProgrammableVertexShaderGenerateGuardedProbeEnabled();
    const bool programmable_spirv_probe = IsProgrammableVertexShaderSpirvOnlyProbeEnabled();
    const bool programmable_module_probe = IsProgrammableVertexShaderModuleOnlyProbeEnabled();
    const bool programmable_generate_probe = IsProgrammableVertexShaderGenerateProbeEnabled();

    V115DA7ZShaderTraceRaw("v115d_a7z setup_vs_begin");
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 setup_vs_begin");
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 setup_vs_binding_count",
                               pipeline_info.vertex_layout.binding_count);
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 setup_vs_attribute_count",
                               pipeline_info.vertex_layout.attribute_count);
    V115DA7ZShaderTraceNumber("v115d_a7z setup_vs_binding_count",
                              pipeline_info.vertex_layout.binding_count);
    V115DA7ZShaderTraceNumber("v115d_a7z setup_vs_attribute_count",
                              pipeline_info.vertex_layout.attribute_count);
    V115DA7ZShaderTraceNumber("v115d_a7z setup_vs_generate_guarded_probe",
                              static_cast<u64>(programmable_generate_guarded_probe));
    V115DA7ZShaderTraceNumber("v115d_a7z setup_vs_before_generate_probe",
                              static_cast<u64>(programmable_before_generate_probe));

    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_begin");
        V115DA7XShaderTraceRaw("v115d_a7x before_setup_vertex_shader");
        V115DA7YShaderTraceRaw("v115d_a7y before_setup_vertex_shader");
        if (programmable_config_probe) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_probe=config_only");
        }
        if (programmable_before_generate_probe) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_probe=before_generate_only");
        }
        if (programmable_generate_guarded_probe) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_probe=generate_guarded_only");
        }
        if (programmable_spirv_probe) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_probe=spirv_only");
        }
        if (programmable_module_probe) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vs_probe=shader_module_only");
        }
        V114ShaderMultiplexFileTraceNumber("v115d_mux setup_vs_binding_count",
                                           pipeline_info.vertex_layout.binding_count);
        V114ShaderMultiplexFileTraceNumber("v115d_mux setup_vs_attribute_count",
                                           pipeline_info.vertex_layout.attribute_count);
        V115DA7XShaderTraceNumber("v115d_a7x setup_vs_binding_count",
                                  pipeline_info.vertex_layout.binding_count);
        V115DA7XShaderTraceNumber("v115d_a7x setup_vs_attribute_count",
                                  pipeline_info.vertex_layout.attribute_count);
        V115DA7YShaderTraceNumber("v115d_a7y setup_vs_binding_count",
                                  pipeline_info.vertex_layout.binding_count);
        V115DA7YShaderTraceNumber("v115d_a7y setup_vs_attribute_count",
                                  pipeline_info.vertex_layout.attribute_count);
    }

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 vertex_shader_setup_begin trivial_probe={} programmable_config_probe={} programmable_before_generate_probe={} programmable_generate_guarded_probe={} programmable_spirv_probe={} programmable_module_probe={} programmable_generate_probe={} binding_count={} attribute_count={} accurate_mul={} strict_compat={}",
                    static_cast<u32>(trivial_vs_probe),
                    static_cast<u32>(programmable_config_probe),
                    static_cast<u32>(programmable_before_generate_probe),
                    static_cast<u32>(programmable_generate_guarded_probe),
                    static_cast<u32>(programmable_spirv_probe),
                    static_cast<u32>(programmable_module_probe),
                    static_cast<u32>(programmable_generate_probe),
                    pipeline_info.vertex_layout.binding_count,
                    pipeline_info.vertex_layout.attribute_count, static_cast<u32>(accurate_mul),
                    static_cast<u32>(IsStrictCompatEnabled()));
    }

    if (trivial_vs_probe) {
        pipeline_cache.UseTrivialVertexShader();
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_trivial_probe_used result=1");
        }
        return true;
    }

    auto build_programmable_vs_config = [&]() {
        V115DA7YShaderTraceRaw("v115d_a7y before_vs_config");
        V115DA7ZShaderTraceRaw("v115d_a7z before_vs_config");
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_vs_config");
        V115DA7Z3ShaderTraceRaw("v115d_a7z3 before_vs_config");
        V115DA7YShaderTraceNumber("v115d_a7y vs_config_attribute_count",
                                  pipeline_info.vertex_layout.attribute_count);
        V115DA7YShaderTraceNumber("v115d_a7y vs_config_binding_count",
                                  pipeline_info.vertex_layout.binding_count);

        const bool use_geometry_shader = instance.UseGeometryShaders() && !regs.lighting.disable &&
                                         !instance.IsFragmentShaderBarycentricSupported();
        PicaVSConfig config{regs, pica.vs_setup, instance.IsShaderClipDistanceSupported(),
                            use_geometry_shader, accurate_mul};

        u32 converted_attribs = 0;
        u32 zero_w_attribs = 0;

        for (u32 i = 0; i < pipeline_info.vertex_layout.attribute_count; i++) {
            const VertexAttribute& attr = pipeline_info.vertex_layout.attributes[i];
            const FormatTraits& traits = instance.GetTraits(attr.type, attr.size);
            const u32 location = attr.location.Value();
            AttribLoadFlags& flags = config.state.load_flags[location];

            if (traits.needs_conversion) {
                flags = MakeAccelAttribLoadFlag(attr.type);
                converted_attribs++;
            }
            if (traits.needs_emulation) {
                flags |= AttribLoadFlags::ZeroW;
                zero_w_attribs++;
            }
        }

        V115DA7XShaderTraceRaw("v115d_a7x after_build_programmable_vs_config");
        V115DA7XShaderTraceNumber("v115d_a7x config_hash", config.Hash());
        V115DA7XShaderTraceNumber("v115d_a7x use_geometry_shader", static_cast<u32>(use_geometry_shader));
        V115DA7XShaderTraceNumber("v115d_a7x converted_attribs", converted_attribs);
        V115DA7XShaderTraceNumber("v115d_a7x zero_w_attribs", zero_w_attribs);
        V115DA7YShaderTraceRaw("v115d_a7y after_vs_config");
        V115DA7ZShaderTraceRaw("v115d_a7z after_vs_config");
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_vs_config");
        V115DA7Z3ShaderTraceRaw("v115d_a7z3 after_vs_config");
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 config_hash", config.Hash());
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 main_offset", config.state.main_offset);
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 num_outputs", config.state.num_outputs);
        V115DA7Z3ShaderTraceBool("v115d_a7z3 use_geometry_shader", config.state.use_geometry_shader);
        V115DA7Z3ShaderTraceBool("v115d_a7z3 use_clip_planes", config.state.use_clip_planes);
        V115DA7Z3ShaderTraceBool("v115d_a7z3 sanitize_mul", config.state.sanitize_mul);
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 converted_attribs", converted_attribs);
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 zero_w_attribs", zero_w_attribs);
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 gs_output_attributes", config.state.gs_state.gs_output_attributes);
        V115DA7Z3ShaderTraceNumber("v115d_a7z3 vs_output_attributes", config.state.gs_state.vs_output_attributes);
        V115DA7ZShaderTraceNumber("v115d_a7z config_hash", config.Hash());
        V115DA7ZShaderTraceNumber("v115d_a7z use_geometry_shader", static_cast<u32>(use_geometry_shader));
        V115DA7ZShaderTraceNumber("v115d_a7z converted_attribs", converted_attribs);
        V115DA7ZShaderTraceNumber("v115d_a7z zero_w_attribs", zero_w_attribs);
        V115DA7YShaderTraceNumber("v115d_a7y config_hash", config.Hash());
        V115DA7YShaderTraceNumber("v115d_a7y use_geometry_shader", static_cast<u32>(use_geometry_shader));
        V115DA7YShaderTraceNumber("v115d_a7y converted_attribs", converted_attribs);
        V115DA7YShaderTraceNumber("v115d_a7y zero_w_attribs", zero_w_attribs);

        return std::tuple<PicaVSConfig, bool, u32, u32>{
            config, use_geometry_shader, converted_attribs, zero_w_attribs};
    };

    if (programmable_before_generate_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_before_generate_only_begin");
        }

        auto [config, use_geometry_shader, converted_attribs, zero_w_attribs] =
            build_programmable_vs_config();

        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_config_built");
            V114ShaderMultiplexFileTraceNumber("v115d_mux shader_probe_config_hash", config.Hash());
            V114ShaderMultiplexFileTraceNumber("v115d_mux shader_probe_converted_attribs",
                                               converted_attribs);
            V114ShaderMultiplexFileTraceNumber("v115d_mux shader_probe_zero_w_attribs",
                                               zero_w_attribs);
        }

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_before_generate_only_config use_geometry_shader={} converted_attribs={} zero_w_attribs={} config_hash={}",
                        static_cast<u32>(use_geometry_shader), converted_attribs, zero_w_attribs,
                        config.Hash());
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_before_generate_only_before_generate_call skipped=1");
        }

        pipeline_cache.UseTrivialVertexShader();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_before_generate_only_trivial_bind result=1");
        }

        return true;
    }

    if (programmable_config_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_config_only_begin");
        }

        auto [config, use_geometry_shader, converted_attribs, zero_w_attribs] =
            build_programmable_vs_config();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_config_only_end use_geometry_shader={} converted_attribs={} zero_w_attribs={} config_hash={}",
                        static_cast<u32>(use_geometry_shader), converted_attribs, zero_w_attribs,
                        config.Hash());
        }

        pipeline_cache.UseTrivialVertexShader();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_config_only_trivial_bind result=1");
        }

        return true;
    }

    auto run_shader_multiplex_probe = [&](const char* mode_name, bool compile_spirv,
                                           bool create_module) -> bool {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_begin");
            V114ShaderMultiplexFileTraceRaw(mode_name);
            V115DA7XShaderTraceRaw("v115d_a7x shader_probe_begin");
            V115DA7XShaderTraceRaw(mode_name);
            V115DA7XShaderTraceRaw("v115d_a7x before_build_programmable_vs_config");
            V115DA7YShaderTraceRaw("v115d_a7y shader_probe_begin");
            V115DA7YShaderTraceRaw(mode_name);
            V115DA7YShaderTraceRaw("v115d_a7y before_vs_config_from_probe");
            V115DA7ZShaderTraceRaw("v115d_a7z shader_probe_begin");
            V115DA7ZShaderTraceRaw(mode_name);
            V115DA7ZShaderTraceRaw("v115d_a7z before_vs_config_from_probe");
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 shader_probe_begin");
            V115DA7Z3ShaderTraceRaw(mode_name);
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 before_vs_config_from_probe");
        }
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_begin", mode_name);
        }

        auto [config, use_geometry_shader, converted_attribs, zero_w_attribs] =
            build_programmable_vs_config();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_config use_geometry_shader={} converted_attribs={} zero_w_attribs={} config_hash={}",
                        mode_name, static_cast<u32>(use_geometry_shader), converted_attribs,
                        zero_w_attribs, config.Hash());
        }

        std::string program;

        try {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_before_generate_call");
                V115DA7XShaderTraceRaw("v115d_a7x before_generate_vertex_shader");
                V115DA7XShaderTraceRaw("v115d_a7x before_generate_call");
                V115DA7YShaderTraceRaw("v115d_a7y before_generate_vertex_shader");
                V115DA7YShaderTraceRaw("v115d_a7y before_generate_call");
                V115DA7YShaderTraceNumber("v115d_a7y generate_config_hash", config.Hash());
                V115DA7ZShaderTraceRaw("v115d_a7z before_generate_vertex_shader");
                V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_generate_vertex_shader");
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 before_generate_vertex_shader");
                V115DA7Z3ShaderTraceNumber("v115d_a7z3 before_generate_config_hash", config.Hash());
                V115DA7Z3ShaderTraceNumber("v115d_a7z3 before_generate_main_offset", config.state.main_offset);
                V115DA7Z3ShaderTraceNumber("v115d_a7z3 before_generate_num_outputs", config.state.num_outputs);
                V115DA7Z3ShaderTraceNumber("v115d_a7z3 before_generate_binding_count",
                                            pipeline_info.vertex_layout.binding_count);
                V115DA7Z3ShaderTraceNumber("v115d_a7z3 before_generate_attribute_count",
                                            pipeline_info.vertex_layout.attribute_count);
                V115DA7ZShaderTraceRaw("v115d_a7z before_generate_call");
                V115DA7ZShaderTraceNumber("v115d_a7z generate_config_hash", config.Hash());
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 immediately_before_generate_call_flushed");
            }
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_before_generate_call",
                            mode_name);
            }

            if (IsV115DA7Z3SkipGenerateWithTrivialVSEnabled()) {
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 skip_generate_with_trivial_vs_begin");
                pipeline_cache.UseTrivialVertexShader();
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 skip_generate_with_trivial_vs_return_true");
                return true;
            }

            program = GLSL::GenerateVertexShader(pica.vs_setup, config, true);

            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_after_generate_call");
                V115DA7XShaderTraceRaw("v115d_a7x after_generate_vertex_shader");
                V115DA7XShaderTraceRaw("v115d_a7x after_generate_call");
                V115DA7YShaderTraceRaw("v115d_a7y after_generate_vertex_shader");
                V115DA7YShaderTraceRaw("v115d_a7y after_generate_call");
                V115DA7ZShaderTraceRaw("v115d_a7z after_generate_vertex_shader");
                V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_generate_vertex_shader");
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 after_generate_vertex_shader");
                V115DA7Z3ShaderTraceRaw("v115d_a7z3 after_generate_call");
                V115DA7ZShaderTraceRaw("v115d_a7z after_generate_call");
            }
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_after_generate_call",
                            mode_name);
            }
        } catch (const std::exception& e) {
            V115DA7YShaderTraceRaw("v115d_a7y generate_vertex_shader_std_exception");
            V115DA7ZShaderTraceRaw("v115d_a7z generate_vertex_shader_std_exception");
            V115DA7Z2ShaderTraceRaw("v115d_a7z2 generate_vertex_shader_std_exception");
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 generate_vertex_shader_std_exception");
            LOG_ERROR(Render_Vulkan,
                      "TRACE_ACCEL_STAGE v114 programmable VS GLSL generation threw std::exception: {}",
                      e.what());
            return false;
        } catch (...) {
            V115DA7YShaderTraceRaw("v115d_a7y generate_vertex_shader_unknown_exception");
            V115DA7ZShaderTraceRaw("v115d_a7z generate_vertex_shader_unknown_exception");
            V115DA7Z2ShaderTraceRaw("v115d_a7z2 generate_vertex_shader_unknown_exception");
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 generate_vertex_shader_unknown_exception");
            LOG_ERROR(Render_Vulkan,
                      "TRACE_ACCEL_STAGE v114 programmable VS GLSL generation threw unknown exception");
            return false;
        }

        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_glsl_end");
            V114ShaderMultiplexFileTraceNumber("v115d_mux shader_probe_program_bytes", program.size());
            V114ShaderMultiplexFileTraceNumber("v115d_mux shader_probe_program_empty",
                                               static_cast<u64>(program.empty()));
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 shader_probe_glsl_end");
            V115DA7Z3ShaderTraceNumber("v115d_a7z3 shader_probe_program_bytes", program.size());
            V115DA7Z3ShaderTraceNumber("v115d_a7z3 shader_probe_program_empty",
                                        static_cast<u64>(program.empty()));
            V115DA7XShaderTraceRaw("v115d_a7x generate_vertex_shader_completed");
            V115DA7XShaderTraceNumber("v115d_a7x generated_program_bytes", program.size());
            V115DA7XShaderTraceNumber("v115d_a7x generated_program_empty",
                                      static_cast<u64>(program.empty()));
            V115DA7YShaderTraceRaw("v115d_a7y generate_vertex_shader_completed");
            V115DA7ZShaderTraceRaw("v115d_a7z generate_vertex_shader_completed");
            V115DA7Z2ShaderTraceRaw("v115d_a7z2 generate_vertex_shader_completed");
            V115DA7ZShaderTraceNumber("v115d_a7z generated_glsl_size", program.size());
            V115DA7Z2ShaderTraceNumber("v115d_a7z2 generated_glsl_size", program.size());
            V115DA7YShaderTraceNumber("v115d_a7y generated_glsl_size", program.size());
            V115DA7YShaderTraceNumber("v115d_a7y generated_program_empty",
                                      static_cast<u64>(program.empty()));
        }
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_glsl_end program_bytes={} empty={}",
                        mode_name, program.size(), static_cast<u32>(program.empty()));
        }

        if (program.empty()) {
            V115DA7YShaderTraceRaw("v115d_a7y generated_program_empty_return_false");
            V115DA7ZShaderTraceRaw("v115d_a7z generated_program_empty_return_false");
            LOG_ERROR(Render_Vulkan,
                      "TRACE_ACCEL_STAGE v114 programmable VS GLSL generation returned empty program");
            return false;
        }

        std::vector<u32> spirv;
        if (compile_spirv || create_module) {
            try {
                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_before_spirv_compile program_bytes={}",
                                mode_name, program.size());
                }
                V115DA7YShaderTraceRaw("v115d_a7y before_spirv_compile");
                V115DA7YShaderTraceNumber("v115d_a7y spirv_input_glsl_size", program.size());

                spirv = CompileGLSLtoSPIRV(program, vk::ShaderStageFlagBits::eVertex,
                                           instance.GetDevice());

                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_after_spirv_compile words={} bytes={} empty={}",
                                mode_name, spirv.size(), spirv.size() * sizeof(u32),
                                static_cast<u32>(spirv.empty()));
                }
                V115DA7YShaderTraceRaw("v115d_a7y after_spirv_compile");
                V115DA7YShaderTraceNumber("v115d_a7y spirv_words", spirv.size());
                V115DA7YShaderTraceNumber("v115d_a7y spirv_empty", static_cast<u64>(spirv.empty()));
            } catch (const std::exception& e) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v114 programmable VS GLSL->SPIR-V threw std::exception: {}",
                          e.what());
                return false;
            } catch (...) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v114 programmable VS GLSL->SPIR-V threw unknown exception");
                return false;
            }

            if (spirv.empty()) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v114 programmable VS GLSL->SPIR-V returned empty code");
                return false;
            }
        }

        if (create_module) {
            try {
                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_before_shader_module words={}",
                                mode_name, spirv.size());
                }

                vk::ShaderModule module =
                    CompileSPV(std::span<const u32>{spirv.data(), spirv.size()}, instance.GetDevice());

                const bool module_valid = static_cast<bool>(module);

                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_after_shader_module module_valid={}",
                                mode_name, static_cast<u32>(module_valid));
                }

                if (module_valid) {
                    instance.GetDevice().destroyShaderModule(module);
                    if (trace_accel) {
                        LOG_WARNING(Render_Vulkan,
                                    "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_destroyed_shader_module result=1",
                                    mode_name);
                    }
                }

                if (!module_valid) {
                    return false;
                }
            } catch (const std::exception& e) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v114 programmable VS VkShaderModule probe threw std::exception: {}",
                          e.what());
                return false;
            } catch (...) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v114 programmable VS VkShaderModule probe threw unknown exception");
                return false;
            }
        }

        pipeline_cache.UseTrivialVertexShader();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_{}_trivial_bind result=1",
                        mode_name);
        }
        V115DA7XShaderTraceRaw("v115d_a7x setup_vertex_shader_probe_return_true");
        V115DA7YShaderTraceRaw("v115d_a7y setup_vertex_shader_probe_return_true");
        V115DA7ZShaderTraceRaw("v115d_a7z setup_vertex_shader_probe_return_true");
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 setup_vertex_shader_probe_return_true");

        return true;
    };

    if (programmable_generate_guarded_probe) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux shader_probe_generate_guarded_selected");
        }
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_generate_guarded_probe_selected");
        }
        return run_shader_multiplex_probe("programmable_generate_guarded_only", false, false);
    }

    if (programmable_spirv_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_spirv_only_probe_selected");
        }
        return run_shader_multiplex_probe("programmable_spirv_only", true, false);
    }

    if (programmable_module_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_shader_module_only_probe_selected");
        }
        return run_shader_multiplex_probe("programmable_shader_module_only", true, true);
    }

    if (programmable_generate_probe) {
        return run_shader_multiplex_probe("programmable_generate_only", false, false);
    }

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_begin");
    }

    const bool result = pipeline_cache.UseProgrammableVertexShader(
        regs, pica.vs_setup, pipeline_info.vertex_layout, accurate_mul);

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 vertex_shader_setup_programmable_end result={}",
                    static_cast<u32>(result));
    }

    return result;
}

bool RasterizerVulkan::SetupGeometryShader() {
    BORKED3DS_PROFILE("Vulkan", "Geometry Shader Setup");

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        LOG_ERROR(Render_Vulkan, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    if (regs.lighting.disable || instance.IsFragmentShaderBarycentricSupported()) {
        pipeline_cache.UseTrivialGeometryShader();
        return true;
    }

    return pipeline_cache.UseFixedGeometryShader(regs);
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    // v114 diagnostic:
    // v110 proved raw_enter_noargs is safe. Now emit raw_enter_noargs plus raw_enter_simple,
    // then return before stage=1. No shader setup, no SPIR-V, no pipeline, no descriptors,
    // and no Vulkan draw command are reached.
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 accelerate_draw_batch_enter_pre_silent_gate");
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 accelerate_draw_batch_indexed_pre_silent_gate",
                               static_cast<u64>(is_indexed));
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 accelerate_draw_batch_num_vertices_pre_silent_gate",
                               regs.pipeline.num_vertices);
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 accelerate_draw_batch_stop_after_pre_silent_gate",
                               GetAccelStageStopAfter());
    if (IsAccelSilentEntryReturnEnabled()) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 accelerate_draw_batch_silent_entry_return_true");
        return true;
    }

    const bool v114_entry_safe = IsV114ShaderMultiplexEntrySafeEnabled();
    const bool v114_silent_stages = IsV114ShaderMultiplexSilentStagesEnabled();
    const bool v114_file_trace = IsV114ShaderMultiplexFileTraceEnabled();

    // v115-D-A7Z27C: cache the fragile post-before_record gates at function entry.
    // The previous rebuild proved the A7Z27 flag is visible at accel entry, but the sidecar
    // still cut immediately after real_vertex_bind_mux_before_record and before the A7Z27
    // return markers. Reusing the cached booleans below removes repeated getenv()/strict
    // gate evaluation from the fragile post-before_record boundary.
    const bool a7z23_return_false_after_pipeline_bind =
        IsV115DA7Z23MuxReturnFalseAfterPipelineBindEnabled();
    const bool a7z23b_return_false_before_pipeline_bind =
        IsV115DA7Z23BReturnFalseBeforePipelineBindEnabled();
    const bool a7z26_return_false_after_before_record =
        IsV115DA7Z26MuxReturnFalseAfterBeforeRecordEnabled();
    const bool a7z26f_return_false_after_selected_step =
        IsV115DA7Z26FReturnFalseAfterSelectedStepEnabled();
    const bool a7z26g_return_false_after_final_count =
        IsV115DA7Z26GReturnFalseAfterFinalCountEnabled();
    const bool a7z26h_return_false_after_final_vertex_offset =
        IsV115DA7Z26HReturnFalseAfterFinalVertexOffsetEnabled();
    const bool a7z26i_return_false_after_internal_binding_count =
        IsV115DA7Z26IReturnFalseAfterInternalBindingCountEnabled();
    const bool a7z26j_return_false_after_internal_flags =
        IsV115DA7Z26JReturnFalseAfterInternalFlagCacheEnabled();
    const bool a7z26k_return_false_after_internal_stage10 =
        IsV115DA7Z26KReturnFalseAfterInternalStage10Enabled();
    const bool a7z26l_return_false_after_internal_stage10_single_marker =
        IsV115DA7Z26LReturnFalseAfterInternalStage10SingleMarkerEnabled();
    const u32 a7z26_multi_probe_step =
        GetEnvU32("BORKED3DS_V3DV_A7Z26_MULTI_PROBE_STEP", 0);
    const u32 a7z26_mp3l_substep =
        GetEnvU32("BORKED3DS_V3DV_A7Z26_MP3L_SUBSTEP", 0);
    const bool a7z27_return_false_before_binding_count_number =
        IsV115DA7Z27MuxReturnFalseBeforeBindingCountNumberEnabled();

    if (!v114_entry_safe) {
        LOG_WARNING(Render_Vulkan, "TRACE_ACCEL_STAGE v114 raw_enter_noargs");
    }

    if (IsAccelRawEnterReturnEnabled()) {
        return true;
    }

    const u64 accel_id = ++g_vk_accel_draw_counter;
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 accelerate_draw_batch_after_accel_id");
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 accel_id", accel_id);
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 indexed", static_cast<u64>(is_indexed));
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 num_vertices", regs.pipeline.num_vertices);
    V115DA7Z2ShaderTraceNumber("v115d_a7z2 topology",
                               static_cast<u64>(regs.pipeline.triangle_topology.Value()));

    if (v114_file_trace && accel_id == 1) {
        V114ShaderMultiplexFileTraceReset();
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux accel_enter");
        V114ShaderMultiplexFileTraceNumber("v115d_mux accel_id", accel_id);
        V114ShaderMultiplexFileTraceNumber("v115d_mux accel_indexed", static_cast<u64>(is_indexed));
        V114ShaderMultiplexFileTraceNumber("v115d_mux accel_num_vertices",
                                           regs.pipeline.num_vertices);
        V114ShaderMultiplexFileTraceNumber("v115d_mux accel_topology",
                                           static_cast<u64>(regs.pipeline.triangle_topology.Value()));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z23_return_false_after_pipeline_bind",
            static_cast<u64>(a7z23_return_false_after_pipeline_bind));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z23b_return_false_before_pipeline_bind",
            static_cast<u64>(a7z23b_return_false_before_pipeline_bind));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26_return_false_after_before_record",
            static_cast<u64>(a7z26_return_false_after_before_record));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26f_return_false_after_selected_step",
            static_cast<u64>(a7z26f_return_false_after_selected_step));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26g_return_false_after_final_count",
            static_cast<u64>(a7z26g_return_false_after_final_count));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26h_return_false_after_final_vertex_offset",
            static_cast<u64>(a7z26h_return_false_after_final_vertex_offset));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26i_return_false_after_internal_binding_count",
            static_cast<u64>(a7z26i_return_false_after_internal_binding_count));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26j_return_false_after_internal_flags",
            static_cast<u64>(a7z26j_return_false_after_internal_flags));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26k_return_false_after_internal_stage10",
            static_cast<u64>(a7z26k_return_false_after_internal_stage10));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z26l_return_false_after_internal_stage10_single_marker",
            static_cast<u64>(a7z26l_return_false_after_internal_stage10_single_marker));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux a7z26_multi_probe_step",
            static_cast<u64>(a7z26_multi_probe_step));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux a7z26_mp3l_substep",
            static_cast<u64>(a7z26_mp3l_substep));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux flag_a7z27_return_false_before_binding_count_number",
            static_cast<u64>(a7z27_return_false_before_binding_count_number));
        V114ShaderMultiplexFileTraceRaw("v115d_mux flags_cached_for_post_before_record");
    }

    if (!v114_entry_safe) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 raw_enter_simple accel_id={} indexed={} stop_after={} force_stage_trace={} entry_only_probe={}",
                    accel_id, is_indexed, GetAccelStageStopAfter(),
                    static_cast<u32>(IsForceAccelStageTraceEnabled()),
                    static_cast<u32>(IsAccelEntryOnlyProbeEnabled()));
    }

    if (IsAccelRawEnterSimpleReturnEnabled()) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 raw_enter_simple_return_true");
        return true;
    }

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage_lambda_setup");

    if (IsAccelEntryOnlyProbeEnabled()) {
        // Entry-only probe: prove the call boundary and return before stage=1. Keep this log
        // deliberately simple: no regs, no shader state, no framebuffer addresses.
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 entry_only_probe_consumed before_stage1 indexed={} result=1",
                    is_indexed);
        return true;
    }

    const bool trace_accel = IsAccelStageTraceEnabled() && !v114_file_trace;

    const auto log_stage = [&](u32 stage, const char* name) {
        if (v114_silent_stages && stage <= 6) {
            return;
        }
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 accel_id={} stage={} name={} indexed={} num_vertices={} topology={} use_gs={} preflight_expected={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        accel_id, stage, name, is_indexed, regs.pipeline.num_vertices,
                        static_cast<u32>(regs.pipeline.triangle_topology.Value()),
                        static_cast<u32>(regs.pipeline.use_gs.Value()),
                        static_cast<u32>(IsAccelEntryPreflightExpected()),
                        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }
    };

    const auto consume_if_stage_limited = [&](u32 stage, const char* name) {
        log_stage(stage, name);
        if (ShouldStopAfterAccelStage(stage)) {
            if (trace_accel && !(v114_silent_stages && stage <= 6)) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 stage_limit consumed accel_id={} stage={} name={} stop_after={} before_vulkan_command=1",
                            accel_id, stage, name, GetAccelStageStopAfter());
            }
            return true;
        }
        return false;
    };

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage1_consume");
    if (consume_if_stage_limited(1, "enter_accelerate_draw_batch")) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage1_return_true");
        return true;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_stage1_consume");

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 rejected_gs_mode accel_id={} gs_mode={}",
                            accel_id, static_cast<u32>(regs.pipeline.gs_config.mode.Value()));
            }
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 rejected_gs_topology accel_id={} topology={}",
                            accel_id, static_cast<u32>(regs.pipeline.triangle_topology.Value()));
            }
            return false;
        }
    }

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage2_consume");
    if (consume_if_stage_limited(2, "geometry_shader_gate_ok")) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage2_return_true");
        return true;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_stage2_consume");

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_topology_assign");
    pipeline_info.rasterization.topology.Assign(regs.pipeline.triangle_topology);
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_topology_assign");
    if (consume_if_stage_limited(3, "topology_assigned")) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage3_return_true");
        return true;
    }

    if (regs.pipeline.triangle_topology == TriangleTopology::Fan &&
        !instance.IsTriangleFanSupported()) {
        LOG_DEBUG(Render_Vulkan,
                  "Skipping accelerated draw with unsupported triangle fan topology");
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 rejected_triangle_fan accel_id={}", accel_id);
        }
        return false;
    }

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage4_consume");
    if (consume_if_stage_limited(4, "topology_supported")) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage4_return_true");
        return true;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_stage4_consume");

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_analyze_vertex_array");
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux before_analyze_vertex_array");
    }
    vertex_info = AnalyzeVertexArray(is_indexed, instance.GetMinVertexStrideAlignment());
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_analyze_vertex_array");
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux after_analyze_vertex_array");
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage5_consume");
    if (consume_if_stage_limited(5, "vertex_array_analyzed")) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage5_return_true");
        return true;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_stage5_consume");

    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_setup_vertex_array");
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux before_setup_vertex_array");
    }
    SetupVertexArray();
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_setup_vertex_array");
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux after_setup_vertex_array");
        V115DA7XShaderTraceRaw("v115d_a7x accel_stage6_done");
        V115DA7YShaderTraceRaw("v115d_a7y accel_stage6_done");
        V115DA7ZShaderTraceRaw("v115d_a7z accel_stage6_done");
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage6_consume");
    if (consume_if_stage_limited(6, "vertex_array_setup_done")) {
        V115DA7XShaderTraceRaw("v115d_a7x stage6_return_true");
        V115DA7YShaderTraceRaw("v115d_a7y stage6_return_true");
        V115DA7ZShaderTraceRaw("v115d_a7z stage6_return_true");
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage6_return_true");
        return true;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 after_stage6_consume_before_setup_vertex_shader");

    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux before_setup_vertex_shader");
        V115DA7XShaderTraceRaw("v115d_a7x before_setup_vertex_shader");
        V115DA7YShaderTraceRaw("v115d_a7y before_setup_vertex_shader_from_accel");
        V115DA7ZShaderTraceRaw("v115d_a7z before_setup_vertex_shader_from_accel");
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 immediately_before_setup_vertex_shader_call");
    if (!SetupVertexShader()) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 setup_vertex_shader_result=0");
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vertex_shader_result=0");
            V115DA7YShaderTraceRaw("v115d_a7y setup_vertex_shader_result=0");
            V115DA7ZShaderTraceRaw("v115d_a7z setup_vertex_shader_result=0");
        }
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 vertex_shader_setup_failed accel_id={} trivial_probe={} programmable_config_probe={} programmable_before_generate_probe={} programmable_generate_guarded_probe={} programmable_spirv_probe={} programmable_module_probe={} programmable_generate_probe={} force_stage_trace={}",
                        accel_id, static_cast<u32>(IsTrivialVertexShaderProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderConfigProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderBeforeGenerateOnlyProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderSpirvOnlyProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderModuleOnlyProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateProbeEnabled()),
                        static_cast<u32>(IsForceAccelStageTraceEnabled()));
        }
        return false;
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 setup_vertex_shader_result=1");
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux setup_vertex_shader_result=1");
        V115DA7XShaderTraceRaw("v115d_a7x setup_vertex_shader_result=1");
        V115DA7YShaderTraceRaw("v115d_a7y setup_vertex_shader_result=1");
        V115DA7ZShaderTraceRaw("v115d_a7z setup_vertex_shader_result=1");
    }
    const char* stage7_name = "vertex_shader_setup_ok";
    if (IsTrivialVertexShaderProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_trivial_probe";
    } else if (IsProgrammableVertexShaderConfigProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_config_only";
    } else if (IsProgrammableVertexShaderBeforeGenerateOnlyProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_before_generate_only";
    } else if (IsProgrammableVertexShaderGenerateGuardedProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_generate_guarded_only";
    } else if (IsProgrammableVertexShaderSpirvOnlyProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_spirv_only";
    } else if (IsProgrammableVertexShaderModuleOnlyProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_shader_module_only";
    } else if (IsProgrammableVertexShaderGenerateProbeEnabled()) {
        stage7_name = "vertex_shader_setup_ok_programmable_generate_only";
    }

    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux before_stage7_consume");
        V115DA7XShaderTraceRaw("v115d_a7x before_stage7_consume");
        V115DA7YShaderTraceRaw("v115d_a7y before_stage7_consume");
        V115DA7ZShaderTraceRaw("v115d_a7z before_stage7_consume");
    }
    V115DA7Z2ShaderTraceRaw("v115d_a7z2 before_stage7_consume");
    if (consume_if_stage_limited(7, stage7_name)) {
        V115DA7Z2ShaderTraceRaw("v115d_a7z2 stage7_return_true");
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux stage7_consumed_return_true");
            V115DA7Z3ShaderTraceRaw("v115d_a7z3 stage7_return_true");
            V115DA7XShaderTraceRaw("v115d_a7x stage7_return_true");
            V115DA7YShaderTraceRaw("v115d_a7y stage7_return_true");
            V115DA7ZShaderTraceRaw("v115d_a7z stage7_return_true");
        }
        return true;
    }

    if (!SetupGeometryShader()) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 geometry_shader_setup_failed accel_id={}", accel_id);
        }
        return false;
    }
    if (consume_if_stage_limited(8, "geometry_shader_setup_ok")) {
        return true;
    }

    if (consume_if_stage_limited(9, "before_draw_wrapper")) {
        return true;
    }

    const bool result = Draw(true, is_indexed);
    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 accel_id={} stage=18 name=after_draw_wrapper result={}",
                    accel_id, result);
    }
    return result;
}

bool RasterizerVulkan::AccelerateDrawBatchInternal(bool is_indexed) {
    const bool trace_accel = IsAccelStageTraceEnabled();
    const bool v114_file_trace = IsV114ShaderMultiplexFileTraceEnabled();

    // v115-D-A7Z27C2: cache the fragile post-before_record gates inside the internal
    // draw path as well. The previous patch cached them in AccelerateDrawBatch(),
    // but the post-before_record mux lives in AccelerateDrawBatchInternal(), so the
    // cached booleans must be local to this function to compile and to reflect the
    // exact path being tested.
    const bool a7z23_return_false_after_pipeline_bind =
        IsV115DA7Z23MuxReturnFalseAfterPipelineBindEnabled();
    const bool a7z23b_return_false_before_pipeline_bind =
        IsV115DA7Z23BReturnFalseBeforePipelineBindEnabled();
    const bool a7z26_return_false_after_before_record =
        IsV115DA7Z26MuxReturnFalseAfterBeforeRecordEnabled();
    const bool a7z26f_return_false_after_selected_step =
        IsV115DA7Z26FReturnFalseAfterSelectedStepEnabled();
    const bool a7z26g_return_false_after_final_count =
        IsV115DA7Z26GReturnFalseAfterFinalCountEnabled();
    const bool a7z26h_return_false_after_final_vertex_offset =
        IsV115DA7Z26HReturnFalseAfterFinalVertexOffsetEnabled();
    const bool a7z26i_return_false_after_internal_binding_count =
        IsV115DA7Z26IReturnFalseAfterInternalBindingCountEnabled();
    const bool a7z26j_return_false_after_internal_flags =
        IsV115DA7Z26JReturnFalseAfterInternalFlagCacheEnabled();
    const bool a7z26k_return_false_after_internal_stage10 =
        IsV115DA7Z26KReturnFalseAfterInternalStage10Enabled();
    const bool a7z26l_return_false_after_internal_stage10_single_marker =
        IsV115DA7Z26LReturnFalseAfterInternalStage10SingleMarkerEnabled();
    const u32 a7z26_multi_probe_step =
        GetEnvU32("BORKED3DS_V3DV_A7Z26_MULTI_PROBE_STEP", 0);
    const u32 a7z26_mp3l_substep =
        GetEnvU32("BORKED3DS_V3DV_A7Z26_MP3L_SUBSTEP", 0);
    const u32 a7z26_mp3m_chain =
        GetEnvU32("BORKED3DS_V3DV_A7Z26_MP3M_CHAIN", 0);
    const bool a7z27_return_false_before_binding_count_number =
        IsV115DA7Z27MuxReturnFalseBeforeBindingCountNumberEnabled();

    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_mux internal_flags_cached_for_post_before_record");
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_a7z26_multi_probe_step",
            static_cast<u64>(a7z26_multi_probe_step));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_a7z26_mp3l_substep",
            static_cast<u64>(a7z26_mp3l_substep));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_a7z26_mp3m_chain",
            static_cast<u64>(a7z26_mp3m_chain));
    }

    // A7Z26MP2: fast multi-probe gate. Keep the multi-probe path quiet before
    // the fragile internal trace area. The previous MP build proved that the
    // sidecar can stop while writing the long list of internal flag breadcrumbs,
    // before the selected step is actually reached. When a multi-probe step is
    // active, avoid that long breadcrumb list and jump directly to the requested
    // gate below.
    if (a7z26_multi_probe_step == 1) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step01_return_false_after_internal_flags_before_stage10");
        }
        return false;
    }

    if (v114_file_trace && a7z26_multi_probe_step == 0) {
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z23_return_false_after_pipeline_bind",
            static_cast<u64>(a7z23_return_false_after_pipeline_bind));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z23b_return_false_before_pipeline_bind",
            static_cast<u64>(a7z23b_return_false_before_pipeline_bind));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26_return_false_after_before_record",
            static_cast<u64>(a7z26_return_false_after_before_record));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26f_return_false_after_selected_step",
            static_cast<u64>(a7z26f_return_false_after_selected_step));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26g_return_false_after_final_count",
            static_cast<u64>(a7z26g_return_false_after_final_count));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26h_return_false_after_final_vertex_offset",
            static_cast<u64>(a7z26h_return_false_after_final_vertex_offset));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26i_return_false_after_internal_binding_count",
            static_cast<u64>(a7z26i_return_false_after_internal_binding_count));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26j_return_false_after_internal_flags",
            static_cast<u64>(a7z26j_return_false_after_internal_flags));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26k_return_false_after_internal_stage10",
            static_cast<u64>(a7z26k_return_false_after_internal_stage10));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z26l_return_false_after_internal_stage10_single_marker",
            static_cast<u64>(a7z26l_return_false_after_internal_stage10_single_marker));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_a7z26_multi_probe_step",
            static_cast<u64>(a7z26_multi_probe_step));
        V114ShaderMultiplexFileTraceNumber(
            "v115d_mux internal_flag_a7z27_return_false_before_binding_count_number",
            static_cast<u64>(a7z27_return_false_before_binding_count_number));
    }

    if (a7z26j_return_false_after_internal_flags) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26j return_false_after_internal_flags_before_stage10");
        }
        return false;
    }

    if (a7z26_multi_probe_step == 1) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp step01_return_false_after_internal_flags_before_stage10");
        }
        return false;
    }

    const auto log_stage = [&](u32 stage, const char* name) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 internal stage={} name={} indexed={} vertex_count={} binding_count={} dry_run={} stop_after={}",
                        stage, name, is_indexed, regs.pipeline.num_vertices,
                        pipeline_info.vertex_layout.binding_count,
                        static_cast<u32>(IsStrictAccelInternalDryRunEnabled()),
                        GetAccelStageStopAfter());
        }
    };

    const auto consume_if_stage_limited = [&](u32 stage, const char* name) {
        log_stage(stage, name);
        if (ShouldStopAfterAccelStage(stage)) {
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v114 internal stage_limit consumed stage={} name={} before_vulkan_command=1",
                            stage, name);
            }
            return true;
        }
        return false;
    };

    if (consume_if_stage_limited(10, "internal_enter")) {
        return true;
    }
    if (a7z26_multi_probe_step == 2) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step02_return_false_after_internal_stage10_before_vertex_count");
        }
        return false;
    }
    if (a7z26l_return_false_after_internal_stage10_single_marker) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26l return_false_after_internal_stage10_single_marker_before_vertex_count");
        }
        return false;
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_after_stage10");
    }
    if (a7z26k_return_false_after_internal_stage10) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26k return_false_after_internal_stage10_before_vertex_count");
        }
        return false;
    }

    if (regs.pipeline.num_vertices == 0) {
        if (trace_accel) {
            LOG_INFO(Render_Vulkan, "TRACE_ACCEL_STAGE v114 internal skipped empty draw");
        }
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_zero_vertices_return_true");
        }
        return true;
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_vertex_count_nonzero");
    }
    if (a7z26_multi_probe_step == 3) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step03_return_false_after_vertex_count_before_stage11");
        }
        return false;
    }

    // v115-D-D-A7Z26MP3D:
    // MP3D: keep the already validated MP2 steps intact. The MP3C high-step
    // marker reached step60 but did not return cleanly to PICA, while the old
    // MP2 step3 did return cleanly on the same build. Use shorter, single-marker
    // high-step probes and remove range breadcrumbs so the next tests isolate
    // control flow rather than the tracing payload.
    if (a7z26_multi_probe_step == 60) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s60_after_vertex");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 61) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s61_before_stage11");
        }
        return false;
    }

    if (consume_if_stage_limited(11, "vertex_count_ok")) {
        return true;
    }
    if (a7z26_multi_probe_step == 62) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s62_after_stage11_consume");
        }
        return false;
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_after_stage11");
    }
    if (a7z26_multi_probe_step == 63) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s63_after_stage11_marker");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 4) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp step04_return_false_after_internal_stage11_before_binding_count");
        }
        return false;
    }

    const u32 binding_count = pipeline_info.vertex_layout.binding_count;
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceNumber("v115d_a7z23b internal_binding_count",
                                           binding_count);
    }
    if (a7z26_multi_probe_step == 64) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s64_after_binding_count");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 5) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step05_return_false_after_internal_binding_count_before_vertex_buffer_count");
        }
        return false;
    }
    if (a7z26i_return_false_after_internal_binding_count) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26i return_false_after_internal_binding_count_before_vertex_buffer_count");
        }
        return false;
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceNumber("v115d_a7z23b internal_vertex_buffer_count",
                                           static_cast<u64>(vertex_buffers.size()));
    }
    if (a7z26_multi_probe_step == 65) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s65_after_vertex_buffers");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 6) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step06_return_false_after_vertex_buffer_count_before_binding_count_valid");
        }
        return false;
    }
    if (binding_count == 0 || binding_count > vertex_buffers.size()) {
        LOG_ERROR(Render_Vulkan, "Accelerated draw has invalid binding_count={} (max={})",
                  binding_count, vertex_buffers.size());
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 internal invalid_binding_count binding_count={} max={}",
                        binding_count, vertex_buffers.size());
        }
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_invalid_binding_count_return_false");
        }
        return false;
    }
    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_after_binding_count_valid");
    }
    if (a7z26_multi_probe_step == 66) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s66_after_binding_valid");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 7) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp2 step07_return_false_after_binding_count_valid_before_stage12");
        }
        return false;
    }

    // v115-D-D-A7Z26MP3D stage12 split:
    // MP3 step 80 was requested after stage12, but the observed log stopped after
    // internal_after_binding_count_valid and never reached internal_after_stage12.
    // Split the tiny gap around consume_if_stage_limited(12) so we do not rework
    // MP2 steps 2-7 and can identify whether the fragile point is before, inside,
    // or immediately after the stage12 consume check.
    if (a7z26_multi_probe_step == 70) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s70_before_stage12");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 71) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s71_before_stage12_call");
        }
        return false;
    }

    if (consume_if_stage_limited(12, "binding_count_ok")) {
        return true;
    }
    if (a7z26_multi_probe_step == 72) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s72_after_stage12_consume");
        }
        return false;
    }

    // v115-D-D-A7Z26MP3E:
    // Step 80 in MP3D was intended to return immediately after stage12, but the uploaded log
    // for step 80 stopped at internal_after_binding_count_valid and never emitted the
    // internal_after_stage12 breadcrumb. Step 72 and the stable MP2 step 8 both proved the
    // stage12 consume path can return cleanly, so make the high-step bridge cut at the same
    // quiet post-consume point before the longer internal_after_stage12 breadcrumb. This keeps
    // the 80+ progression usable without reworking the already validated 60-72/8 sequence.
    if (a7z26_multi_probe_step == 80) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3e_s80_after_stage12_consume_before_marker");
        }
        return false;
    }

    if (v114_file_trace) {
        V114ShaderMultiplexFileTraceRaw("v115d_a7z23b internal_after_stage12");
    }
    if (a7z26_multi_probe_step == 73) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s73_after_stage12_marker");
        }
        return false;
    }
    if (a7z26_multi_probe_step == 8) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z26mp step08_return_false_after_internal_stage12_before_index_setup");
        }
        return false;
    }

    if (IsPipelineBindProbeOnlyEnabled()) {
        const bool wait_built = true;
        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux pipeline_bind_probe_begin");
            V114ShaderMultiplexFileTraceNumber("v115d_mux pipeline_bind_probe_indexed",
                                            static_cast<u32>(is_indexed));
            V114ShaderMultiplexFileTraceNumber("v115d_mux pipeline_bind_probe_num_vertices",
                                            regs.pipeline.num_vertices);
            V114ShaderMultiplexFileTraceNumber("v115d_mux pipeline_bind_probe_binding_count",
                                            binding_count);
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_pipeline_bind");
        }
        if (trace_accel || IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_mux pipeline_bind_probe before_bind indexed={} num_vertices={} binding_count={} wait_built={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        is_indexed, regs.pipeline.num_vertices, binding_count,
                        static_cast<u32>(wait_built),
                        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        if (a7z23_return_false_after_pipeline_bind && v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z23 before_bind_pipeline_call");
        }

        const bool pipeline_ready = pipeline_cache.BindPipeline(pipeline_info, wait_built);

        if (a7z23_return_false_after_pipeline_bind && v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z23 after_bind_pipeline_call");
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux after_pipeline_bind");
            V114ShaderMultiplexFileTraceNumber("v115d_mux pipeline_bind_ready",
                                            static_cast<u32>(pipeline_ready));
            V114ShaderMultiplexFileTraceRaw("v115d_mux stage8_pipeline_bind_consumed_return_true");
        }
        if (trace_accel || IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_mux pipeline_bind_probe after_bind result={} before_vertex_or_index_bind=1 before_vkcmd_draw=1",
                        static_cast<u32>(pipeline_ready));
        }

        return pipeline_ready;
    }

    // v115-D-D-A7Z26MP3D/MP3F selected-step split:
    // Step 9 from MP2 stopped after internal_after_stage12 and before the first
    // D-step/index setup breadcrumb. Keep the same single env knob.
    //
    // MP3D/MP3E validated the pre-index path through:
    //   internal_after_stage12 -> any_step -> zero_count_draw
    //   -> real_vertex_bind_ultra_quiet_draw -> is_indexed=1
    //
    // MP3F now isolates the real indexed setup path:
    //   90: before SetupIndexArray()
    //   91: after SetupIndexArray()
    //   92: after stage13 consume/check
    //   93: after selected_step, before final_indexed/final_count/final_vertex_offset
    if (a7z26_multi_probe_step == 80) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s80_after_stage12");
        }
        return false;
    }

    const bool v115d_mux_step_a = IsV115DAMuxRealVertexBindDrawZeroEnabled();
    if (a7z26_multi_probe_step == 81) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s81_after_step_a");
        }
        return false;
    }

    const bool v115d_mux_step_b = IsV115DBMuxRealVertexBindDraw3Enabled();
    if (a7z26_multi_probe_step == 82) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s82_after_step_b");
        }
        return false;
    }

    const bool v115d_mux_step_c = IsV115DCMuxRealVertexBindDraw6Enabled();
    if (a7z26_multi_probe_step == 83) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s83_after_step_c");
        }
        return false;
    }

    const bool v115d_mux_step_d = IsV115DDMuxRealVertexBindDrawIndexedZeroEnabled();
    if (a7z26_multi_probe_step == 84) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s84_after_step_d");
        }
        return false;
    }

    const bool v115d_mux_step_e = IsV115DEMuxRealVertexBindDrawIndexed3Enabled();
    if (a7z26_multi_probe_step == 85) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s85_after_step_e");
        }
        return false;
    }

    const bool v115d_mux_any_step = IsV115DMuxAnyDrawCommandProbeEnabled();
    if (a7z26_multi_probe_step == 86) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s86_after_any_step");
        }
        return false;
    }

    const bool v115d_mux_zero_count_draw = IsFirstVkCmdDrawZeroCountProbeOnlyEnabled() ||
                                           IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled() ||
                                           v115d_mux_step_a || v115d_mux_step_d;
    if (a7z26_multi_probe_step == 87) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceNumber(
                "v115d_mp3d_s87_zero", static_cast<u32>(v115d_mux_zero_count_draw));
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s87_after_zero");
        }
        return false;
    }

    const bool v115d_mux_real_vertex_bind_ultra_quiet_draw =
        IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled() || v115d_mux_any_step;
    if (a7z26_multi_probe_step == 88) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceNumber(
                "v115d_mp3d_s88_realbind",
                static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s88_after_realbind");
        }
        return false;
    }

    if (a7z26_multi_probe_step == 89) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceNumber("v115d_mp3d_s89_indexed",
                                            static_cast<u32>(is_indexed));
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3d_s89_before_indexed");
        }
        return false;
    }

    if (is_indexed) {
        // v115-D-MUX: keep the indexed setup path available for every multiplex step.
        // For D-A/D-B/D-C the final Vulkan command is deliberately non-indexed, but the
        // original PICA command may still be indexed; keeping SetupIndexArray() unchanged
        // avoids changing two variables at once.
        if ((v115d_mux_zero_count_draw || v115d_mux_any_step) && IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux keep_setup_index_array_for_mux_step");
        }
        if (a7z26_multi_probe_step == 90) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3f_s90_before_setup_index");
            }
            return false;
        }

        SetupIndexArray();

        if (a7z26_multi_probe_step == 91) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3f_s91_after_setup_index");
            }
            return false;
        }

        // v115-D-D-A7Z26MP3G:
        // Step 92 reached internal_after_stage12 but did not return cleanly to PICA before
        // emitting the old after-stage13 marker. Split the fragile stage13 boundary into
        // shorter probes so we do not retest the already validated SetupIndexArray path.
        if (a7z26_multi_probe_step == 94) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3g_s94_before_stage13_consume");
            }
            return false;
        }

        const bool stage13_consumed = consume_if_stage_limited(
            13, v115d_mux_zero_count_draw ? "zero_count_index_array_setup_done"
                                          : "index_array_setup_done");

        if (a7z26_multi_probe_step == 95) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3g_s95_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3g_s95_after_stage13_consume_call");
            }

            // v115-D-D-A7Z26MP3L:
            // MP3K step 95 is validated, but MP3K step 96 still behaves like a fragile
            // high-step probe and can stop back at internal_after_stage12 before logging its
            // direct after-stage13 marker. Keep the stable numeric step 95 and use a tiny
            // substep selector to advance from the same validated location without changing
            // the surrounding indexed setup or stage13 consume path.
            // v115-D-D-A7Z26MP3M:
            // MP3L substep 1 is stable and validates the after_stage13_direct marker.
            // MP3L substep 2, when selected directly with SUBSTEP=2, can stop much earlier
            // at internal_after_binding_count_valid before reaching the same stage13 point.
            // Keep the known-good SUBSTEP=1 selector and add a tiny chained probe from that
            // already validated location. This avoids reusing the fragile SUBSTEP=2 path.
            if (a7z26_mp3l_substep == 1 && a7z26_mp3m_chain == 1) {
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3m_s95_chain_stage13_consumed",
                        static_cast<u32>(stage13_consumed));
                    V114ShaderMultiplexFileTraceRaw(
                        stage13_consumed
                            ? "v115d_mp3m_s95_chain_stage13_consumed_return_true"
                            : "v115d_mp3m_s95_chain_after_stage13_direct");
                }
                if (stage13_consumed) {
                    return true;
                }
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3m_s95_chain_after_stage13_consumed_branch");
                }
                return false;
            }

            if (a7z26_mp3l_substep == 1) {
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3l_s95_sub1_stage13_consumed",
                        static_cast<u32>(stage13_consumed));
                    V114ShaderMultiplexFileTraceRaw(
                        stage13_consumed
                            ? "v115d_mp3l_s95_sub1_stage13_consumed_return_true"
                            : "v115d_mp3l_s95_sub1_after_stage13_direct");
                }
                return stage13_consumed;
            }

            if (a7z26_mp3l_substep == 2) {
                if (stage13_consumed) {
                    if (v114_file_trace) {
                        V114ShaderMultiplexFileTraceRaw(
                            "v115d_mp3l_s95_sub2_stage13_consumed_return_true");
                    }
                    return true;
                }
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3l_s95_sub2_after_stage13_consumed_branch");
                }
                return false;
            }

            if (a7z26_mp3l_substep == 3) {
                if (stage13_consumed) {
                    if (v114_file_trace) {
                        V114ShaderMultiplexFileTraceRaw(
                            "v115d_mp3l_s95_sub3_stage13_consumed_return_true");
                    }
                    return true;
                }
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3l_s95_sub3_realbind",
                        static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3l_s95_sub3_before_realbind_branch");
                }
                return false;
            }

            if (a7z26_mp3l_substep == 4) {
                if (stage13_consumed) {
                    if (v114_file_trace) {
                        V114ShaderMultiplexFileTraceRaw(
                            "v115d_mp3l_s95_sub4_stage13_consumed_return_true");
                    }
                    return true;
                }
                if (v114_file_trace) {
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3l_s95_sub4_realbind",
                        static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
                    V114ShaderMultiplexFileTraceRaw(
                        v115d_mux_real_vertex_bind_ultra_quiet_draw
                            ? "v115d_mp3l_s95_sub4_realbind_branch_enter"
                            : "v115d_mp3l_s95_sub4_realbind_branch_not_taken");
                }
                return false;
            }

            return false;
        }

        // v115-D-D-A7Z26MP3K:
        // MP3J step 95 is now validated in the same build, while MP3J steps 97/98 were
        // still fragile high-step probes. Reuse the nearby step 96 number, but move its
        // probe immediately after the validated stage13 consume call. This avoids the old
        // later MP3G step96 location and gives a shorter direct cut:
        // stage13_consumed value -> after_stage13_direct.
        if (a7z26_multi_probe_step == 96) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3k_s96_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                if (stage13_consumed) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3k_s96_stage13_consumed_return_true");
                } else {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3k_s96_after_stage13_direct");
                }
            }
            return stage13_consumed;
        }

        // v115-D-D-A7Z26MP3I:
        // MP3H made step 95 fragile again in this user's Pi5/V3DV run even though
        // the MP3G version of step 95 was already validated. Restart from the MP3G
        // stable source and add only minimal after-stage13 probes. Do not retest or
        // alter the validated SetupIndexArray path.
        if (a7z26_multi_probe_step == 97) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3i_s97_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3i_s97_before_stage13_consumed_branch");
            }
            return false;
        }

        // v115-D-D-A7Z26MP3J:
        // MP3I step 97 proved the stage13 consume call and stage13_consumed value can be
        // logged cleanly. MP3I step 98 was placed only a few lines later, but on the Pi5/V3DV
        // test it became a high-step style fragile probe and stopped back at
        // internal_after_binding_count_valid. Keep the validated MP3I source and add direct,
        // same-scope probes that log and return immediately without falling through the
        // generic stage13 / realbind path. This avoids retesting SetupIndexArray or stage13.
        if (a7z26_multi_probe_step == 98) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3j_s98_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                if (stage13_consumed) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3j_s98_stage13_consumed_return_true");
                } else {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3j_s98_after_stage13_direct");
                }
            }
            return stage13_consumed;
        }

        if (a7z26_multi_probe_step == 99) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3j_s99_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                if (stage13_consumed) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3j_s99_stage13_consumed_return_true");
                } else {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3j_s99_after_stage13_before_realbind_check");
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3j_s99_realbind",
                        static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
                }
            }
            return stage13_consumed;
        }

        if (a7z26_multi_probe_step == 100) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3j_s100_stage13_consumed",
                    static_cast<u32>(stage13_consumed));
                if (stage13_consumed) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_mp3j_s100_stage13_consumed_return_true");
                } else {
                    V114ShaderMultiplexFileTraceNumber(
                        "v115d_mp3j_s100_realbind",
                        static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
                    V114ShaderMultiplexFileTraceRaw(
                        v115d_mux_real_vertex_bind_ultra_quiet_draw
                            ? "v115d_mp3j_s100_realbind_branch_enter"
                            : "v115d_mp3j_s100_realbind_branch_not_taken");
                }
            }
            return stage13_consumed;
        }

        if (stage13_consumed) {
            return true;
        }

        if (a7z26_multi_probe_step == 101) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3k_s101_after_stage13_consumed_branch");
            }
            return false;
        }

        if (a7z26_multi_probe_step == 92) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw("v115d_mp3f_s92_after_stage13");
            }
            return false;
        }
    } else if (consume_if_stage_limited(13, "nonindexed_no_index_setup")) {
        return true;
    }

    if (a7z26_multi_probe_step == 102) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceNumber(
                "v115d_mp3k_s102_realbind",
                static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3k_s102_before_realbind_branch");
        }
        return false;
    }

    if (a7z26_multi_probe_step == 99) {
        if (v114_file_trace) {
            V114ShaderMultiplexFileTraceNumber(
                "v115d_mp3i_s99_realbind",
                static_cast<u32>(v115d_mux_real_vertex_bind_ultra_quiet_draw));
            V114ShaderMultiplexFileTraceRaw(
                "v115d_mp3i_s99_before_realbind_branch");
        }
        return false;
    }

    if (v115d_mux_real_vertex_bind_ultra_quiet_draw) {
        if (a7z26_multi_probe_step == 103) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3k_s103_realbind_branch_enter");
            }
            return false;
        }

        if (a7z26_multi_probe_step == 100) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3i_s100_realbind_branch_enter");
            }
            return false;
        }

        // v115-D-MUX:
        // One build now contains the five draw-command probes agreed for the D series:
        //   D-A: vkCmdDraw(0)
        //   D-B: vkCmdDraw(3)
        //   D-C: vkCmdDraw(6)
        //   D-D: vkCmdDrawIndexed(0)
        //   D-E: vkCmdDrawIndexed(3)
        // Activate exactly one step from emulators.cfg. The old C15 rollback flag is
        // preserved and behaves like D-A.
        const bool wait_built = true;
        const bool final_indexed = v115d_mux_step_d || v115d_mux_step_e;
        const u32 final_count = (v115d_mux_step_b || v115d_mux_step_e) ? 3u
                              : v115d_mux_step_c                  ? 6u
                                                                   : 0u;
        const s32 final_vertex_offset = -static_cast<s32>(vertex_info.vs_input_index_min);
        u32 selected_step = 0;
        if (v115d_mux_step_a || IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled()) {
            selected_step = 1;
        }
        if (v115d_mux_step_b) {
            selected_step = 2;
        }
        if (v115d_mux_step_c) {
            selected_step = 3;
        }
        if (v115d_mux_step_d) {
            selected_step = 4;
        }
        if (v115d_mux_step_e) {
            selected_step = 5;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_before_bind_pipeline");
            V114ShaderMultiplexFileTraceNumber("v115d_mux selected_step", selected_step);
        }

        if (a7z26_multi_probe_step == 93) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceNumber(
                    "v115d_mp3f_s93_selected_step", selected_step);
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_mp3f_s93_after_selected_step_before_final_indexed");
            }
            return false;
        }

        if (a7z26_multi_probe_step == 9) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp2 step09_return_false_after_selected_step_before_final_indexed");
            }
            return false;
        }

        // v115-D-A7Z26F:
        // The latest uploaded test was not a valid A7Z26E run: the sidecar showed the A7Z26
        // flag as 0 and the reset list did not include the A7Z26B/C/D/E markers. This gate is
        // deliberately moved one breadcrumb earlier than A7Z26E so the next run can prove both
        // source and emulators.cfg alignment with the smallest possible boundary:
        //
        //   selected_step=4 -> A7Z26F marker -> return false
        //
        // It avoids final_indexed, final_count, final_vertex_offset, BindPipeline, before_record,
        // binding_count, offsets, scheduler.Record, vkCmdBindVertexBuffers, and all draw commands.
        if (a7z26f_return_false_after_selected_step) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26f return_false_after_selected_step_before_final_indexed");
            }
            return false;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceNumber("v115d_mux final_indexed",
                                               static_cast<u32>(final_indexed));
        }
        if (a7z26_multi_probe_step == 10) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp2 step10_return_false_after_final_indexed_before_final_count");
            }
            return false;
        }

        // v115-D-A7Z26E:
        // A7Z26F has now proven that source, binary, and emulators.cfg are aligned and that
        // returning false immediately after selected_step is clean. Advance one micro-step:
        // emit final_indexed, then return false before final_count, final_vertex_offset,
        // BindPipeline, before_record, binding_count, offsets, scheduler.Record,
        // vkCmdBindVertexBuffers, and vkCmdDrawIndexed.
        if (a7z26_return_false_after_before_record) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26e return_false_after_final_indexed_before_final_count");
            }
            return false;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceNumber("v115d_mux final_count", final_count);
        }
        if (a7z26_multi_probe_step == 11) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp2 step11_return_false_after_final_count_before_final_vertex_offset");
            }
            return false;
        }

        // v115-D-A7Z26G:
        // A7Z26E proved final_indexed is safe and returns false before final_count.
        // Advance exactly one breadcrumb: allow final_count, then return false before
        // final_vertex_offset, BindPipeline, before_record, binding_count, offsets,
        // scheduler.Record, vkCmdBindVertexBuffers, and vkCmdDrawIndexed.
        if (a7z26g_return_false_after_final_count) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26g return_false_after_final_count_before_final_vertex_offset");
            }
            return false;
        }

        // v115-D-A7Z26D:
        // A7Z26C proved the build is active and reached the pre-final-vertex-offset gate, but the
        // sidecar stopped after the second breadcrumb and never reached the explicit false marker.
        // Keep the same A7Z26 env switch, but make the gate ultra-minimal: one flushed sidecar
        // breadcrumb, no numeric breadcrumb, no LOG_WARNING, no final_vertex_offset trace, no
        // BindPipeline, no before_record, no binding_count, no offsets, no scheduler.Record, and no
        // Vulkan draw command. This isolates whether the crash is caused by the multi-breadcrumb /
        // normal-log path, or by returning false to PICA at this exact boundary.
        if (a7z26_return_false_after_before_record) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26d single_marker_return_false_before_final_vertex_offset");
            }
            return false;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceNumber("v115d_mux final_vertex_offset",
                                               static_cast<u64>(static_cast<s64>(final_vertex_offset)));
        }
        if (a7z26_multi_probe_step == 12) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp2 step12_return_false_after_final_vertex_offset_before_bind_pipeline");
            }
            return false;
        }

        // v115-D-A7Z26H:
        // A7Z26G proved final_count=0 is safe. Advance one breadcrumb: allow
        // final_vertex_offset to be emitted, then return false before BindPipeline,
        // before_record, binding_count, offsets, scheduler.Record, vkCmdBindVertexBuffers,
        // and vkCmdDrawIndexed.
        if (a7z26h_return_false_after_final_vertex_offset) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26h return_false_after_final_vertex_offset_before_bind_pipeline");
            }
            return false;
        }

        if (a7z23b_return_false_before_pipeline_bind) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z23b return_false_before_pipeline_bind_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z23b return_false_before_pipeline_bind_false");
            }
            return false;
        }

        if (IsV115DA7Z17MuxUltraCleanReturnBeforePipelineBindEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z17 mux_ultra_clean_return_before_pipeline_bind_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z17 mux_ultra_clean_return_before_pipeline_bind_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z17 mux ultra-clean return before pipeline bind result=1 selected_step={} final_indexed={} final_count={} final_vertex_offset={}",
                            selected_step, static_cast<u32>(final_indexed), final_count,
                            final_vertex_offset);
            }
            return true;
        }

        if (IsV115DA7Z18MuxUltraCleanReturnFalseBeforePipelineBindEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z18 mux_ultra_clean_return_false_before_pipeline_bind_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z18 mux_ultra_clean_return_false_before_pipeline_bind_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z18 mux ultra-clean return false before pipeline bind result=0 selected_step={} final_indexed={} final_count={} final_vertex_offset={}",
                            selected_step, static_cast<u32>(final_indexed), final_count,
                            final_vertex_offset);
            }
            return false;
        }

        if (a7z26_multi_probe_step == 13) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp step13_return_false_before_bind_pipeline_call");
            }
            return false;
        }

        const bool pipeline_ready = pipeline_cache.BindPipeline(pipeline_info, wait_built);

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            if (pipeline_ready) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_bind_pipeline_ready_true");
            } else {
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_bind_pipeline_ready_false");
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_return_false");
            }
        }
        if (!pipeline_ready) {
            return false;
        }
        if (a7z26_multi_probe_step == 14) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp step14_return_false_after_bind_pipeline_before_a7z23");
            }
            return false;
        }

        if (a7z23_return_false_after_pipeline_bind) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z23 mux_return_false_after_pipeline_bind_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z23 mux_return_false_after_pipeline_bind_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z23 mux return false after pipeline bind result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        if (IsV115DA7Z12MuxReturnAfterPipelineBindEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_return_after_pipeline_bind_begin");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_return_after_pipeline_bind_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z12 mux return after pipeline bind result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z25MuxReturnFalseBeforeBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z25 mux_return_false_before_binding_count_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z25 mux_return_false_before_binding_count_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z25 mux return false before binding_count result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        // v115-D-A7Z26B:
        // The corrected A7Z26 run proved the flag is visible in both the outer and internal
        // scopes, but the sidecar stops immediately after bind_pipeline_ready_true and before
        // the generic real_vertex_bind_mux_before_record breadcrumb. A7Z25 already proved a
        // return before this breadcrumb is safe. Therefore, when the A7Z26 flag is selected,
        // return at this same boundary using an A7Z26-specific marker and deliberately skip
        // the generic before_record breadcrumb. This isolates whether the wall is the
        // breadcrumb itself or the first instruction after it, without advancing to
        // binding_count, offsets, scheduler.Record, vkCmdBindVertexBuffers, or vkCmdDrawIndexed.
        if (a7z26_return_false_after_before_record) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26b pre_before_record_gate_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26b skipped_generic_before_record_marker");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26b pre_before_record_gate_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z26b pre-before-record gate result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        if (a7z26_multi_probe_step == 15) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp step15_return_false_before_before_record_marker");
            }
            return false;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_before_record");
        }
        if (a7z26_multi_probe_step == 16) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26mp step16_return_false_after_before_record_before_binding_count_number");
            }
            return false;
        }

        if (a7z27_return_false_before_binding_count_number) {
            if (v114_file_trace) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z27c cached_return_false_before_binding_count_number_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z27c cached_return_false_before_binding_count_number_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z27c cached mux return false before binding_count number result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        if (a7z26_return_false_after_before_record) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26 mux_return_false_after_before_record_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z26 mux_return_false_after_before_record_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z26 mux return false after before_record result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        if (IsV115DA7Z15MuxUltraCleanReturnBeforeBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z15 mux_ultra_clean_return_before_binding_count_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z15 mux_ultra_clean_return_before_binding_count_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z15 mux ultra-clean return before binding_count result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z14MuxReturnBeforeBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z14 mux_return_before_binding_count_begin");
                V114ShaderMultiplexFileTraceNumber("v115d_a7z14 flag_a7z12_draw_raw",
                                                   static_cast<u32>(IsV115DA7Z12MuxDrawRawEnabled()));
                V114ShaderMultiplexFileTraceNumber("v115d_a7z14 flag_a7z13_return_after_binding_count",
                                                   static_cast<u32>(IsV115DA7Z13MuxReturnAfterBindingCountEnabled()));
                V114ShaderMultiplexFileTraceRaw("v115d_a7z14 mux_return_before_binding_count_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z14 mux return before binding_count result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z28MuxSkipBindingCountNumberReturnFalseEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z28 mux_binding_count_number_skipped");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z28 mux_skip_binding_count_number_return_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z28 mux skip binding_count number return false result=0 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return false;
        }

        if (IsV115DA7Z31B2EmptyRecordReturnFalseEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31b2 mux_after_before_record_entry");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31b2 mux_binding_count_number_skipped_continue");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31b2 mux_manual_offsets_begin");
            }
            std::array<vk::DeviceSize, 16> a7z31b2_offsets{};
            for (size_t offset_index = 0; offset_index < a7z31b2_offsets.size(); ++offset_index) {
                a7z31b2_offsets[offset_index] =
                    static_cast<vk::DeviceSize>(binding_offsets[offset_index]);
            }
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31b2 mux_manual_offsets_end");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31b2 mux_empty_record_begin");
            }
            scheduler.Record([](vk::CommandBuffer cmdbuf) { (void)cmdbuf; });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31b2 mux_empty_record_after_record");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31b2 mux_empty_record_return_false");
            }
            return false;
        }

        if (IsV115DA7Z31C3BindVertexBuffer0OnlyReturnFalseEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c3 mux_after_before_record_entry");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c3 mux_binding_count_number_skipped_continue");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31c3 mux_manual_offsets_begin");
            }
            const vk::Buffer a7z31c3_buffer0 = vertex_buffers[0];
            const vk::DeviceSize a7z31c3_offset0 =
                static_cast<vk::DeviceSize>(binding_offsets[0]);
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31c3 mux_manual_offsets_end");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c3 mux_bind_vertex_buffer0_record_begin");
            }
            scheduler.Record([a7z31c3_buffer0, a7z31c3_offset0](vk::CommandBuffer cmdbuf) {
                cmdbuf.bindVertexBuffers(0, 1, &a7z31c3_buffer0, &a7z31c3_offset0);
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c3 mux_bind_vertex_buffer0_after_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c3 mux_bind_vertex_buffer0_return_false");
            }
            return false;
        }

        if (IsV115DA7Z31C2BindVertexBuffersOnlyReturnFalseEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c2 mux_after_before_record_entry");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c2 mux_binding_count_number_skipped_continue");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31c2 mux_manual_offsets_begin");
            }
            std::array<vk::DeviceSize, 16> a7z31c2_offsets{};
            for (size_t offset_index = 0; offset_index < a7z31c2_offsets.size(); ++offset_index) {
                a7z31c2_offsets[offset_index] =
                    static_cast<vk::DeviceSize>(binding_offsets[offset_index]);
            }
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z31c2 mux_manual_offsets_end");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c2 mux_bind_vertex_buffers_record_begin");
            }
            scheduler.Record([this, binding_count, a7z31c2_offsets](vk::CommandBuffer cmdbuf) {
                if (binding_count != 0) {
                    cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(),
                                             a7z31c2_offsets.data());
                }
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c2 mux_bind_vertex_buffers_after_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z31c2 mux_bind_vertex_buffers_return_false");
            }
            return false;
        }

        if (IsV115DA7Z29CMuxImmediateReturnFalseBeforeOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29c mux_after_before_record_entry");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29c mux_binding_count_number_skipped");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29c mux_return_false_before_offsets_immediate");
            }
            return false;
        }

        const bool a7z29_skip_binding_count_number =
            IsV115DA7Z29MuxSkipBindingCountNumberReturnFalseBeforeOffsetsEnabled();
        const bool a7z29b_skip_binding_count_number =
            IsV115DA7Z29BMuxSkipBindingCountNumberReturnFalseBeforeOffsetsSafeEnabled();
        const bool a7z30_manual_offsets =
            IsV115DA7Z30MuxManualOffsetsReturnFalseAfterOffsetsEnabled();
        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            if (a7z30_manual_offsets) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z30 mux_binding_count_number_skipped_continue");
            } else if (a7z29b_skip_binding_count_number) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29b mux_binding_count_number_skipped_continue_safe");
            } else if (a7z29_skip_binding_count_number) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29 mux_binding_count_number_skipped_continue");
            } else {
                V114ShaderMultiplexFileTraceNumber("v115d_mux real_vertex_bind_mux_binding_count",
                                                   binding_count);
            }
        }

        if (IsV115DA7Z16MuxUltraCleanReturnAfterBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z16 mux_ultra_clean_return_after_binding_count_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z16 mux_ultra_clean_return_after_binding_count_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z16 mux ultra-clean return after binding_count result=1 selected_step={} final_indexed={} final_count={} binding_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count,
                            binding_count);
            }
            return true;
        }

        if (IsV115DA7Z24MuxReturnFalseAfterBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z24 mux_return_false_after_binding_count_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z24 mux_return_false_after_binding_count_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z24 mux return false after binding_count result=0 selected_step={} final_indexed={} final_count={} binding_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count,
                            binding_count);
            }
            return false;
        }

        if (IsV115DA7Z13MuxTraceAfterBindingCountEnabled() &&
            IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_after_binding_count_checkpoint");
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z12_return_after_pipeline_bind",
                static_cast<u32>(IsV115DA7Z12MuxReturnAfterPipelineBindEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z12_return_before_offsets",
                static_cast<u32>(IsV115DA7Z12MuxReturnBeforeOffsetsEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z12_return_after_offsets",
                static_cast<u32>(IsV115DA7Z12MuxReturnAfterOffsetsEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z12_bind_only_record",
                static_cast<u32>(IsV115DA7Z12MuxBindOnlyRecordEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z12_draw_raw",
                static_cast<u32>(IsV115DA7Z12MuxDrawRawEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z13_manual_offsets",
                static_cast<u32>(IsV115DA7Z13MuxManualOffsetsEnabled()));
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z13 flag_a7z13_return_after_manual_offsets",
                static_cast<u32>(IsV115DA7Z13MuxReturnAfterManualOffsetsEnabled()));
        }

        if (IsV115DA7Z13MuxReturnAfterBindingCountEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_return_after_binding_count_begin");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_return_after_binding_count_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z13 mux return after binding_count result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z12MuxReturnBeforeOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_return_before_offsets_begin");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_return_before_offsets_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z12 mux return before offsets result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (a7z29b_skip_binding_count_number) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29b mux_return_false_before_offsets_safe_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29b mux_return_false_before_offsets_safe_false");
            }
            return false;
        }

        if (a7z29_skip_binding_count_number) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29 mux_return_false_before_offsets");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z29 mux_return_false_before_offsets_false");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z29 mux skip binding_count number return false before offsets result=0 selected_step={} final_indexed={} final_count={} binding_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count,
                            binding_count);
            }
            return false;
        }

        if (IsV115DA7Z11DAReturnBeforeOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_return_before_offsets_begin");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_return_before_offsets_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z11 D-A return before offsets result=1 selected_step={} final_count={}",
                            selected_step, final_count);
            }
            return true;
        }

        std::array<vk::DeviceSize, 16> real_offsets{};
        if (a7z30_manual_offsets || IsV115DA7Z13MuxManualOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                if (a7z30_manual_offsets) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z30 mux_manual_offsets_begin");
                } else {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_manual_offsets_begin");
                    V114ShaderMultiplexFileTraceNumber("v115d_a7z13 mux_manual_offsets_binding_count",
                                                       binding_count);
                }
            }
            for (size_t offset_index = 0; offset_index < real_offsets.size(); ++offset_index) {
                real_offsets[offset_index] =
                    static_cast<vk::DeviceSize>(binding_offsets[offset_index]);
            }
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                if (a7z30_manual_offsets) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z30 mux_manual_offsets_end");
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_a7z30 mux_return_false_after_offsets");
                } else {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_manual_offsets_end");
                }
            }
            if (a7z30_manual_offsets) {
                return false;
            }
            if (IsV115DA7Z13MuxReturnAfterManualOffsetsEnabled()) {
                if (IsV114ShaderMultiplexFileTraceEnabled()) {
                    V114ShaderMultiplexFileTraceRaw(
                        "v115d_a7z13 mux_return_after_manual_offsets_true");
                }
                if (trace_accel || IsDrawTraceEnabled()) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_DRAW strict_compat v115d_a7z13 mux return after manual offsets result=1 selected_step={} final_indexed={} final_count={}",
                                selected_step, static_cast<u32>(final_indexed), final_count);
                }
                return true;
            }
        } else {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_std_offsets_begin");
            }
            std::transform(binding_offsets.begin(), binding_offsets.end(), real_offsets.begin(),
                           [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_std_offsets_end");
            }
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_after_offsets_build");
            V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_after_offsets_build");
            V114ShaderMultiplexFileTraceRaw("v115d_a7z13 mux_after_offsets_build");
        }

        if (IsV115DA7Z12MuxReturnAfterOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_return_after_offsets_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z12 mux return after offsets result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z11DAReturnAfterOffsetsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_return_after_offsets_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z11 D-A return after offsets result=1 selected_step={} final_count={}",
                            selected_step, final_count);
            }
            return true;
        }

        if (IsV115DA7Z12MuxBindOnlyRecordEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_bind_only_record_begin");
            }
            scheduler.Record([this, binding_count, real_offsets](vk::CommandBuffer cmdbuf) {
                if (binding_count != 0) {
                    cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
                }
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_bind_only_after_record");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_bind_only_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z12 mux bind-only record result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z11DABindOnlyRecordEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_bind_only_record_begin");
            }
            scheduler.Record([this, binding_count, real_offsets](vk::CommandBuffer cmdbuf) {
                if (binding_count != 0) {
                    cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
                }
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_bind_only_after_record");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_bind_only_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z11 D-A bind-only record result=1 selected_step={} final_count={}",
                            selected_step, final_count);
            }
            return true;
        }

        if (IsV115DA7Z12MuxDrawRawEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_draw_raw_record_begin");
            }
            scheduler.Record([this, binding_count, real_offsets, final_indexed, final_count,
                              final_vertex_offset](vk::CommandBuffer cmdbuf) {
                if (binding_count != 0) {
                    cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
                }
                if (final_indexed) {
                    cmdbuf.drawIndexed(final_count, 1, 0, final_vertex_offset, 0);
                } else {
                    cmdbuf.draw(final_count, 1, 0, 0);
                }
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_draw_raw_after_record");
                if (selected_step == 1) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z12 D_A_draw0_raw_return_true");
                } else if (selected_step == 2) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z12 D_B_draw3_raw_return_true");
                } else if (selected_step == 3) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z12 D_C_draw6_raw_return_true");
                } else if (selected_step == 4) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z12 D_D_drawindexed0_raw_return_true");
                } else if (selected_step == 5) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z12 D_E_drawindexed3_raw_return_true");
                }
                V114ShaderMultiplexFileTraceRaw("v115d_a7z12 mux_draw_raw_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z12 mux raw draw record result=1 selected_step={} final_indexed={} final_count={}",
                            selected_step, static_cast<u32>(final_indexed), final_count);
            }
            return true;
        }

        if (IsV115DA7Z11DADraw0RecordRawEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_draw0_raw_record_begin");
            }
            scheduler.Record([this, binding_count, real_offsets](vk::CommandBuffer cmdbuf) {
                if (binding_count != 0) {
                    cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
                }
                cmdbuf.draw(0, 1, 0, 0);
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_draw0_raw_after_record");
                V114ShaderMultiplexFileTraceRaw("v115d_a7z11 da_draw0_raw_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z11 D-A draw0 raw record result=1 selected_step={} final_count={}",
                            selected_step, final_count);
            }
            return true;
        }

        scheduler.Record([this, binding_count, real_offsets, final_indexed, final_count,
                          final_vertex_offset](vk::CommandBuffer cmdbuf) {
            if (binding_count != 0) {
                cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
            }
            if (final_indexed) {
                cmdbuf.drawIndexed(final_count, 1, 0, final_vertex_offset, 0);
            } else {
                cmdbuf.draw(final_count, 1, 0, 0);
            }
        });

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux real_vertex_bind_mux_after_record");
            if (selected_step == 1) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux D_A_draw0_recorded_return_true");
            } else if (selected_step == 2) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux D_B_draw3_recorded_return_true");
            } else if (selected_step == 3) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux D_C_draw6_recorded_return_true");
            } else if (selected_step == 4) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux D_D_drawindexed0_recorded_return_true");
            } else if (selected_step == 5) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux D_E_drawindexed3_recorded_return_true");
            } else {
                V114ShaderMultiplexFileTraceRaw("v115d_mux legacy_zero_count_recorded_return_true");
            }
        }
        return true;
    }

    if (IsDescriptorBindProbeOnlyEnabled() || IsFirstVkCmdDrawProbeOnlyEnabled() ||
        IsFirstVkCmdDrawZeroCountProbeOnlyEnabled() ||
        IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled()) {
        const bool v115d_mux_first_vkcmd_draw = IsFirstVkCmdDrawProbeOnlyEnabled() ||
                                            IsFirstVkCmdDrawZeroCountProbeOnlyEnabled() ||
                                            IsFirstVkCmdDrawZeroCountMinimalProbeOnlyEnabled();
        const bool v115d_mux_first_vkcmd_draw_zero_count = v115d_mux_zero_count_draw;
        const bool v115d_mux_first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet =
            v115d_mux_real_vertex_bind_ultra_quiet_draw;
        const bool wait_built = true;
        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux descriptor_bind_probe_begin");
            V114ShaderMultiplexFileTraceNumber("v115d_mux descriptor_bind_probe_indexed",
                                               static_cast<u32>(is_indexed));
            V114ShaderMultiplexFileTraceNumber("v115d_mux descriptor_bind_probe_num_vertices",
                                               regs.pipeline.num_vertices);
            V114ShaderMultiplexFileTraceNumber("v115d_mux descriptor_bind_probe_binding_count",
                                               binding_count);
            V114ShaderMultiplexFileTraceNumber("v115d_mux first_vkcmd_draw_probe",
                                               static_cast<u32>(v115d_mux_first_vkcmd_draw));
            V114ShaderMultiplexFileTraceNumber("v115d_mux first_vkcmd_draw_zero_count_probe",
                                               static_cast<u32>(v115d_mux_first_vkcmd_draw_zero_count));
            V114ShaderMultiplexFileTraceNumber("v115d_mux first_vkcmd_draw_zero_count_real_vertex_bind_ultra_quiet_probe",
                                               static_cast<u32>(v115d_mux_first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet));
            if (v115d_mux_first_vkcmd_draw_zero_count) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux zero_count_after_index_setup_before_pipeline_bind");
            }
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_descriptor_pipeline_bind");
        }
        if (trace_accel || IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_mux descriptor_bind_probe before_pipeline_bind indexed={} num_vertices={} binding_count={} wait_built={} first_vkcmd_draw={} first_vkcmd_draw_zero_count={} first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        is_indexed, regs.pipeline.num_vertices, binding_count,
                        static_cast<u32>(wait_built), static_cast<u32>(v115d_mux_first_vkcmd_draw),
                        static_cast<u32>(v115d_mux_first_vkcmd_draw_zero_count),
                        static_cast<u32>(v115d_mux_first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet),
                        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        const bool pipeline_ready = pipeline_cache.BindPipeline(pipeline_info, wait_built);

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux after_descriptor_pipeline_bind");
        }
        if (IsV115DA7Z9DescriptorReturnAfterPipelineBindRawEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z9 descriptor_return_after_pipeline_bind_raw_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z9 stage9_descriptor_after_pipeline_bind_raw_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z9 descriptor return after raw pipeline bind result=1 pipeline_ready={}",
                            static_cast<u32>(pipeline_ready));
            }
            return true;
        }
        if (IsV114ShaderMultiplexFileTraceEnabled() &&
            !IsV115DA7Z9DescriptorSkipPipelineReadyNumberEnabled()) {
            V114ShaderMultiplexFileTraceNumber("v115d_mux descriptor_pipeline_bind_ready",
                                               static_cast<u32>(pipeline_ready));
        } else if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw(
                "v115d_a7z9 descriptor_pipeline_bind_ready_number_skipped");
        }
        if (!pipeline_ready) {
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_mux descriptor_bind_probe pipeline_not_ready before_vertex_or_draw result=0");
            }
            return false;
        }

        if (v115d_mux_first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet) {
            // v115-D-MUX: C10 failed before/inside BindPipeline when the next step tried
            // to reintroduce a real vertex bind. Return immediately after a quiet
            // successful pipeline bind to prove the C8-good path still crosses pipeline bind
            // under the new zero-count branch, before DrawParams, vertex bind, or draw record.
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux pipeline_quiet_after_pipeline_bind_return_true");
            }
            return true;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_draw_params_build");
        }
        const DrawParams params = {
            .vertex_count = regs.pipeline.num_vertices,
            .vertex_offset = -static_cast<s32>(vertex_info.vs_input_index_min),
            .binding_count = binding_count,
            .bindings = binding_offsets,
            .is_indexed = is_indexed,
        };
        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux after_draw_params_build");
        }

        if (IsV115DA7Z10DescriptorReturnAfterDrawParamsRawEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z10 descriptor_return_after_drawparams_raw_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z10 stage9_descriptor_after_drawparams_raw_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z10 descriptor return after DrawParams raw result=1");
            }
            return true;
        }

        if (IsV115DA7Z10DescriptorMinimalVertexBindEarlyRawEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z10 minimal_vertex_bind_early_raw_begin");
            }
            scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
                std::array<vk::DeviceSize, 16> offsets{};
                std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                               [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
                cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(),
                                         offsets.data());
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z10 minimal_vertex_bind_early_raw_after_scheduler_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z10 stage9_descriptor_minimal_vertex_bind_early_raw_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z10 descriptor minimal vertex bind early raw result=1");
            }
            return true;
        }

        if (IsV115DA7Z8DescriptorReturnAfterDrawParamsEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z8 descriptor_return_after_drawparams_begin");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z8 stage9_descriptor_after_drawparams_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z8 descriptor return after DrawParams result=1");
            }
            return true;
        }

        if (IsV115DA7Z8DescriptorMinimalVertexBindEarlyEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z8 minimal_vertex_bind_early_begin");
            }
            scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
                std::array<vk::DeviceSize, 16> offsets{};
                std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                               [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
                cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(),
                                         offsets.data());
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z8 minimal_vertex_bind_early_after_scheduler_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z8 stage9_descriptor_minimal_vertex_bind_early_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z8 descriptor minimal vertex bind early result=1");
            }
            return true;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceNumber("v115d_mux params_binding_count", params.binding_count);
            V114ShaderMultiplexFileTraceNumber("v115d_mux params_vertex_count", params.vertex_count);
            V114ShaderMultiplexFileTraceNumber("v115d_mux params_indexed",
                                               static_cast<u32>(params.is_indexed));
        }

        if (v115d_mux_first_vkcmd_draw_zero_count_no_vertex_bind_ultra_quiet) {
            // v115-D-MUX: start from the v115-C8 success path and add only a dummy
            // bindVertexBuffers command. Do not use the real binding_offsets yet and do not issue
            // an indexed draw. This isolates whether the crash is caused by the Vulkan vertex-buffer
            // bind command itself or by the real binding offsets/indexed fetch path.
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux zero_count_real_vertex_bind_ultra_quiet_before_record");
                V114ShaderMultiplexFileTraceNumber("v115d_mux pipeline_quiet_count", binding_count);
            }
            std::array<vk::DeviceSize, 16> real_offsets{};
            std::transform(params.bindings.begin(), params.bindings.end(), real_offsets.begin(),
                           [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
            scheduler.Record([this, binding_count, real_offsets](vk::CommandBuffer cmdbuf) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_record_before_vertex_bind");
                cmdbuf.bindVertexBuffers(0, binding_count, vertex_buffers.data(), real_offsets.data());
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_record_after_vertex_bind");
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_record_before_zero_count_draw");
                cmdbuf.draw(0, 1, 0, 0);
                V114ShaderMultiplexFileTraceRaw("v115d_mux real_record_after_zero_count_draw");
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux zero_count_real_vertex_bind_ultra_quiet_after_record");
                V114ShaderMultiplexFileTraceRaw("v115d_mux zero_count_real_vertex_bind_ultra_quiet_recorded_return_true");
            }
            return true;
        }


        if (IsV115DA7Z7DescriptorMinimalVertexBindRecordEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z7 minimal_vertex_bind_record_begin");
                V114ShaderMultiplexFileTraceNumber("v115d_a7z7 minimal_binding_count",
                                                   params.binding_count);
            }
            scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
                std::array<vk::DeviceSize, 16> offsets{};
                std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                               [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
                cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(),
                                         offsets.data());
            });
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z7 minimal_vertex_bind_record_after_scheduler_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z7 stage9_descriptor_minimal_vertex_bind_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z7 descriptor minimal vertex bind record indexed={} vertex_count={} binding_count={} result=1",
                            params.is_indexed, params.vertex_count, params.binding_count);
            }
            return true;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_vertex_buffer_bind_record");
            V114ShaderMultiplexFileTraceNumber("v115d_mux vertex_bind_count", params.binding_count);
            V114ShaderMultiplexFileTraceNumber("v115d_mux vertex_count", params.vertex_count);
            V114ShaderMultiplexFileTraceNumber("v115d_mux indexed_path", static_cast<u32>(params.is_indexed));
            V114ShaderMultiplexFileTraceRaw("v115d_a7z5 before_zero_count_draw_trace");
            V114ShaderMultiplexFileTraceNumber("v115d_mux zero_count_draw",
                                               static_cast<u32>(v115d_mux_first_vkcmd_draw_zero_count));
            V114ShaderMultiplexFileTraceRaw("v115d_a7z5 after_zero_count_draw_trace");
            V114ShaderMultiplexFileTraceNumber("v115d_mux draw_vertex_count_requested",
                                               v115d_mux_first_vkcmd_draw_zero_count ? 0 : params.vertex_count);
            V114ShaderMultiplexFileTraceRaw("v115d_a7z5 after_draw_vertex_count_requested_trace");
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z5 return_before_vertex_bind_env",
                static_cast<u32>(IsV115DA7Z5DescriptorReturnBeforeVertexBindEnabled()));
        }

        if (IsV115DA7Z5DescriptorReturnBeforeVertexBindEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z5 descriptor_return_before_vertex_bind_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z5 stage9_descriptor_pre_vertex_bind_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z5 descriptor probe return before vertex bind indexed={} vertex_count={} binding_count={} result=1",
                            params.is_indexed, params.vertex_count, params.binding_count);
            }
            return true;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled() && v115d_mux_first_vkcmd_draw) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_first_vkcmd_draw_record");
        }

        if (IsV114ShaderMultiplexFileTraceEnabled() &&
            IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z5 before_scheduler_record_vertex_bind");
        }

        scheduler.Record([this, params, v115d_mux_first_vkcmd_draw, v115d_mux_first_vkcmd_draw_zero_count](vk::CommandBuffer cmdbuf) {
            if (IsV114ShaderMultiplexFileTraceEnabled() &&
                IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_lambda_enter");
            }
            std::array<vk::DeviceSize, 16> offsets{};
            std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                           [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
            if (IsV114ShaderMultiplexFileTraceEnabled() &&
                IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_after_offsets_build");
                V114ShaderMultiplexFileTraceNumber("v115d_a7z5 record_binding_count",
                                                   params.binding_count);
                V114ShaderMultiplexFileTraceNumber("v115d_a7z5 record_vertex_count",
                                                   params.vertex_count);
                V114ShaderMultiplexFileTraceNumber("v115d_a7z5 record_indexed",
                                                   static_cast<u32>(params.is_indexed));
                V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_before_bind_vertex_buffers");
            }
            cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(), offsets.data());
            if (IsV114ShaderMultiplexFileTraceEnabled() &&
                IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_after_bind_vertex_buffers");
            }
            if (v115d_mux_first_vkcmd_draw) {
                const u32 draw_vertex_count = v115d_mux_first_vkcmd_draw_zero_count ? 0 : params.vertex_count;
                if (IsV114ShaderMultiplexFileTraceEnabled() &&
                    IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                    V114ShaderMultiplexFileTraceNumber("v115d_a7z5 record_draw_vertex_count",
                                                       draw_vertex_count);
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_before_draw_command");
                }
                if (params.is_indexed) {
                    cmdbuf.drawIndexed(draw_vertex_count, 1, 0, params.vertex_offset, 0);
                } else {
                    cmdbuf.draw(draw_vertex_count, 1, 0, 0);
                }
                if (IsV114ShaderMultiplexFileTraceEnabled() &&
                    IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                    V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_after_draw_command");
                }
            }
            if (IsV114ShaderMultiplexFileTraceEnabled() &&
                IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_a7z5 record_lambda_exit");
            }
            // v115-B rollback intentionally stops after bindVertexBuffers(). v115-D-MUX records the
            // same first guarded draw command but with vertex/index count optionally forced to 0.
        });

        if (IsV114ShaderMultiplexFileTraceEnabled() &&
            IsV115DA7Z5DescriptorVerboseRecordTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_a7z5 after_scheduler_record_vertex_bind");
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceNumber(
                "v115d_a7z6 return_after_vertex_bind_record_env",
                static_cast<u32>(IsV115DA7Z6DescriptorReturnAfterVertexBindRecordEnabled()));
        }

        if (IsV115DA7Z6DescriptorReturnAfterVertexBindRecordEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z6 descriptor_return_after_vertex_bind_record");
                V114ShaderMultiplexFileTraceRaw(
                    "v115d_a7z6 stage9_descriptor_post_vertex_bind_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_a7z6 descriptor probe return after vertex bind record indexed={} vertex_count={} binding_count={} result=1",
                            params.is_indexed, params.vertex_count, params.binding_count);
            }
            return true;
        }

        if (v115d_mux_first_vkcmd_draw) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux after_first_vkcmd_draw_record");
            }

            if (v115d_mux_first_vkcmd_draw_zero_count && IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux zero_count_draw_recorded_return_true");
            }
            if (v115d_mux_first_vkcmd_draw_zero_count) {
                return true;
            }

            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux stage10_first_vkcmd_draw_consumed_return_true");
            }
            if (trace_accel || IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_mux first_vkcmd_draw_probe recorded indexed={} vertex_count={} binding_count={} result=1",
                            params.is_indexed, params.vertex_count, params.binding_count);
            }
            return true;
        }

        if (IsV114ShaderMultiplexFileTraceEnabled()) {
            V114ShaderMultiplexFileTraceRaw("v115d_mux after_vertex_buffer_bind_record");
            V114ShaderMultiplexFileTraceRaw("v115d_mux before_vkcmd_draw_return_true");
            V114ShaderMultiplexFileTraceRaw("v115d_mux stage9_descriptor_bind_consumed_return_true");
        }
        if (trace_accel || IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v115d_mux descriptor_bind_probe after_vertex_bind_record indexed={} vertex_count={} binding_count={} before_vkcmd_draw=1 result=1",
                        params.is_indexed, params.vertex_count, params.binding_count);
        }

        return true;
    }

    const bool wait_built = IsStrictCompatEnabled() ? true
                                                    : (!async_shaders || regs.pipeline.num_vertices <= 6);

    if (consume_if_stage_limited(14, "before_bind_pipeline")) {
        return true;
    }

    if (!pipeline_cache.BindPipeline(pipeline_info, wait_built)) {
        if (trace_accel) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_ACCEL_STAGE v114 pipeline_not_ready wait_built={} strict_compat={}",
                     wait_built, static_cast<u32>(IsStrictCompatEnabled()));
        }
        return false;
    }

    if (consume_if_stage_limited(15, "pipeline_bound")) {
        return true;
    }

    const DrawParams params = {
        .vertex_count = regs.pipeline.num_vertices,
        .vertex_offset = -static_cast<s32>(vertex_info.vs_input_index_min),
        .binding_count = binding_count,
        .bindings = binding_offsets,
        .is_indexed = is_indexed,
    };

    if (IsStrictAccelInternalDryRunEnabled()) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v114 internal dry-run consumed before vkCmdDraw stage=16 indexed={} vertex_count={} vertex_offset={} binding_count={} wait_built={}",
                        params.is_indexed, params.vertex_count, params.vertex_offset,
                        params.binding_count, wait_built);
        }
        return true;
    }

    if (consume_if_stage_limited(16, "before_record_vkcmd")) {
        return true;
    }

    scheduler.Record([this, params](vk::CommandBuffer cmdbuf) {
        std::array<vk::DeviceSize, 16> offsets{};
        std::transform(params.bindings.begin(), params.bindings.end(), offsets.begin(),
                       [](u32 offset) { return static_cast<vk::DeviceSize>(offset); });
        cmdbuf.bindVertexBuffers(0, params.binding_count, vertex_buffers.data(), offsets.data());
        if (params.is_indexed) {
            cmdbuf.drawIndexed(params.vertex_count, 1, 0, params.vertex_offset, 0);
        } else {
            cmdbuf.draw(params.vertex_count, 1, 0, 0);
        }
    });

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v114 internal stage=17 name=vkcmd_recorded indexed={} vertex_count={} binding_count={}",
                    params.is_indexed, params.vertex_count, params.binding_count);
    }

    return true;
}

void RasterizerVulkan::SetupIndexArray() {
    const bool index_u8 = regs.pipeline.index_array.format == 0;
    const bool native_u8 = index_u8 && instance.IsIndexTypeUint8Supported();
    const u32 source_index_size = regs.pipeline.num_vertices * (index_u8 ? 1u : 2u);
    const u32 index_buffer_size = regs.pipeline.num_vertices * (native_u8 ? 1u : 2u);
    const vk::IndexType index_type = native_u8 ? vk::IndexType::eUint8EXT : vk::IndexType::eUint16;
    const PAddr index_addr =
        regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() + regs.pipeline.index_array.offset;

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW setup_index_array addr=0x{:08x} num_vertices={} index_u8={} native_u8={} src_size={} dst_size={}",
                 index_addr, regs.pipeline.num_vertices, static_cast<u32>(index_u8),
                 static_cast<u32>(native_u8), source_index_size, index_buffer_size);
    }

    auto [index_ptr, index_offset, _] = stream_buffer.Map(index_buffer_size, 2);
    std::memset(index_ptr, 0, index_buffer_size);

    if (source_index_size != 0) {
        const MemoryRef index_ref = memory.GetPhysicalRef(index_addr);
        if (index_ref.GetSize() < source_index_size) {
            LOG_ERROR(Render_Vulkan,
                      "Index buffer size {} exceeds available space {} at address {:#016X}",
                      source_index_size, index_ref.GetSize(), index_addr);
        } else {
            const u8* index_data = index_ref.GetPtr();
            if (index_u8 && !native_u8) {
                u16* index_ptr_u16 = reinterpret_cast<u16*>(index_ptr);
                for (u32 i = 0; i < regs.pipeline.num_vertices; i++) {
                    index_ptr_u16[i] = index_data[i];
                }
            } else {
                std::memcpy(index_ptr, index_data, source_index_size);
            }
        }
    }

    stream_buffer.Commit(index_buffer_size);

    scheduler.Record(
        [this, index_offset = index_offset, index_type = index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindIndexBuffer(stream_buffer.Handle(), index_offset, index_type);
        });
}

void RasterizerVulkan::DrawTriangles() {
    LOG_DEBUG(Render_Vulkan, "Starting DrawTriangles with batch size {}", vertex_batch.size());

    if (vertex_batch.empty()) {
        LOG_DEBUG(Render_Vulkan, "Empty vertex batch, skipping draw");
        return;
    }

    try {
        pipeline_info.rasterization.topology.Assign(Pica::PipelineRegs::TriangleTopology::List);
        pipeline_info.vertex_layout = software_layout;

        pipeline_cache.UseTrivialVertexShader();
        pipeline_cache.UseTrivialGeometryShader();
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW draw_triangles software_batch_size={}",
                     vertex_batch.size());
        }
        Draw(false, false);
        LOG_DEBUG(Render_Vulkan, "RasterizerVulkan::DrawTriangles draw_submitted");
    } catch (const vk::SystemError& e) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in DrawTriangles: {}", e.what());
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Error in DrawTriangles: {}", e.what());
    }
}

bool RasterizerVulkan::Draw(bool accelerate, bool is_indexed) {
    BORKED3DS_PROFILE("Vulkan", "Drawing");
    if (IsDrawTraceEnabled()) {
        const u64 draw_index = ++g_vk_draw_counter;
        if (draw_index <= 16 || (draw_index % 256) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW begin index={} accelerate={} indexed={} num_vertices={} topology={} shadow={} color_mask=0x{:x} depth_test={} depth_write={} stencil_test={}",
                     draw_index, accelerate, is_indexed, regs.pipeline.num_vertices,
                     static_cast<u32>(regs.pipeline.triangle_topology.Value()),
                     regs.framebuffer.IsShadowRendering(),
                     static_cast<u32>(pipeline_info.blending.color_write_mask),
                     static_cast<bool>(pipeline_info.depth_stencil.depth_test_enable),
                     static_cast<bool>(pipeline_info.depth_stencil.depth_write_enable),
                     static_cast<bool>(pipeline_info.depth_stencil.stencil_test_enable));
        }
    }

    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    const bool has_stencil = regs.framebuffer.HasStencil();

    const bool write_color_fb = shadow_rendering || pipeline_info.blending.color_write_mask;
    const bool write_depth_fb = pipeline_info.IsDepthWriteEnabled();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW targets write_color={} write_depth={} has_stencil={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                 write_color_fb, write_depth_fb, has_stencil,
                 regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
    }

    const bool using_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && pipeline_info.depth_stencil.stencil_test_enable));

    const auto fb_helper = res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb);
    const Framebuffer* framebuffer = fb_helper.Framebuffer();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW framebuffer using_color={} using_depth={} fb_valid={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                 using_color_fb, using_depth_fb, static_cast<bool>(framebuffer->Handle()),
                 regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
    }
    if (!framebuffer->Handle()) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW skipped: framebuffer handle invalid");
        }
        return true;
    }

    // v82: hard descriptorless proof path for Pi5/V3DV strict software fallback.
    // v77 proved that small descriptorless tiles can keep the render target/present path
    // alive and visible. v82 keeps that stable bridge, but periodically lets one very
    // small null-texture software draw continue into the real shader/pipeline/vkCmdDraw path.
    // This is the next controlled step: prove whether the wall is still descriptor/pipeline
    // submit, without enabling real PICA texture surfaces yet.
    bool strict_software_null_draw_probe = false;
    const bool strict_software_early_debug_clear =
        !accelerate && IsStrictCompatEnabled() && IsSoftwareClearProbeEnabled() &&
        !IsSoftwareTexturesAllowed() && using_color_fb;
    if (strict_software_early_debug_clear) {
        const auto draw_rect = fb_helper.DrawRect();
        const u64 clear_index = ++g_vk_strict_software_debug_clear_counter;
        const u32 vertex_count = static_cast<u32>(vertex_batch.size());

        const bool full_clear_probe = IsFullSoftwareClearProbeEnabled();
        const u32 enabled_primary_textures = CountEnabledPrimaryTextures(regs);
        const bool primary_textures_disabled = ArePrimaryTexturesDisabled(regs);
        const bool depth_active = HasActiveDepthState(regs);
        u64 eligible_null_probe_index = 0;
        bool textured_null_probe_candidate = false;
        strict_software_null_draw_probe = ShouldAttemptNullSoftwareDrawProbe(
            clear_index, vertex_count, enabled_primary_textures, primary_textures_disabled,
            depth_active, eligible_null_probe_index, textured_null_probe_candidate);
        if (strict_software_null_draw_probe) {
            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 allowing controlled untextured null-software draw probe clear_index={} eligible_index={} vertex_count={} draw_w={} draw_h={} enabled_textures={} textures_disabled={} depth_active={} textured_candidate={} color_addr=0x{:08x}; bypassing tile clear with textures disabled and strict fixed-null descriptors",
                            clear_index, eligible_null_probe_index, vertex_count,
                            draw_rect.GetWidth(), draw_rect.GetHeight(), enabled_primary_textures,
                            static_cast<u32>(primary_textures_disabled),
                            static_cast<u32>(depth_active),
                            static_cast<u32>(textured_null_probe_candidate),
                            regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
            }
        } else {
            if (textured_null_probe_candidate && IsDrawTraceEnabled() &&
                clear_index >= 96 && (clear_index % 96) == 0) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 textured null-software draw probe vetoed clear_index={} vertex_count={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x}; v78/v79 crashed when real software probes were allowed, keeping descriptorless tile clear unless explicitly opted in",
                            clear_index, vertex_count, enabled_primary_textures,
                            static_cast<u32>(primary_textures_disabled),
                            static_cast<u32>(depth_active),
                            regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
            }
            const u32 tile_clear_budget = GetSoftwareClearTileBudget();
            const u32 tile_clear_period = GetSoftwareClearTilePeriod();
            const bool budget_exhausted = tile_clear_budget == 0 || clear_index > tile_clear_budget;
            const bool period_skipped = !full_clear_probe && tile_clear_period > 1 &&
                                        ((clear_index - 1) % tile_clear_period) != 0;

            if (budget_exhausted || period_skipped) {
                if (IsDrawTraceEnabled() &&
                    (clear_index <= 16 || clear_index == static_cast<u64>(tile_clear_budget) + 1 ||
                     (clear_index % 32) == 0)) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_DRAW strict_compat v82 descriptorless render-target tile clear throttled clear_index={} vertex_count={} draw_w={} draw_h={} enabled_textures={} textures_disabled={} depth_active={} full_clear={} budget={} period={} budget_exhausted={} period_skipped={} color_addr=0x{:08x}; clearing skipped and software draw consumed as safe no-op before SyncTextureUnits/descriptors/shaders/pipeline/vkCmdDraw",
                                clear_index, vertex_count, draw_rect.GetWidth(), draw_rect.GetHeight(),
                                CountEnabledPrimaryTextures(regs),
                                static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                                static_cast<u32>(HasActiveDepthState(regs)),
                                static_cast<u32>(full_clear_probe), tile_clear_budget, tile_clear_period,
                                static_cast<u32>(budget_exhausted),
                                static_cast<u32>(period_skipped),
                                regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }

            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 early descriptorless render-target tile clear clear_index={} vertex_count={} draw_w={} draw_h={} enabled_textures={} textures_disabled={} depth_active={} full_clear={} budget={} period={} color_addr=0x{:08x}; bypassing quarantine, SyncTextureUnits, descriptors, shaders, pipeline and vkCmdDraw",
                            clear_index, vertex_count, draw_rect.GetWidth(), draw_rect.GetHeight(),
                            CountEnabledPrimaryTextures(regs),
                            static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                            static_cast<u32>(HasActiveDepthState(regs)),
                            static_cast<u32>(full_clear_probe), tile_clear_budget, tile_clear_period,
                            regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
            }

            if (draw_rect.GetWidth() > 0 && draw_rect.GetHeight() > 0) {
                scheduler.Finish();
                renderpass_cache.BeginRendering(framebuffer, draw_rect);
                scheduler.Record([draw_rect, clear_index, full_clear_probe](vk::CommandBuffer cmdbuf) {
                    vk::ClearAttachment color_attachment{};
                    color_attachment.aspectMask = vk::ImageAspectFlagBits::eColor;
                    color_attachment.colorAttachment = 0;
                    const float phase = static_cast<float>(clear_index % 4);
                    const std::array<float, 4> color{
                        phase == 0.0f ? 0.35f : 0.02f,
                        phase == 1.0f ? 0.28f : 0.01f,
                        phase == 2.0f ? 0.40f : 0.04f,
                        1.0f,
                    };
                    color_attachment.clearValue.color = vk::ClearColorValue{color};

                    const u32 target_width = static_cast<u32>(draw_rect.GetWidth());
                    const u32 target_height = static_cast<u32>(draw_rect.GetHeight());
                    const u32 tile_width = full_clear_probe ? target_width : std::min<u32>(32, target_width);
                    const u32 tile_height = full_clear_probe ? target_height : std::min<u32>(32, target_height);
                    const u32 max_x = target_width > tile_width ? target_width - tile_width : 0;
                    const u32 max_y = target_height > tile_height ? target_height - tile_height : 0;
                    const u32 tile_x = full_clear_probe ? 0 : static_cast<u32>((clear_index * 37) % (max_x + 1));
                    const u32 tile_y = full_clear_probe ? 0 : static_cast<u32>((clear_index * 19) % (max_y + 1));

                    vk::ClearRect clear_rect{};
                    clear_rect.rect.offset = vk::Offset2D{
                        static_cast<s32>(draw_rect.left + tile_x),
                        static_cast<s32>(draw_rect.bottom + tile_y),
                    };
                    clear_rect.rect.extent = vk::Extent2D{tile_width, tile_height};
                    clear_rect.baseArrayLayer = 0;
                    clear_rect.layerCount = 1;
                    const std::array<vk::ClearAttachment, 1> clear_attachments{color_attachment};
                    const std::array<vk::ClearRect, 1> clear_rects{clear_rect};
                    cmdbuf.clearAttachments(clear_attachments, clear_rects);
                });
            }

            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW strict_compat v82 early descriptorless clear submitted clear_index={} vertex_count={}",
                         clear_index, vertex_count);
            }
            vertex_batch.clear();
            return true;
        }
    }

    // v84: v83 stabilized Sonic but kept the framebuffer black because every fallback draw
    // was consumed. Reintroduce only the least fragile real software draws: untextured,
    // no-depth batches, bounded by an explicit budget. Textured and depth draws stay no-op
    // unless the broad BORKED3DS_V3DV_ALLOW_REAL_SOFTWARE_DRAWS diagnosis switch is enabled.
    const bool strict_safe_untextured_real_draw_candidate =
        !accelerate && IsStrictCompatEnabled() && IsStrictSafeUntexturedSoftwareDrawAllowed() &&
        using_color_fb && ArePrimaryTexturesDisabled(regs) && !HasActiveDepthState(regs) &&
        vertex_batch.size() > 0 && vertex_batch.size() <= 96;
    u64 strict_safe_untextured_real_draw_index = 0;
    const bool strict_safe_untextured_real_draw = [&] {
        if (!strict_safe_untextured_real_draw_candidate) {
            return false;
        }
        strict_safe_untextured_real_draw_index = ++g_vk_strict_safe_untextured_real_draw_counter;
        const u32 budget = GetStrictSafeUntexturedSoftwareDrawBudget();
        return budget != 0 && strict_safe_untextured_real_draw_index <= budget;
    }();

    if (strict_safe_untextured_real_draw && IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v114 allowing safe untextured real software draw safe_index={} budget={} vertex_batch_size={} num_vertices={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                    strict_safe_untextured_real_draw_index,
                    GetStrictSafeUntexturedSoftwareDrawBudget(), vertex_batch.size(),
                    regs.pipeline.num_vertices,
                    regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                    regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
    }

    if (!accelerate && IsStrictCompatEnabled() && !IsStrictSoftwareNoopGuardDisabled() &&
        !IsStrictSoftwareRealDrawAllowed() && !strict_safe_untextured_real_draw &&
        !IsSoftwareTexturesAllowed() && using_color_fb) {
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v114 software fallback consumed as safe no-op vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}; allow_safe_untextured={} safe_candidate={} set BORKED3DS_V3DV_ALLOW_REAL_SOFTWARE_DRAWS=1 only for full diagnosis",
                        vertex_batch.size(), regs.pipeline.num_vertices,
                        CountEnabledPrimaryTextures(regs),
                        static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                        static_cast<u32>(HasActiveDepthState(regs)),
                        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress(),
                        static_cast<u32>(IsStrictSafeUntexturedSoftwareDrawAllowed()),
                        static_cast<u32>(strict_safe_untextured_real_draw_candidate));
        }
        vertex_batch.clear();
        return true;
    }

    const bool tiny_textured_software_draw =
        !accelerate && ShouldAttemptTinyTexturedSoftwareDraw(regs, vertex_batch.size());
    const bool medium_textured_software_draw =
        !accelerate && ShouldAttemptMediumTexturedSoftwareDraw(regs, vertex_batch.size());
    u64 large_textured_software_draw_index = 0;
    const bool large_textured_software_draw = [&] {
        if (accelerate || !ShouldAttemptLargeTexturedSoftwareDraw(regs, vertex_batch.size())) {
            return false;
        }
        large_textured_software_draw_index = ++g_vk_large_textured_software_allow_counter;
        return large_textured_software_draw_index <= 1;
    }();

    if (!accelerate) {
        const bool strict_quarantine_candidate =
            IsStrictCompatEnabled() && !IsSoftwareSkipAllowed() && !strict_safe_untextured_real_draw &&
            !IsStartupSoftwareQuarantineDisabled() && !IsStartupSoftwareQuarantineForcedOff();
        const bool strict_quarantine_fragile =
            HasPrimaryTexturesEnabled(regs) || ArePrimaryTexturesDisabled(regs) ||
            HasActiveDepthState(regs) || vertex_batch.size() <= 6 || vertex_batch.size() >= 512;
        if (strict_quarantine_candidate && strict_quarantine_fragile && !strict_software_null_draw_probe) {
            const u64 quarantine_index = ++g_vk_strict_software_quarantine_counter;
            // v82: the v57 logs prove that the first real software draws now reach vk_rasterizer,
            // but they crash before the first Vulkan draw submit. Quarantine only the very first
            // startup batches so the emulator can progress to later, safer batches and so the next
            // log tells us whether the wall is still the first draw or a later render-target write.
            if (quarantine_index <= 32) {
                if (IsDrawTraceEnabled()) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_DRAW strict_compat v82 quarantining startup software draw quarantine_index={} vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}; set BORKED3DS_V3DV_DISABLE_SOFTWARE_QUARANTINE=1 only for diagnosis",
                                quarantine_index, vertex_batch.size(), regs.pipeline.num_vertices,
                                CountEnabledPrimaryTextures(regs),
                                static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                                static_cast<u32>(HasActiveDepthState(regs)),
                                regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                                regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            } else if (IsDrawTraceEnabled() && quarantine_index == 33) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 startup software quarantine exhausted; allowing subsequent software draws");
            }
        }

        if (IsStrictCompatEnabled() && !IsSoftwareSkipAllowed() && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat v114 software skip disabled; drawing software batch vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                     vertex_batch.size(), regs.pipeline.num_vertices,
                     CountEnabledPrimaryTextures(regs), static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                     static_cast<u32>(HasActiveDepthState(regs)),
                     regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        if (IsSoftwareSkipAllowed() && ShouldBypassFragileSoftwareDraw(regs, vertex_batch.size())) {
            const u64 bypass_index = ++g_vk_software_bypass_counter;
            if (bypass_index <= 32) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_bypass_software_draw bypass_index={} vertex_batch_size={} num_vertices={} textures_disabled=1 color_addr=0x{:08x} depth_addr=0x{:08x}",
                             bypass_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldBypassFragileTexturedSoftwareDraw(regs, vertex_batch.size()) &&
            !tiny_textured_software_draw && !medium_textured_software_draw &&
            !large_textured_software_draw) {
            const u64 bypass_index = ++g_vk_textured_software_bypass_counter;
            if (bypass_index <= 6) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_bypass_textured_software_draw bypass_index={} vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled=0 depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                             bypass_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs),
                             static_cast<u32>(HasActiveDepthState(regs)),
                             regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                             regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldSkipStartupTexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 skip_index = ++g_vk_startup_textured_software_skip_counter;
            if (skip_index <= 320) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_startup_textured_software_draw skip_index={} vertex_batch_size={} num_vertices={} enabled_textures={}",
                             skip_index, vertex_batch.size(), regs.pipeline.num_vertices,
                             CountEnabledPrimaryTextures(regs));
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldSkipBatch42TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 skip_index = ++g_vk_batch42_textured_software_skip_counter;
            if (skip_index <= 4) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_batch42_textured_software_draw skip_index={} vertex_batch_size={} num_vertices={}",
                             skip_index, vertex_batch.size(), regs.pipeline.num_vertices);
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldSkipNonIndexed96TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 skip_index = ++g_vk_nonindexed96_textured_software_skip_counter;
            if (skip_index <= 16) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_nonindexed96_textured_software_draw skip_index={} vertex_batch_size={} num_vertices={}",
                             skip_index, vertex_batch.size(), regs.pipeline.num_vertices);
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldSkipNonIndexed36TexturedSoftwareDraw(regs, vertex_batch.size())) {
            const u64 skip_index = ++g_vk_nonindexed36_textured_software_skip_counter;
            if (skip_index <= 16) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_nonindexed36_textured_software_draw skip_index={} vertex_batch_size={} num_vertices={}",
                             skip_index, vertex_batch.size(), regs.pipeline.num_vertices);
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (IsSoftwareSkipAllowed() && ShouldSkipIndexed6TexturedLateStartupSoftwareDraw(regs, vertex_batch.size())) {
            const u64 skip_index = ++g_vk_indexed6_textured_late_startup_skip_counter;
            if (skip_index <= 128) {
                if (IsDrawTraceEnabled()) {
                    LOG_INFO(Render_Vulkan,
                             "TRACE_DRAW strict_compat early_skip_indexed6_textured_late_startup_software_draw_v82_allowed_only skip_index={} vertex_batch_size={} num_vertices={}",
                             skip_index, vertex_batch.size(), regs.pipeline.num_vertices);
                }
                vertex_batch.clear();
                return true;
            }
        }

        if (tiny_textured_software_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat allowing_tiny_textured_software_draw vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                     vertex_batch.size(), regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                     static_cast<u32>(HasActiveDepthState(regs)),
                     regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        if (medium_textured_software_draw && IsDrawTraceEnabled()) {
            const u64 skip_index = ++g_vk_medium_textured_software_skip_counter;
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat allowing_medium_textured_software_draw_v82 trace_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={}",
                     skip_index, vertex_batch.size(), regs.pipeline.num_vertices,
                     CountEnabledPrimaryTextures(regs), static_cast<u32>(HasActiveDepthState(regs)));
        }

        if (large_textured_software_draw && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat allowing_first_large_textured_software_draw_v82 large_index={} vertex_batch_size={} num_vertices={} enabled_textures={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                     large_textured_software_draw_index, vertex_batch.size(),
                     regs.pipeline.num_vertices, CountEnabledPrimaryTextures(regs),
                     static_cast<u32>(HasActiveDepthState(regs)),
                     regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }

        if (IsDrawTraceEnabled() && !large_textured_software_draw) {
            const u64 trace_index = ++g_vk_non_bypassed_software_trace_counter;
            if (trace_index <= 12) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW software_draw_after_bypass trace_index={} vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                         trace_index, vertex_batch.size(), regs.pipeline.num_vertices,
                         CountEnabledPrimaryTextures(regs),
                         static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                         static_cast<u32>(HasActiveDepthState(regs)),
                         regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                         regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
        }
    }

    pipeline_info.attachments.color = framebuffer->Format(SurfaceType::Color);
    pipeline_info.attachments.depth = framebuffer->Format(SurfaceType::Depth);
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW attachments color_format={} depth_format={} using_color={} using_depth={}",
                 static_cast<u32>(pipeline_info.attachments.color),
                 static_cast<u32>(pipeline_info.attachments.depth), using_color_fb, using_depth_fb);
    }

    const auto [scissor_x1, scissor_y2, scissor_x2, scissor_y1] = fb_helper.Scissor();
    if (fs_uniform_block_data.data.scissor_x1 != scissor_x1 ||
        fs_uniform_block_data.data.scissor_x2 != scissor_x2 ||
        fs_uniform_block_data.data.scissor_y1 != scissor_y1 ||
        fs_uniform_block_data.data.scissor_y2 != scissor_y2) {

        fs_uniform_block_data.data.scissor_x1 = scissor_x1;
        fs_uniform_block_data.data.scissor_x2 = scissor_x2;
        fs_uniform_block_data.data.scissor_y1 = scissor_y1;
        fs_uniform_block_data.data.scissor_y2 = scissor_y2;
        fs_uniform_block_data.dirty = true;
    }

    const bool strict_software_null_texture_path =
        !accelerate && IsStrictCompatEnabled() && !IsSoftwareTexturesAllowed();
    if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v114 using forced-null texture path before shader/pipeline setup vertex_batch_size={} enabled_textures={} textures_disabled={}",
                    vertex_batch.size(), CountEnabledPrimaryTextures(regs),
                    static_cast<u32>(ArePrimaryTexturesDisabled(regs)));
    }

    SyncTextureUnits(framebuffer);
    if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW strict_compat v82 after SyncTextureUnits before utility/shader path vertex_batch_size={}",
                 vertex_batch.size());
    }
    if (strict_software_null_texture_path) {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat v82 skipping SyncUtilityTextures on software null-texture path");
        }
    } else {
        SyncUtilityTextures(framebuffer);
    }

    if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW strict_compat v82 before fragment shader path shader_dirty={}",
                 static_cast<u32>(shader_dirty));
    }

    if (shader_dirty) {
        Pica::Shader::UserConfig user_config{};
        const bool lighting_disabled = static_cast<bool>(regs.lighting.disable.Value());
        const bool use_custom_normal =
            (!lighting_disabled) && !instance.IsFragmentShaderBarycentricSupported();
        user_config.use_custom_normal.Assign(use_custom_normal);
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW use_fragment_shader shader_dirty=1 use_custom_normal={} lighting_disabled={} barycentric_supported={}",
                     static_cast<u32>(use_custom_normal), static_cast<u32>(lighting_disabled),
                     static_cast<u32>(instance.IsFragmentShaderBarycentricSupported()));
        }
        pipeline_cache.UseFragmentShader(regs, user_config);
        if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat v82 after UseFragmentShader");
        }
        shader_dirty = false;
    } else if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW use_fragment_shader shader_dirty=0");
    }

    if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW strict_compat v82 before LUT/uniform upload");
    }
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();
    UploadUniforms(accelerate);
    if (strict_software_null_texture_path && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW strict_compat v82 after LUT/uniform upload before descriptor flush");
    }

    update_queue.Flush();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW descriptors_flushed accelerate={}",
                 static_cast<u32>(accelerate));
    }
    if (IsStrictCompatEnabled() && !accelerate) {
        scheduler.Finish();
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat serialized_before_software_draw vertex_batch_size={}",
                     vertex_batch.size());
        }
    }

    const auto draw_rect = fb_helper.DrawRect();
    renderpass_cache.BeginRendering(framebuffer, draw_rect);

    const auto viewport = fb_helper.Viewport();
    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW render_area x={} y={} w={} h={} viewport=({}, {}, {}, {})",
                 draw_rect.left, draw_rect.bottom, draw_rect.GetWidth(), draw_rect.GetHeight(),
                 viewport.x, viewport.y, viewport.width, viewport.height);
    }
    pipeline_info.dynamic.viewport = Common::Rectangle<s32>{
        viewport.x,
        viewport.y,
        viewport.x + viewport.width,
        viewport.y + viewport.height,
    };
    pipeline_info.dynamic.scissor = draw_rect;

    const bool strict_software_debug_clear_fallback =
        !accelerate && IsStrictCompatEnabled() && IsSoftwareClearProbeEnabled() &&
        !IsSoftwareTexturesAllowed() && using_color_fb && !strict_software_null_draw_probe;
    if (strict_software_debug_clear_fallback) {
        const u64 clear_index = ++g_vk_strict_software_debug_clear_counter;
        const u32 vertex_count = static_cast<u32>(vertex_batch.size());
        const bool full_clear_probe = IsFullSoftwareClearProbeEnabled();
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 descriptorless software debug tile-clear fallback clear_index={} vertex_count={} draw_w={} draw_h={} enabled_textures={} textures_disabled={} depth_active={} full_clear={} color_addr=0x{:08x}; bypassing texture descriptors, fragment shader, pipeline bind and vkCmdDraw",
                        clear_index, vertex_count, draw_rect.GetWidth(), draw_rect.GetHeight(),
                        CountEnabledPrimaryTextures(regs),
                        static_cast<u32>(ArePrimaryTexturesDisabled(regs)),
                        static_cast<u32>(HasActiveDepthState(regs)),
                        static_cast<u32>(full_clear_probe),
                        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
        }

        if (draw_rect.GetWidth() > 0 && draw_rect.GetHeight() > 0) {
            scheduler.Record([draw_rect, clear_index, full_clear_probe](vk::CommandBuffer cmdbuf) {
                vk::ClearAttachment color_attachment{};
                color_attachment.aspectMask = vk::ImageAspectFlagBits::eColor;
                color_attachment.colorAttachment = 0;
                const float phase = static_cast<float>(clear_index % 3);
                const std::array<float, 4> color{
                    phase == 0.0f ? 0.06f : 0.00f,
                    phase == 1.0f ? 0.10f : 0.00f,
                    phase == 2.0f ? 0.18f : 0.06f,
                    1.0f,
                };
                color_attachment.clearValue.color = vk::ClearColorValue{color};

                const u32 target_width = static_cast<u32>(draw_rect.GetWidth());
                const u32 target_height = static_cast<u32>(draw_rect.GetHeight());
                const u32 tile_width = full_clear_probe ? target_width : std::min<u32>(32, target_width);
                const u32 tile_height = full_clear_probe ? target_height : std::min<u32>(32, target_height);
                const u32 max_x = target_width > tile_width ? target_width - tile_width : 0;
                const u32 max_y = target_height > tile_height ? target_height - tile_height : 0;
                const u32 tile_x = full_clear_probe ? 0 : static_cast<u32>((clear_index * 23) % (max_x + 1));
                const u32 tile_y = full_clear_probe ? 0 : static_cast<u32>((clear_index * 11) % (max_y + 1));

                vk::ClearRect clear_rect{};
                clear_rect.rect.offset = vk::Offset2D{
                    static_cast<s32>(draw_rect.left + tile_x),
                    static_cast<s32>(draw_rect.bottom + tile_y),
                };
                clear_rect.rect.extent = vk::Extent2D{tile_width, tile_height};
                clear_rect.baseArrayLayer = 0;
                clear_rect.layerCount = 1;
                const std::array<vk::ClearAttachment, 1> clear_attachments{color_attachment};
                const std::array<vk::ClearRect, 1> clear_rects{clear_rect};
                cmdbuf.clearAttachments(clear_attachments, clear_rects);
            });
        }

        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW strict_compat v82 descriptorless clear submitted clear_index={} vertex_count={}",
                     clear_index, vertex_count);
        }
        vertex_batch.clear();
        return true;
    }

    bool succeeded = true;
    if (accelerate) {
        if (IsPipelineBindProbeOnlyEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux draw_wrapper_before_pipeline_bind_probe");
            }
            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_mux draw_wrapper entering pipeline_bind_probe indexed={} num_vertices={} before_vkcmd_draw=1",
                            is_indexed, regs.pipeline.num_vertices);
            }
        }
        if (IsDescriptorBindProbeOnlyEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux draw_wrapper_before_descriptor_bind_probe");
            }
            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_mux draw_wrapper entering descriptor_bind_probe indexed={} num_vertices={} before_vkcmd_draw=1",
                            is_indexed, regs.pipeline.num_vertices);
            }
        }
        if (IsFirstVkCmdDrawProbeOnlyEnabled() || IsFirstVkCmdDrawZeroCountProbeOnlyEnabled()) {
            if (IsV114ShaderMultiplexFileTraceEnabled()) {
                V114ShaderMultiplexFileTraceRaw("v115d_mux draw_wrapper_before_first_vkcmd_draw_probe");
                V114ShaderMultiplexFileTraceNumber("v115d_mux draw_wrapper_first_vkcmd_draw_zero_count_probe",
                                                   static_cast<u32>(IsFirstVkCmdDrawZeroCountProbeOnlyEnabled()));
            }
            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v115d_mux draw_wrapper entering first_vkcmd_draw_probe indexed={} num_vertices={} nopost={} before_vkcmd_draw=1",
                            is_indexed, regs.pipeline.num_vertices,
                            static_cast<u32>(IsFirstVkCmdDrawZeroCountProbeOnlyEnabled()));
            }
        }
        succeeded = AccelerateDrawBatchInternal(is_indexed);
    } else {
        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan, "TRACE_DRAW software_path vertex_batch_size={}",
                     vertex_batch.size());
        }

        const bool pipeline_ready = pipeline_cache.BindPipeline(pipeline_info, true);
        if (!pipeline_ready) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW software_path pipeline_not_ready vertex_batch_size={} strict_compat={}",
                         vertex_batch.size(), static_cast<u32>(IsStrictCompatEnabled()));
            }
            return false;
        }

        const u32 vertex_count = static_cast<u32>(vertex_batch.size());
        if (vertex_count == 0) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan, "TRACE_DRAW software_path skipped empty vertex batch");
            }
            return true;
        }

        const u32 vertex_size = vertex_count * sizeof(HardwareVertex);
        const auto [buffer, offset, _] = stream_buffer.Map(vertex_size, sizeof(HardwareVertex));

        std::memcpy(buffer, vertex_batch.data(), vertex_size);
        stream_buffer.Commit(vertex_size);

        scheduler.Record([this, offset = offset, vertex_count](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, stream_buffer.Handle(), offset);
            cmdbuf.draw(vertex_count, 1, 0, 0);
        });

        if (IsDrawTraceEnabled()) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW software_path submitted vertex_count={} buffer_offset={}",
                     vertex_count, offset);
        }
    }

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW end succeeded={} remaining_batch={}", succeeded,
                 vertex_batch.size());
    }
    vertex_batch.clear();
    return succeeded;
}

void RasterizerVulkan::SyncTextureUnits(const Framebuffer* framebuffer) {
    // v82 strict fallback: if the startup quarantine is exhausted and we still reach the
    // software path, populate the current texture descriptor set with fixed null resources.
    // v57 reused constructor descriptors, but the acquired per-draw texture set can still be
    // a different set. Leaving it unwritten is risky on V3DV, so v82 writes only the three
    // primary PICA units and never reads PICA texture state or creates texture surfaces here.
    if (IsStrictCompatEnabled() && !IsSoftwareTexturesAllowed()) {
        const auto texture_set = pipeline_cache.Acquire(DescriptorHeapType::Texture);
        const Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
        const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);
        const vk::ImageView null_view = null_surface.ImageView();
        const vk::Sampler null_handle = null_sampler.Handle();

        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 binding fixed null texture descriptors framebuffer_valid={} null_view_valid={} null_sampler_valid={}",
                        framebuffer != nullptr, static_cast<bool>(null_view),
                        static_cast<bool>(null_handle));
        }

        for (u32 texture_index = 0; texture_index < 3; ++texture_index) {
            if (IsDrawTraceEnabled()) {
                LOG_INFO(Render_Vulkan,
                         "TRACE_DRAW tex{} -> null reason=strict_compat_v82_fixed_null_descriptor",
                         texture_index);
            }
            update_queue.AddImageSampler(texture_set, texture_index, 0, null_view, null_handle);
        }
        return;
    }

    using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan, "TRACE_DRAW sync_textures begin framebuffer_valid={}",
                 framebuffer != nullptr);
    }

    const auto pica_textures = regs.texturing.GetTextures();
    const bool use_cube_heap =
        pica_textures[0].enabled && pica_textures[0].config.type == TextureType::ShadowCube;
    const auto texture_set = pipeline_cache.Acquire(use_cube_heap ? DescriptorHeapType::Texture
                                                                  : DescriptorHeapType::Texture);

    const Surface& null_surface = res_cache.GetSurface(VideoCore::NULL_SURFACE_ID);
    const Sampler& null_sampler = res_cache.GetSampler(VideoCore::NULL_SAMPLER_ID);
    const vk::ImageView null_view = null_surface.ImageView();
    const vk::Sampler null_handle = null_sampler.Handle();
    const vk::ImageView color_view =
        framebuffer ? framebuffer->ImageView(SurfaceType::Color) : vk::ImageView{};

    for (u32 texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        auto bind_null = [&](const char* reason) {
            if (IsDrawTraceEnabled() && texture_index < 3) {
                LOG_INFO(Render_Vulkan, "TRACE_DRAW tex{} -> null reason={} type={} format={}",
                         texture_index, reason, static_cast<u32>(texture.config.type.Value()),
                         static_cast<u32>(texture.format));
            }
            update_queue.AddImageSampler(texture_set, texture_index, 0, null_view, null_handle);
        };

        if (!texture.enabled) {
            bind_null("disabled");
            continue;
        }

        const bool strict_compat = IsStrictCompatEnabled();
        const u32 texture_format = static_cast<u32>(texture.format);
        const u32 texture_type = static_cast<u32>(texture.config.type.Value());

        if (strict_compat && !IsSoftwareTexturesAllowed() &&
            IsStrictCompatFragileTextureFormat(texture_format)) {
            if (IsDrawTraceEnabled() && texture_index < 3) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 binding fragile texture to null before surface_create tex{} type={} format={} addr=0x{:08X}; set BORKED3DS_V3DV_ALLOW_SOFTWARE_TEXTURES=1 only for diagnosis",
                            texture_index, texture_type, texture_format,
                            texture.config.GetPhysicalAddress());
            }
            bind_null("strict_compat_v82_fragile_texture_before_surface_create");
            continue;
        }

        if (texture_index == 0) {
            switch (texture.config.type.Value()) {
            case TextureType::Shadow2D: {
                Surface& surface = res_cache.GetTextureSurface(texture);
                Sampler& sampler = res_cache.GetSampler(texture.config);
                surface.flags |= VideoCore::SurfaceFlagBits::ShadowMap;
                const vk::ImageView view = surface.ImageView();
                update_queue.AddImageSampler(texture_set, texture_index, 0,
                                             IsValidImageView(view) ? view : null_view,
                                             IsValidImageView(view) ? sampler.Handle() : null_handle);
                continue;
            }
            case TextureType::ShadowCube: {
                BindShadowCube(texture, texture_set);
                continue;
            }
            case TextureType::TextureCube: {
                BindTextureCube(texture, texture_set);
                continue;
            }
            default:
                break;
            }
        }

        Surface& surface = res_cache.GetTextureSurface(texture);
        Sampler& sampler = res_cache.GetSampler(texture.config);
        const vk::ImageView base_view = surface.ImageView();
        const vk::ImageView copy_view = surface.CopyImageView();

        if (!IsValidImageView(base_view) && !IsValidImageView(copy_view)) {
            bind_null("invalid_base_and_copy_view");
            continue;
        }

        const bool direct_feedback = IsValidImageView(color_view) && color_view == base_view;
        vk::ImageView texture_view = base_view;
        const char* bind_reason = "base_view";

        if (strict_compat && IsValidImageView(copy_view)) {
            texture_view = copy_view;
            bind_reason = "strict_compat_copy";
        } else if (direct_feedback) {
            if (IsValidImageView(copy_view)) {
                texture_view = copy_view;
                bind_reason = "feedback_copy";
            } else if (!IsValidImageView(base_view)) {
                bind_null("feedback_copy_missing");
                continue;
            }
        } else if (!IsValidImageView(base_view) && IsValidImageView(copy_view)) {
            texture_view = copy_view;
            bind_reason = "copy_fallback";
        }

        if (IsDrawTraceEnabled() && texture_index < 3) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_DRAW tex{} bound reason={} sampler_valid={} type={} format={} strict_compat={} direct_feedback={}",
                     texture_index, bind_reason, static_cast<bool>(sampler.Handle()),
                     static_cast<u32>(texture.config.type.Value()), static_cast<u32>(texture.format),
                     static_cast<u32>(strict_compat), static_cast<u32>(direct_feedback));
        }
        update_queue.AddImageSampler(texture_set, texture_index, 0, texture_view, sampler.Handle());
    }
}

void RasterizerVulkan::SyncUtilityTextures(const Framebuffer* framebuffer) {
    const bool shadow_rendering = regs.framebuffer.IsShadowRendering();
    if (!shadow_rendering) {
        return;
    }

    const auto utility_set = pipeline_cache.Acquire(DescriptorHeapType::Utility);
    update_queue.AddStorageImage(utility_set, 0, framebuffer->ImageView(SurfaceType::Color));
}

void RasterizerVulkan::BindShadowCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                      vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config, texture.format);
    constexpr std::array faces = {
        CubeFace::PositiveX, CubeFace::NegativeX, CubeFace::PositiveY,
        CubeFace::NegativeY, CubeFace::PositiveZ, CubeFace::NegativeZ,
    };

    Sampler& sampler = res_cache.GetSampler(texture.config);
    for (CubeFace face : faces) {
        const u32 binding = static_cast<u32>(face);
        info.physical_address = regs.texturing.GetCubePhysicalAddress(face);
        const VideoCore::SurfaceId surface_id = res_cache.GetTextureSurface(info);
        Surface& surface = res_cache.GetSurface(surface_id);
        surface.flags |= VideoCore::SurfaceFlagBits::ShadowMap;
        update_queue.AddImageSampler(texture_set, 0, binding, surface.ImageView(), sampler.Handle());
    }
}

void RasterizerVulkan::BindTextureCube(const Pica::TexturingRegs::FullTextureConfig& texture,
                                       vk::DescriptorSet texture_set) {
    using CubeFace = Pica::TexturingRegs::CubeFace;
    const VideoCore::TextureCubeConfig config = {
        .px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
        .nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
        .py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
        .ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
        .pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
        .nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
        .width = texture.config.width,
        .levels = texture.config.lod.max_level + 1,
        .format = texture.format,
    };
    Surface& surface = res_cache.GetTextureCube(config);
    Sampler& sampler = res_cache.GetSampler(texture.config);
    update_queue.AddImageSampler(texture_set, 0, 0, surface.ImageView(), sampler.Handle());
}

void RasterizerVulkan::NotifyFixedFunctionPicaRegisterChanged(u32 id) {
    switch (id) {
    case PICA_REG_INDEX(rasterizer.cull_mode):
        SyncCullMode();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.alphablend_enable):
        SyncBlendEnabled();
        SyncLogicOp();
        SyncColorWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_blending):
        SyncBlendFuncs();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.blend_const):
        SyncBlendColor();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_func):
        SyncStencilTest();
        SyncStencilWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_op):
    case PICA_REG_INDEX(framebuffer.framebuffer.depth_format):
        SyncStencilTest();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.depth_test_enable):
        SyncDepthTest();
        SyncDepthWriteMask();
        SyncColorWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_depth_stencil_write):
        SyncDepthWriteMask();
        SyncStencilWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_color_write):
        SyncColorWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.logic_op):
        SyncLogicOp();
        SyncColorWriteMask();
        break;
    }
}

void RasterizerVulkan::FlushAll() {
    res_cache.FlushAll();
}

void RasterizerVulkan::FlushRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::InvalidateRegion(PAddr addr, u32 size) {
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::ClearAll(bool flush) {
    res_cache.ClearAll(flush);
}

bool RasterizerVulkan::AccelerateDisplayTransfer(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateDisplayTransfer(config);
}

bool RasterizerVulkan::AccelerateTextureCopy(const Pica::DisplayTransferConfig& config) {
    return res_cache.AccelerateTextureCopy(config);
}

bool RasterizerVulkan::AccelerateFill(const Pica::MemoryFillConfig& config) {
    return res_cache.AccelerateFill(config);
}

bool RasterizerVulkan::AccelerateDisplay(const Pica::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    if (framebuffer_addr == 0) [[unlikely]] {
        return false;
    }

    // v82: keep AccelerateDisplay active by default and stop clearing the renderer-owned
    // present texture automatically. v65 proved that the owned-present clear is submitted
    // and the screen info remains valid, so the next diagnostic must move to the existing
    // renderer_vulkan.cpp present-probe path via emulators.cfg.
    if (IsStrictCompatEnabled() && IsForcedNonAcceleratedDisplay() && !IsAcceleratedDisplayAllowed()) {
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 opt-in non-accelerated display fallback addr=0x{:08x} width={} height={} stride={} pixel_stride={} format={}; remove BORKED3DS_V3DV_FORCE_NON_ACCELERATED_DISPLAY for normal testing",
                        framebuffer_addr, config.width.Value(), static_cast<u32>(config.height.Value()),
                        config.stride, pixel_stride,
                        static_cast<u32>(config.color_format.Value()));
        }
        screen_info.image_view = vk::ImageView{};
        return false;
    }

    const VideoCore::PixelFormat present_pixel_format =
        VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    const u32 present_width = config.width.Value();
    const u32 present_height = static_cast<u32>(config.height.Value());
    const u32 present_effective_width = std::min(present_width, pixel_stride);

    // v82: the v82 mono duplicate-return-false path can still die before renderer_vulkan.cpp
    // finishes LoadFBToScreenInfo(right_eye=true). Do not present or reuse that duplicate by
    // default. Instead, deliberately fail only this duplicate AccelerateDisplay call. The
    // renderer keeps the permanent renderer-owned texture/view for slot 1, while slot 0 remains
    // the valid top-left framebuffer and slot 2 can still load the bottom screen.
    if (IsStrictPresentDisplayDuplicate(framebuffer_addr, present_effective_width, present_height,
                                        pixel_stride, present_pixel_format) &&
        !IsDuplicateExternalReuseAllowed() && !IsDuplicateOwnedReuseAllowed()) {
        screen_info.texcoords = Common::Rectangle<f32>{0.0f, 0.0f, 1.0f, 1.0f};
        screen_info.image_view = screen_info.texture.image_view;

        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 duplicate AccelerateDisplay mono_return_false_owned_texture addr=0x{:08x} width={} height={} stride={} pixel_format={} owned_valid={}; duplicate slot will be handled by renderer-owned texture and should not be drawn in mono mode",
                        framebuffer_addr, present_effective_width, present_height, pixel_stride,
                        static_cast<u32>(present_pixel_format),
                        static_cast<u32>(static_cast<bool>(screen_info.image_view)));
        }

        return false;
    }

    if (TryReuseStrictPresentDisplay(framebuffer_addr, present_effective_width, present_height,
                                     pixel_stride, present_pixel_format, screen_info)) {
        return true;
    }

    if (IsStrictCompatEnabled() && IsOwnedPresentTextureDebugEnabled() &&
        !IsOwnedPresentTextureDebugDisabled() && static_cast<bool>(screen_info.texture.image) &&
        static_cast<bool>(screen_info.texture.image_view)) {
        const VideoCore::PixelFormat owned_pixel_format = present_pixel_format;
        const u64 owned_clear_index = ++g_vk_strict_owned_present_clear_counter;

        RecordStrictOwnedPresentTextureClear(scheduler, renderpass_cache, screen_info.texture.image,
                                             owned_clear_index, framebuffer_addr,
                                             config.width.Value(),
                                             static_cast<u32>(config.height.Value()), pixel_stride,
                                             owned_pixel_format);

        screen_info.texcoords = Common::Rectangle<f32>(0.0f, 0.0f, 1.0f, 1.0f);
        screen_info.image_view = screen_info.texture.image_view;

        if (IsDrawTraceEnabled() && !IsForceQuietDisplayEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW accelerate_display v82 owned-present texture debug path addr=0x{:08x} width={} height={} stride={} pixel_format={} clear_index={} view_valid={}; set BORKED3DS_V3DV_DISABLE_OWNED_PRESENT_TEXTURE_CLEAR=1 only for diagnosis",
                        framebuffer_addr, config.width.Value(),
                        static_cast<u32>(config.height.Value()), pixel_stride,
                        static_cast<u32>(owned_pixel_format), owned_clear_index,
                        static_cast<u32>(static_cast<bool>(screen_info.image_view)));
        }

        RememberStrictPresentDisplay(framebuffer_addr, present_width, present_height, pixel_stride,
                                     owned_pixel_format, screen_info.texcoords,
                                     screen_info.image_view);

        return static_cast<bool>(screen_info.image_view);
    }

    VideoCore::SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    const auto [src_surface_id, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);

    if (!src_surface_id) {
        // v82: Kid Icarus/Sonic can reach AccelerateDisplay with a valid renderer-owned
        // screen texture while the cache lookup for the external framebuffer fails. Returning
        // false leaves renderer_vulkan.cpp with loaded=false even though view_valid=true, which
        // was the exact crash/flash pattern in the v69 logs. In strict Pi5/V3DV mode, keep the
        // accelerated display path alive by falling back to the renderer-owned present texture.
        if (IsStrictCompatEnabled() && static_cast<bool>(screen_info.texture.image_view)) {
            screen_info.texcoords = Common::Rectangle<f32>{0.0f, 0.0f, 1.0f, 1.0f};
            screen_info.image_view = screen_info.texture.image_view;

            RememberStrictPresentDisplay(framebuffer_addr, present_effective_width, present_height,
                                         pixel_stride, present_pixel_format, screen_info.texcoords,
                                         screen_info.image_view);

            if (IsDrawTraceEnabled()) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW strict_compat v82 src_surface_missing owned_present_fallback addr=0x{:08x} width={} height={} stride={} pixel_format={} owned_view_valid=1; keeping accelerated=1 to avoid loaded=false present path",
                            framebuffer_addr, present_effective_width, present_height, pixel_stride,
                            static_cast<u32>(present_pixel_format));
            }

            return true;
        }

        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 src_surface_missing no_owned_present_fallback addr=0x{:08x} width={} height={} stride={} pixel_format={} owned_view_valid=0",
                        framebuffer_addr, present_effective_width, present_height, pixel_stride,
                        static_cast<u32>(present_pixel_format));
        }

        return false;
    }
    Surface& src_surface = res_cache.GetSurface(src_surface_id);
    const u32 scaled_width = src_surface.GetScaledWidth();
    const u32 scaled_height = src_surface.GetScaledHeight();
    const bool strict_compat = IsStrictCompatEnabled();
    const bool strict_present_debug_clear =
        strict_compat && IsPresentImageClearAllowed() && !IsPresentDebugClearDisabled();
    const vk::ImageView base_view = src_surface.ImageView();
    const vk::ImageView copy_view = src_surface.CopyImageView();
    // v84: keep the v83 stable base-view default; use copy view only for diagnosis
    // after the loading screen. Return to the older, more stable base-view behavior by
    // default. The copy view is kept as an explicit comparison knob only.
    const bool use_copy_present_view =
        strict_compat && IsEnvEnabled("BORKED3DS_V3DV_USE_COPY_PRESENT_VIEW") &&
        IsValidImageView(copy_view);
    const bool force_base_present_view =
        strict_compat && !use_copy_present_view && IsValidImageView(base_view);

    if (strict_present_debug_clear) {
        const u64 present_clear_index = ++g_vk_strict_present_debug_clear_counter;
        if (IsDrawTraceEnabled()) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW strict_compat v82 present-path debug clear begin clear_index={} addr=0x{:08x} width={} height={} stride={} pixel_format={} src_rect=({}, {}, {}, {})",
                        present_clear_index, framebuffer_addr, src_params.width, src_params.height,
                        src_params.stride, static_cast<u32>(src_params.pixel_format),
                        src_rect.left, src_rect.bottom, src_rect.right, src_rect.top);
        }
        RecordStrictPresentDebugClear(scheduler, renderpass_cache, src_surface, present_clear_index,
                                      framebuffer_addr, src_params.width, src_params.height,
                                      src_params.stride, src_params.pixel_format);
    }

    screen_info.texcoords = Common::Rectangle<f32>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    screen_info.image_view =
        (use_copy_present_view && IsValidImageView(copy_view)) ? copy_view
                                                              : (IsValidImageView(base_view) ? base_view : copy_view);

    RememberStrictPresentDisplay(framebuffer_addr, src_params.width, src_params.height,
                                 src_params.stride, src_params.pixel_format, screen_info.texcoords,
                                 screen_info.image_view);

    if (IsDrawTraceEnabled() && !IsForceQuietDisplayEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW accelerate_display v114 addr=0x{:08x} width={} height={} stride={} pixel_format={} src_rect=({}, {}, {}, {}) base_valid={} copy_valid={} chosen={} strict_compat={} forced_base_present_view={}",
                 framebuffer_addr, src_params.width, src_params.height, src_params.stride,
                 static_cast<u32>(src_params.pixel_format), src_rect.left, src_rect.bottom,
                 src_rect.right, src_rect.top, static_cast<bool>(base_view),
                 static_cast<bool>(copy_view), static_cast<bool>(screen_info.image_view),
                 static_cast<u32>(strict_compat), static_cast<u32>(force_base_present_view));
    }

    return static_cast<bool>(screen_info.image_view);
}

void RasterizerVulkan::MakeSoftwareVertexLayout() {
    constexpr std::array sizes = {4, 4, 2, 2, 2, 1, 4, 3};

    software_layout = VertexLayout{
        .binding_count = 1,
        .attribute_count = 8,
    };

    for (u32 i = 0; i < software_layout.binding_count; i++) {
        VertexBinding& binding = software_layout.bindings[i];
        binding.binding.Assign(i);
        binding.fixed.Assign(0);
        binding.stride.Assign(sizeof(HardwareVertex));
    }

    u32 offset = 0;
    for (u32 i = 0; i < 8; i++) {
        VertexAttribute& attribute = software_layout.attributes[i];
        attribute.binding.Assign(0);
        attribute.location.Assign(i);
        attribute.offset.Assign(offset);
        attribute.type.Assign(Pica::PipelineRegs::VertexAttributeFormat::FLOAT);
        attribute.size.Assign(sizes[i]);
        offset += sizes[i] * sizeof(float);
    }
}

void RasterizerVulkan::SyncCullMode() {
    pipeline_info.rasterization.cull_mode.Assign(regs.rasterizer.cull_mode);
}

void RasterizerVulkan::SyncBlendEnabled() {
    pipeline_info.blending.blend_enable = regs.framebuffer.output_merger.alphablend_enable;
}

void RasterizerVulkan::SyncBlendFuncs() {
    pipeline_info.blending.color_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb);
    pipeline_info.blending.alpha_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_a);
    pipeline_info.blending.src_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    pipeline_info.blending.dst_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    pipeline_info.blending.src_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    pipeline_info.blending.dst_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
}

void RasterizerVulkan::SyncBlendColor() {
    pipeline_info.dynamic.blend_color = regs.framebuffer.output_merger.blend_const.raw;
}

void RasterizerVulkan::SyncLogicOp() {
    if (instance.NeedsLogicOpEmulation()) {
        shader_dirty = true;
    }

    pipeline_info.blending.logic_op = regs.framebuffer.output_merger.logic_op;

    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !regs.framebuffer.output_merger.alphablend_enable;
    const bool is_logic_op_noop =
        regs.framebuffer.output_merger.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        pipeline_info.blending.color_write_mask = 0;
    }
}

void RasterizerVulkan::SyncColorWriteMask() {
    const u32 color_mask = regs.framebuffer.framebuffer.allow_color_write != 0
                               ? (regs.framebuffer.output_merger.depth_color_mask >> 8) & 0xF
                               : 0;

    const bool is_logic_op_emulated =
        instance.NeedsLogicOpEmulation() && !regs.framebuffer.output_merger.alphablend_enable;
    const bool is_logic_op_noop =
        regs.framebuffer.output_merger.logic_op == Pica::FramebufferRegs::LogicOp::NoOp;
    if (is_logic_op_emulated && is_logic_op_noop) {
        return;
    }

    pipeline_info.blending.color_write_mask = color_mask;
}

void RasterizerVulkan::SyncStencilWriteMask() {
    pipeline_info.dynamic.stencil_write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;
}

void RasterizerVulkan::SyncDepthWriteMask() {
    const bool write_enable = regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                              regs.framebuffer.output_merger.depth_write_enable;
    pipeline_info.depth_stencil.depth_write_enable.Assign(write_enable);
}

void RasterizerVulkan::SyncStencilTest() {
    const auto& stencil_test = regs.framebuffer.output_merger.stencil_test;
    const bool test_enable = stencil_test.enable &&
                             regs.framebuffer.framebuffer.depth_format ==
                                 Pica::FramebufferRegs::DepthFormat::D24S8;
    pipeline_info.depth_stencil.stencil_test_enable.Assign(test_enable);
    pipeline_info.depth_stencil.stencil_fail_op.Assign(stencil_test.action_stencil_fail);
    pipeline_info.depth_stencil.stencil_pass_op.Assign(stencil_test.action_depth_pass);
    pipeline_info.depth_stencil.stencil_depth_fail_op.Assign(stencil_test.action_depth_fail);
    pipeline_info.depth_stencil.stencil_compare_op.Assign(stencil_test.func);
    pipeline_info.dynamic.stencil_reference = stencil_test.reference_value;
    pipeline_info.dynamic.stencil_compare_mask = stencil_test.input_mask;
}

void RasterizerVulkan::SyncDepthTest() {
    const bool test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                              regs.framebuffer.output_merger.depth_write_enable == 1;
    const auto compare_op = regs.framebuffer.output_merger.depth_test_enable == 1
                                ? regs.framebuffer.output_merger.depth_test_func.Value()
                                : Pica::FramebufferRegs::CompareFunc::Always;
    pipeline_info.depth_stencil.depth_test_enable.Assign(test_enabled);
    pipeline_info.depth_stencil.depth_compare_op.Assign(compare_op);
}

void RasterizerVulkan::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size = sizeof(Common::Vec2f) * 256 *
                                         Pica::LightingRegs::NumLightingSampler +
                                     sizeof(Common::Vec2f) * 128;
    if (!fs_uniform_block_data.lighting_lut_dirty_any && !fs_uniform_block_data.fog_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_lf_buffer.Map(max_size, sizeof(Common::Vec4f));

    if (fs_uniform_block_data.lighting_lut_dirty_any || invalidate) {
        for (unsigned index = 0; index < fs_uniform_block_data.lighting_lut_dirty.size(); index++) {
            if (fs_uniform_block_data.lighting_lut_dirty[index] || invalidate) {
                std::array<Common::Vec2f, 256> new_data;
                const auto& source_lut = pica.lighting.luts[index];
                std::transform(source_lut.begin(), source_lut.end(), new_data.begin(),
                               [](const auto& entry) {
                                   return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                               });

                if (new_data != lighting_lut_data[index] || invalidate) {
                    lighting_lut_data[index] = new_data;
                    std::memcpy(buffer + bytes_used, new_data.data(),
                                new_data.size() * sizeof(Common::Vec2f));
                    fs_uniform_block_data.data.lighting_lut_offset[index / 4][index % 4] =
                        static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
                    fs_uniform_block_data.dirty = true;
                    bytes_used += new_data.size() * sizeof(Common::Vec2f);
                }
                fs_uniform_block_data.lighting_lut_dirty[index] = false;
            }
        }
        fs_uniform_block_data.lighting_lut_dirty_any = false;
    }

    if (fs_uniform_block_data.fog_lut_dirty || invalidate) {
        std::array<Common::Vec2f, 128> new_data;
        std::transform(pica.fog.lut.begin(), pica.fog.lut.end(), new_data.begin(),
                       [](const auto& entry) {
                           return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                       });
        if (new_data != fog_lut_data || invalidate) {
            fog_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec2f));
            fs_uniform_block_data.data.fog_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
        fs_uniform_block_data.fog_lut_dirty = false;
    }

    texture_lf_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::SyncAndUploadLUTs() {
    const auto& proctex = pica.proctex;
    constexpr std::size_t max_size = sizeof(Common::Vec2f) * 128 * 3 +
                                     sizeof(Common::Vec4f) * 256 +
                                     sizeof(Common::Vec4f) * 256;
    if (!fs_uniform_block_data.proctex_noise_lut_dirty &&
        !fs_uniform_block_data.proctex_color_map_dirty &&
        !fs_uniform_block_data.proctex_alpha_map_dirty &&
        !fs_uniform_block_data.proctex_lut_dirty &&
        !fs_uniform_block_data.proctex_diff_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_buffer.Map(max_size, sizeof(Common::Vec4f));

    auto sync_proctex_value_lut = [this, buffer = buffer, offset = offset,
                                   invalidate = invalidate,
                                   &bytes_used](const auto& lut, auto& lut_data, int& lut_offset) {
        std::array<Common::Vec2f, 128> new_data;
        std::transform(lut.begin(), lut.end(), new_data.begin(), [](const auto& entry) {
            return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
        });
        if (new_data != lut_data || invalidate) {
            lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec2f));
            lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
    };

    if (fs_uniform_block_data.proctex_noise_lut_dirty || invalidate) {
        sync_proctex_value_lut(proctex.noise_table, proctex_noise_lut_data,
                               fs_uniform_block_data.data.proctex_noise_lut_offset);
        fs_uniform_block_data.proctex_noise_lut_dirty = false;
    }

    if (fs_uniform_block_data.proctex_color_map_dirty || invalidate) {
        sync_proctex_value_lut(proctex.color_map_table, proctex_color_map_data,
                               fs_uniform_block_data.data.proctex_color_map_offset);
        fs_uniform_block_data.proctex_color_map_dirty = false;
    }

    if (fs_uniform_block_data.proctex_alpha_map_dirty || invalidate) {
        sync_proctex_value_lut(proctex.alpha_map_table, proctex_alpha_map_data,
                               fs_uniform_block_data.data.proctex_alpha_map_offset);
        fs_uniform_block_data.proctex_alpha_map_dirty = false;
    }

    if (fs_uniform_block_data.proctex_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;
        std::transform(proctex.color_table.begin(), proctex.color_table.end(), new_data.begin(),
                       [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_lut_data || invalidate) {
            proctex_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            fs_uniform_block_data.data.proctex_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        fs_uniform_block_data.proctex_lut_dirty = false;
    }

    if (fs_uniform_block_data.proctex_diff_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::transform(proctex.color_diff_table.begin(), proctex.color_diff_table.end(),
                       new_data.begin(), [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_diff_lut_data || invalidate) {
            proctex_diff_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            fs_uniform_block_data.data.proctex_diff_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            fs_uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        fs_uniform_block_data.proctex_diff_lut_dirty = false;
    }

    texture_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::UploadUniforms(bool accelerate_draw) {
    const bool sync_vs_pica = accelerate_draw;
    const bool sync_vs = vs_uniform_block_data.dirty;
    const bool sync_fs = fs_uniform_block_data.dirty;
    if (!sync_vs_pica && !sync_vs && !sync_fs) {
        return;
    }

    const u32 uniform_size =
        uniform_size_aligned_vs_pica + uniform_size_aligned_vs + uniform_size_aligned_fs;
    auto [uniforms, offset, invalidate] =
        uniform_buffer.Map(uniform_size, uniform_buffer_alignment);

    u32 used_bytes = 0;

    if (sync_vs || invalidate) {
        std::memcpy(uniforms + used_bytes, &vs_uniform_block_data.data,
                    sizeof(vs_uniform_block_data.data));

        pipeline_cache.UpdateRange(1, offset + used_bytes);
        vs_uniform_block_data.dirty = false;
        used_bytes += uniform_size_aligned_vs;
    }

    if (sync_fs || invalidate) {
        std::memcpy(uniforms + used_bytes, &fs_uniform_block_data.data,
                    sizeof(fs_uniform_block_data.data));

        pipeline_cache.UpdateRange(2, offset + used_bytes);
        fs_uniform_block_data.dirty = false;
        used_bytes += uniform_size_aligned_fs;
    }

    if (sync_vs_pica) {
        VSPicaUniformData vs_uniforms;
        vs_uniforms.uniforms.SetFromRegs(regs.vs, pica.vs_setup);
        std::memcpy(uniforms + used_bytes, &vs_uniforms, sizeof(vs_uniforms));

        pipeline_cache.UpdateRange(0, offset + used_bytes);
        used_bytes += uniform_size_aligned_vs_pica;
    }

    uniform_buffer.Commit(used_bytes);
}

} // namespace Vulkan
