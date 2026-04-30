// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <string>
#include <tuple>

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

[[nodiscard]] bool IsStrictCompatEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_STRICT_COMPAT");
}

[[nodiscard]] bool IsForceAccelStageTraceEnabled() {
    // v100/v103 diagnostic:
    // Keep forced stage tracing available. v100 still stopped after the PICA pre_call before
    // the raw-enter marker, so v103 adds an even earlier no-argument entry-only probe.
    // This does not execute any extra Vulkan work by itself.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE");
}

[[nodiscard]] bool IsAccelEntryOnlyProbeEnabled() {
    // v103 diagnostic:
    // pica_core reached TRACE_DRAW_PICA v100 pre_call, but vk_rasterizer raw_enter was absent.
    // Return immediately after a no-argument entry marker. This proves the C++ call boundary
    // and avoids touching regs, shader config, GLSL generation, SPIR-V, pipeline, descriptors,
    // Draw(), vkCmdDraw, and vkCmdDrawIndexed.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE");
}

[[nodiscard]] bool IsAccelStageTraceEnabled() {
    return IsDrawTraceEnabled() || IsEnvEnabled("BORKED3DS_V3DV_TRACE_ACCEL_STAGE") ||
           IsForceAccelStageTraceEnabled();
}

[[nodiscard]] bool IsTrivialVertexShaderProbeEnabled() {
    // v96 diagnostic only:
    // v95 reached TRACE_DRAW_PICA pre_call with stop_after=7, but the log did not show
    // stage=7/post_call. v96 used a trivial vertex shader to prove the generic VS bind
    // path is stable. Keep this switch as an explicit fallback, but v103 should normally
    // leave it disabled.
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_USE_TRIVIAL_VERTEX_SHADER_PROBE");
}

[[nodiscard]] bool IsProgrammableVertexShaderGenerateProbeEnabled() {
    // v97 diagnostic:
    // v96 proved the generic/trivial vertex-shader path reaches stage=7 and returns true.
    // v97 attempted to run GLSL::GenerateVertexShader() only. Keep this switch available,
    // but v103 should normally leave it disabled.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderConfigProbeEnabled() {
    // v98 diagnostic:
    // Build only PicaVSConfig and the attribute load_flags for the programmable PICA VS,
    // then bind the trivial VS. v98 proved this part is safe. Keep this switch available,
    // but v103 should normally leave it disabled.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_CONFIG_ONLY");
}

[[nodiscard]] bool IsProgrammableVertexShaderGenerateGuardedProbeEnabled() {
    // v99/v100 diagnostic:
    // This guarded GLSL generation probe is kept as the next shader-path step, but v103 should
    // normally leave it disabled because v100 did not even reach the raw-enter marker.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY");
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
    // v103: plan de travail 1 does not advance beyond function entry. v100 reached the
    // PICA pre_call but no raw-enter marker appeared. The recommended v103 emulators.cfg keeps
    // BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=7 for continuity but enables
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
           IsEnvEnabled("BORKED3DS_V3DV_ALLOW_V103_REAL_SOFTWARE_DRAWS") &&
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
    // v103 grouped test: v102 revalidated stage=1 after the entry-only detour.
    // As discussed, regroup the already-proven stages and stop at stage=6.
    // This passes geometry-shader gate, topology assignment/support, vertex analysis,
    // and SetupVertexArray(), but still returns before stage=7 / SetupVertexShader(),
    // GLSL generation, SPIR-V, shader modules, pipeline state, descriptors, Draw(), or vkCmdDraw.
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

    if (IsDrawTraceEnabled()) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW strict_compat v103 RasterizerVulkan constructor marker strict_compat={} allow_software_textures={} quarantine_disabled={}",
                    static_cast<u32>(IsStrictCompatEnabled()),
                    static_cast<u32>(IsSoftwareTexturesAllowed()),
                    static_cast<u32>(IsStartupSoftwareQuarantineDisabled()));
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

    const bool trace_accel = IsAccelStageTraceEnabled();
    const bool trivial_vs_probe = IsTrivialVertexShaderProbeEnabled();
    const bool programmable_config_probe = IsProgrammableVertexShaderConfigProbeEnabled();
    const bool programmable_generate_guarded_probe =
        IsProgrammableVertexShaderGenerateGuardedProbeEnabled();
    const bool programmable_generate_probe = IsProgrammableVertexShaderGenerateProbeEnabled();

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v103 vertex_shader_setup_begin trivial_probe={} programmable_config_probe={} programmable_generate_guarded_probe={} programmable_generate_probe={} binding_count={} attribute_count={} accurate_mul={} strict_compat={}",
                    static_cast<u32>(trivial_vs_probe),
                    static_cast<u32>(programmable_config_probe),
                    static_cast<u32>(programmable_generate_guarded_probe),
                    static_cast<u32>(programmable_generate_probe),
                    pipeline_info.vertex_layout.binding_count,
                    pipeline_info.vertex_layout.attribute_count, static_cast<u32>(accurate_mul),
                    static_cast<u32>(IsStrictCompatEnabled()));
    }

    if (trivial_vs_probe) {
        pipeline_cache.UseTrivialVertexShader();
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_trivial_probe_used result=1");
        }
        return true;
    }

    auto build_programmable_vs_config = [&]() {
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

        return std::tuple<PicaVSConfig, bool, u32, u32>{
            config, use_geometry_shader, converted_attribs, zero_w_attribs};
    };

    if (programmable_config_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_config_only_begin");
        }

        auto [config, use_geometry_shader, converted_attribs, zero_w_attribs] =
            build_programmable_vs_config();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_config_only_end use_geometry_shader={} converted_attribs={} zero_w_attribs={} config_hash={}",
                        static_cast<u32>(use_geometry_shader), converted_attribs, zero_w_attribs,
                        config.Hash());
        }

        pipeline_cache.UseTrivialVertexShader();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_config_only_trivial_bind result=1");
        }

        return true;
    }

    auto run_generate_only_probe = [&](const char* mode_name) -> bool {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_begin", mode_name);
        }

        auto [config, use_geometry_shader, converted_attribs, zero_w_attribs] =
            build_programmable_vs_config();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_config use_geometry_shader={} converted_attribs={} zero_w_attribs={} config_hash={}",
                        mode_name, static_cast<u32>(use_geometry_shader), converted_attribs,
                        zero_w_attribs, config.Hash());
        }

        std::string program;

        if (programmable_generate_guarded_probe) {
            try {
                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_before_generate_call",
                                mode_name);
                }

                program = GLSL::GenerateVertexShader(pica.vs_setup, config, true);

                if (trace_accel) {
                    LOG_WARNING(Render_Vulkan,
                                "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_after_generate_call",
                                mode_name);
                }
            } catch (const std::exception& e) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v103 programmable VS GLSL generation threw std::exception: {}",
                          e.what());
                return false;
            } catch (...) {
                LOG_ERROR(Render_Vulkan,
                          "TRACE_ACCEL_STAGE v103 programmable VS GLSL generation threw unknown exception");
                return false;
            }
        } else {
            program = GLSL::GenerateVertexShader(pica.vs_setup, config, true);
        }

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_end program_bytes={} empty={}",
                        mode_name, program.size(), static_cast<u32>(program.empty()));
        }

        if (program.empty()) {
            LOG_ERROR(Render_Vulkan,
                      "TRACE_ACCEL_STAGE v103 programmable VS GLSL generation returned empty program");
            return false;
        }

        // Diagnostic only: GLSL generation succeeded, but do not compile SPIR-V, do not create a
        // VkShaderModule, and do not bind the programmable shader yet. Keep the trivial VS bound
        // so stage=7 can still be consumed safely.
        pipeline_cache.UseTrivialVertexShader();

        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_{}_trivial_bind result=1",
                        mode_name);
        }

        return true;
    };

    if (programmable_generate_guarded_probe) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_generate_guarded_probe_selected");
        }
        return run_generate_only_probe("programmable_generate_guarded_only");
    }

    if (programmable_generate_probe) {
        return run_generate_only_probe("programmable_generate_only");
    }

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_begin");
    }

    const bool result = pipeline_cache.UseProgrammableVertexShader(
        regs, pica.vs_setup, pipeline_info.vertex_layout, accurate_mul);

    if (trace_accel) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v103 vertex_shader_setup_programmable_end result={}",
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
    // v103 diagnostic: keep the absolutely first marker from v101. v101 proved this
    // function body is reached; v102 proved stage=1. v103 expects
    // BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE=0 and BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=6.
    LOG_WARNING(Render_Vulkan, "TRACE_ACCEL_STAGE v103 raw_enter_noargs");

    if (IsAccelEntryOnlyProbeEnabled()) {
        // Entry-only probe: prove the call boundary and return before stage=1. Keep this log
        // deliberately simple: no regs, no shader state, no framebuffer addresses.
        LOG_WARNING(Render_Vulkan,
                    "TRACE_ACCEL_STAGE v103 entry_only_probe_consumed before_stage1 indexed={} result=1",
                    is_indexed);
        return true;
    }

    const u64 accel_id = ++g_vk_accel_draw_counter;

    // v103 secondary marker: entry-only mode should be disabled now; this confirms env state
    // before the grouped stage path reuses stages 1 through 6.
    LOG_WARNING(Render_Vulkan,
                "TRACE_ACCEL_STAGE v103 raw_enter_simple accel_id={} indexed={} stop_after={} force_stage_trace={} entry_only_probe={} generate_guarded_probe={}",
                accel_id, is_indexed, GetAccelStageStopAfter(),
                static_cast<u32>(IsForceAccelStageTraceEnabled()),
                static_cast<u32>(IsAccelEntryOnlyProbeEnabled()),
                static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()));

    const bool trace_accel = IsAccelStageTraceEnabled();

    const auto log_stage = [&](u32 stage, const char* name) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 accel_id={} stage={} name={} indexed={} num_vertices={} topology={} use_gs={} preflight_expected={} color_addr=0x{:08x} depth_addr=0x{:08x}",
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
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v103 stage_limit consumed accel_id={} stage={} name={} stop_after={} before_vulkan_command=1",
                            accel_id, stage, name, GetAccelStageStopAfter());
            }
            return true;
        }
        return false;
    };

    if (consume_if_stage_limited(1, "enter_accelerate_draw_batch")) {
        return true;
    }

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v103 rejected_gs_mode accel_id={} gs_mode={}",
                            accel_id, static_cast<u32>(regs.pipeline.gs_config.mode.Value()));
            }
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            if (trace_accel) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_ACCEL_STAGE v103 rejected_gs_topology accel_id={} topology={}",
                            accel_id, static_cast<u32>(regs.pipeline.triangle_topology.Value()));
            }
            return false;
        }
    }

    if (consume_if_stage_limited(2, "geometry_shader_gate_ok")) {
        return true;
    }

    pipeline_info.rasterization.topology.Assign(regs.pipeline.triangle_topology);
    if (consume_if_stage_limited(3, "topology_assigned")) {
        return true;
    }

    if (regs.pipeline.triangle_topology == TriangleTopology::Fan &&
        !instance.IsTriangleFanSupported()) {
        LOG_DEBUG(Render_Vulkan,
                  "Skipping accelerated draw with unsupported triangle fan topology");
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 rejected_triangle_fan accel_id={}", accel_id);
        }
        return false;
    }

    if (consume_if_stage_limited(4, "topology_supported")) {
        return true;
    }

    vertex_info = AnalyzeVertexArray(is_indexed, instance.GetMinVertexStrideAlignment());
    if (consume_if_stage_limited(5, "vertex_array_analyzed")) {
        return true;
    }

    SetupVertexArray();
    if (consume_if_stage_limited(6, "vertex_array_setup_done")) {
        return true;
    }

    if (!SetupVertexShader()) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 vertex_shader_setup_failed accel_id={} trivial_probe={} programmable_config_probe={} programmable_generate_guarded_probe={} programmable_generate_probe={} force_stage_trace={}",
                        accel_id, static_cast<u32>(IsTrivialVertexShaderProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderConfigProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateGuardedProbeEnabled()),
                        static_cast<u32>(IsProgrammableVertexShaderGenerateProbeEnabled()),
                        static_cast<u32>(IsForceAccelStageTraceEnabled()));
        }
        return false;
    }
    if (consume_if_stage_limited(7, IsTrivialVertexShaderProbeEnabled() ? "vertex_shader_setup_ok_trivial_probe" : (IsProgrammableVertexShaderConfigProbeEnabled() ? "vertex_shader_setup_ok_programmable_config_only" : (IsProgrammableVertexShaderGenerateGuardedProbeEnabled() ? "vertex_shader_setup_ok_programmable_generate_guarded_only" : (IsProgrammableVertexShaderGenerateProbeEnabled() ? "vertex_shader_setup_ok_programmable_generate_only" : "vertex_shader_setup_ok"))))) {
        return true;
    }

    if (!SetupGeometryShader()) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 geometry_shader_setup_failed accel_id={}", accel_id);
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
                    "TRACE_ACCEL_STAGE v103 accel_id={} stage=18 name=after_draw_wrapper result={}",
                    accel_id, result);
    }
    return result;
}

bool RasterizerVulkan::AccelerateDrawBatchInternal(bool is_indexed) {
    const bool trace_accel = IsAccelStageTraceEnabled();

    const auto log_stage = [&](u32 stage, const char* name) {
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 internal stage={} name={} indexed={} vertex_count={} binding_count={} dry_run={} stop_after={}",
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
                            "TRACE_ACCEL_STAGE v103 internal stage_limit consumed stage={} name={} before_vulkan_command=1",
                            stage, name);
            }
            return true;
        }
        return false;
    };

    if (consume_if_stage_limited(10, "internal_enter")) {
        return true;
    }

    if (regs.pipeline.num_vertices == 0) {
        if (trace_accel) {
            LOG_INFO(Render_Vulkan, "TRACE_ACCEL_STAGE v103 internal skipped empty draw");
        }
        return true;
    }

    if (consume_if_stage_limited(11, "vertex_count_ok")) {
        return true;
    }

    const u32 binding_count = pipeline_info.vertex_layout.binding_count;
    if (binding_count == 0 || binding_count > vertex_buffers.size()) {
        LOG_ERROR(Render_Vulkan, "Accelerated draw has invalid binding_count={} (max={})",
                  binding_count, vertex_buffers.size());
        if (trace_accel) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_ACCEL_STAGE v103 internal invalid_binding_count binding_count={} max={}",
                        binding_count, vertex_buffers.size());
        }
        return false;
    }

    if (consume_if_stage_limited(12, "binding_count_ok")) {
        return true;
    }

    if (is_indexed) {
        SetupIndexArray();
        if (consume_if_stage_limited(13, "index_array_setup_done")) {
            return true;
        }
    } else if (consume_if_stage_limited(13, "nonindexed_no_index_setup")) {
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
                     "TRACE_ACCEL_STAGE v103 pipeline_not_ready wait_built={} strict_compat={}",
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
                        "TRACE_ACCEL_STAGE v103 internal dry-run consumed before vkCmdDraw stage=16 indexed={} vertex_count={} vertex_offset={} binding_count={} wait_built={}",
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
                    "TRACE_ACCEL_STAGE v103 internal stage=17 name=vkcmd_recorded indexed={} vertex_count={} binding_count={}",
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
                    "TRACE_DRAW strict_compat v103 allowing safe untextured real software draw safe_index={} budget={} vertex_batch_size={} num_vertices={} color_addr=0x{:08x} depth_addr=0x{:08x}",
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
                        "TRACE_DRAW strict_compat v103 software fallback consumed as safe no-op vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}; allow_safe_untextured={} safe_candidate={} set BORKED3DS_V3DV_ALLOW_REAL_SOFTWARE_DRAWS=1 only for full diagnosis",
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
                     "TRACE_DRAW strict_compat v103 software skip disabled; drawing software batch vertex_batch_size={} num_vertices={} enabled_textures={} textures_disabled={} depth_active={} color_addr=0x{:08x} depth_addr=0x{:08x}",
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
                    "TRACE_DRAW strict_compat v103 using forced-null texture path before shader/pipeline setup vertex_batch_size={} enabled_textures={} textures_disabled={}",
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

        if (IsDrawTraceEnabled()) {
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

    if (IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_DRAW accelerate_display v103 addr=0x{:08x} width={} height={} stride={} pixel_format={} src_rect=({}, {}, {}, {}) base_valid={} copy_valid={} chosen={} strict_compat={} forced_base_present_view={}",
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
