// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <boost/container/static_vector.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_set>

#include "common/hash.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

[[nodiscard]] bool IsPi5StrictCompatEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_STRICT_COMPAT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsDrawTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_TRACE_DRAW");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}


[[nodiscard]] bool IsV115DA7Z48GraphicsPipelineTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_A7Z48_GRAPHICS_PIPELINE_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsV115DA7Z51GraphicsPipelineForceTraceEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_A7Z51_GRAPHICS_PIPELINE_FORCE_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsV115DA7Z57GraphicsPipelineMainLogOnlyEnabled() {
    const char* value = std::getenv("BORKED3DS_V3DV_A7Z57_GRAPHICS_PIPELINE_MAINLOG_ONLY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsV115DA7Z61GraphicsPipelineUltraQuietTryBuildEnabled() {
    const char* value =
        std::getenv("BORKED3DS_V3DV_A7Z61_GRAPHICS_PIPELINE_ULTRA_QUIET_TRYBUILD");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// ---------------------------------------------------------------------------
// Sonde de compilation de pipeline -- BORKED3DS_V3DV_TRACE_PIPELINE_BUILD
//
// Cause confirmee du figement (pile gdb + test du disk shader cache) : la
// compilation d'un pipeline V3DV neuf est synchrone et lente sur Pi 5, et
// bloque la presentation a la premiere rencontre. Cette sonde chiffre, par
// seconde :
//   compile        : nombre de vraies compilations (createGraphicsPipeline succes)
//   compile_ms     : temps de compilation cumule (= duree du gel attendue)
//   compile_max_ms : pire compilation unique
//   cache_hit      : succes via fail_on_compile_required (deja en cache driver)
//   shaderwait_ms  : temps passe a attendre la compilation des shaders (SPIR-V)
// Elle distingue donc "pipeline lent a compiler" de "shader lent a compiler",
// et donne la TAILLE de la rafale par gel -- deux inconnues qui decident la
// forme du correctif (pipeline de secours). Purement numerique (compatible
// daltonisme), inerte hors variable, getenv lu une seule fois.
// ---------------------------------------------------------------------------
struct PipelineBuildStats {
    [[nodiscard]] static bool Enabled() noexcept {
        static const bool enabled =
            std::getenv("BORKED3DS_V3DV_TRACE_PIPELINE_BUILD") != nullptr;
        return enabled;
    }

    static inline std::atomic<u64> compile_calls{0};
    static inline std::atomic<u64> compile_ns{0};
    static inline std::atomic<u64> compile_max_ns{0};
    static inline std::atomic<u64> cache_hit_calls{0};
    static inline std::atomic<u64> shaderwait_calls{0};
    static inline std::atomic<u64> shaderwait_ns{0};
    static inline std::atomic<u64> shaderwait_max_ns{0};

    static void RecordCompile(u64 ns) noexcept {
        compile_calls.fetch_add(1, std::memory_order_relaxed);
        compile_ns.fetch_add(ns, std::memory_order_relaxed);
        UpdateMax(compile_max_ns, ns);
    }
    static void RecordCacheHit() noexcept {
        cache_hit_calls.fetch_add(1, std::memory_order_relaxed);
    }
    static void RecordShaderWait(u64 ns) noexcept {
        shaderwait_calls.fetch_add(1, std::memory_order_relaxed);
        shaderwait_ns.fetch_add(ns, std::memory_order_relaxed);
        UpdateMax(shaderwait_max_ns, ns);
    }

    static void ReportIfDue() {
        using clock = std::chrono::steady_clock;
        static std::mutex report_mutex;
        static clock::time_point last = clock::now();

        const clock::time_point now = clock::now();
        if (now - last < std::chrono::seconds(1)) {
            return;
        }
        std::unique_lock lock{report_mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            return;
        }
        if (now - last < std::chrono::seconds(1)) {
            return;
        }
        const double window_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(now - last).count() / 1000.0;
        last = now;

        const u64 n_c = compile_calls.exchange(0, std::memory_order_relaxed);
        const u64 t_c = compile_ns.exchange(0, std::memory_order_relaxed);
        const u64 t_cmax = compile_max_ns.exchange(0, std::memory_order_relaxed);
        const u64 n_hit = cache_hit_calls.exchange(0, std::memory_order_relaxed);
        const u64 n_sw = shaderwait_calls.exchange(0, std::memory_order_relaxed);
        const u64 t_sw = shaderwait_ns.exchange(0, std::memory_order_relaxed);
        const u64 t_swmax = shaderwait_max_ns.exchange(0, std::memory_order_relaxed);

        // Ne rien emettre sur une fenetre sans activite de compilation.
        if (n_c == 0 && n_hit == 0 && n_sw == 0) {
            return;
        }

        LOG_INFO(Render_Vulkan,
                 "TRACE_PIPELINE_BUILD compile={} compile_ms={:.1f} compile_max_ms={:.1f} "
                 "cache_hit={} shaderwait={} shaderwait_ms={:.1f} shaderwait_max_ms={:.1f} "
                 "window_ms={:.1f}",
                 n_c, t_c / 1.0e6, t_cmax / 1.0e6, n_hit, n_sw, t_sw / 1.0e6, t_swmax / 1.0e6,
                 window_ms);
    }

private:
    static void UpdateMax(std::atomic<u64>& target, u64 value) noexcept {
        u64 prev = target.load(std::memory_order_relaxed);
        while (value > prev &&
               !target.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
        }
    }
};

// Seuil au-dela duquel une compilation de pipeline est consideree "poison" :
// V3DV passe alors de ~0,2 ms a plusieurs secondes sur certaines combinaisons.
constexpr u64 POISON_THRESHOLD_NS = 2'000'000'000ull; // 2 s

// Dumpe la signature complete d'un pipeline dont la compilation a explose, pour
// identifier CE qui empoisonne le compilateur V3DV (shader ? etat baked ?).
// Ne dumpe qu'UNE fois par hash. Ecrit aussi le SPIR-V de chaque stage present
// dans /tmp pour desassemblage hors ligne (spirv-dis). Purement diagnostique,
// n'est appele que depuis le chemin trace, sur une compilation > 2 s (donc rare).
void DumpPoisonPipeline(const Instance& instance, const PipelineInfo& info,
                        const std::array<Shader*, 3>& stages, u64 elapsed_ns) {
    const u64 hash = info.Hash(instance);

    {
        static std::mutex dumped_mutex;
        static std::unordered_set<u64> dumped_hashes;
        std::scoped_lock lock{dumped_mutex};
        if (!dumped_hashes.insert(hash).second) {
            return; // deja dumpe ce pipeline
        }
    }

    // Tailles SPIR-V par stage (en mots u32) + dump binaire de chaque stage.
    std::array<u32, 3> stage_words{0, 0, 0};
    u32 present = 0;
    std::string dumped_files;
    for (u32 i = 0; i < 3; ++i) {
        if (!stages[i]) {
            continue;
        }
        present++;
        const std::vector<u32>& prog = stages[i]->program;
        stage_words[i] = static_cast<u32>(prog.size());
        if (!prog.empty()) {
            const std::string path =
                fmt::format("/tmp/borked3ds_poison_{:016x}_s{}.spv", hash, i);
            std::ofstream out{path, std::ios::binary | std::ios::trunc};
            if (out) {
                out.write(reinterpret_cast<const char*>(prog.data()),
                          static_cast<std::streamsize>(prog.size() * sizeof(u32)));
                if (!dumped_files.empty()) {
                    dumped_files += ',';
                }
                dumped_files += path;
            }
        }
    }
    if (dumped_files.empty()) {
        dumped_files = "-";
    }

    LOG_INFO(
        Render_Vulkan,
        "TRACE_PIPELINE_POISON hash={:#018x} compile_ms={:.1f} shaders={} "
        "s0_words={} s1_words={} s2_words={} color={} depth={} "
        "topology={} cull={} depth_test={} depth_write={} stencil_test={} "
        "depth_cmp={} blend_enable={} color_write_mask={} logic_op={} blend_factors={:#x} "
        "vtx_bind={} vtx_attr={} dumped={}",
        hash, elapsed_ns / 1.0e6, present, stage_words[0], stage_words[1], stage_words[2],
        static_cast<u32>(info.attachments.color), static_cast<u32>(info.attachments.depth),
        static_cast<u32>(info.rasterization.topology.Value()),
        static_cast<u32>(info.rasterization.cull_mode.Value()),
        static_cast<u32>(info.depth_stencil.depth_test_enable.Value()),
        static_cast<u32>(info.depth_stencil.depth_write_enable.Value()),
        static_cast<u32>(info.depth_stencil.stencil_test_enable.Value()),
        static_cast<u32>(info.depth_stencil.depth_compare_op.Value()),
        static_cast<u32>(info.blending.blend_enable),
        static_cast<u32>(info.blending.color_write_mask),
        static_cast<u32>(info.blending.logic_op), static_cast<u32>(info.blending.value),
        static_cast<u32>(info.vertex_layout.binding_count),
        static_cast<u32>(info.vertex_layout.attribute_count), dumped_files);
}

void LogV115DA7Z57GraphicsPipeline(const char* message) {
    LOG_WARNING(Render_Vulkan, "TRACE_DRAW {}", message);
}

void AppendV115DA7Z48GraphicsPipelineTrace(const char* message) {
    std::ofstream out{"/tmp/borked3ds_v115d_mux_shader_probe.log", std::ios::app};
    if (!out) {
        return;
    }
    out << message << '\n';
}

void AppendV115DA7Z48GraphicsPipelineTraceBool(const char* message, bool value) {
    std::ofstream out{"/tmp/borked3ds_v115d_mux_shader_probe.log", std::ios::app};
    if (!out) {
        return;
    }
    out << message << '=' << (value ? 1 : 0) << '\n';
}

void AppendV115DA7Z48GraphicsPipelineTraceU32(const char* message, u32 value) {
    std::ofstream out{"/tmp/borked3ds_v115d_mux_shader_probe.log", std::ios::app};
    if (!out) {
        return;
    }
    out << message << '=' << value << '\n';
}

} // namespace

vk::ShaderStageFlagBits MakeShaderStage(std::size_t index) {
    switch (index) {
    case 0:
        return vk::ShaderStageFlagBits::eVertex;
    case 1:
        return vk::ShaderStageFlagBits::eFragment;
    case 2:
        return vk::ShaderStageFlagBits::eGeometry;
    default:
        LOG_CRITICAL(Render_Vulkan, "Invalid shader stage index!");
        UNREACHABLE();
    }
    return vk::ShaderStageFlagBits::eVertex;
}

u64 PipelineInfo::Hash(const Instance& instance) const {
    u64 info_hash = 0;
    const auto append_hash = [&info_hash](const auto& data) {
        const u64 data_hash = Common::ComputeStructHash64(data);
        info_hash = Common::HashCombine(info_hash, data_hash);
    };

    append_hash(vertex_layout);
    append_hash(attachments);
    append_hash(blending);

    if (!instance.IsExtendedDynamicStateSupported()) {
        append_hash(rasterization);
        append_hash(depth_stencil);
    }

    return info_hash;
}

Shader::Shader(const Instance& instance) : device{instance.GetDevice()} {}

Shader::Shader(const Instance& instance, vk::ShaderStageFlagBits stage, std::string code)
    : Shader{instance} {
    module = Compile(code, stage, instance.GetDevice());
    MarkDone();
}

Shader::Shader(const Instance& instance, std::span<const u32> code) : Shader{instance} {
    module = CompileSPV(code, instance.GetDevice());
    MarkDone();
}

Shader::~Shader() {
    if (device && module) {
        device.destroyShaderModule(module);
    }
}

GraphicsPipeline::GraphicsPipeline(const Instance& instance_, RenderManager& renderpass_cache_,
                                   const PipelineInfo& info_, vk::PipelineCache pipeline_cache_,
                                   vk::PipelineLayout layout_, std::array<Shader*, 3> stages_,
                                   Common::ThreadWorker* worker_)
    : instance{instance_}, renderpass_cache{renderpass_cache_}, worker{worker_},
      pipeline_layout{layout_}, pipeline_cache{pipeline_cache_}, info{info_}, stages{stages_} {
    const bool a7z61_ultra_quiet = IsV115DA7Z61GraphicsPipelineUltraQuietTryBuildEnabled();
    const bool a7z57_trace =
        IsV115DA7Z57GraphicsPipelineMainLogOnlyEnabled() && !a7z61_ultra_quiet;
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 graphics_pipeline_constructor_enter");
    }
    if (IsV115DA7Z51GraphicsPipelineForceTraceEnabled() && !a7z57_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 graphics_pipeline_constructor_enter");
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 graphics_pipeline_constructor_leave");
    }
}

GraphicsPipeline::~GraphicsPipeline() = default;

bool GraphicsPipeline::TryBuild(bool wait_built) {
    const bool a7z61_ultra_quiet = IsV115DA7Z61GraphicsPipelineUltraQuietTryBuildEnabled();
    const bool a7z57_trace =
        IsV115DA7Z57GraphicsPipelineMainLogOnlyEnabled() && !a7z61_ultra_quiet;
    const bool a7z51_trace = IsV115DA7Z51GraphicsPipelineForceTraceEnabled() &&
                             !a7z57_trace && !a7z61_ultra_quiet;
    const bool a7z48_trace =
        (IsV115DA7Z48GraphicsPipelineTraceEnabled() || a7z51_trace) && !a7z57_trace &&
        !a7z61_ultra_quiet;
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_enter");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_enter");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_wait_built", wait_built);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_is_pending_before", is_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_is_done_before", IsDone());
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_enter");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_wait_built", wait_built);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_is_pending_before", is_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_is_done_before", IsDone());
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_initial_state");
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_before_pending_branch");
    }

    if (is_pending) {
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_pending_branch");
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_pending_return", wait_built);
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_pending_branch");
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_pending_return", wait_built);
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_pending_branch_return");
        }
        return wait_built;
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_pending_branch");
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_before_shader_scan");
    }

    u32 present_shader_count = 0;
    u32 pending_shader_count = 0;
    for (Shader* shader : stages) {
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_shader_scan_slot_begin");
        }
        if (!shader) {
            if (a7z57_trace) {
                LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_shader_scan_slot_null");
            }
            continue;
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_shader_scan_slot_present");
        }
        present_shader_count++;
        if (!shader->IsDone()) {
            pending_shader_count++;
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_shader_scan_slot_done_checked");
        }
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_shader_scan");
    }

    const bool shaders_pending = pending_shader_count != 0;
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_shaders_pending_calc");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 trybuild_present_shader_count", present_shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 trybuild_pending_shader_count", pending_shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_shaders_pending", shaders_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool(
            "v115d_a7z51 trybuild_cache_control_supported",
            instance.IsPipelineCreationCacheControlSupported());
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 trybuild_present_shader_count", present_shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 trybuild_pending_shader_count", pending_shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_shaders_pending", shaders_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool(
            "v115d_a7z48 trybuild_cache_control_supported",
            instance.IsPipelineCreationCacheControlSupported());
    }

    if (!wait_built && shaders_pending) {
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_return_false_shaders_pending_nowait");
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_return_false_shaders_pending_nowait");
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_return_false_shaders_pending_nowait");
        }
        return false;
    }

    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_before_cache_control_branch");
    }
    if (!shaders_pending && instance.IsPipelineCreationCacheControlSupported()) {
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_before_build_fail_on_compile_required");
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_before_build_fail_on_compile_required");
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_before_fail_on_compile_required_build");
        }
        const bool built_with_fail_on_compile_required = Build(true);
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_fail_on_compile_required_build");
        }
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTraceBool(
                "v115d_a7z51 trybuild_build_fail_on_compile_required_result",
                built_with_fail_on_compile_required);
            AppendV115DA7Z48GraphicsPipelineTraceBool(
                "v115d_a7z51 trybuild_is_done_after_fail_on_compile_required", IsDone());
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTraceBool(
                "v115d_a7z48 trybuild_build_fail_on_compile_required_result",
                built_with_fail_on_compile_required);
            AppendV115DA7Z48GraphicsPipelineTraceBool(
                "v115d_a7z48 trybuild_is_done_after_fail_on_compile_required", IsDone());
        }
        if (built_with_fail_on_compile_required) {
            if (a7z51_trace) {
                AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_return_true_sync_build");
            }
            if (a7z48_trace) {
                AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_return_true_sync_build");
            }
            if (a7z57_trace) {
                LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_return_true_sync_build");
            }
            return true;
        }
    }

    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_before_queue_worker_build");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_before_queue_worker_build");
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_before_queue_worker_build");
    }
    worker->QueueWork([this] { Build(); });
    is_pending = true;
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_after_queue_worker_build");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 trybuild_after_queue_worker_build");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_is_pending_after_queue", is_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 trybuild_return_after_queue", wait_built);
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 trybuild_after_queue_worker_build");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_is_pending_after_queue", is_pending);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 trybuild_return_after_queue", wait_built);
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 trybuild_return_after_queue");
    }
    return wait_built;
}

bool GraphicsPipeline::Build(bool fail_on_compile_required) {
    BORKED3DS_PROFILE("Vulkan", "Pipeline Building");
    const bool a7z61_ultra_quiet = IsV115DA7Z61GraphicsPipelineUltraQuietTryBuildEnabled();
    const bool a7z57_trace =
        IsV115DA7Z57GraphicsPipelineMainLogOnlyEnabled() && !a7z61_ultra_quiet;
    const bool a7z51_trace = IsV115DA7Z51GraphicsPipelineForceTraceEnabled() &&
                             !a7z57_trace && !a7z61_ultra_quiet;
    const bool a7z48_trace =
        (IsV115DA7Z48GraphicsPipelineTraceEnabled() || a7z51_trace) && !a7z57_trace &&
        !a7z61_ultra_quiet;
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_enter");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_enter");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_fail_on_compile_required", fail_on_compile_required);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_is_done_before", IsDone());
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_enter");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_fail_on_compile_required", fail_on_compile_required);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_is_done_before", IsDone());
    }
    const vk::Device device = instance.GetDevice();
    const bool pi5_strict_compat = IsPi5StrictCompatEnabled();
    const bool use_extended_dynamic_state =
        instance.IsExtendedDynamicStateSupported() && !pi5_strict_compat;
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_pi5_strict_compat", pi5_strict_compat);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_use_extended_dynamic_state", use_extended_dynamic_state);
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_pi5_strict_compat", pi5_strict_compat);
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_use_extended_dynamic_state", use_extended_dynamic_state);
    }

    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_before_vertex_bindings");
    }
    std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BINDINGS> bindings;
    for (u32 i = 0; i < info.vertex_layout.binding_count; i++) {
        const auto& binding = info.vertex_layout.bindings[i];
        // v116-B: on Pi5/V3DV strict_compat a fixed (constant) attribute binding is bound
        // per-vertex (eVertex). SetupFixedAttribs then replicates the constant block once per
        // vertex with the real stride, so vertex N reads its own identical copy. This avoids
        // relying on eInstance / stride-0 fixed-attribute fetch, which V3DV does not deliver to
        // every vertex on the non-instanced safe-draw path (flat texcoord reg2 / color reg1 ->
        // invisible dialogue text and washed-out Sonic). Other backends keep eInstance.
        const bool force_vertex_rate = pi5_strict_compat && binding.fixed.Value();
        bindings[i] = vk::VertexInputBindingDescription{
            .binding = binding.binding,
            .stride = binding.stride,
            .inputRate = (binding.fixed.Value() && !force_vertex_rate)
                             ? vk::VertexInputRate::eInstance
                             : vk::VertexInputRate::eVertex,
        };
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_after_vertex_bindings");
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_before_vertex_attributes");
    }

    std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> attributes;
    for (u32 i = 0; i < info.vertex_layout.attribute_count; i++) {
        const auto& attr = info.vertex_layout.attributes[i];
        const FormatTraits& traits = instance.GetTraits(attr.type, attr.size);
        attributes[i] = vk::VertexInputAttributeDescription{
            .location = attr.location,
            .binding = attr.binding,
            .format = traits.native,
            .offset = attr.offset,
        };

        if (traits.needs_emulation) {
            const FormatTraits& comp_four_traits = instance.GetTraits(attr.type, 4);
            attributes[i].format = comp_four_traits.native;
        }
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_after_vertex_attributes");
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = info.vertex_layout.binding_count,
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = info.vertex_layout.attribute_count,
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = PicaToVK::PrimitiveTopology(info.rasterization.topology),
        .primitiveRestartEnable = false,
    };

    vk::CullModeFlags raster_cull_mode = PicaToVK::CullMode(info.rasterization.cull_mode);
    vk::FrontFace raster_front_face = PicaToVK::FrontFace(info.rasterization.cull_mode);
    if (pi5_strict_compat) {
        // Pi 5 / V3DV conservative fallback:
        // prefer showing too much geometry over accidentally culling everything
        // when Vulkan pipeline state diverges from GLES behavior.
        raster_cull_mode = vk::CullModeFlagBits::eNone;
        raster_front_face = vk::FrontFace::eCounterClockwise;
    }

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = raster_cull_mode,
        .frontFace = raster_front_face,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = false,
        .alphaToOneEnable = false,
    };

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = info.blending.blend_enable,
        .srcColorBlendFactor = PicaToVK::BlendFunc(info.blending.src_color_blend_factor),
        .dstColorBlendFactor = PicaToVK::BlendFunc(info.blending.dst_color_blend_factor),
        .colorBlendOp = PicaToVK::BlendEquation(info.blending.color_blend_eq),
        .srcAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.src_alpha_blend_factor),
        .dstAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.dst_alpha_blend_factor),
        .alphaBlendOp = PicaToVK::BlendEquation(info.blending.alpha_blend_eq),
        .colorWriteMask = static_cast<vk::ColorComponentFlags>(info.blending.color_write_mask),
    };

    const vk::Bool32 logic_op_enable =
        (!pi5_strict_compat && !info.blending.blend_enable && !instance.NeedsLogicOpEmulation());

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = logic_op_enable,
        .logicOp = PicaToVK::LogicOp(info.blending.logic_op),
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = 1.0f,
        .height = 1.0f,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = {1, 1},
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    boost::container::static_vector<vk::DynamicState, 14> dynamic_states = {
        vk::DynamicState::eViewport,           vk::DynamicState::eScissor,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilReference,   vk::DynamicState::eBlendConstants,
    };

    if (use_extended_dynamic_state) {
        constexpr std::array extended = {
            vk::DynamicState::eCullModeEXT,        vk::DynamicState::eDepthCompareOpEXT,
            vk::DynamicState::eDepthTestEnableEXT, vk::DynamicState::eDepthWriteEnableEXT,
            vk::DynamicState::eFrontFaceEXT,       vk::DynamicState::ePrimitiveTopologyEXT,
            vk::DynamicState::eStencilOpEXT,       vk::DynamicState::eStencilTestEnableEXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());
    }

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::StencilOpState stencil_op_state = {
        .failOp = PicaToVK::StencilOp(info.depth_stencil.stencil_fail_op),
        .passOp = PicaToVK::StencilOp(info.depth_stencil.stencil_pass_op),
        .depthFailOp = PicaToVK::StencilOp(info.depth_stencil.stencil_depth_fail_op),
        .compareOp = PicaToVK::CompareFunc(info.depth_stencil.stencil_compare_op),
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = static_cast<vk::Bool32>(info.depth_stencil.depth_test_enable.Value()),
        .depthWriteEnable = static_cast<vk::Bool32>(info.depth_stencil.depth_write_enable.Value()),
        .depthCompareOp = PicaToVK::CompareFunc(info.depth_stencil.depth_compare_op),
        .depthBoundsTestEnable = false,
        .stencilTestEnable = static_cast<vk::Bool32>(info.depth_stencil.stencil_test_enable.Value()),
        .front = stencil_op_state,
        .back = stencil_op_state,
    };

    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_before_shader_stage_loop");
    }
    u32 shader_count = 0;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> shader_stages;
    for (std::size_t i = 0; i < stages.size(); i++) {
        Shader* shader = stages[i];
        if (!shader) {
            continue;
        }

        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_shader_index", static_cast<u32>(i));
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_shader_done_before_wait", shader->IsDone());
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_before_shader_wait_done");
        }
        if (PipelineBuildStats::Enabled()) {
            const auto sw_t0 = std::chrono::steady_clock::now();
            shader->WaitDone();
            const auto sw_t1 = std::chrono::steady_clock::now();
            PipelineBuildStats::RecordShaderWait(
                u64(std::chrono::duration_cast<std::chrono::nanoseconds>(sw_t1 - sw_t0).count()));
        } else {
            shader->WaitDone();
        }
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_after_shader_wait_done");
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_shader_done_after_wait", shader->IsDone());
        }
        shader_stages[shader_count++] = vk::PipelineShaderStageCreateInfo{
            .stage = MakeShaderStage(i),
            .module = shader->Handle(),
            .pName = "main",
        };
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_after_shader_stage_loop");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_shader_count", shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_vertex_binding_count", info.vertex_layout.binding_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_vertex_attribute_count", info.vertex_layout.attribute_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_color_attachment", static_cast<u32>(info.attachments.color));
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_depth_attachment", static_cast<u32>(info.attachments.depth));
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_shader_count", shader_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_vertex_binding_count", info.vertex_layout.binding_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_vertex_attribute_count", info.vertex_layout.attribute_count);
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_color_attachment", static_cast<u32>(info.attachments.color));
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_depth_attachment", static_cast<u32>(info.attachments.depth));
    }

    vk::GraphicsPipelineCreateInfo pipeline_info = {
        .stageCount = shader_count,
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_state,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout,
        .renderPass =
            renderpass_cache.GetRenderpass(info.attachments.color, info.attachments.depth, false),
    };

    if (pi5_strict_compat && IsDrawTraceEnabled()) {
        LOG_INFO(Render_Vulkan,
                 "TRACE_PIPELINE strict_compat build shader_count={} color_attachment={} depth_attachment={} cull_mode={} front_face={} depth_test={} depth_write={} stencil_test={} blend_enable={} logic_op_enable={} sample_count=1 sample_shading=0 use_extended_dynamic_state={}",
                 shader_count, static_cast<u32>(info.attachments.color),
                 static_cast<u32>(info.attachments.depth), static_cast<u32>(static_cast<VkCullModeFlags>(raster_cull_mode)),
                 static_cast<u32>(static_cast<VkFrontFace>(raster_front_face)),
                 static_cast<u32>(info.depth_stencil.depth_test_enable.Value()),
                 static_cast<u32>(info.depth_stencil.depth_write_enable.Value()),
                 static_cast<u32>(info.depth_stencil.stencil_test_enable.Value()),
                 static_cast<u32>(info.blending.blend_enable),
                 static_cast<u32>(logic_op_enable),
                 static_cast<u32>(use_extended_dynamic_state));
    }

    if (fail_on_compile_required) {
        pipeline_info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT;
    }

    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_before_create_graphics_pipeline");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_create_fail_on_compile_required", fail_on_compile_required);
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_before_create_graphics_pipeline");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_create_fail_on_compile_required", fail_on_compile_required);
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_before_create_graphics_pipeline");
    }
    const bool trace_build = PipelineBuildStats::Enabled();
    const auto build_t0 = trace_build ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    auto result = device.createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
    if (trace_build) {
        const auto build_t1 = std::chrono::steady_clock::now();
        const u64 build_elapsed_ns = u64(
            std::chrono::duration_cast<std::chrono::nanoseconds>(build_t1 - build_t0).count());
        if (result.result == vk::Result::eSuccess) {
            if (fail_on_compile_required) {
                // Succes AVEC le flag = pipeline deja en cache driver, aucune
                // compilation reelle : c'est le cas rapide (pas de gel).
                PipelineBuildStats::RecordCacheHit();
            } else {
                // Vraie compilation V3DV -- c'est CE temps qui fige la presentation.
                PipelineBuildStats::RecordCompile(build_elapsed_ns);
                if (build_elapsed_ns > POISON_THRESHOLD_NS) {
                    // Compilation pathologique (> 2 s) : capturer sa signature
                    // complete + son SPIR-V pour identifier ce qui empoisonne V3DV.
                    DumpPoisonPipeline(instance, info, stages, build_elapsed_ns);
                }
            }
        }
        PipelineBuildStats::ReportIfDue();
    }
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_after_create_graphics_pipeline");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z51 build_vk_result", static_cast<u32>(result.result));
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTraceU32("v115d_a7z48 build_vk_result", static_cast<u32>(result.result));
    }
    if (result.result == vk::Result::eSuccess) {
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_result_success");
        }
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_result_success");
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_result_success");
        }
        pipeline = std::move(result.value);
    } else if (result.result == vk::Result::eErrorPipelineCompileRequiredEXT) {
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_result_error_compile_required");
        }
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_result_error_compile_required");
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_is_done_on_compile_required", IsDone());
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_result_error_compile_required");
            AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_is_done_on_compile_required", IsDone());
        }
        return false;
    } else {
        if (a7z57_trace) {
            LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_result_unreachable_error");
        }
        if (a7z51_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_result_unreachable_error");
        }
        if (a7z48_trace) {
            AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_result_unreachable_error");
        }
        UNREACHABLE_MSG("Graphics pipeline creation failed!");
    }

    MarkDone();
    if (a7z57_trace) {
        LogV115DA7Z57GraphicsPipeline("v115d_a7z57 build_mark_done");
    }
    if (a7z51_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z51 build_mark_done");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z51 build_is_done_after_mark", IsDone());
    }
    if (a7z48_trace) {
        AppendV115DA7Z48GraphicsPipelineTrace("v115d_a7z48 build_mark_done");
        AppendV115DA7Z48GraphicsPipelineTraceBool("v115d_a7z48 build_is_done_after_mark", IsDone());
    }
    return true;
}

} // namespace Vulkan
