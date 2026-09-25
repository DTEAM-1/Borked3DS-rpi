// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <memory>
#include <unordered_map>

#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/profiling.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_descriptor_update_queue.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/shader/generator/glsl_fs_shader_gen.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/shader/generator/spv_fs_shader_gen.h"
#include "video_core/shader/generator/spv_shader_gen.h"

using namespace Pica::Shader::Generator;
using Pica::Shader::FSConfig;

namespace Vulkan {

namespace {

[[nodiscard]] bool IsEnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] bool IsPi5StrictCompatEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_STRICT_COMPAT");
    return cached;
}

// BORKED3DS_V3DV_DISABLE_EDS -- voir la note detaillee dans vk_graphics_pipeline.cpp.
// Doit imperativement etre lu AUX DEUX sites : ici (emission des setXxxEXT) et
// dans Build() (declaration des dynamic states). Les desynchroniser produirait un
// pipeline declarant un etat dynamique jamais emis, ou l'inverse.
[[nodiscard]] bool IsV3dvEdsDisabled() {
    static const bool disabled = std::getenv("BORKED3DS_V3DV_DISABLE_EDS") != nullptr;
    return disabled;
}

[[nodiscard]] bool IsV115DA7Z41PipelineCacheTraceEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z41_PIPELINE_CACHE_TRACE");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z41PipelineForceNoWaitOnWaitEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z41_PIPELINE_FORCE_NOWAIT_ON_WAIT");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z44PipelineNoWaitRetryEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z44_PIPELINE_NOWAIT_RETRY");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z54PipelineCacheMinimalSafeEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z54_PIPELINE_CACHE_MINIMAL_SAFE");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z55PipelineCacheMainLogOnlyEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z55_PIPELINE_CACHE_MAINLOG_ONLY");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z56PipelineCachePlainMarkersEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z56_PIPELINE_CACHE_PLAIN_MARKERS");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z58PipelineCacheSplitTryEmplaceEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z58_PIPELINE_CACHE_SPLIT_TRY_EMPLACE");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z59PipelineCachePostTryFastReturnEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z59_PIPELINE_CACHE_POST_TRY_FAST_RETURN");
    return cached;
}

[[nodiscard]] bool IsV115DA7Z60PipelineCacheUltraQuietFastReturnEnabled() {
    static const bool cached = IsEnvFlagEnabled("BORKED3DS_V3DV_A7Z60_PIPELINE_CACHE_ULTRA_QUIET_FAST_RETURN");
    return cached;
}

[[nodiscard]] u32 GetEnvU32Limited(const char* name, u32 default_value, u32 max_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return default_value;
    }

    const u32 limited = static_cast<u32>(std::min<unsigned long>(parsed, max_value));
    return limited;
}

// v331 : taille du pool de compilation.
// L'expression d'origine, std::max(std::thread::hardware_concurrency(), 2U) >> 1, donne
// DEUX threads sur un Pi5 quatre coeurs. Et ce meme pool sert QUATRE sortes de travail :
// compilation des vertex shaders, des geometry shaders, des fragment shaders, et
// construction des pipelines.
// Mesure v330, 15 s de jeu sur Kid Icarus : 10 671 appels a TryBuild, dont
//     4 733 refuses parce qu'un shader n'est pas encore compile (shaders_pending),
//     5 804 refuses parce qu'un build est deja en file (is_pending),
//       134 seulement ont atteint la mise en file.
// Soit 98 % de refus pour cause de compilation en retard, pendant que le thread de rendu
// tourne a 24-30 % de CPU. Le plafond n'est donc ni le GPU ni le rendu : c'est ce pool.
// Autre contrainte mesuree : trybuild_cache_control_supported=0 sur toutes les occurrences.
// V3DV n'expose pas VK_EXT_pipeline_creation_cache_control, donc le chemin de build
// synchrone rapide est inaccessible et la file asynchrone est la SEULE voie disponible.
// Defaut porte a hardware_concurrency - 1 (3 sur Pi5), en laissant un coeur au reste.
// BORKED3DS_V3DV_PIPELINE_WORKER_THREADS permet de revenir a 2, ou d'essayer 4, sans rebuild.
[[nodiscard]] std::size_t V331PipelineWorkerThreadCount() {
    const u32 hw = std::max(std::thread::hardware_concurrency(), 2u);
    const u32 fallback = hw > 2u ? hw - 1u : 2u;
    const u32 requested =
        GetEnvU32Limited("BORKED3DS_V3DV_PIPELINE_WORKER_THREADS", fallback, 8u);
    return static_cast<std::size_t>(std::max(requested, 1u));
}

void AppendV115DPipelineCacheTraceLine(const std::string& line) {
    static std::mutex trace_mutex;
    std::lock_guard<std::mutex> lock{trace_mutex};

    std::ofstream file{"/tmp/borked3ds_v115d_mux_shader_probe.log", std::ios::app};
    if (!file.is_open()) {
        return;
    }

    file << line << '\n';
}

void AppendV115DPipelineCacheTraceU64(const std::string& key, u64 value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z41 " + key + "=" + std::to_string(value));
}

void AppendV115DPipelineCacheTraceBool(const std::string& key, bool value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z41 " + key + "=" + std::to_string(value ? 1 : 0));
}

void AppendV115DPipelineCacheTraceA7Z44U32(const std::string& key, u32 value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z44 " + key + "=" + std::to_string(value));
}

void AppendV115DPipelineCacheTraceA7Z44Bool(const std::string& key, bool value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z44 " + key + "=" + std::to_string(value ? 1 : 0));
}

void AppendV115DPipelineCacheTraceA7Z54Line(const std::string& line) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z54 " + line);
}

void AppendV115DPipelineCacheTraceA7Z54U64(const std::string& key, u64 value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z54 " + key + "=" + std::to_string(value));
}

void AppendV115DPipelineCacheTraceA7Z54Bool(const std::string& key, bool value) {
    AppendV115DPipelineCacheTraceLine("v115d_a7z54 " + key + "=" + std::to_string(value ? 1 : 0));
}

} // namespace

u32 AttribBytes(Pica::PipelineRegs::VertexAttributeFormat format, u32 size) {
    switch (format) {
    case Pica::PipelineRegs::VertexAttributeFormat::FLOAT:
        return sizeof(float) * size;
    case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
        return sizeof(u16) * size;
    case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
    case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
        return sizeof(u8) * size;
    }
    return 0;
}

AttribLoadFlags MakeAttribLoadFlag(Pica::PipelineRegs::VertexAttributeFormat format) {
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

constexpr std::array<vk::DescriptorSetLayoutBinding, 7> BUFFER_BINDINGS = {{
    {0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex},
    {1, vk::DescriptorType::eUniformBufferDynamic, 1,
     vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry},
    {2, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eFragment},
    {3, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    {4, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    {5, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment},
    {6, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eVertex},
}};

template <u32 NumTex0>
constexpr std::array<vk::DescriptorSetLayoutBinding, 3> TEXTURE_BINDINGS = {{
    {0, vk::DescriptorType::eCombinedImageSampler, NumTex0,
     vk::ShaderStageFlagBits::eFragment},                                                  // tex0
    {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex1
    {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex2
}};

constexpr std::array<vk::DescriptorSetLayoutBinding, 2> UTILITY_BINDINGS = {{
    {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eFragment}, // shadow_buffer
    {1, vk::DescriptorType::eCombinedImageSampler, 1,
     vk::ShaderStageFlagBits::eFragment}, // tex_normal
}};

namespace {
/// vDIRA v123 (BORKED3DS_V3DV_DIRA_TRIVIAL_VS_GLSL=1): build the trivial vertex shader from the
/// GLSL generator compiled through glslang instead of the direct SPIR-V generator. Rationale: this
/// fork already migrated every PROGRAMMABLE vertex shader to the GLSL->glslang path (see
/// UseProgrammableVertexShader: the SPIRV generator is commented out) and those shaders work on
/// V3DV -- the trivial VS was the only one left on the direct SPIR-V generator, and it feeds
/// EXCLUSIVELY the software-vertex draw path, which is precisely the path that produces zero
/// fragments on V3DV (proven by the v122 fullscreen-triangle substitution: correct pipeline,
/// correct state, correct data injected straight into the GPU buffer, still nothing). The
/// spirv_shader_gen setting never covered this shader, so it was never innocented. Default
/// behaviour (flag absent) is unchanged.
std::vector<u32> MakeTrivialVertexShaderCode(const Instance& instance) {
    const bool has_clip = instance.IsShaderClipDistanceSupported();
    static const bool use_glsl_trivial_vs =
        std::getenv("BORKED3DS_V3DV_DIRA_TRIVIAL_VS_GLSL") != nullptr;
    if (use_glsl_trivial_vs) {
        const std::string program = GLSL::GenerateTrivialVertexShader(has_clip, true);
        return CompileGLSLtoSPIRV(program, vk::ShaderStageFlagBits::eVertex,
                                  instance.GetDevice());
    }
    const auto spirv = SPIRV::GenerateTrivialVertexShader(has_clip);
    return std::vector<u32>(spirv.begin(), spirv.end());
}
} // namespace

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             RenderManager& renderpass_cache_, DescriptorUpdateQueue& update_queue_)
    : instance{instance_}, scheduler{scheduler_}, renderpass_cache{renderpass_cache_},
      update_queue{update_queue_},
      num_worker_threads{V331PipelineWorkerThreadCount()},
      workers{num_worker_threads, "Pipeline workers"},
      descriptor_heaps{
          DescriptorHeap{instance, scheduler.GetMasterSemaphore(), BUFFER_BINDINGS, 32},
          DescriptorHeap{instance, scheduler.GetMasterSemaphore(), TEXTURE_BINDINGS<1>},
          DescriptorHeap{instance, scheduler.GetMasterSemaphore(), UTILITY_BINDINGS, 32}},
      trivial_vertex_shader{instance, MakeTrivialVertexShaderCode(instance)} {
    scheduler.RegisterOnDispatch([this] { update_queue.Flush(); });
    const bool pi5_strict_compat = IsPi5StrictCompatEnabled();
    profile = Pica::Shader::Profile{
        .has_separable_shaders = true,
        .has_clip_planes = instance.IsShaderClipDistanceSupported(),
        .has_geometry_shader = instance.UseGeometryShaders(),
        .has_custom_border_color = instance.IsCustomBorderColorSupported(),
        .has_fragment_shader_interlock = instance.IsFragmentShaderInterlockSupported(),
        .has_fragment_shader_barycentric = instance.IsFragmentShaderBarycentricSupported(),
        .has_blend_minmax_factor = false,
        .has_minus_one_to_one_range = false,
        .has_logic_op = !pi5_strict_compat && !instance.NeedsLogicOpEmulation(),
        .is_vulkan = true,
    };
    BuildLayout();

    // v370 -- CACHE PIPELINE PERDUE APRES CHARGEMENT D'UN SAVESTATE.
    //
    // La cache pipeline Vulkan (pipeline_cache) n'etait creee que par LoadDiskCache(), appelee
    // une seule fois au demarrage par EmuThread (bootmanager.cpp, LoadDiskResources). Or charger
    // un savestate fait System::serialize -> Shutdown(true) -> Init() : TOUT le renderer est
    // recree, donc un nouveau PipelineCache, et LoadDiskResources n'est plus jamais rappele.
    // Consequences mesurees (runs Z4b/Z4c, Kid Icarus) :
    //   - toute la partie tourne avec pipeline_cache = VK_NULL_HANDLE : aucun pipeline n'est
    //     mis en cache, les memes pipelines de 14 a 34 s se recompilent a chaque session ;
    //   - a la sortie, SaveDiskCache() voit !pipeline_cache et n'ecrit rien ;
    //   - le seul fichier ecrit est celui de l'ancienne instance detruite au chargement du
    //     savestate : 40 octets (en-tete seul), date = heure du chargement du savestate.
    // Correctif : creer et charger la cache des la construction. LoadDiskCache() devient
    // idempotente, donc l'appel de LoadDiskResources au demarrage ne change rien.
    LoadDiskCache();
}

void PipelineCache::BuildLayout() {
    std::array<vk::DescriptorSetLayout, NumRasterizerSets> descriptor_set_layouts;
    descriptor_set_layouts[0] = descriptor_heaps[0].Layout();
    descriptor_set_layouts[1] = descriptor_heaps[1].Layout();
    descriptor_set_layouts[2] = descriptor_heaps[2].Layout();

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = NumRasterizerSets,
        .pSetLayouts = descriptor_set_layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    pipeline_layout = instance.GetDevice().createPipelineLayoutUnique(layout_info);
}

PipelineCache::~PipelineCache() {
    // V384 : une sauvegarde de fond peut etre en cours ; la terminer avant la derniere.
    if (v384_save_thread.joinable()) {
        v384_save_thread.join();
    }
    SaveDiskCache();
}

// ---------------------------------------------------------------------------------------
// V384 (methode E) -- SAUVEGARDER LA CACHE PIPELINE PENDANT LA PARTIE.
//
// La cache n'etait ecrite qu'a la destruction du PipelineCache (sortie propre). Un plantage,
// une coupure de courant ou un arret brutal perdait TOUTES les compilations de la session
// (jusqu'a plusieurs minutes de poisons au premier passage). Ici : au plus toutes les 30 s,
// et seulement si de nouveaux pipelines ont ete crees depuis la derniere sauvegarde, la
// cache est ecrite sur un fil de fond (aucun gel du rendu). Ecriture atomique (.tmp puis
// rename) : un arret pendant l'ecriture laisse l'ancien fichier intact.
// Echappatoire (A/B seulement) : BORKED3DS_V3DV_V384_NO_CACHE_SAVE=1.
// ---------------------------------------------------------------------------------------
void PipelineCache::V384MaybeSaveDiskCache() {
    static const bool disabled = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_V384_NO_CACHE_SAVE");
        return v != nullptr && v[0] != '\0';
    }();
    constexpr auto kPeriod = std::chrono::seconds(30);
    if (disabled || !pipeline_cache || !Settings::values.use_disk_shader_cache) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (v384_last_save.time_since_epoch().count() == 0) {
        v384_last_save = now; // premiere image : on laisse passer une periode
        v384_saved_pipelines = graphics_pipelines.size();
        return;
    }
    if (now - v384_last_save < kPeriod) {
        return;
    }
    const std::size_t count = graphics_pipelines.size();
    if (count == v384_saved_pipelines || v384_save_running.load(std::memory_order_acquire)) {
        return;
    }
    if (v384_save_thread.joinable()) {
        v384_save_thread.join(); // fil precedent deja termine (save_running faux)
    }
    v384_last_save = now;
    v384_saved_pipelines = count;
    v384_save_running.store(true, std::memory_order_release);
    v384_save_thread = std::thread([this, count] {
        const auto t0 = std::chrono::steady_clock::now();
        SaveDiskCache();
        LOG_WARNING(Render_Vulkan, "V384_CACHE_EN_PARTIE pipelines={} ms={}", count,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count());
        v384_save_running.store(false, std::memory_order_release);
    });
}

void PipelineCache::LoadDiskCache() {
    // v370 : idempotent. Deja creee (par le constructeur) -> ne pas la remplacer : des
    // pipelines en cours de compilation sur les workers utilisent deja ce handle.
    if (pipeline_cache) {
        return;
    }
    if (!Settings::values.use_disk_shader_cache || !EnsureDirectories()) {
        return;
    }

    const auto cache_dir = GetPipelineCacheDir();
    const u32 vendor_id = instance.GetVendorID();
    const u32 device_id = instance.GetDeviceID();
    const auto cache_file_path = fmt::format("{}{:x}{:x}.bin", cache_dir, vendor_id, device_id);

    vk::PipelineCacheCreateInfo cache_info{};
    std::vector<u8> cache_data;

    SCOPE_EXIT({
        const vk::Device device = instance.GetDevice();
        pipeline_cache = device.createPipelineCacheUnique(cache_info);
    });

    FileUtil::IOFile cache_file{cache_file_path, "rb"};
    if (!cache_file.IsOpen()) {
        LOG_INFO(Render_Vulkan, "No pipeline cache found for device");
        return;
    }

    const u64 cache_file_size = cache_file.GetSize();
    cache_data.resize(cache_file_size);
    if (cache_file.ReadBytes(cache_data.data(), cache_file_size) != cache_file_size) {
        LOG_ERROR(Render_Vulkan, "Error during pipeline cache read");
        return;
    }

    if (!IsCacheValid(cache_data)) {
        LOG_WARNING(Render_Vulkan, "Pipeline cache provided invalid, removing");
        cache_file.Close();
        FileUtil::Delete(cache_file_path);
        return;
    }

    LOG_INFO(Render_Vulkan, "Loading pipeline cache with size {} KB", cache_file_size / 1024);
    LOG_WARNING(Render_Vulkan, "V370_PIPELINE_CACHE_CHARGEE octets={}", cache_file_size);
    cache_info.initialDataSize = cache_file_size;
    cache_info.pInitialData = cache_data.data();
}

void PipelineCache::SaveDiskCache() {
    if (!Settings::values.use_disk_shader_cache || !EnsureDirectories() || !pipeline_cache) {
        return;
    }
    // V384 : serialise les sauvegardes (fil de fond et sortie ne s'ecrivent jamais en meme temps).
    std::scoped_lock lock{v384_save_mutex};

    const auto cache_dir = GetPipelineCacheDir();
    const u32 vendor_id = instance.GetVendorID();
    const u32 device_id = instance.GetDeviceID();
    const auto cache_file_path = fmt::format("{}{:x}{:x}.bin", cache_dir, vendor_id, device_id);
    // V384 : ecriture dans un fichier temporaire puis renommage atomique.
    const auto tmp_file_path = cache_file_path + ".tmp";

    const vk::Device device = instance.GetDevice();
    const auto cache_data = device.getPipelineCacheData(*pipeline_cache);
    {
        FileUtil::IOFile cache_file{tmp_file_path, "wb"};
        if (!cache_file.IsOpen()) {
            LOG_ERROR(Render_Vulkan, "Unable to open pipeline cache for writing");
            return;
        }
        if (cache_file.WriteBytes(cache_data.data(), cache_data.size()) != cache_data.size()) {
            LOG_ERROR(Render_Vulkan, "Error during pipeline cache write");
            cache_file.Close();
            FileUtil::Delete(tmp_file_path);
            return;
        }
    }
    if (!FileUtil::Rename(tmp_file_path, cache_file_path)) {
        FileUtil::Delete(tmp_file_path);
        return;
    }
    LOG_WARNING(Render_Vulkan, "V370_PIPELINE_CACHE_SAUVEE octets={}", cache_data.size());
}

bool PipelineCache::BindPipeline(const PipelineInfo& info, bool wait_built) {
    BORKED3DS_PROFILE("Vulkan", "Pipeline Bind");

    const bool a7z60_ultra_quiet_fast_return =
        IsV115DA7Z60PipelineCacheUltraQuietFastReturnEnabled();
    const bool a7z59_post_try_fast_return =
        IsV115DA7Z59PipelineCachePostTryFastReturnEnabled() &&
        !a7z60_ultra_quiet_fast_return;
    const bool a7z58_split_try_emplace =
        IsV115DA7Z58PipelineCacheSplitTryEmplaceEnabled() && !a7z59_post_try_fast_return && !a7z60_ultra_quiet_fast_return;
    const bool a7z56_plain_markers =
        IsV115DA7Z56PipelineCachePlainMarkersEnabled() && !a7z58_split_try_emplace &&
        !a7z59_post_try_fast_return && !a7z60_ultra_quiet_fast_return;
    const bool a7z55_mainlog_only =
        IsV115DA7Z55PipelineCacheMainLogOnlyEnabled() && !a7z56_plain_markers &&
        !a7z58_split_try_emplace && !a7z59_post_try_fast_return && !a7z60_ultra_quiet_fast_return;
    const bool a7z54_minimal_safe =
        IsV115DA7Z54PipelineCacheMinimalSafeEnabled() && !a7z55_mainlog_only &&
        !a7z56_plain_markers && !a7z58_split_try_emplace && !a7z59_post_try_fast_return && !a7z60_ultra_quiet_fast_return;
    const bool a7z41_trace =
        IsV115DA7Z41PipelineCacheTraceEnabled() && !a7z54_minimal_safe &&
        !a7z55_mainlog_only && !a7z56_plain_markers && !a7z58_split_try_emplace &&
        !a7z59_post_try_fast_return && !a7z60_ultra_quiet_fast_return;
    const bool a7z41_force_nowait_on_wait = IsV115DA7Z41PipelineForceNoWaitOnWaitEnabled();
    const bool a7z44_nowait_retry = IsV115DA7Z44PipelineNoWaitRetryEnabled();
    // A1 (etendu) : ces deux lectures etaient faites par draw. Les valeurs viennent de
    // l'environnement, fixe au lancement -- une lecture unique suffit.
    static const u32 a7z44_retry_count =
        GetEnvU32Limited("BORKED3DS_V3DV_A7Z44_PIPELINE_NOWAIT_RETRY_COUNT", 4, 16);
    static const u32 a7z44_retry_sleep_ms =
        GetEnvU32Limited("BORKED3DS_V3DV_A7Z44_PIPELINE_NOWAIT_RETRY_SLEEP_MS", 1, 10);
    const bool effective_wait_built = wait_built && !a7z41_force_nowait_on_wait;

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceLine("v115d_a7z41 bind_pipeline_enter");
        AppendV115DPipelineCacheTraceBool("wait_built_requested", wait_built);
        AppendV115DPipelineCacheTraceBool("force_nowait_on_wait", a7z41_force_nowait_on_wait);
        AppendV115DPipelineCacheTraceBool("effective_wait_built", effective_wait_built);
        AppendV115DPipelineCacheTraceA7Z44Bool("nowait_retry_enabled", a7z44_nowait_retry);
        AppendV115DPipelineCacheTraceA7Z44U32("retry_count", a7z44_retry_count);
        AppendV115DPipelineCacheTraceA7Z44U32("retry_sleep_ms", a7z44_retry_sleep_ms);
    }

    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Line("bind_pipeline_enter");
        AppendV115DPipelineCacheTraceA7Z54Bool("wait_built_requested", wait_built);
        AppendV115DPipelineCacheTraceA7Z54Bool("force_nowait_on_wait",
                                               a7z41_force_nowait_on_wait);
        AppendV115DPipelineCacheTraceA7Z54Bool("effective_wait_built", effective_wait_built);
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW v115d_a7z54 bind_pipeline_enter effective_wait_built={}",
                    effective_wait_built ? 1 : 0);
    }

    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW v115d_a7z55 bind_pipeline_enter wait={} force_nowait={} "
                    "effective_wait={}",
                    wait_built ? 1 : 0, a7z41_force_nowait_on_wait ? 1 : 0,
                    effective_wait_built ? 1 : 0);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 bind_pipeline_enter");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 bind_pipeline_enter");
    }
    if (a7z59_post_try_fast_return) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z59 bind_pipeline_enter");
    }
    if (a7z60_ultra_quiet_fast_return) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z60 bind_pipeline_enter");
    }

    u64 shader_hash = 0;
    for (u32 i = 0; i < MAX_SHADER_STAGES; i++) {
        shader_hash = Common::HashCombine(shader_hash, shader_hashes[i]);
    }

    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54U64("shader_hash", shader_hash);
        AppendV115DPipelineCacheTraceA7Z54Line("before_info_hash");
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z54 before_info_hash");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 before_info_hash shader_hash={}",
                    shader_hash);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 before_info_hash");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 before_info_hash");
    }

    const u64 info_hash = info.Hash(instance);

    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54U64("info_hash", info_hash);
        AppendV115DPipelineCacheTraceA7Z54Line("after_info_hash");
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z54 after_info_hash");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 after_info_hash info_hash={}",
                    info_hash);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 after_info_hash");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 after_info_hash");
    }

    const u64 pipeline_hash = Common::HashCombine(shader_hash, info_hash);

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceU64("shader_hash", shader_hash);
        AppendV115DPipelineCacheTraceU64("info_hash", info_hash);
        AppendV115DPipelineCacheTraceU64("pipeline_hash", pipeline_hash);
    }

    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54U64("pipeline_hash", pipeline_hash);
        AppendV115DPipelineCacheTraceA7Z54Line("before_try_emplace");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW v115d_a7z55 before_try_emplace pipeline_hash={}",
                    pipeline_hash);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 before_try_emplace");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 before_find");
    }

    auto it = graphics_pipelines.find(pipeline_hash);
    bool new_pipeline = it == graphics_pipelines.end();

    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 after_find");
    }

    if (new_pipeline) {
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 before_emplace_null");
        }
        it = graphics_pipelines.emplace(pipeline_hash, nullptr).first;
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 after_emplace_null");
        }
    }

    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Bool("new_pipeline", new_pipeline);
        AppendV115DPipelineCacheTraceA7Z54Line("after_try_emplace");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 after_try_emplace new_pipeline={}",
                    new_pipeline ? 1 : 0);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 after_try_emplace");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 after_try_emplace_split");
    }
    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceBool("new_pipeline", new_pipeline);
    }

    if (new_pipeline) {
        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 new_graphics_pipeline_begin");
        }
        if (a7z54_minimal_safe) {
            AppendV115DPipelineCacheTraceA7Z54Line("new_graphics_pipeline_begin");
        }
        if (a7z55_mainlog_only) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 new_graphics_pipeline_begin");
        }
        if (a7z56_plain_markers) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 new_graphics_pipeline_begin");
        }
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 new_graphics_pipeline_begin");
        }

        it.value() =
            std::make_unique<GraphicsPipeline>(instance, renderpass_cache, info, *pipeline_cache,
                                               *pipeline_layout, current_shaders, &workers);

        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 new_graphics_pipeline_end");
        }
        if (a7z54_minimal_safe) {
            AppendV115DPipelineCacheTraceA7Z54Line("new_graphics_pipeline_end");
        }
        if (a7z55_mainlog_only) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 new_graphics_pipeline_end");
        }
        if (a7z56_plain_markers) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 new_graphics_pipeline_end");
        }
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 new_graphics_pipeline_end");
        }
    }

    GraphicsPipeline* const pipeline{it->second.get()};
    if (pipeline == nullptr) {
        LOG_ERROR(Render_Vulkan, "TRACE_DRAW v115d_a7z58 null_pipeline_after_emplace");
        return false;
    }
    const bool pipeline_done_before_try = pipeline->IsDone();

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceBool("pipeline_done_before_try", pipeline_done_before_try);
    }
    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Bool("pipeline_done_before_try",
                                               pipeline_done_before_try);
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW v115d_a7z55 pipeline_done_before_try={}",
                    pipeline_done_before_try ? 1 : 0);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 pipeline_done_before_try");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 pipeline_done_before_try");
    }

    if (!pipeline_done_before_try) {
        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 try_build_begin");
            AppendV115DPipelineCacheTraceBool("try_build_wait", effective_wait_built);
        }
        if (a7z54_minimal_safe) {
            AppendV115DPipelineCacheTraceA7Z54Line("try_build_begin");
            AppendV115DPipelineCacheTraceA7Z54Bool("try_build_wait", effective_wait_built);
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z54 try_build_begin wait={}",
                        effective_wait_built ? 1 : 0);
        }
        if (a7z55_mainlog_only) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 try_build_begin wait={}",
                        effective_wait_built ? 1 : 0);
        }
        if (a7z56_plain_markers) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 try_build_begin");
        }
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 try_build_begin");
        }
        if (a7z59_post_try_fast_return) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z59 try_build_begin_no_postlog");
        }

        bool try_build_result = pipeline->TryBuild(effective_wait_built);

        // v371 -- COURTE ATTENTE A LA PREMIERE DEMANDE D'UN PIPELINE.
        //
        // Avec A7Z41_PIPELINE_FORCE_NOWAIT_ON_WAIT, un pipeline qui n'est pas pret fait sauter
        // son draw, meme si sa construction ne prend que 0,1 ms (cache V3DV chaude, v370). Test
        // S1 (Sonic, a chaud, census par image) : 28 images avec draws sautes en 10 grappes,
        // exactement aux instants ou de nouveaux pipelines sont construits, alors que tous se
        // construisent en 0,0-0,3 ms (compile_max_ms). Symptome visible : une boite de texte
        // affichee sans son texte puis corrigee, un petit flash noir, 3-4 fois par session.
        //
        // Correctif : a la PREMIERE demande d'un pipeline (new_pipeline) seulement, attendre
        // qu'il soit pret, au plus BORKED3DS_V3DV_PIPELINE_FIRST_WAIT_MS (defaut 16 ms, 0 =
        // desactive), en sondant toutes les 250 us. Un pipeline lent (poison de 10-40 s) coute
        // donc au plus 16 ms une seule fois, puis retrouve le comportement actuel (draw saute,
        // aucun gel). Plafond global : BORKED3DS_V3DV_PIPELINE_FIRST_WAIT_CAP_MS (defaut 48 ms
        // d'attente cumulee par fenetre d'une seconde) pour qu'une scene neuve avec beaucoup de
        // pipelines lents ne fabrique pas un gel. BindPipeline tourne sur EmuThread seulement.
        if (!try_build_result && new_pipeline && !effective_wait_built) {
            static const u32 v371_budget_ms =
                GetEnvU32Limited("BORKED3DS_V3DV_PIPELINE_FIRST_WAIT_MS", 16, 100);
            static const u32 v371_cap_ms =
                GetEnvU32Limited("BORKED3DS_V3DV_PIPELINE_FIRST_WAIT_CAP_MS", 48, 1000);
            if (v371_budget_ms > 0) {
                using v371_clock = std::chrono::steady_clock;
                static v371_clock::time_point v371_window_start = v371_clock::now();
                static u64 v371_window_spent_us = 0;
                static u64 v371_events = 0;
                static u64 v371_ready = 0;
                static u64 v371_expired = 0;
                static u64 v371_capped = 0;
                static u64 v371_total_us = 0;
                static u64 v371_max_us = 0;

                const auto t0 = v371_clock::now();
                if (t0 - v371_window_start >= std::chrono::seconds(1)) {
                    v371_window_start = t0;
                    v371_window_spent_us = 0;
                }
                const u64 cap_us = static_cast<u64>(v371_cap_ms) * 1000;
                const u64 left_us =
                    v371_window_spent_us < cap_us ? cap_us - v371_window_spent_us : 0;
                const u64 budget_us =
                    std::min<u64>(static_cast<u64>(v371_budget_ms) * 1000, left_us);

                const char* outcome = "plafond";
                if (budget_us > 0) {
                    const auto deadline = t0 + std::chrono::microseconds(budget_us);
                    outcome = "expire";
                    while (v371_clock::now() < deadline) {
                        std::this_thread::sleep_for(std::chrono::microseconds(250));
                        if (pipeline->IsDone() || pipeline->TryBuild(false)) {
                            try_build_result = true;
                            outcome = "pret";
                            break;
                        }
                    }
                }
                const u64 waited_us = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::microseconds>(v371_clock::now() - t0)
                        .count());
                v371_window_spent_us += waited_us;
                v371_total_us += waited_us;
                v371_max_us = std::max(v371_max_us, waited_us);
                ++v371_events;
                if (try_build_result) {
                    ++v371_ready;
                } else if (budget_us > 0) {
                    ++v371_expired;
                } else {
                    ++v371_capped;
                }
                if (v371_events <= 24 || (v371_events % 128) == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "V371_ATTENTE_PIPELINE #{} resultat={} attente_us={} | cumul: "
                                "prets={} expires={} plafonnes={} total_ms={:.1f} max_us={}",
                                v371_events, outcome, waited_us, v371_ready, v371_expired,
                                v371_capped, static_cast<double>(v371_total_us) / 1000.0,
                                v371_max_us);
                }
            }
        }

        if (a7z60_ultra_quiet_fast_return) {
            if (!try_build_result) {
                return false;
            }
            current_info = info;
            current_pipeline = pipeline;
            return true;
        }

        if (a7z59_post_try_fast_return) {
            if (!try_build_result) {
                return false;
            }
            current_info = info;
            current_pipeline = pipeline;
            return true;
        }

        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 try_build_end");
            AppendV115DPipelineCacheTraceBool("try_build_result", try_build_result);
            AppendV115DPipelineCacheTraceBool("pipeline_done_after_try", pipeline->IsDone());
        }
        if (a7z54_minimal_safe) {
            AppendV115DPipelineCacheTraceA7Z54Line("try_build_end");
            AppendV115DPipelineCacheTraceA7Z54Bool("try_build_result", try_build_result);
            AppendV115DPipelineCacheTraceA7Z54Bool("pipeline_done_after_try", pipeline->IsDone());
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW v115d_a7z54 try_build_end result={} done={}",
                        try_build_result ? 1 : 0, pipeline->IsDone() ? 1 : 0);
        }
        if (a7z55_mainlog_only) {
            LOG_WARNING(Render_Vulkan,
                        "TRACE_DRAW v115d_a7z55 try_build_end result={} done={}",
                        try_build_result ? 1 : 0, pipeline->IsDone() ? 1 : 0);
        }
        if (a7z56_plain_markers) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 try_build_end");
        }
        if (a7z58_split_try_emplace) {
            LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 try_build_end");
        }

        if (!try_build_result && a7z44_nowait_retry && !effective_wait_built) {
            for (u32 retry_index = 1; retry_index <= a7z44_retry_count; ++retry_index) {
                if (a7z44_retry_sleep_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(a7z44_retry_sleep_ms));
                }

                const bool retry_done_before_try = pipeline->IsDone();
                if (a7z41_trace) {
                    AppendV115DPipelineCacheTraceA7Z44U32("retry_index", retry_index);
                    AppendV115DPipelineCacheTraceA7Z44Bool("retry_done_before_try",
                                                             retry_done_before_try);
                }

                if (retry_done_before_try) {
                    try_build_result = true;
                    if (a7z41_trace) {
                        AppendV115DPipelineCacheTraceLine(
                            "v115d_a7z44 retry_done_before_try_accept_ready");
                    }
                    break;
                }

                const bool retry_try_build_result = pipeline->TryBuild(false);
                const bool retry_done_after_try = pipeline->IsDone();
                if (a7z41_trace) {
                    AppendV115DPipelineCacheTraceA7Z44Bool("retry_try_build_result",
                                                             retry_try_build_result);
                    AppendV115DPipelineCacheTraceA7Z44Bool("retry_done_after_try",
                                                             retry_done_after_try);
                }

                if (retry_try_build_result || retry_done_after_try) {
                    try_build_result = true;
                    if (a7z41_trace) {
                        AppendV115DPipelineCacheTraceLine(
                            "v115d_a7z44 retry_accept_ready_after_try");
                    }
                    break;
                }
            }

            if (a7z41_trace) {
                AppendV115DPipelineCacheTraceA7Z44Bool("final_try_build_result_after_retries",
                                                         try_build_result);
                AppendV115DPipelineCacheTraceA7Z44Bool("final_pipeline_done_after_retries",
                                                         pipeline->IsDone());
            }
        }

        if (!try_build_result) {
            if (a7z41_trace) {
                AppendV115DPipelineCacheTraceLine("v115d_a7z41 bind_pipeline_return_false_not_ready");
            }
            if (a7z54_minimal_safe) {
                AppendV115DPipelineCacheTraceA7Z54Line("bind_pipeline_return_false_not_ready");
            }
            if (a7z55_mainlog_only) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW v115d_a7z55 bind_pipeline_return_false_not_ready");
            }
            if (a7z56_plain_markers) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW v115d_a7z56 bind_pipeline_return_false_not_ready");
            }
            if (a7z58_split_try_emplace) {
                LOG_WARNING(Render_Vulkan,
                            "TRACE_DRAW v115d_a7z58 bind_pipeline_return_false_not_ready");
            }
            return false;
        }
    }

    const bool pi5_strict_compat = IsPi5StrictCompatEnabled();
    const bool use_extended_dynamic_state =
        instance.IsExtendedDynamicStateSupported() &&
        (!pi5_strict_compat || !IsV3dvEdsDisabled());

    const bool is_dirty = scheduler.IsStateDirty(StateFlags::Pipeline);
    const bool pipeline_dirty = (current_pipeline != pipeline) || is_dirty;

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceBool("state_dirty", is_dirty);
        AppendV115DPipelineCacheTraceBool("pipeline_dirty", pipeline_dirty);
        AppendV115DPipelineCacheTraceLine("v115d_a7z41 scheduler_record_begin");
    }
    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Bool("state_dirty", is_dirty);
        AppendV115DPipelineCacheTraceA7Z54Bool("pipeline_dirty", pipeline_dirty);
        AppendV115DPipelineCacheTraceA7Z54Line("scheduler_record_begin");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan,
                    "TRACE_DRAW v115d_a7z55 scheduler_record_begin state_dirty={} "
                    "pipeline_dirty={}",
                    is_dirty ? 1 : 0, pipeline_dirty ? 1 : 0);
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 scheduler_record_begin");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 scheduler_record_begin");
    }

    scheduler.Record([this, is_dirty, pipeline_dirty, pipeline, use_extended_dynamic_state,
                      a7z41_trace,
                      current_dynamic = current_info.dynamic, dynamic = info.dynamic,
                      descriptor_sets = bound_descriptor_sets, offsets = offsets,
                      current_rasterization = current_info.rasterization,
                      current_depth_stencil = current_info.depth_stencil,
                      rasterization = info.rasterization,
                      depth_stencil = info.depth_stencil](vk::CommandBuffer cmdbuf) {
        if (dynamic.viewport != current_dynamic.viewport || is_dirty) {
            const vk::Viewport vk_viewport = {
                .x = static_cast<f32>(dynamic.viewport.left),
                .y = static_cast<f32>(dynamic.viewport.top),
                .width = static_cast<f32>(dynamic.viewport.GetWidth()),
                .height = static_cast<f32>(dynamic.viewport.GetHeight()),
                .minDepth = 0.f,
                .maxDepth = 1.f,
            };
            cmdbuf.setViewport(0, vk_viewport);
        }

        if (dynamic.scissor != current_dynamic.scissor || is_dirty) {
            const vk::Rect2D scissor = {
                .offset{
                    .x = static_cast<s32>(dynamic.scissor.left),
                    .y = static_cast<s32>(dynamic.scissor.bottom),
                },
                .extent{
                    .width = dynamic.scissor.GetWidth(),
                    .height = dynamic.scissor.GetHeight(),
                },
            };
            cmdbuf.setScissor(0, scissor);
        }

        if (dynamic.stencil_compare_mask != current_dynamic.stencil_compare_mask || is_dirty) {
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         dynamic.stencil_compare_mask);
        }

        if (dynamic.stencil_write_mask != current_dynamic.stencil_write_mask || is_dirty) {
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       dynamic.stencil_write_mask);
        }

        if (dynamic.stencil_reference != current_dynamic.stencil_reference || is_dirty) {
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       dynamic.stencil_reference);
        }

        if (dynamic.blend_color != current_dynamic.blend_color || is_dirty) {
            const Common::Vec4f color = PicaToVK::ColorRGBA8(dynamic.blend_color);
            cmdbuf.setBlendConstants(color.AsArray());
        }

        if (use_extended_dynamic_state) {
            if (rasterization.cull_mode != current_rasterization.cull_mode || is_dirty) {
                cmdbuf.setCullModeEXT(PicaToVK::CullMode(rasterization.cull_mode));
                cmdbuf.setFrontFaceEXT(PicaToVK::FrontFace(rasterization.cull_mode));
            }

            if (depth_stencil.depth_compare_op != current_depth_stencil.depth_compare_op ||
                is_dirty) {
                cmdbuf.setDepthCompareOpEXT(PicaToVK::CompareFunc(depth_stencil.depth_compare_op));
            }

            if (depth_stencil.depth_test_enable != current_depth_stencil.depth_test_enable ||
                is_dirty) {
                cmdbuf.setDepthTestEnableEXT(depth_stencil.depth_test_enable);
            }

            if (depth_stencil.depth_write_enable != current_depth_stencil.depth_write_enable ||
                is_dirty) {
                cmdbuf.setDepthWriteEnableEXT(depth_stencil.depth_write_enable);
            }

            if (rasterization.topology != current_rasterization.topology || is_dirty) {
                cmdbuf.setPrimitiveTopologyEXT(PicaToVK::PrimitiveTopology(rasterization.topology));
            }

            if (depth_stencil.stencil_test_enable != current_depth_stencil.stencil_test_enable ||
                is_dirty) {
                cmdbuf.setStencilTestEnableEXT(depth_stencil.stencil_test_enable);
            }

            if (depth_stencil.stencil_fail_op != current_depth_stencil.stencil_fail_op ||
                depth_stencil.stencil_pass_op != current_depth_stencil.stencil_pass_op ||
                depth_stencil.stencil_depth_fail_op !=
                    current_depth_stencil.stencil_depth_fail_op ||
                depth_stencil.stencil_compare_op != current_depth_stencil.stencil_compare_op ||
                is_dirty) {
                cmdbuf.setStencilOpEXT(vk::StencilFaceFlagBits::eFrontAndBack,
                                       PicaToVK::StencilOp(depth_stencil.stencil_fail_op),
                                       PicaToVK::StencilOp(depth_stencil.stencil_pass_op),
                                       PicaToVK::StencilOp(depth_stencil.stencil_depth_fail_op),
                                       PicaToVK::CompareFunc(depth_stencil.stencil_compare_op));
            }
        }

        if (pipeline_dirty) {
            if (a7z41_trace) {
                AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_pipeline_dirty_enter");
                AppendV115DPipelineCacheTraceBool("record_pipeline_done_before_wait",
                                                  pipeline->IsDone());
            }

            if (!pipeline->IsDone()) {
                if (a7z41_trace) {
                    AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_wait_done_begin");
                }
                pipeline->WaitDone();
                if (a7z41_trace) {
                    AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_wait_done_end");
                }
            }

            if (a7z41_trace) {
                AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_bind_pipeline_begin");
            }
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
            if (a7z41_trace) {
                AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_bind_pipeline_end");
            }
        }

        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_bind_descriptor_sets_begin");
        }
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0,
                                  descriptor_sets, offsets);
        if (a7z41_trace) {
            AppendV115DPipelineCacheTraceLine("v115d_a7z41 record_bind_descriptor_sets_end");
        }
    });

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceLine("v115d_a7z41 scheduler_record_end");
    }
    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Line("scheduler_record_end");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 scheduler_record_end");
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 scheduler_record_end");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 scheduler_record_end");
    }

    current_info = info;
    current_pipeline = pipeline;
    scheduler.MarkStateNonDirty(StateFlags::Pipeline | StateFlags::DescriptorSets);

    if (a7z41_trace) {
        AppendV115DPipelineCacheTraceLine("v115d_a7z41 bind_pipeline_return_true");
    }
    if (a7z54_minimal_safe) {
        AppendV115DPipelineCacheTraceA7Z54Line("bind_pipeline_return_true");
    }
    if (a7z55_mainlog_only) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z55 bind_pipeline_return_true");
    }
    if (a7z56_plain_markers) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z56 bind_pipeline_return_true");
    }
    if (a7z58_split_try_emplace) {
        LOG_WARNING(Render_Vulkan, "TRACE_DRAW v115d_a7z58 bind_pipeline_return_true");
    }

    return true;
}

bool PipelineCache::UseProgrammableVertexShader(const Pica::RegsInternal& regs,
                                                Pica::ShaderSetup& setup,
                                                const VertexLayout& layout, bool accurate_mul) {
    const bool use_geometry_shader = instance.UseGeometryShaders() && !regs.lighting.disable &&
                                     !instance.IsFragmentShaderBarycentricSupported();
    PicaVSConfig config{regs, setup, instance.IsShaderClipDistanceSupported(), use_geometry_shader,
                        accurate_mul};

    for (u32 i = 0; i < layout.attribute_count; i++) {
        const VertexAttribute& attr = layout.attributes[i];
        const FormatTraits& traits = instance.GetTraits(attr.type, attr.size);
        const u32 location = attr.location.Value();
        AttribLoadFlags& flags = config.state.load_flags[location];

        if (traits.needs_conversion) {
            flags = MakeAttribLoadFlag(attr.type);
        }
        if (traits.needs_emulation) {
            flags |= AttribLoadFlags::ZeroW;
        }
    }

    // v373 -- VOIE HYBRIDE POUR LES VERTEX SHADERS (equivalent de l'ubershader de Dolphin ou de
    // l'interpreteur de shaders de RPCS3).
    //
    // Mesure S2b/S3a (Sonic) : un gel de ~1 s = six conversions GLSL -> SPIR-V de vertex shaders
    // PICA (~27 000 mots, ~170 ms chacune) faites en serie sur EmuThread. v372 les met en cache,
    // ce qui regle le jeu A CHAUD ; a froid, la conversion doit encore etre faite une fois.
    //
    // Au lieu de geler, on la fait en arriere-plan : tant que le SPIR-V n'est pas pret, on
    // renvoie false. RasterizerVulkan::AccelerateDrawBatch echoue alors a SetupVertexShader() et
    // PicaCore::DrawArrays bascule sur LoadVertices() : le vertex shader PICA est execute par le
    // CPU (JIT) et l'objet est dessine normalement par le chemin logiciel -- le meme que celui de
    // tous les draws avec geometry shader. Ni gel, ni objet manquant. Des que le SPIR-V est
    // pret, le draw suivant reprend le chemin Vulkan.
    //   - A chaud (SPIR-V deja en cache v372) : pris tout de suite, aucun passage par le CPU.
    //   - Conversions sur un pool dedie (BORKED3DS_V3DV_VS_TRANSLATE_THREADS, defaut 2, max 4),
    //     separe des workers de pipelines qu'un pipeline poison peut occuper 30 s.
    //   - BORKED3DS_V3DV_DISABLE_HYBRID_VS=1 : retour a la conversion synchrone (comparaison).
    //   - Journal V373_VS_HYBRIDE : lance / pret (avec le nombre de draws passes par le CPU).
    // Etat partage entre instances de PipelineCache (un savestate recree le renderer) : la cle
    // est config.Hash(), qui designe le meme programme PICA et donc le meme SPIR-V.
    struct V373Pending {
        std::atomic<bool> done{false};
        std::vector<u32> code;
        std::chrono::steady_clock::time_point t_submit{};
        u64 cpu_draws{0};
    };
    static const bool v373_disabled = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_DISABLE_HYBRID_VS");
        return v != nullptr && v[0] != '\0';
    }();
    static std::unordered_map<u64, std::shared_ptr<V373Pending>> v373_pending;
    static u64 v373_launched = 0;
    static u64 v373_ready = 0;
    static u64 v373_cpu_draws_total = 0;

    auto it = programmable_vertex_map.find(config);
    const bool new_config = it == programmable_vertex_map.end();
    if (new_config) {
        const bool use_spirv = Settings::values.spirv_shader_gen.GetValue();
        const vk::Device device = instance.GetDevice();

        std::vector<u32> code;
        const u64 v373_key = config.Hash();

        if (use_spirv && false) {
            // TODO: Generate vertex shader SPIRV from the given VS program
            // code = SPIRV::GenerateVertexShader(setup, config, profile);
        } else if (const auto pit = v373_pending.find(v373_key); pit != v373_pending.end()) {
            V373Pending& p = *pit->second;
            if (!p.done.load(std::memory_order_acquire)) {
                ++p.cpu_draws;
                ++v373_cpu_draws_total;
                return false; // conversion en cours : ce draw passe par le chemin CPU
            }
            code = std::move(p.code);
            ++v373_ready;
            LOG_WARNING(Render_Vulkan,
                        "V373_VS_HYBRIDE pret cle={:016x} mots={} delai_ms={:.1f} "
                        "draws_cpu={} | lances={} prets={} draws_cpu_total={}",
                        v373_key, code.size(),
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - p.t_submit)
                            .count(),
                        p.cpu_draws, v373_launched, v373_ready, v373_cpu_draws_total);
            v373_pending.erase(pit);
            if (code.empty()) {
                LOG_ERROR(Render_Vulkan, "Failed to translate programmable vertex shader");
                programmable_vertex_map.emplace(config, nullptr);
                return false;
            }
        } else {
            const std::string program = GLSL::GenerateVertexShader(setup, config, true);
            if (program.empty()) {
                LOG_ERROR(Render_Vulkan, "Failed to retrieve programmable vertex shader");
                programmable_vertex_map.emplace(config, nullptr);
                return false;
            }
            if (v373_disabled) {
                code = CompileGLSLtoSPIRV(program, vk::ShaderStageFlagBits::eVertex, device);
            } else if (!TryGetCachedSPIRV(program, vk::ShaderStageFlagBits::eVertex, code)) {
                static Common::ThreadWorker v373_workers(
                    GetEnvU32Limited("BORKED3DS_V3DV_VS_TRANSLATE_THREADS", 2, 4),
                    "VS translate");
                auto pending = std::make_shared<V373Pending>();
                pending->t_submit = std::chrono::steady_clock::now();
                pending->cpu_draws = 1;
                ++v373_cpu_draws_total;
                ++v373_launched;
                v373_pending.emplace(v373_key, pending);
                v373_workers.QueueWork([pending, program, device] {
                    pending->code =
                        CompileGLSLtoSPIRV(program, vk::ShaderStageFlagBits::eVertex, device);
                    pending->done.store(true, std::memory_order_release);
                });
                LOG_WARNING(Render_Vulkan,
                            "V373_VS_HYBRIDE lance cle={:016x} glsl_octets={} | lances={} "
                            "prets={} draws_cpu_total={}",
                            v373_key, program.size(), v373_launched, v373_ready,
                            v373_cpu_draws_total);
                return false; // ce draw passe par le chemin CPU
            }
        }

        if (code.empty()) {
            LOG_ERROR(Render_Vulkan, "Failed to translate programmable vertex shader");
            programmable_vertex_map.emplace(config, nullptr);
            return false;
        }

        const u64 code_hash = Common::ComputeHash64(std::as_bytes(std::span(code)));

        const auto [iter, new_program] = programmable_vertex_cache.try_emplace(code_hash, instance);
        auto& shader = iter->second;

        if (new_program) {
            shader.program = std::move(code);
            // V384 (mesure A) : cle complete, et famille = cle sans les formats de sommets.
            shader.v384_cle = config.Hash();
            {
                // memcpy : copie aussi les octets de bourrage (le hachage lit la structure
                // brute), pour que la famille soit stable d'un lancement a l'autre.
                Pica::Shader::Generator::PicaVSConfigState famille;
                std::memcpy(&famille, &config.state, sizeof(famille));
                famille.load_flags.fill(Pica::Shader::Generator::AttribLoadFlags{});
                shader.v384_famille = Common::ComputeStructHash64(famille);
            }
            workers.QueueWork([device, &shader] {
                shader.module = CompileSPV(shader.program, device);
                shader.MarkDone();
            });
        }

        it = programmable_vertex_map.emplace(config, &shader).first;
    }

    Shader* const shader{it->second};
    if (!shader) {
        LOG_ERROR(Render_Vulkan, "Failed to retrieve programmable vertex shader");
        return false;
    }

    current_shaders[ProgramType::VS] = shader;
    shader_hashes[ProgramType::VS] = config.Hash();

    return true;
}

void PipelineCache::UseTrivialVertexShader() {
    current_shaders[ProgramType::VS] = &trivial_vertex_shader;
    shader_hashes[ProgramType::VS] = 0;
}

bool PipelineCache::UseFixedGeometryShader(const Pica::RegsInternal& regs) {
    if (!instance.UseGeometryShaders()) {
        UseTrivialGeometryShader();
        return true;
    }

    const PicaFixedGSConfig gs_config{regs, instance.IsShaderClipDistanceSupported()};
    auto [it, new_shader] = fixed_geometry_shaders.try_emplace(gs_config, instance);
    auto& shader = it->second;

    if (new_shader) {
        shader.v384_cle = gs_config.Hash(); // V384 (mesure A)
        workers.QueueWork([gs_config, device = instance.GetDevice(), &shader]() {
            const auto code = GLSL::GenerateFixedGeometryShader(gs_config, true);
            shader.module = Compile(code, vk::ShaderStageFlagBits::eGeometry, device);
            shader.MarkDone();
        });
    }

    current_shaders[ProgramType::GS] = &shader;
    shader_hashes[ProgramType::GS] = gs_config.Hash();

    return true;
}

void PipelineCache::UseTrivialGeometryShader() {
    current_shaders[ProgramType::GS] = nullptr;
    shader_hashes[ProgramType::GS] = 0;
}

void PipelineCache::UseFragmentShader(const Pica::RegsInternal& regs,
                                      const Pica::Shader::UserConfig& user) {
    const FSConfig fs_config{regs, user, profile};
    const auto [it, new_shader] = fragment_shaders.try_emplace(fs_config, instance);
    auto& shader = it->second;

    if (new_shader) {
        shader.v384_cle = fs_config.Hash(); // V384 (mesure A)
        workers.QueueWork([fs_config, this, &shader]() {
            const bool use_spirv = Settings::values.spirv_shader_gen.GetValue();
            const bool is_v3dv_driver = instance.GetDriverID() == vk::DriverId::eMesaV3Dv ||
                                        instance.GetDriverID() ==
                                            vk::DriverId::eBroadcomProprietary;

            const bool can_use_spirv_fs =
                use_spirv && !is_v3dv_driver && !fs_config.UsesShadowPipeline() &&
                instance.IsShaderStencilExportSupported();

            if (can_use_spirv_fs) {
                const std::vector<u32> code = SPIRV::GenerateFragmentShader(fs_config, profile);
                shader.module = CompileSPV(code, instance.GetDevice());
            } else {
                const std::string code = GLSL::GenerateFragmentShader(fs_config, profile);
                shader.module =
                    Compile(code, vk::ShaderStageFlagBits::eFragment, instance.GetDevice());
            }
            shader.MarkDone();
        });
    }

    current_shaders[ProgramType::FS] = &shader;
    shader_hashes[ProgramType::FS] = fs_config.Hash();
}

bool PipelineCache::IsCacheValid(std::span<const u8> data) const {
    if (data.size() < sizeof(vk::PipelineCacheHeaderVersionOne)) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header");
        return false;
    }

    vk::PipelineCacheHeaderVersionOne header;
    std::memcpy(&header, data.data(), sizeof(header));
    if (header.headerSize < sizeof(header)) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header length");
        return false;
    }

    if (header.headerVersion != vk::PipelineCacheHeaderVersion::eOne) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header version");
        return false;
    }

    if (u32 vendor_id = instance.GetVendorID(); header.vendorID != vendor_id) {
        LOG_ERROR(
            Render_Vulkan,
            "Pipeline cache failed validation: Incorrect vendor ID (file: {:#X}, device: {:#X})",
            header.vendorID, vendor_id);
        return false;
    }

    if (u32 device_id = instance.GetDeviceID(); header.deviceID != device_id) {
        LOG_ERROR(
            Render_Vulkan,
            "Pipeline cache failed validation: Incorrect device ID (file: {:#X}, device: {:#X})",
            header.deviceID, device_id);
        return false;
    }

    if (header.pipelineCacheUUID != instance.GetPipelineCacheUUID()) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Incorrect UUID");
        return false;
    }

    return true;
}

bool PipelineCache::EnsureDirectories() const {
    const auto create_dir = [](const std::string& dir) {
        if (!FileUtil::CreateDir(dir)) {
            LOG_ERROR(Render_Vulkan, "Failed to create directory={}", dir);
            return false;
        }

        return true;
    };

    return create_dir(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir)) &&
           create_dir(GetPipelineCacheDir());
}

std::string PipelineCache::GetPipelineCacheDir() const {
    return FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "vulkan" + DIR_SEP;
}

} // namespace Vulkan
