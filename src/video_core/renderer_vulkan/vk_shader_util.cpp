// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <utility>
#include <vector>
#include <string_view>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <string>
#include <pthread.h>
#include <unistd.h>
#include <SPIRV/GlslangToSpv.h>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv-tools/optimizer.hpp>

#include <unordered_map>

#include "common/assert.h"
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/hash.h"
#include "common/literals.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using namespace Common::Literals;

namespace {
constexpr TBuiltInResource DefaultTBuiltInResource = {
    .maxLights = 32,
    .maxClipPlanes = 6,
    .maxTextureUnits = 32,
    .maxTextureCoords = 32,
    .maxVertexAttribs = 64,
    .maxVertexUniformComponents = 4096,
    .maxVaryingFloats = 64,
    .maxVertexTextureImageUnits = 32,
    .maxCombinedTextureImageUnits = 80,
    .maxTextureImageUnits = 32,
    .maxFragmentUniformComponents = 4096,
    .maxDrawBuffers = 32,
    .maxVertexUniformVectors = 128,
    .maxVaryingVectors = 8,
    .maxFragmentUniformVectors = 16,
    .maxVertexOutputVectors = 16,
    .maxFragmentInputVectors = 15,
    .minProgramTexelOffset = -8,
    .maxProgramTexelOffset = 7,
    .maxClipDistances = 8,
    .maxComputeWorkGroupCountX = 65535,
    .maxComputeWorkGroupCountY = 65535,
    .maxComputeWorkGroupCountZ = 65535,
    .maxComputeWorkGroupSizeX = 1024,
    .maxComputeWorkGroupSizeY = 1024,
    .maxComputeWorkGroupSizeZ = 64,
    .maxComputeUniformComponents = 1024,
    .maxComputeTextureImageUnits = 16,
    .maxComputeImageUniforms = 8,
    .maxComputeAtomicCounters = 8,
    .maxComputeAtomicCounterBuffers = 1,
    .maxVaryingComponents = 60,
    .maxVertexOutputComponents = 64,
    .maxGeometryInputComponents = 64,
    .maxGeometryOutputComponents = 128,
    .maxFragmentInputComponents = 128,
    .maxImageUnits = 8,
    .maxCombinedImageUnitsAndFragmentOutputs = 8,
    .maxCombinedShaderOutputResources = 8,
    .maxImageSamples = 0,
    .maxVertexImageUniforms = 0,
    .maxTessControlImageUniforms = 0,
    .maxTessEvaluationImageUniforms = 0,
    .maxGeometryImageUniforms = 0,
    .maxFragmentImageUniforms = 8,
    .maxCombinedImageUniforms = 8,
    .maxGeometryTextureImageUnits = 16,
    .maxGeometryOutputVertices = 256,
    .maxGeometryTotalOutputComponents = 1024,
    .maxGeometryUniformComponents = 1024,
    .maxGeometryVaryingComponents = 64,
    .maxTessControlInputComponents = 128,
    .maxTessControlOutputComponents = 128,
    .maxTessControlTextureImageUnits = 16,
    .maxTessControlUniformComponents = 1024,
    .maxTessControlTotalOutputComponents = 4096,
    .maxTessEvaluationInputComponents = 128,
    .maxTessEvaluationOutputComponents = 128,
    .maxTessEvaluationTextureImageUnits = 16,
    .maxTessEvaluationUniformComponents = 1024,
    .maxTessPatchComponents = 120,
    .maxPatchVertices = 32,
    .maxTessGenLevel = 64,
    .maxViewports = 16,
    .maxVertexAtomicCounters = 0,
    .maxTessControlAtomicCounters = 0,
    .maxTessEvaluationAtomicCounters = 0,
    .maxGeometryAtomicCounters = 0,
    .maxFragmentAtomicCounters = 8,
    .maxCombinedAtomicCounters = 8,
    .maxAtomicCounterBindings = 1,
    .maxVertexAtomicCounterBuffers = 0,
    .maxTessControlAtomicCounterBuffers = 0,
    .maxTessEvaluationAtomicCounterBuffers = 0,
    .maxGeometryAtomicCounterBuffers = 0,
    .maxFragmentAtomicCounterBuffers = 1,
    .maxCombinedAtomicCounterBuffers = 1,
    .maxAtomicCounterBufferSize = 16384,
    .maxTransformFeedbackBuffers = 4,
    .maxTransformFeedbackInterleavedComponents = 64,
    .maxCullDistances = 8,
    .maxCombinedClipAndCullDistances = 8,
    .maxSamples = 4,
    .maxMeshOutputVerticesNV = 256,
    .maxMeshOutputPrimitivesNV = 512,
    .maxMeshWorkGroupSizeX_NV = 32,
    .maxMeshWorkGroupSizeY_NV = 1,
    .maxMeshWorkGroupSizeZ_NV = 1,
    .maxTaskWorkGroupSizeX_NV = 32,
    .maxTaskWorkGroupSizeY_NV = 1,
    .maxTaskWorkGroupSizeZ_NV = 1,
    .maxMeshViewCountNV = 4,
    .maxDualSourceDrawBuffersEXT = 1,
    .limits =
        TLimits{
            .nonInductiveForLoops = 1,
            .whileLoops = 1,
            .doWhileLoops = 1,
            .generalUniformIndexing = 1,
            .generalAttributeMatrixVectorIndexing = 1,
            .generalVaryingIndexing = 1,
            .generalSamplerIndexing = 1,
            .generalVariableIndexing = 1,
            .generalConstantMatrixVectorIndexing = 1,
        },
};

EShLanguage ToEshShaderStage(vk::ShaderStageFlagBits stage) {
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex:
        return EShLanguage::EShLangVertex;
    case vk::ShaderStageFlagBits::eGeometry:
        return EShLanguage::EShLangGeometry;
    case vk::ShaderStageFlagBits::eFragment:
        return EShLanguage::EShLangFragment;
    case vk::ShaderStageFlagBits::eCompute:
        return EShLanguage::EShLangCompute;
    default:
        UNREACHABLE_MSG("Unkown shader stage {}", stage);
    }
    return EShLanguage::EShLangVertex;
}

bool SpirvContainsExtensionString(std::span<const u32> code, std::string_view needle) {
    if (code.empty() || needle.empty()) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const char*>(code.data());
    const std::size_t size_bytes = code.size() * sizeof(u32);
    const std::string_view haystack{bytes, size_bytes};
    return haystack.find(needle) != std::string_view::npos;
}

u64 HashSpirvWords(std::span<const u32> code) {
    constexpr u64 kOffset = 1469598103934665603ull;
    constexpr u64 kPrime = 1099511628211ull;
    u64 hash = kOffset;
    for (u32 word : code) {
        hash ^= static_cast<u64>(word);
        hash *= kPrime;
    }
    return hash;
}

const char* ShaderStageName(vk::ShaderStageFlagBits stage) {
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex:
        return "vertex";
    case vk::ShaderStageFlagBits::eGeometry:
        return "geometry";
    case vk::ShaderStageFlagBits::eFragment:
        return "fragment";
    case vk::ShaderStageFlagBits::eCompute:
        return "compute";
    default:
        return "unknown";
    }
}

void LogSpirvTrace(std::span<const u32> code, std::string_view origin,
                   vk::ShaderStageFlagBits stage) {
    LOG_INFO(Render_Vulkan,
             "SPIR-V trace: origin='{}', stage={}, words={}, bytes={}, hash=0x{:016x}",
             origin, ShaderStageName(stage), code.size(), code.size_bytes(), HashSpirvWords(code));
    if (SpirvContainsExtensionString(code, "SPV_EXT_shader_stencil_export")) {
        LOG_ERROR(Render_Vulkan,
                  "SPIR-V trace: origin='{}', stage={} contains SPV_EXT_shader_stencil_export",
                  origin, ShaderStageName(stage));
    }
}

void DumpProblematicShaderSource(std::string_view origin, vk::ShaderStageFlagBits stage,
                                 std::string_view source) {
    LOG_ERROR(Render_Vulkan,
              "Problematic shader source dump: origin='{}', stage={}, bytes={}",
              origin, ShaderStageName(stage), source.size());
    LOG_ERROR(Render_Vulkan, "Problematic shader source BEGIN\n{}\nProblematic shader source END",
              source);
}

bool InitializeCompiler() {
    static bool glslang_initialized = false;

    if (glslang_initialized) {
        return true;
    }

    if (!glslang::InitializeProcess()) {
        LOG_CRITICAL(Render_Vulkan, "Failed to initialize glslang shader compiler");
        return false;
    }

    std::atexit([]() { glslang::FinalizeProcess(); });

    glslang_initialized = true;
    return true;
}
// BORKED3DS_V3DV_A7Z9_DUMP_COMPILESPV -- ecrit dans /tmp le SPIR-V FINAL de CHAQUE
// module compile. CompileSPV est l'entonnoir UNIQUE (VS l.971, FS SPIR-V direct l.1044,
// FS issu du GLSL via Compile l.385) : ce dump capture donc le FS meme quand
// shader.program n'est PAS conserve -- contrairement au dump _s0 du pipeline qui le
// ratait. Un .spv par hash (dedup), pour desassemblage hors ligne (spirv-dis) et
// recherche RelaxedPrecision / OpTypeFloat 16 / OpFConvert. Inerte hors variable,
// getenv lu une seule fois.
void DumpCompileSpvIfEnabled(std::span<const u32> code) {
    static const bool enabled =
        std::getenv("BORKED3DS_V3DV_A7Z9_DUMP_COMPILESPV") != nullptr;
    if (!enabled || code.empty()) {
        return;
    }
    const u64 hash = HashSpirvWords(code);
    {
        static std::mutex dumped_mutex;
        static std::unordered_set<u64> dumped_hashes;
        std::scoped_lock lock{dumped_mutex};
        if (!dumped_hashes.insert(hash).second) {
            return; // deja dumpe ce module
        }
    }
    char path[64];
    std::snprintf(path, sizeof(path), "/tmp/borked3ds_compilespv_%016llx.spv",
                  static_cast<unsigned long long>(hash));
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if (out) {
        out.write(reinterpret_cast<const char*>(code.data()),
                  static_cast<std::streamsize>(code.size_bytes()));
    }
}

} // Anonymous namespace

/**
 * @brief Optimizes SPIR-V code using spirv-opt from SPIRV-Tools
 * @param code The string containing SPIR-V code
 */
std::vector<u32> OptimizeSPIRV(std::vector<u32> code) {

    std::vector<u32> result = code;
    std::vector<u32> spirv = code;
    spv_target_env vulkanEnv = SPV_ENV_UNIVERSAL_1_0;

    const u32 available_version = vk::enumerateInstanceVersion();

    if (VK_API_VERSION_MAJOR(available_version) == 1 &&
        VK_API_VERSION_MINOR(available_version) == 1) {
        vulkanEnv = SPV_ENV_VULKAN_1_1;
        LOG_DEBUG(Render_Vulkan, "SPIRV-OPT Vulkan 1.1");
    } else if (VK_API_VERSION_MAJOR(available_version) == 1 &&
               VK_API_VERSION_MINOR(available_version) == 2) {
        vulkanEnv = SPV_ENV_VULKAN_1_2;
        LOG_DEBUG(Render_Vulkan, "SPIRV-OPT Vulkan 1.2");
    } else if (VK_API_VERSION_MAJOR(available_version) == 1 &&
               VK_API_VERSION_MINOR(available_version) >= 3) {
        vulkanEnv = SPV_ENV_VULKAN_1_3;
        LOG_DEBUG(Render_Vulkan, "SPIRV-OPT Vulkan 1.3");
    }

    spvtools::Optimizer spv_opt(vulkanEnv);

    spv_opt.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&,
                                  const char* m) { LOG_ERROR(HW_GPU, "spirv-opt: {}", m); });

    // SPIR-V Legalization
    if (Settings::values.spirv_output_legalization.GetValue()) {
        spv_opt.RegisterLegalizationPasses();
    }

    // Optimize SPIR-V for Size or Performance. Equivalent to passing -Os or -O to spirv-opt
    // respectively.
    if (Settings::values.optimize_spirv_output.GetValue() == Settings::OptimizeSpirv::Size) {
        spv_opt.RegisterSizePasses();
    } else if (Settings::values.optimize_spirv_output.GetValue() ==
               Settings::OptimizeSpirv::Performance) {
        spv_opt.RegisterPerformancePasses();
    }

    spvtools::OptimizerOptions opt_options;

    // SPIR-V Validation
    if (Settings::values.spirv_output_validation.GetValue()) {
        opt_options.set_run_validator(true);
    } else {
        opt_options.set_run_validator(false);
    }

    if (!spv_opt.Run(spirv.data(), spirv.size(), &result, opt_options)) {
        LOG_ERROR(HW_GPU,
                  "Failed to optimize SPIRV shader output, continuing without optimization");
        result = std::move(spirv);
    }

    return result;
}

// ---------------------------------------------------------------------------------------------
// v363 -- SHADER_INFLIGHT_DUMP. Sonde de MESURE, inerte si la variable d'environnement est absente.
//
//   BORKED3DS_V3DV_SHADER_INFLIGHT_DUMP=<repertoire existant>
//
// Avant chaque conversion GLSL -> SPIR-V, le source exact est ecrit dans
// <rep>/inflight_tid<N>.glsl (ecrase par la conversion suivante du meme thread). Apres le retour de
// GlslangToSpv, une ligne est ajoutee a <rep>/done.csv :
//   seq,epoch_s,tid,thread,stage,glsl_bytes,spirv_words,ms
// Deux usages :
//   1. Gel majeur : si le processus meurt PENDANT une conversion (emballement a 12 Gio), le fichier
//      inflight dont le seq manque dans done.csv contient le shader coupable, pret a etre rejoue
//      hors ligne (glslangValidator + spirv-opt). Les ecritures sont dans le cache de pages du
//      noyau : elles survivent a la mort du processus, pas besoin de fsync.
//   2. Petits gels : la conversion des vertex shaders PICA est SYNCHRONE sur le thread appelant
//      (vk_pipeline_cache.cpp, UseProgrammableVertexShader -> CompileGLSLtoSPIRV), et
//      TRACE_PIPELINE_BUILD ne mesure que createGraphicsPipeline. Le temps passe ici, sur
//      EmuThread, n'a jamais ete mesure. La colonne ms le donne, par shader et par thread.
//      Chaque ligne de done.csv correspond 1:1, dans l'ordre, a une trace
//      'CompileGLSLtoSPIRV/raw' du log : c'est l'ancrage temporel avec le census par frame.
// ---------------------------------------------------------------------------------------------
namespace {

const char* InflightDumpDir() {
    static const char* const dir = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_SHADER_INFLIGHT_DUMP");
        return (v != nullptr && v[0] != '\0') ? v : static_cast<const char*>(nullptr);
    }();
    return dir;
}

std::string InflightThreadName() {
    char name[32] = {};
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "?";
    }
    std::string out(name);
    for (char& c : out) {
        if (c == ',' || c == '\n') {
            c = '_';
        }
    }
    return out;
}

std::atomic<u64> g_inflight_seq{0};
std::mutex g_inflight_csv_mutex;

} // namespace

/**
 * @brief Compiles GLSL into SPIRV
 * @param code The string containing GLSL code.
 * @param stage The pipeline stage the shader will be used in.
 * @param device The vulkan device handle.
 */
// ---------------------------------------------------------------------------------------------
// v372 -- CACHE DES CONVERSIONS GLSL -> SPIR-V (memoire + disque).
//
// Mesure S2b (Sonic, a chaud, v371) : image de 1 087 ms = SIX conversions de vertex shaders PICA
// (26 919 a 27 944 mots) faites l'une apres l'autre sur EmuThread, ~160-180 ms chacune, alors
// que les pipelines se construisaient en 0,1 ms (cache V3DV chaude, v370). Runs Z5a/Z5b (Kid
// Icarus) : 24-30 grosses conversions par session, a chaud comme a froid, ~la moitie des images
// > 100 ms. La cache pipeline Vulkan ne couvre PAS cette etape : chaque session reconvertit tout.
//
// La conversion est deterministe : meme source GLSL + meme preambule + meme stage + meme reglage
// optimize_spirv_output + meme binaire => meme SPIR-V. On garde donc le resultat :
//   - en memoire (doublons de la session, 59 % mesures en v365) ;
//   - sur disque, <ShaderDir>/vulkan/spirv/<stage>_<hash>.spv. Ce repertoire est sous
//     shaders/vulkan, que le scriptmodule purge a chaque build : un nouveau binaire (autre
//     glslang, autre generateur) repart donc toujours d'une cache vide.
// En-tete verifie a la lecture (magie, version, stage, reglage, tailles, deux hashes, mot
// magique SPIR-V) : un fichier douteux est ignore et reconverti. Ecriture atomique (fichier
// temporaire + rename). Actif si use_disk_shader_cache=true ; desactivable sans rebuild par
// BORKED3DS_V3DV_DISABLE_SPIRV_DISK_CACHE=1 (la memoisation en memoire reste active).
// Journal : V372_SPIRV_CACHE (chaque vertex shader, puis bilan tous les 64 evenements).
// ---------------------------------------------------------------------------------------------
namespace {

constexpr u32 V372_MAGIC = 0x56535342; // "BSSV"
constexpr u32 V372_VERSION = 1;
constexpr u32 V372_SPIRV_MAGIC = 0x07230203;
constexpr std::size_t V372_MEM_MAX_ENTRIES = 4096;

struct V372Header {
    u32 magic;
    u32 version;
    u32 stage;
    u32 optimize;
    u32 code_bytes;
    u32 preamble_bytes;
    u64 code_hash;
    u64 preamble_hash;
    u32 words;
    u32 reserved;
};
static_assert(sizeof(V372Header) == 48);

struct V372Key {
    u32 stage;
    u32 optimize;
    u32 code_bytes;
    u32 preamble_bytes;
    u64 code_hash;
    u64 preamble_hash;
    bool operator==(const V372Key&) const = default;
};

struct V372KeyHash {
    std::size_t operator()(const V372Key& k) const noexcept {
        return static_cast<std::size_t>(Common::HashCombine(
            Common::HashCombine(k.code_hash, k.preamble_hash),
            (static_cast<u64>(k.stage) << 40) ^ (static_cast<u64>(k.optimize) << 32) ^
                k.code_bytes));
    }
};

std::mutex g_v372_mutex;
std::unordered_map<V372Key, std::vector<u32>, V372KeyHash> g_v372_mem;
std::atomic<u64> g_v372_hit_mem{0};
std::atomic<u64> g_v372_hit_disk{0};
std::atomic<u64> g_v372_miss{0};
std::atomic<u64> g_v372_write_fail{0};
std::atomic<u64> g_v372_events{0};
std::atomic<u64> g_v372_miss_us{0};

bool V372DiskEnabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_DISABLE_SPIRV_DISK_CACHE");
        return v != nullptr && v[0] != '\0';
    }();
    return !disabled && Settings::values.use_disk_shader_cache.GetValue();
}

const std::string& V372Dir() {
    static const std::string dir = [] {
        std::string d = FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + "vulkan" +
                        DIR_SEP + "spirv" + DIR_SEP;
        FileUtil::CreateFullPath(d);
        return d;
    }();
    return dir;
}

std::string V372Path(const V372Key& k) {
    return fmt::format("{}{}_{:016x}_{:016x}_{}.spv", V372Dir(), ShaderStageName(
                           static_cast<vk::ShaderStageFlagBits>(k.stage)),
                       k.code_hash, k.preamble_hash, k.optimize);
}

bool V372ReadDisk(const V372Key& k, std::vector<u32>& out) {
    const std::string path = V372Path(k);
    FileUtil::IOFile f(path, "rb");
    if (!f.IsOpen()) {
        return false;
    }
    V372Header h{};
    if (f.ReadBytes(&h, sizeof(h)) != sizeof(h)) {
        return false;
    }
    if (h.magic != V372_MAGIC || h.version != V372_VERSION || h.stage != k.stage ||
        h.optimize != k.optimize || h.code_bytes != k.code_bytes ||
        h.preamble_bytes != k.preamble_bytes || h.code_hash != k.code_hash ||
        h.preamble_hash != k.preamble_hash || h.words == 0 || h.words > (16u << 20)) {
        return false;
    }
    if (f.GetSize() != sizeof(h) + static_cast<u64>(h.words) * sizeof(u32)) {
        return false;
    }
    std::vector<u32> words(h.words);
    if (f.ReadBytes(words.data(), words.size() * sizeof(u32)) != words.size() * sizeof(u32)) {
        return false;
    }
    if (words[0] != V372_SPIRV_MAGIC) {
        return false;
    }
    out = std::move(words);
    return true;
}

void V372WriteDisk(const V372Key& k, const std::vector<u32>& words) {
    const std::string path = V372Path(k);
    const std::string tmp = fmt::format("{}.tmp{}", path, static_cast<long>(gettid()));
    bool ok = false;
    {
        FileUtil::IOFile f(tmp, "wb");
        if (f.IsOpen()) {
            const V372Header h{V372_MAGIC,         V372_VERSION,     k.stage,
                               k.optimize,         k.code_bytes,     k.preamble_bytes,
                               k.code_hash,        k.preamble_hash,  static_cast<u32>(words.size()),
                               0};
            ok = f.WriteBytes(&h, sizeof(h)) == sizeof(h) &&
                 f.WriteBytes(words.data(), words.size() * sizeof(u32)) ==
                     words.size() * sizeof(u32);
        }
    }
    if (ok) {
        ok = FileUtil::Rename(tmp, path);
    }
    if (!ok) {
        FileUtil::Delete(tmp);
        g_v372_write_fail.fetch_add(1, std::memory_order_relaxed);
    }
}

void V372Log(const char* result, vk::ShaderStageFlagBits stage, std::size_t words, double ms) {
    const u64 n = g_v372_events.fetch_add(1, std::memory_order_relaxed) + 1;
    if (stage == vk::ShaderStageFlagBits::eVertex || (n % 64) == 0) {
        LOG_WARNING(Render_Vulkan,
                    "V372_SPIRV_CACHE #{} {} stage={} mots={} ms={:.1f} | memoire={} disque={} "
                    "conversions={} ({:.0f} ms) echecs_ecriture={}",
                    n, result, ShaderStageName(stage), words, ms,
                    g_v372_hit_mem.load(std::memory_order_relaxed),
                    g_v372_hit_disk.load(std::memory_order_relaxed),
                    g_v372_miss.load(std::memory_order_relaxed),
                    static_cast<double>(g_v372_miss_us.load(std::memory_order_relaxed)) / 1000.0,
                    g_v372_write_fail.load(std::memory_order_relaxed));
    }
}

std::vector<u32> CompileGLSLtoSPIRVUncached(std::string_view code, vk::ShaderStageFlagBits stage,
                                            vk::Device device, std::string_view premable);

} // namespace

std::vector<u32> CompileGLSLtoSPIRV(std::string_view code, vk::ShaderStageFlagBits stage,
                                    vk::Device device, std::string_view premable) {
    const auto t0 = std::chrono::steady_clock::now();
    const V372Key key{
        static_cast<u32>(stage),
        static_cast<u32>(Settings::values.optimize_spirv_output.GetValue()),
        static_cast<u32>(code.size()),
        static_cast<u32>(premable.size()),
        Common::ComputeHash64(code.data(), code.size()),
        Common::ComputeHash64(premable.data(), premable.size()),
    };

    {
        std::scoped_lock lock(g_v372_mutex);
        if (const auto it = g_v372_mem.find(key); it != g_v372_mem.end()) {
            std::vector<u32> copy = it->second;
            g_v372_hit_mem.fetch_add(1, std::memory_order_relaxed);
            V372Log("memoire", stage, copy.size(),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              t0)
                        .count());
            LogSpirvTrace(copy, "CompileGLSLtoSPIRV/cache", stage);
            return copy;
        }
    }

    const bool disk = V372DiskEnabled();
    std::vector<u32> result;
    const char* origin = "conversion";
    if (disk && V372ReadDisk(key, result)) {
        g_v372_hit_disk.fetch_add(1, std::memory_order_relaxed);
        origin = "disque";
        LogSpirvTrace(result, "CompileGLSLtoSPIRV/cache", stage);
    } else {
        result = CompileGLSLtoSPIRVUncached(code, stage, device, premable);
        if (result.empty()) {
            return result; // echec : ne rien memoriser
        }
        g_v372_miss.fetch_add(1, std::memory_order_relaxed);
        g_v372_miss_us.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count()),
            std::memory_order_relaxed);
        if (disk) {
            V372WriteDisk(key, result);
        }
    }

    {
        std::scoped_lock lock(g_v372_mutex);
        if (g_v372_mem.size() < V372_MEM_MAX_ENTRIES) {
            g_v372_mem.emplace(key, result);
        }
    }
    V372Log(origin, stage, result.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count());
    return result;
}

namespace {

std::vector<u32> CompileGLSLtoSPIRVUncached(std::string_view code, vk::ShaderStageFlagBits stage,
                                            vk::Device device, std::string_view premable) {
    if (!InitializeCompiler()) {
        return {};
    }

    EProfile profile = ECoreProfile;
    EShMessages messages =
        static_cast<EShMessages>(EShMsgDefault | EShMsgSpvRules | EShMsgVulkanRules);
    EShLanguage lang = ToEshShaderStage(stage);

    const int default_version = 450;
    const char* pass_source_code = code.data();
    int pass_source_code_length = static_cast<int>(code.size());

    auto shader = std::make_unique<glslang::TShader>(lang);
    shader->setEnvTarget(glslang::EShTargetSpv,
                         glslang::EShTargetLanguageVersion::EShTargetSpv_1_3);
    shader->setStringsWithLengths(&pass_source_code, &pass_source_code_length, 1);
    shader->setPreamble(premable.data());

    glslang::TShader::ForbidIncluder includer;
    if (!shader->parse(&DefaultTBuiltInResource, default_version, profile, false, true, messages,
                       includer)) [[unlikely]] {
        LOG_INFO(Render_Vulkan, "Shader Info Log:\n{}\n{}", shader->getInfoLog(),
                 shader->getInfoDebugLog());
        LOG_INFO(Render_Vulkan, "Shader Source:\n{}", code);
        return {};
    }

    // Even though there's only a single shader, we still need to link it to generate SPV
    auto program = std::make_unique<glslang::TProgram>();
    program->addShader(shader.get());
    if (!program->link(messages)) {
        LOG_INFO(Render_Vulkan, "Program Info Log:\n{}\n{}", program->getInfoLog(),
                 program->getInfoDebugLog());
        return {};
    }

    glslang::TIntermediate* intermediate = program->getIntermediate(lang);
    std::vector<u32> out_code;
    spv::SpvBuildLogger logger;
    glslang::SpvOptions options;

    if (Settings::values.optimize_spirv_output.GetValue() == Settings::OptimizeSpirv::Disabled) {
        // Use built-in glslang to enable default optimizations on the generated SPIR-V code
        options.disableOptimizer = false;
        options.validate = false;
        options.optimizeSize = true;
    } else {
        // Use external SPIRV-Tools to perform optimizations
        options.disableOptimizer = true;
        options.validate = false;
        options.optimizeSize = false;
    }

    out_code.reserve(8_KiB);
    // v363 SHADER_INFLIGHT_DUMP -- source ecrit AVANT la conversion (voir en-tete de la sonde).
    const char* const inflight_dir = InflightDumpDir();
    u64 inflight_seq = 0;
    long inflight_tid = 0;
    std::string inflight_thread;
    double inflight_epoch = 0.0;
    std::chrono::steady_clock::time_point inflight_t0{};
    if (inflight_dir != nullptr) {
        inflight_seq = ++g_inflight_seq;
        inflight_tid = static_cast<long>(gettid());
        inflight_thread = InflightThreadName();
        inflight_epoch = std::chrono::duration<double>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        const std::string path = std::string(inflight_dir) + "/inflight_tid" +
                                 std::to_string(inflight_tid) + ".glsl";
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (f) {
            f << "// inflight seq=" << inflight_seq << " tid=" << inflight_tid
              << " thread=" << inflight_thread << " stage=" << ShaderStageName(stage)
              << " code_bytes=" << code.size() << " preamble_bytes=" << premable.size()
              << "\n";
            // Reassemblage fidele : glslang insere le preambule apres la ligne #version.
            const std::string_view src = code;
            if (!premable.empty() && src.starts_with("#version")) {
                const auto nl = src.find('\n');
                if (nl == std::string_view::npos) {
                    f << src << "\n" << premable << "\n";
                } else {
                    f << src.substr(0, nl + 1) << premable << "\n" << src.substr(nl + 1);
                }
            } else {
                if (!src.starts_with("#version")) {
                    f << "#version 450\n";
                }
                f << premable << "\n" << src;
            }
        }
        inflight_t0 = std::chrono::steady_clock::now();
    }

    glslang::GlslangToSpv(*intermediate, out_code, &logger, &options);

    // v363 SHADER_INFLIGHT_DUMP -- conversion terminee : une ligne dans done.csv.
    if (inflight_dir != nullptr) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - inflight_t0)
                              .count();
        std::scoped_lock lock(g_inflight_csv_mutex);
        std::ofstream csv(std::string(inflight_dir) + "/done.csv", std::ios::out | std::ios::app);
        if (csv) {
            csv.setf(std::ios::fixed);
            csv.precision(3);
            csv << inflight_seq << ',' << inflight_epoch << ',' << inflight_tid << ','
                << inflight_thread << ',' << ShaderStageName(stage) << ',' << code.size()
                << ',' << out_code.size() << ',' << ms << '\n';
        }
    }

    const std::string spv_messages = logger.getAllMessages();
    if (!spv_messages.empty()) {
        LOG_INFO(Render_Vulkan, "SPIR-V conversion messages: {}", spv_messages);
    }

    LogSpirvTrace(out_code, "CompileGLSLtoSPIRV/raw", stage);
    if (stage == vk::ShaderStageFlagBits::eFragment &&
        SpirvContainsExtensionString(out_code, "SPV_EXT_shader_stencil_export")) {
        DumpProblematicShaderSource("CompileGLSLtoSPIRV/raw", stage, code);
    }

    // Final pass through SPIRV-Optimizer
    if (Settings::values.optimize_spirv_output.GetValue() == Settings::OptimizeSpirv::Disabled) {
        return out_code;
    } else {
        std::vector<u32> result;
        result = OptimizeSPIRV(out_code);
        LogSpirvTrace(result, "CompileGLSLtoSPIRV/optimized", stage);
        if (stage == vk::ShaderStageFlagBits::eFragment &&
            SpirvContainsExtensionString(result, "SPV_EXT_shader_stencil_export")) {
            DumpProblematicShaderSource("CompileGLSLtoSPIRV/optimized", stage, code);
        }
        return result;
    }
}

} // namespace

vk::ShaderModule Compile(std::string_view code, vk::ShaderStageFlagBits stage, vk::Device device,
                         std::string_view premable) {
    const std::vector<u32> spirv = CompileGLSLtoSPIRV(code, stage, device, premable);
    LogSpirvTrace(spirv, "Compile", stage);
    return CompileSPV(spirv, device);
}

vk::ShaderModule MakeFallbackFragmentModule(vk::Device device) {
    static constexpr std::string_view fallback_source = R"(
layout(location = 0) out vec4 color;
void main() {
    color = vec4(1.0);
    gl_FragDepth = gl_FragCoord.z;
}
)";
    LOG_WARNING(Render_Vulkan,
                "Compiling fallback fragment shader to replace unsupported stencil-export shader");
    const std::vector<u32> fallback_spirv =
        CompileGLSLtoSPIRV(fallback_source, vk::ShaderStageFlagBits::eFragment, device, "");
    if (fallback_spirv.empty()) {
        LOG_ERROR(Render_Vulkan, "Fallback fragment shader compilation failed");
        return {};
    }
    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = fallback_spirv.size() * sizeof(u32),
        .pCode = fallback_spirv.data(),
    };
    try {
        return device.createShaderModule(shader_info);
    } catch (vk::SystemError& err) {
        LOG_ERROR(Render_Vulkan, "Fallback shader module creation failed: {}", err.what());
        return {};
    }
}

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    if (code.empty()) {
        LOG_ERROR(Render_Vulkan, "CompileSPV received empty SPIR-V bytecode");
        return {};
    }

    LogSpirvTrace(code, "CompileSPV", vk::ShaderStageFlagBits::eFragment);
    DumpCompileSpvIfEnabled(code);

    if (SpirvContainsExtensionString(code, "SPV_EXT_shader_stencil_export")) {
        LOG_ERROR(Render_Vulkan,
                  "Refusing to create shader module: SPV_EXT_shader_stencil_export is present in SPIR-V");
        LOG_ERROR(Render_Vulkan,
                  "Trace hint: this module reached CompileSPV already contaminated; inspect the most recent SPIR-V trace lines above");
        return MakeFallbackFragmentModule(device);
    }

    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    };

    try {
        return device.createShaderModule(shader_info);
    } catch (vk::SystemError& err) {
        UNREACHABLE_MSG("{}", err.what());
    }

    return {};
}

} // namespace Vulkan
