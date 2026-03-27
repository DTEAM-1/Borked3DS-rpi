// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <memory>
#include <utility>
#include <vector>
#include <string_view>
#include <SPIRV/GlslangToSpv.h>
#include <glslang/Include/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv-tools/optimizer.hpp>

#include "common/assert.h"
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

/**
 * @brief Compiles GLSL into SPIRV
 * @param code The string containing GLSL code.
 * @param stage The pipeline stage the shader will be used in.
 * @param device The vulkan device handle.
 */
std::vector<u32> CompileGLSLtoSPIRV(std::string_view code, vk::ShaderStageFlagBits stage,
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
    glslang::GlslangToSpv(*intermediate, out_code, &logger, &options);

    const std::string spv_messages = logger.getAllMessages();
    if (!spv_messages.empty()) {
        LOG_INFO(Render_Vulkan, "SPIR-V conversion messages: {}", spv_messages);
    }

    LogSpirvTrace(out_code, "CompileGLSLtoSPIRV/raw", stage);

    // Final pass through SPIRV-Optimizer
    if (Settings::values.optimize_spirv_output.GetValue() == Settings::OptimizeSpirv::Disabled) {
        return out_code;
    } else {
        std::vector<u32> result;
        result = OptimizeSPIRV(out_code);
        LogSpirvTrace(result, "CompileGLSLtoSPIRV/optimized", stage);
        return result;
    }
}

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
    color = vec4(0.0);
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
