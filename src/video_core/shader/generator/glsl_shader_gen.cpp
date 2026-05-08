// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string_view>
#include <fmt/format.h>

#ifndef __APPLE__
#include <glad/gl.h>
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_vars.h"
#endif

#include "video_core/pica/regs_rasterizer.h"
#include "video_core/shader/generator/glsl_shader_decompiler.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/shader/generator/shader_gen.h"

using VSOutputAttributes = Pica::RasterizerRegs::VSOutputAttributes;

namespace Pica::Shader::Generator::GLSL {

namespace {

bool IsEnabledEnv(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool ShouldSkipGLSLGLVersionQuery() {
    return IsEnabledEnv("BORKED3DS_V3DV_SKIP_GLSL_GL_VERSION_QUERY");
}

bool ShouldForceSkipLegacyGLESQueryForVulkan() {
    // v115-D-A7Z4: this GLSL generator is also used by the Vulkan/SPIR-V path.
    // On Pi5/V3DV there is no active OpenGL context at this point, so querying
    // GL_MAJOR_VERSION / GL_MINOR_VERSION via glGetIntegerv can terminate before
    // GenerateVertexShader() returns. Treat strict V3DV mode as a hard signal to
    // skip the legacy GLES separable-output workaround.
    return IsEnabledEnv("BORKED3DS_V3DV_STRICT_COMPAT") ||
           IsEnabledEnv("BORKED3DS_V3DV_A7Z4_FORCE_SKIP_LEGACY_GLES_QUERY");
}

// v115-D-A7Z4 build-fix: ShouldEmitLegacyGLESSeparableShaderOutputs() is near the top of
// this file, but the A7Z3 sidecar helpers are defined a little later in the same anonymous
// namespace. Declare them here before first use so GCC can compile the file with -Werror.
void V115DA7Z3GLSLTraceRaw(const char* message);
void V115DA7Z3GLSLTraceNumber(const char* label, std::uint64_t value);
void V115DA7Z3GLSLTraceBool(const char* label, bool value);

bool ShouldEmitLegacyGLESSeparableShaderOutputs() {
#ifndef __APPLE__
    V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_query_enter");

    if (ShouldSkipGLSLGLVersionQuery()) {
        V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_query_skip_env");
        return false;
    }

    if (ShouldForceSkipLegacyGLESQueryForVulkan()) {
        V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_query_skip_vulkan_strict");
        return false;
    }

    GLint majorVersion = 0;
    GLint minorVersion = 0;
    V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_before_glget_major");
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    V115DA7Z3GLSLTraceNumber("v115d_a7z4 legacy_gles_major", static_cast<std::uint64_t>(majorVersion));
    V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_before_glget_minor");
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    V115DA7Z3GLSLTraceNumber("v115d_a7z4 legacy_gles_minor", static_cast<std::uint64_t>(minorVersion));

    const bool result = OpenGL::GLES && majorVersion == 3 && minorVersion < 2;
    V115DA7Z3GLSLTraceBool("v115d_a7z4 legacy_gles_result", result);
    return result;
#else
    V115DA7Z3GLSLTraceRaw("v115d_a7z4 legacy_gles_query_skip_apple");
    return false;
#endif
}


constexpr const char* V115DA7Z3GLSLTracePath = "/tmp/borked3ds_v115d_a7z3_glsl_gen.log";

bool ShouldTraceV115DA7Z3GLSL() {
    return IsEnabledEnv("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE") ||
           IsEnabledEnv("BORKED3DS_V3DV_A7Z3_GLSL_FILE_TRACE");
}

void V115DA7Z3GLSLTraceResetOnce() {
    if (!ShouldTraceV115DA7Z3GLSL()) {
        return;
    }

    static bool reset_done = false;
    if (reset_done) {
        return;
    }
    reset_done = true;

    std::ofstream file(V115DA7Z3GLSLTracePath, std::ios::out | std::ios::trunc);
    if (file) {
        file << "v115d_a7z3_glsl glsl_trace_reset" << std::endl;
    }
}

void V115DA7Z3GLSLTraceRaw(const char* message) {
    if (!ShouldTraceV115DA7Z3GLSL()) {
        return;
    }

    std::ofstream file(V115DA7Z3GLSLTracePath, std::ios::out | std::ios::app);
    if (file) {
        file << message << std::endl;
    }
}

void V115DA7Z3GLSLTraceNumber(const char* label, std::uint64_t value) {
    if (!ShouldTraceV115DA7Z3GLSL()) {
        return;
    }

    std::ofstream file(V115DA7Z3GLSLTracePath, std::ios::out | std::ios::app);
    if (file) {
        file << label << "=" << value << std::endl;
    }
}

void V115DA7Z3GLSLTraceBool(const char* label, bool value) {
    V115DA7Z3GLSLTraceNumber(label, value ? 1U : 0U);
}

bool ShouldA7Z3ReturnTrivialVSFromGLSLGenerator() {
    return ShouldTraceV115DA7Z3GLSL() &&
           IsEnabledEnv("BORKED3DS_V3DV_A7Z3_GLSL_RETURN_TRIVIAL_VS");
}

bool ShouldA7Z3DumpGeneratedVertexShader() {
    return ShouldTraceV115DA7Z3GLSL() &&
           IsEnabledEnv("BORKED3DS_V3DV_A7Z3_DUMP_GENERATED_VS_GLSL");
}

void V115DA7Z3DumpGeneratedVertexShader(const std::string& source) {
    if (!ShouldA7Z3DumpGeneratedVertexShader()) {
        return;
    }

    std::ofstream file("/tmp/borked3ds_v115d_a7z3_generated_vs.glsl",
                       std::ios::out | std::ios::trunc);
    if (file) {
        file << source;
        file.flush();
    }
}

} // Anonymous namespace

constexpr std::string_view VSPicaUniformBlockDef = R"(
struct pica_uniforms {
    bool b[16];
    uvec4 i[4];
    vec4 f[96];
};

#ifdef VULKAN
layout (set = 0, binding = 0, std140) uniform vs_pica_data {
#else
layout (binding = 0, std140) uniform vs_pica_data {
#endif
    pica_uniforms uniforms;
};
)";

constexpr std::string_view VSUniformBlockDef = R"(
#ifdef VULKAN
layout (set = 0, binding = 1, std140) uniform vs_data {
#else
layout (binding = 1, std140) uniform vs_data {
#endif
    bool enable_clip1;
    vec4 clip_coef;
};

const vec2 EPSILON_Z = vec2(0.000001f, -1.00001f);

vec4 SanitizeVertex(vec4 vtx_pos) {
    float ndc_z = vtx_pos.z / vtx_pos.w;
    if (ndc_z > 0.0f && ndc_z < EPSILON_Z[0]) {
        vtx_pos.z = 0.0f;
    }
    if (ndc_z < -1.0f && ndc_z > EPSILON_Z[1]) {
        vtx_pos.z = -vtx_pos.w;
    }
    return vtx_pos;
}
)";

static std::string GetVertexInterfaceDeclaration(bool is_output, bool use_clip_planes,
                                                 bool separable_shader) {
    std::string out;

    const auto append_variable = [&](std::string_view var, int location) {
        if (separable_shader) {
            out += fmt::format("layout (location={}) ", location);
        }
        out += fmt::format("{}{};\n", is_output ? "out " : "in ", var);
    };

    append_variable("vec4 primary_color", ATTRIBUTE_COLOR);
    append_variable("vec2 texcoord0", ATTRIBUTE_TEXCOORD0);
    append_variable("vec2 texcoord1", ATTRIBUTE_TEXCOORD1);
    append_variable("vec2 texcoord2", ATTRIBUTE_TEXCOORD2);
    append_variable("float texcoord0_w", ATTRIBUTE_TEXCOORD0_W);
    append_variable("vec4 normquat", ATTRIBUTE_NORMQUAT);
    append_variable("vec3 view", ATTRIBUTE_VIEW);

    if (is_output && separable_shader) {
        // gl_PerVertex redeclaration is required for separate shader object
        out += "out gl_PerVertex {\n";
        // Apple Silicon GPU drivers optimize more aggressively, which can create
        // too much variance and cause visual artifacting in games like Pokemon.
#ifdef __APPLE__
        out += "    invariant vec4 gl_Position;\n";
#else
        out += "    vec4 gl_Position;\n";
#endif
        if (use_clip_planes) {
            out += "    float gl_ClipDistance[2];\n";
        }
        out += "};\n";
    }

    return out;
}

std::string GenerateTrivialVertexShader(bool use_clip_planes, bool separable_shader) {
    std::string out;

    if (separable_shader) {
        out += "#extension GL_ARB_separate_shader_objects : enable\n";

        if (ShouldEmitLegacyGLESSeparableShaderOutputs()) {
            out += R"(
#extension GL_ARB_separate_shader_objects : enable
layout(location = ATTRIBUTE_COLOR) out vec4 primary_color;
layout(location = ATTRIBUTE_TEXCOORD0) out vec2 texcoord0;
layout(location = ATTRIBUTE_TEXCOORD1) out vec2 texcoord1;
layout(location = ATTRIBUTE_TEXCOORD2) out vec2 texcoord2;
layout(location = ATTRIBUTE_TEXCOORD0_W) out float texcoord0_w;
layout(location = ATTRIBUTE_NORMQUAT) out vec4 normquat;
layout(location = ATTRIBUTE_VIEW) out vec3 view;
)";
        }
    }

    out +=
        fmt::format("layout(location = {}) in vec4 vert_position;\n"
                    "layout(location = {}) in vec4 vert_color;\n"
                    "layout(location = {}) in vec2 vert_texcoord0;\n"
                    "layout(location = {}) in vec2 vert_texcoord1;\n"
                    "layout(location = {}) in vec2 vert_texcoord2;\n"
                    "layout(location = {}) in float vert_texcoord0_w;\n"
                    "layout(location = {}) in vec4 vert_normquat;\n"
                    "layout(location = {}) in vec3 vert_view;\n",
                    ATTRIBUTE_POSITION, ATTRIBUTE_COLOR, ATTRIBUTE_TEXCOORD0, ATTRIBUTE_TEXCOORD1,
                    ATTRIBUTE_TEXCOORD2, ATTRIBUTE_TEXCOORD0_W, ATTRIBUTE_NORMQUAT, ATTRIBUTE_VIEW);

    out += GetVertexInterfaceDeclaration(true, use_clip_planes, separable_shader);
    out += VSUniformBlockDef;

    out += R"(
void main() {
    primary_color = vert_color;
    texcoord0 = vert_texcoord0;
    texcoord1 = vert_texcoord1;
    texcoord2 = vert_texcoord2;
    texcoord0_w = vert_texcoord0_w;
    normquat = vert_normquat;
    view = vert_view;
    vec4 vtx_pos = SanitizeVertex(vert_position);
    gl_Position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);
)";
    if (use_clip_planes) {
        out += R"(
        gl_ClipDistance[0] = -vtx_pos.z; // fixed PICA clipping plane z <= 0
        if (enable_clip1) {
            gl_ClipDistance[1] = dot(clip_coef, vtx_pos);
        } else {
            gl_ClipDistance[1] = 0.0;
        }
        )";
    }

    out += "}\n";

    return out;
}

std::string_view MakeLoadPrefix(AttribLoadFlags flag) {
    if (True(flag & AttribLoadFlags::Float)) {
        return "";
    } else if (True(flag & AttribLoadFlags::Sint)) {
        return "i";
    } else if (True(flag & AttribLoadFlags::Uint)) {
        return "u";
    }
    return "";
}

std::string GenerateVertexShader(const ShaderSetup& setup, const PicaVSConfig& config,
                                 bool separable_shader) {
    V115DA7Z3GLSLTraceResetOnce();
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl generate_vertex_shader_enter");
    V115DA7Z3GLSLTraceRaw("v115d_a7z4 glsl_generator_patch_active");
    V115DA7Z3GLSLTraceBool("v115d_a7z3_glsl separable_shader", separable_shader);
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl config_hash", config.Hash());
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl main_offset", config.state.main_offset);
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl num_outputs", config.state.num_outputs);
    V115DA7Z3GLSLTraceBool("v115d_a7z3_glsl use_geometry_shader", config.state.use_geometry_shader);
    V115DA7Z3GLSLTraceBool("v115d_a7z3_glsl use_clip_planes", config.state.use_clip_planes);
    V115DA7Z3GLSLTraceBool("v115d_a7z3_glsl sanitize_mul", config.state.sanitize_mul);
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl gs_output_attributes",
                             config.state.gs_state.gs_output_attributes);
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl vs_output_attributes",
                             config.state.gs_state.vs_output_attributes);
    V115DA7Z3GLSLTraceBool("v115d_a7z3_glsl return_trivial_vs_env",
                           ShouldA7Z3ReturnTrivialVSFromGLSLGenerator());

    if (ShouldA7Z3ReturnTrivialVSFromGLSLGenerator()) {
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl return_trivial_vs_begin");
        V115DA7Z3GLSLTraceRaw("v115d_a7z4 return_trivial_before_generate_trivial");
        std::string trivial = GenerateTrivialVertexShader(config.state.use_clip_planes, separable_shader);
        V115DA7Z3GLSLTraceRaw("v115d_a7z4 return_trivial_after_generate_trivial");
        V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl return_trivial_vs_size", trivial.size());
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl return_trivial_vs_end");
        return trivial;
    }

    std::string out;
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_separable_extension_block");
    if (separable_shader) {
        out += "#extension GL_ARB_separate_shader_objects : enable\n";

        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_legacy_gles_query");
        V115DA7Z3GLSLTraceRaw("v115d_a7z4 before_safe_legacy_gles_query");
        if (ShouldEmitLegacyGLESSeparableShaderOutputs()) {
            V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl legacy_gles_outputs_enabled");
            out += R"(
#extension GL_ARB_separate_shader_objects : enable
layout(location = ATTRIBUTE_COLOR) out vec4 primary_color;
layout(location = ATTRIBUTE_TEXCOORD0) out vec2 texcoord0;
layout(location = ATTRIBUTE_TEXCOORD1) out vec2 texcoord1;
layout(location = ATTRIBUTE_TEXCOORD2) out vec2 texcoord2;
layout(location = ATTRIBUTE_TEXCOORD0_W) out float texcoord0_w;
layout(location = ATTRIBUTE_NORMQUAT) out vec4 normquat;
layout(location = ATTRIBUTE_VIEW) out vec3 view;
)";
        } else {
            V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl legacy_gles_outputs_disabled");
        }
    }
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_separable_extension_block");

    out += VSPicaUniformBlockDef;
    out += VSUniformBlockDef;
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_uniform_blocks");

    std::array<bool, 16> used_regs{};
    const auto get_input_reg = [&used_regs](u32 reg) {
        V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl get_input_reg", reg);
        ASSERT(reg < 16);
        used_regs[reg] = true;
        return fmt::format("vs_in_reg{}", reg);
    };

    const auto get_output_reg = [&](u32 reg) -> std::string {
        V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl get_output_reg", reg);
        ASSERT(reg < 16);
        V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl get_output_reg_map", config.state.output_map[reg]);
        if (config.state.output_map[reg] < config.state.num_outputs) {
            return fmt::format("vs_out_attr{}", config.state.output_map[reg]);
        }
        return "";
    };

    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_decompile_program");
    auto program_source =
        DecompileProgram(setup.program_code, setup.swizzle_data, config.state.main_offset,
                         get_input_reg, get_output_reg, config.state.sanitize_mul);
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_decompile_program");
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl program_source_size", program_source.size());

    if (program_source.empty()) {
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl program_source_empty_return_empty");
        return "";
    }

    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_input_attributes_declaration");
    // input attributes declaration
    for (std::size_t i = 0; i < used_regs.size(); ++i) {
        if (used_regs[i]) {
            V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl declare_input_reg", i);
            const auto flags = config.state.load_flags[i];
            const std::string_view prefix = MakeLoadPrefix(flags);
            out +=
                fmt::format("layout(location = {0}) in {1}vec4 vs_in_typed_reg{0};\n", i, prefix);
            out += fmt::format("vec4 vs_in_reg{0};\n", i);
        }
    }
    out += '\n';
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_input_attributes_declaration");

    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_output_branch");
    if (config.state.use_geometry_shader) {
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl output_branch_geometry_shader");
        // output attributes declaration
        for (u32 i = 0; i < config.state.num_outputs; ++i) {
            V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl declare_gs_output_attr", i);
            if (separable_shader) {
                out += fmt::format("layout(location = {}) ", i);
            }
//gvx64            out += fmt::format("vec4 vs_out_attr{} = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n", i);
        out += fmt::format("out vec4 vs_out_attr{};\n", i);  // Changed: 'out' and no initialization //gvx64
        }
        out += "void EmitVtx() {}\n";
    } else {
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl output_branch_no_geometry_shader");
        out += GetVertexInterfaceDeclaration(true, config.state.use_clip_planes, separable_shader);
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_vertex_interface_declaration");

        // output attributes declaration
        for (u32 i = 0; i < config.state.num_outputs; ++i) {
            V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl declare_vs_output_attr", i);
            out += fmt::format("vec4 vs_out_attr{} = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n", i);
        }

        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_semantic_lambda");
        const auto semantic =
            [&state = config.state](VSOutputAttributes::Semantic slot_semantic) -> std::string {
            const u32 slot = static_cast<u32>(slot_semantic);
            const u32 attrib = state.gs_state.semantic_maps[slot].attribute_index;
            const u32 comp = state.gs_state.semantic_maps[slot].component_index;
            if (attrib < state.gs_state.gs_output_attributes) {
                return fmt::format("vs_out_attr{}.{}", attrib, "xyzw"[comp]);
            }
            return "1.0";
        };
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_semantic_lambda");

        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_emit_quaternion_source");
        out += "vec4 GetVertexQuaternion() {\n";
        out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
               semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
        out += "}\n\n";

        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_emit_vtx_source");
        out += "void EmitVtx() {\n";
        out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
               semantic(VSOutputAttributes::POSITION_Y) + ", " +
               semantic(VSOutputAttributes::POSITION_Z) + ", " +
               semantic(VSOutputAttributes::POSITION_W) + ");\n";
        out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
        out += "    gl_Position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);\n";
        if (config.state.use_clip_planes) {
            V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl emit_clip_planes_source");
            out += "    gl_ClipDistance[0] = -vtx_pos.z;\n"; // fixed PICA clipping plane z <= 0
            out += "    if (enable_clip1) {\n";
            out += "        gl_ClipDistance[1] = dot(clip_coef, vtx_pos);\n";
            out += "    } else {\n";
            out += "        gl_ClipDistance[1] = 0.0;\n";
            out += "    }\n\n";
        }

        out += "    normquat = GetVertexQuaternion();\n";
        out += "    vec4 vtx_color = vec4(" + semantic(VSOutputAttributes::COLOR_R) + ", " +
               semantic(VSOutputAttributes::COLOR_G) + ", " +
               semantic(VSOutputAttributes::COLOR_B) + ", " + semantic(VSOutputAttributes::COLOR_A) + ");\n";
        out += "    primary_color = min(abs(vtx_color), vec4(1.0f));\n\n";

        out += "    texcoord0 = vec2(" + semantic(VSOutputAttributes::TEXCOORD0_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD0_V) + ");\n";
        out += "    texcoord1 = vec2(" + semantic(VSOutputAttributes::TEXCOORD1_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD1_V) + ");\n\n";

        out += "    texcoord0_w = " + semantic(VSOutputAttributes::TEXCOORD0_W) + ";\n";
        out += "    view = vec3(" + semantic(VSOutputAttributes::VIEW_X) + ", " +
               semantic(VSOutputAttributes::VIEW_Y) + ", " + semantic(VSOutputAttributes::VIEW_Z) +
               ");\n\n";

        out += "    texcoord2 = vec2(" + semantic(VSOutputAttributes::TEXCOORD2_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD2_V) + ");\n\n";
        out += "}\n";
        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl after_emit_vtx_source");
    }

    out += "bool exec_shader();\n\n";

    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_main_source");
    out += "\nvoid main() {\n";
    for (std::size_t i = 0; i < used_regs.size(); ++i) {
        if (used_regs[i]) {
            V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl main_load_reg", i);
            out += fmt::format("vs_in_reg{0} = vec4(vs_in_typed_reg{0});\n", i);
            if (True(config.state.load_flags[i] & AttribLoadFlags::ZeroW)) {
                V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl main_zero_w_reg", i);
                out += fmt::format("vs_in_reg{0}.w = 0;\n", i);
            }
        }
    }
    out += "    // Initialize all vertex attributes to zero\n";
    for (u32 i = 0; i < config.state.num_outputs; ++i) {
        V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl main_init_output_attr", i);
        out += fmt::format("    vs_out_attr{} = vec4(0.0f, 0.0f, 0.0f, 1.0f);\n", i);
    }
    out += "\n    // Execute shader and emit vertex\n"
           "    exec_shader();\n"
           "    EmitVtx();\n"
           "}\n\n";

    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_append_program_source");
    out += program_source;
    V115DA7Z3GLSLTraceNumber("v115d_a7z3_glsl final_source_size", out.size());
    V115DA7Z3DumpGeneratedVertexShader(out);
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl return_success");

    return out;
}

static std::string GetGSCommonSource(const PicaGSConfigState& state, bool separable_shader) {
    std::string out = GetVertexInterfaceDeclaration(true, state.use_clip_planes, separable_shader);
    out += VSUniformBlockDef;

    // REMOVED: The vs_out_attr declarations that were here
    // Those are input arrays from the vertex shader, declared in GenerateFixedGeometryShader()

    out += R"(
struct Vertex {
)";
    out += fmt::format("    vec4 attributes[{}];\n", state.gs_output_attributes);
    out += "};\n\n";

    const auto semantic = [&state](VSOutputAttributes::Semantic slot_semantic) -> std::string {
        const u32 slot = static_cast<u32>(slot_semantic);
        const u32 attrib = state.semantic_maps[slot].attribute_index;
        const u32 comp = state.semantic_maps[slot].component_index;
        if (attrib < state.gs_output_attributes) {
            return fmt::format("vtx.attributes[{}].{}", attrib, "xyzw"[comp]);
        }
        return "1.0";
    };

    out += "vec4 GetVertexQuaternion(Vertex vtx) {\n";
    out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
           semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
    out += "}\n\n";

    out += "void EmitVtx(Vertex vtx, bool quats_opposite) {\n";
    out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
           semantic(VSOutputAttributes::POSITION_Y) + ", " +
           semantic(VSOutputAttributes::POSITION_Z) + ", " +
           semantic(VSOutputAttributes::POSITION_W) + ");\n";
    out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
    out += "    gl_Position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);\n";
    if (state.use_clip_planes) {
        out += "    gl_ClipDistance[0] = -vtx_pos.z;\n"; // fixed PICA clipping plane z <= 0
        out += "    if (enable_clip1) {\n";
        out += "        gl_ClipDistance[1] = dot(clip_coef, vtx_pos);\n";
        out += "    } else {\n";
        out += "        gl_ClipDistance[1] = 0.0;\n";
        out += "    }\n\n";
    }

    out += "    vec4 vtx_quat = GetVertexQuaternion(vtx);\n";
    out += "    normquat = mix(vtx_quat, -vtx_quat, bvec4(quats_opposite));\n\n";

    out += "    vec4 vtx_color = vec4(" + semantic(VSOutputAttributes::COLOR_R) + ", " +
           semantic(VSOutputAttributes::COLOR_G) + ", " + semantic(VSOutputAttributes::COLOR_B) +
           ", " + semantic(VSOutputAttributes::COLOR_A) + ");\n";
    out += "    primary_color = min(abs(vtx_color), vec4(1.0f));\n\n";

    out += "    texcoord0 = vec2(" + semantic(VSOutputAttributes::TEXCOORD0_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD0_V) + ");\n";
    out += "    texcoord1 = vec2(" + semantic(VSOutputAttributes::TEXCOORD1_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD1_V) + ");\n\n";

    out += "    texcoord0_w = " + semantic(VSOutputAttributes::TEXCOORD0_W) + ";\n";
    out += "    view = vec3(" + semantic(VSOutputAttributes::VIEW_X) + ", " +
           semantic(VSOutputAttributes::VIEW_Y) + ", " + semantic(VSOutputAttributes::VIEW_Z) +
           ");\n\n";

    out += "    texcoord2 = vec2(" + semantic(VSOutputAttributes::TEXCOORD2_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD2_V) + ");\n\n";

    out += "    EmitVertex();\n";
    out += "}\n";

    out += R"(
bool AreQuaternionsOpposite(vec4 qa, vec4 qb) {
    return (dot(qa, qb) < 0.0);
}

void EmitPrim(Vertex vtx0, Vertex vtx1, Vertex vtx2) {
    EmitVtx(vtx0, false);
    EmitVtx(vtx1, AreQuaternionsOpposite(GetVertexQuaternion(vtx0), GetVertexQuaternion(vtx1)));
    EmitVtx(vtx2, AreQuaternionsOpposite(GetVertexQuaternion(vtx0), GetVertexQuaternion(vtx2)));
    EndPrimitive();
}
)";

    return out;
}

std::string GenerateFixedGeometryShader(const PicaFixedGSConfig& config, bool separable_shader) {
    std::stringstream out;

    if (separable_shader) {
        out << "#extension GL_ARB_separate_shader_objects : enable\n";

        if (ShouldEmitLegacyGLESSeparableShaderOutputs()) {
            out << R"(
#extension GL_ARB_separate_shader_objects : enable
layout(location = ATTRIBUTE_COLOR) out vec4 primary_color;
layout(location = ATTRIBUTE_TEXCOORD0) out vec2 texcoord0;
layout(location = ATTRIBUTE_TEXCOORD1) out vec2 texcoord1;
layout(location = ATTRIBUTE_TEXCOORD2) out vec2 texcoord2;
layout(location = ATTRIBUTE_TEXCOORD0_W) out float texcoord0_w;
layout(location = ATTRIBUTE_NORMQUAT) out vec4 normquat;
layout(location = ATTRIBUTE_VIEW) out vec3 view;
)";
        }
    }

    out << R"(
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

)";

    // ADD THIS: Declare input arrays from vertex shader
    for (u32 i = 0; i < config.state.vs_output_attributes; ++i) { //gvx64 - testing only
        out << "in vec4 vs_out_attr" << i << "[];\n"; //gvx64
    } //gvx64
    out << "\n"; //gvx64

    out << GetGSCommonSource(config.state, separable_shader);

    out << R"(
void main() {
    Vertex prim_buffer[3];
)";

    for (u32 vtx = 0; vtx < 3; ++vtx) {
        out << fmt::format("    prim_buffer[{}].attributes = vec4[{}](", vtx,
                           config.state.gs_output_attributes);
        for (u32 i = 0; i < config.state.vs_output_attributes; ++i) {
            out << fmt::format("{}vs_out_attr{}[{}]", i == 0 ? "" : ", ", i, vtx); //gvx64 - original implementation
//gvx64            out << fmt::format("{}vs_out_attr{}", i == 0 ? "" : ", ", i); //gvx64
        }
        out << ");\n";
    }

    out << "    EmitPrim(prim_buffer[0], prim_buffer[1], prim_buffer[2]);\n";
    out << "}\n";

    return out.str();
}

} // namespace Pica::Shader::Generator::GLSL
