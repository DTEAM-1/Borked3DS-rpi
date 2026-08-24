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

// vDIRA v140 (ROOT CAUSE FIX, BORKED3DS_V3DV_CLIP_EPSILON=1): the PICA fixed clip plane is emitted
// as gl_ClipDistance[0] = -vtx_pos.z, and a user clip plane discards anything with a NEGATIVE
// distance. 2D overlay geometry -- dialogue glyphs -- comes out of the software vertex shader with
// z EXACTLY 0.0 (measured: pos0=(x,y,0.000,1.000)), so all three vertices of every glyph triangle
// land with gl_ClipDistance[0] == 0.0, i.e. exactly ON the clip plane. V3DV then eliminates the
// primitive outright: the triangle has no area inside the clip volume. That is why those draws
// measured occlusion samples=0 while 3D draws (z=-497 -> distance +497) rasterized normally on the
// very same path, pipeline, viewport and scissor; why the fullscreen-triangle substitution also
// vanished (it was written with z=0.0f too, so it was clipped by the same rule); and why the text
// renders correctly under Mesa's GL driver, which keeps the boundary case. A tiny positive bias
// pushes the exactly-zero case just inside the volume while remaining negligible for real 3D
// geometry, whose distances are orders of magnitude larger.
// vDIRA v142 (ROOT CAUSE FIX, BORKED3DS_V3DV_NEG_ZERO_FIX=1): the vertex shaders emit
// gl_Position.z as -vtx_pos.z. 2D overlay geometry -- dialogue glyphs -- arrives with z EXACTLY
// +0.0, and negating it yields -0.0, an IEEE 754 NEGATIVE zero. V3DV evaluates the Vulkan clip
// volume (0 <= z <= w) with the sign bit taken into account, so -0.0 falls below the lower bound
// and the whole primitive is discarded before rasterization. Mesa's GL driver keeps the boundary
// case, which is exactly why this text renders under OpenGL but not Vulkan.
// Measured proof (Sonic Lost World, software path): biasing vertex z by -0.001 or -0.5 (making
// gl_Position.z positive) makes the text appear; +0.5 (making it negative) does not; unbiased
// (-0.0) does not. Occlusion queries read 0 samples in the failing cases and >0 in 3D draws on the
// same path, pipeline, viewport and scissor.
// The fix normalizes only the zero case: when vtx_pos.z is zero of either sign, emit +0.0. Any
// other value is passed through untouched, so real depth and clipping behaviour is unchanged. It
// lives in the shader rather than in the vertex data so it covers BOTH the software path and the
// accelerated path (Kid Icarus Uprising's text VS reads no dynamic upper-bank uniform, so it never
// enters the software path and a data-side bias could never reach it).
// vDIRA v143 (ROOT CAUSE FIX, BORKED3DS_V3DV_NEG_ZERO_FIX=1): 2D overlay geometry -- dialogue
// glyphs -- leaves the vertex stage with z EXACTLY 0.0. The shaders then derive two clip values
// from it: gl_Position.z = -vtx_pos.z and gl_ClipDistance[0] = -vtx_pos.z. At z == 0 BOTH land on
// exactly zero, and V3DV rejects the primitive: it evaluates the boundary as excluded (> 0) where
// the spec admits it (>= 0). Mesa's GL driver keeps the boundary, which is precisely why this text
// renders under OpenGL but not Vulkan.
// Measured proof (Sonic Lost World, software path): biasing the vertex DATA by z -= 0.001 or 0.5
// -- which makes both derived values strictly positive -- makes the text appear; +0.5 (both
// negative) does not; unbiased (both exactly zero) does not. Occlusion queries read 0 samples in
// the failing cases while 3D draws on the same path, pipeline, viewport and scissor read 36..392.
// Two narrower attempts failed and pinned the mechanism down: biasing only gl_ClipDistance (v140)
// and forcing only gl_Position.z to +0.0 (v142) each left the OTHER value at exactly zero, and
// neither restored the text. So the fix must move the source value, exactly like the validated
// data-side bias, which propagates to both expressions at once.
// Applied in the shader rather than the vertex data so it covers BOTH the software path (Sonic)
// and the accelerated path (Kid Icarus Uprising, whose text VS reads no dynamic upper-bank uniform
// and therefore never enters the software path). Only the exactly-zero case is touched; the offset
// is 1e-3 in PICA clip units, far below one depth quantum for real geometry.
// vDIRA v144 (ROOT CAUSE FIX, BORKED3DS_V3DV_NEG_ZERO_FIX=1): OpenGL clips depth against
// -w <= z <= w while Vulkan uses 0 <= z <= w. The shaders emit gl_Position.z = -vtx_pos.z with no
// conversion between the two conventions, so 2D overlay geometry -- dialogue glyphs, which leave
// the vertex stage with z at (or infinitesimally close to) 0 -- lands in the MIDDLE of the GL
// volume but exactly on the near boundary of the Vulkan one. V3DV rejects that boundary and the
// primitive never rasterizes. Real 3D geometry (z=-497, w=497) maps to the far boundary, which is
// accepted, which is why only flat overlay draws vanish and why the text renders under GL.
// Measured proof (Sonic Lost World): biasing the vertex DATA by z -= 0.001 or 0.5 restores the
// text; +0.5 does not; unbiased does not. Occlusion reads 0 samples for those draws and 36..392
// for 3D draws sharing the same path, pipeline, viewport and scissor.
// v143 tried "if (vtx_pos.z == 0.0)" and failed even though the dumped shader proved the line was
// emitted right after SanitizeVertex: the state log prints z with {:.3f}, so a value like 0.0004
// also displays as 0.000, and an exact-equality test never fires. The validated data bias applied
// unconditionally, which is why it worked. Hence a RELATIVE threshold against w rather than an
// equality: anything not already comfortably inside the volume is pushed just inside it, scaled by
// w so it is correct for any clip magnitude, while genuine 3D depths (orders of magnitude larger)
// are left untouched. Applied in the shader so it covers the software path (Sonic) and the
// accelerated path (Kid Icarus Uprising) alike.
// ---------------------------------------------------------------------------------------------
// TG06 -- quaternion de normale NEUTRE quand le jeu n'en mappe aucun.
//
// Mesure (session 17/08, sonde TG05 + desassemblage SPIR-V du GS) : sur les configurations ou
// AUCUNE semantique QUATERNION_X..W n'est mappee (semantic_maps == 16, la valeur de remplissage),
// la lambda semantic() renvoie le litteral "1.0" pour les quatre composantes. Le geometry shader
// compile contient alors, constant-folde par glslang :
//
//     OpStore %normquat %76      avec  %76 = OpConstantComposite %v4float 1.0 1.0 1.0 1.0
//
// Or normalize(vec4(1,1,1,1)) = (0.5,0.5,0.5,0.5) est une rotation de 120 degres autour de
// (1,1,1)/sqrt(3) : une normale constante et INCLINEE, alors que le cas neutre est l'identite
// vec4(0,0,0,1), qui laisse normal = surface_normal. Le fragment shader de ces draws LIT bien
// normquat (verifie : 6 des 8 FS eclaires du dump font OpLoad %normquat + OpNormalize), donc la
// valeur arbitraire est reellement consommee par l'eclairage.
//
// Constate : TG05 mesure quat_never_mapped=1 sur les configs vs_total=3 et vs_total=5 de Metroid,
// et quat_ok=1 sur la config vs_total=7. TG04 (v162) avait rendu ces GS COMPILABLES ; il ne les
// avait pas rendus CORRECTS -- ce qui explique pourquoi la cinematique du vaisseau (config a
// quaternion mappe) a ete corrigee et pas le vaisseau en jeu.
//
// N'agit QUE lorsque les quatre composantes sont non mappees. Une config partiellement mappee
// garde le comportement d'origine.
// RESULTAT MESURE (A/B du 17/08, vaisseau Metroid, async_shader_compilation=false) : l'identite
// est PIRE que le repli d'origine -- environ 2x plus de triangles visibles. Explication coherente :
// normalize(1,1,1,1) tourne la normale vers (1,0,0), la surface est alors peu eclairee et le
// defaut reste discret ; l'identite laisse normal = (0,0,1), face a la camera, donc eclairement
// maximal et defaut pleinement expose. Aucune des deux valeurs n'est "juste" : elles ne changent
// que la VISIBILITE d'un probleme situe ailleurs.
// => TG06 est donc INACTIF PAR DEFAUT. Conserve uniquement comme levier de diagnostic :
//    BORKED3DS_TG06_QUAT_IDENTITY=1 l'active (utile pour RENDRE VISIBLES les draws concernes).
bool IsQuatIdentityFixEnabled() {
    static const bool enabled = IsEnabledEnv("BORKED3DS_TG06_QUAT_IDENTITY");
    return enabled;
}

bool AreQuaternionSemanticsUnmapped(const PicaGSConfigState& state) {
    const auto out_of_range = [&state](VSOutputAttributes::Semantic slot_semantic) {
        const u32 slot = static_cast<u32>(slot_semantic);
        return state.semantic_maps[slot].attribute_index >= state.gs_output_attributes;
    };
    return out_of_range(VSOutputAttributes::QUATERNION_X) &&
           out_of_range(VSOutputAttributes::QUATERNION_Y) &&
           out_of_range(VSOutputAttributes::QUATERNION_Z) &&
           out_of_range(VSOutputAttributes::QUATERNION_W);
}
// ---------------------------------------------------------------------------------------------

std::string ClipZFixupLine(const char* indent) {
    // Enabled by default (correctness fix for the GL/Vulkan clip-convention mismatch).
    // BORKED3DS_V3DV_NO_NEG_ZERO_FIX=1 restores the old behaviour for diagnostics.
    static const bool disabled = IsEnabledEnv("BORKED3DS_V3DV_NO_NEG_ZERO_FIX");
    if (disabled) {
        return std::string{};
    }

    // vDIRA v150 (BANDE SYMETRIQUE -- corrige la regression du plancher).
    //
    // Le fixup v146 poussait z juste a l'interieur du volume de clip Vulkan pour
    // que le texte 2D plat (vtx_pos.z == 0) ne soit plus rejete sur la borne
    // near apres negation. Sa condition etait "si vtx_pos.z > -zlim alors
    // vtx_pos.z = -zlim". Mais en convention PICA, gl_Position.z = -vtx_pos.z :
    // la geometrie 3D DEVANT le plan de clip a vtx_pos.z POSITIF. Or -zlim est
    // negatif, donc "vtx_pos.z > -zlim" est vrai pour TOUT z positif : le
    // plancher de Sonic (vtx_pos.z ~ +200) etait ecrase a -zlim et s'aplatissait
    // sur la borne near -> transparent. Mesure directe : NO_NEG_ZERO_FIX=1 fait
    // reapparaitre le plancher (et disparaitre le texte), prouvant que ce fixup
    // en est la cause.
    //
    // Correctif v150b : n'agir que dans une BANDE etroite autour de zero, ET
    // pousser vers le NEGATIF. Mesure du recap : le texte de Sonic s'affiche a
    // z = -0.001 et -0.5, mais PAS a z = 0 ni a z = +0.5. La cible est donc de
    // rendre vtx_pos.z NEGATIF (apres negation PICA, gl_Position.z devient
    // positif, a l'interieur du volume Vulkan [0, w]). La v150 initiale poussait
    // a +zband -> gl_Position.z negatif -> texte rejete : signe inverse, corrige
    // ici. Ce qui est franchement positif (plancher, ~+200) ou franchement
    // negatif (3D lointaine, ~-497) reste hors bande et intact.
    //
    // BORKED3DS_V3DV_LEGACY_NEG_ZERO_FIX=1 restaure le fixup unilateral v146
    // (bornage d'un seul cote) pour comparaison sans recompilation.
    static const bool legacy = IsEnabledEnv("BORKED3DS_V3DV_LEGACY_NEG_ZERO_FIX");
    if (legacy) {
        return fmt::format("{0}float v3dv_zlim = -1e-3 * abs(vtx_pos.w);\n"
                           "{0}if (vtx_pos.z > v3dv_zlim) vtx_pos.z = v3dv_zlim;\n",
                           indent);
    }
    return fmt::format("{0}float v3dv_zband = 1e-3 * abs(vtx_pos.w);\n"
                       "{0}if (abs(vtx_pos.z) <= v3dv_zband) vtx_pos.z = -v3dv_zband;\n",
                       indent);
}

} // Anonymous namespace

constexpr std::string_view VSPicaUniformBlockDef = R"(
struct pica_uniforms {
    vec4 f[96];
    uvec4 i[4];
    bool b[16];
};

#ifdef VULKAN
layout (set = 0, binding = 0, std140) uniform vs_pica_data {
#else
layout (binding = 0, std140) uniform vs_pica_data {
#endif
pica_uniforms uniforms;
    // v116c-TBO: per-draw texel base of f[0] in the whole-buffer RGBA32F view (vs_pica_f_tbo).
    // Read at a constant index (V3D-safe) to drive texelFetch in get_offset_register_sw.
    uint f_texel_base;
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
)";
    out += ClipZFixupLine("    ");
    out += "    gl_Position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);\n";
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
        if (IsQuatIdentityFixEnabled() && AreQuaternionSemanticsUnmapped(config.state.gs_state)) {
            // TG06 : voir le commentaire au-dessus de IsQuatIdentityFixEnabled().
            out += "    return vec4(0.0, 0.0, 0.0, 1.0);\n";
        } else {
            out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
                   semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
                   semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
                   semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
        }
        out += "}\n\n";

        V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_emit_vtx_source");
        out += "void EmitVtx() {\n";
        out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
               semantic(VSOutputAttributes::POSITION_Y) + ", " +
               semantic(VSOutputAttributes::POSITION_Z) + ", " +
               semantic(VSOutputAttributes::POSITION_W) + ");\n";
        out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
        out += ClipZFixupLine("    ");
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

    // v115-L: Append decompiled shader BEFORE main() so that file-scope globals
    // (conditional_code, address_registers, reg_tmp*) are visible from main() for
    // diagnostic probes. This reordering is safe: all functions and globals are defined
    // before main() sees them, and the forward declaration of exec_shader above handles
    // the reference in the early interface code.
    V115DA7Z3GLSLTraceRaw("v115d_a7z3_glsl before_append_program_source");
    out += program_source;

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
           "    exec_shader();\n";
    // Diagnostic overrides are value-gated ("1" enables; "0"/unset/anything else disables)
    // so that setting a var to 0 in emulators.cfg truly turns it off.
    const char* ftt_env = std::getenv("BORKED3DS_V3DV_FORCE_TEXCOORD_TEST");
    const bool force_texcoord_test = ftt_env != nullptr && ftt_env[0] == '1';
    const char* piu_env = std::getenv("BORKED3DS_V3DV_PROBE_INT_UNIFORM");
    const bool probe_int_uniform = piu_env != nullptr && piu_env[0] == '1';
    // v115-M: probe f[5].y (the loop-counter step / comparison target). If bright (=1.0),
    // then with i[0].x=0 (1 iteration), reg_tmp8 starts at f[5].x=0, never reaches f[5].y=1
    // -> sub_67_86 is never called -> reg_tmp5 stays flat. Text needs i[0].x >= 1.
    const char* pf5_env = std::getenv("BORKED3DS_V3DV_PROBE_F5");
    const bool probe_f5 = pf5_env != nullptr && pf5_env[0] == '1';
    // v115-N: probe reg_tmp5 directly (the texcoord computation result). After the
    // reordering, reg_tmp5 is a file-scope global accessible from main(). Shows the average
    // of reg_tmp5.x and reg_tmp5.y as luminance: 0.5 gray = initial (0,0), brighter/darker
    // = the computation set it. Compare V3DV vs GL.
    const char* pr5_env = std::getenv("BORKED3DS_V3DV_PROBE_REG_TMP5");
    const bool probe_reg_tmp5 = pr5_env != nullptr && pr5_env[0] == '1';
    if (config.state.num_outputs > 2 && force_texcoord_test) {
        out += "    vs_out_attr2 = vec4(float(gl_VertexIndex & 1), "
               "float((gl_VertexIndex >> 1) & 1), 0.0f, 1.0f);\n";
    }
    if (config.state.num_outputs > 2 && probe_int_uniform) {
        out += "    vs_out_attr2 = vec4(uniforms.i[0].x > 0u ? 1.0f : 0.0f);\n";
    }
    if (config.state.num_outputs > 2 && probe_f5) {
        out += "    vs_out_attr2 = vec4(vec3(uniforms.f[5].y), 1.0f);\n";
    }
    if (config.state.num_outputs > 2 && probe_reg_tmp5) {
        out += "    vs_out_attr2 = vec4(vec3((reg_tmp5.x + reg_tmp5.y) * 0.5 + 0.5), 1.0f);\n";
    }
    out += "    EmitVtx();\n"
           "}\n\n";

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
    if (IsQuatIdentityFixEnabled() && AreQuaternionSemanticsUnmapped(state)) {
        // TG06 : aucune semantique quaternion mappee -> identite (rotation nulle),
        // au lieu du repli "1.0" qui donnait une normale constante inclinee de 120 degres.
        out += "    return vec4(0.0, 0.0, 0.0, 1.0);\n";
    } else {
        out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
               semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
    }
    out += "}\n\n";

    out += "void EmitVtx(Vertex vtx, bool quats_opposite) {\n";
    out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
           semantic(VSOutputAttributes::POSITION_Y) + ", " +
           semantic(VSOutputAttributes::POSITION_Z) + ", " +
           semantic(VSOutputAttributes::POSITION_W) + ");\n";
    out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
    out += ClipZFixupLine("    ");
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

    // Declare input arrays from vertex shader.
    // TG04 (axe G, facettes) : les varyings d'entree du GS DOIVENT porter le meme
    // decorateur layout(location = N) que les sorties correspondantes du vertex shader
    // (cf. branche use_geometry_shader plus haut, qui emet bien "layout(location = i)").
    // GLSL/OpenGL tolere leur absence, SPIR-V non : sans eux glslang echoue avec
    // "SPIR-V requires location for user input/output" -> bytecode vide -> le GS de
    // fix-up quaternion n'est jamais compilé -> pas de flip q/-q sur le chemin accelere
    // -> eclairage faux au centre des triangles -> facettes sur les surfaces courbes.
    for (u32 i = 0; i < config.state.vs_output_attributes; ++i) {
        if (separable_shader) {
            out << fmt::format("layout(location = {}) ", i);
        }
        out << "in vec4 vs_out_attr" << i << "[];\n";
    }
    out << "\n";

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
