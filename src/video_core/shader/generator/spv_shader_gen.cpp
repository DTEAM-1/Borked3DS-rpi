// Copyright 2024 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>

#include "common/settings.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/shader/generator/shader_gen.h"
#include "video_core/shader/generator/spv_shader_gen.h"

using VSOutputAttributes = Pica::RasterizerRegs::VSOutputAttributes;

namespace Pica::Shader::Generator::SPIRV {

VertexModule::VertexModule() : Sirit::Module{SPIRV_VERSION_1_3} {
    DefineArithmeticTypes();
    DefineInterface();

    ids.sanitize_vertex = WriteFuncSanitizeVertex();

    DefineEntryPoint();
}

VertexModule::~VertexModule() = default;

void VertexModule::DefineArithmeticTypes() {
    ids.void_ = Name(TypeVoid(), "void_id");
    ids.bool_ = Name(TypeBool(), "bool_id");
    ids.f32 = Name(TypeFloat(32), "f32_id");
    ids.i32 = Name(TypeSInt(32), "i32_id");
    ids.u32 = Name(TypeUInt(32), "u32_id");

    if (Settings::values.relaxed_precision_decorators) {
        Decorate(ids.f32, spv::Decoration::RelaxedPrecision);
    }
    for (u32 size = 2; size <= 4; size++) {
        const u32 i = size - 2;
        ids.bvec.ids[i] = Name(TypeVector(ids.bool_, size), fmt::format("bvec{}_id", size));
        ids.vec.ids[i] = Name(TypeVector(ids.f32, size), fmt::format("vec{}_id", size));
        ids.ivec.ids[i] = Name(TypeVector(ids.i32, size), fmt::format("ivec{}_id", size));
        ids.uvec.ids[i] = Name(TypeVector(ids.u32, size), fmt::format("uvec{}_id", size));

        if (Settings::values.relaxed_precision_decorators) {
            Decorate(ids.vec.ids[i], spv::Decoration::RelaxedPrecision);
        }
    }
}

void VertexModule::DefineEntryPoint() {
    AddCapability(spv::Capability::Shader);
    SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);

    const Id main_type{TypeFunction(TypeVoid())};
    const Id main_func{OpFunction(TypeVoid(), spv::FunctionControlMask::MaskNone, main_type)};

    const Id interface_ids[] = {
        // Inputs
        ids.vert_in_position,
        ids.vert_in_color,
        ids.vert_in_texcoord0,
        ids.vert_in_texcoord1,
        ids.vert_in_texcoord2,
        ids.vert_in_texcoord0_w,
        ids.vert_in_normquat,
        ids.vert_in_view,
        // Outputs
        ids.gl_position,
        ids.gl_clip_distance,
        ids.vert_out_color,
        ids.vert_out_texcoord0,
        ids.vert_out_texcoord1,
        ids.vert_out_texcoord2,
        ids.vert_out_texcoord0_w,
        ids.vert_out_normquat,
        ids.vert_out_view,
    };

    AddEntryPoint(spv::ExecutionModel::Vertex, main_func, "main", interface_ids);
}

void VertexModule::DefineInterface() {
    // Define interface block

    /// Inputs
    ids.vert_in_position =
        Name(DefineInput(ids.vec.Get(4), ATTRIBUTE_POSITION), "vert_in_position");
    ids.vert_in_color = Name(DefineInput(ids.vec.Get(4), ATTRIBUTE_COLOR), "vert_in_color");
    ids.vert_in_texcoord0 =
        Name(DefineInput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD0), "vert_in_texcoord0");
    ids.vert_in_texcoord1 =
        Name(DefineInput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD1), "vert_in_texcoord1");
    ids.vert_in_texcoord2 =
        Name(DefineInput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD2), "vert_in_texcoord2");
    ids.vert_in_texcoord0_w =
        Name(DefineInput(ids.f32, ATTRIBUTE_TEXCOORD0_W), "vert_in_texcoord0_w");
    ids.vert_in_normquat =
        Name(DefineInput(ids.vec.Get(4), ATTRIBUTE_NORMQUAT), "vert_in_normquat");
    ids.vert_in_view = Name(DefineInput(ids.vec.Get(3), ATTRIBUTE_VIEW), "vert_in_view");

    /// Outputs
    ids.vert_out_color = Name(DefineOutput(ids.vec.Get(4), ATTRIBUTE_COLOR), "vert_out_color");
    ids.vert_out_texcoord0 =
        Name(DefineOutput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD0), "vert_out_texcoord0");
    ids.vert_out_texcoord1 =
        Name(DefineOutput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD1), "vert_out_texcoord1");
    ids.vert_out_texcoord2 =
        Name(DefineOutput(ids.vec.Get(2), ATTRIBUTE_TEXCOORD2), "vert_out_texcoord2");
    ids.vert_out_texcoord0_w =
        Name(DefineOutput(ids.f32, ATTRIBUTE_TEXCOORD0_W), "vert_out_texcoord0_w");
    ids.vert_out_normquat =
        Name(DefineOutput(ids.vec.Get(4), ATTRIBUTE_NORMQUAT), "vert_out_normquat");
    ids.vert_out_view = Name(DefineOutput(ids.vec.Get(3), ATTRIBUTE_VIEW), "vert_out_view");

    /// Uniforms

    // vs_data
    const Id type_vs_data = Name(TypeStruct(ids.u32, ids.vec.Get(4)), "vs_data");
    Decorate(type_vs_data, spv::Decoration::Block);

    ids.ptr_vs_data = AddGlobalVariable(TypePointer(spv::StorageClass::Uniform, type_vs_data),
                                        spv::StorageClass::Uniform);

    Decorate(ids.ptr_vs_data, spv::Decoration::DescriptorSet, 0);
    Decorate(ids.ptr_vs_data, spv::Decoration::Binding, 1);

    MemberName(type_vs_data, 0, "enable_clip1");
    MemberName(type_vs_data, 1, "clip_coef");

    MemberDecorate(type_vs_data, 0, spv::Decoration::Offset, 0);
    MemberDecorate(type_vs_data, 1, spv::Decoration::Offset, 16);

    /// Built-ins
    ids.gl_position = DefineVar(ids.vec.Get(4), spv::StorageClass::Output);
    Decorate(ids.gl_position, spv::Decoration::BuiltIn, spv::BuiltIn::Position);

    ids.gl_clip_distance =
        DefineVar(TypeArray(ids.f32, Constant(ids.u32, 2)), spv::StorageClass::Output);
    Decorate(ids.gl_clip_distance, spv::Decoration::BuiltIn, spv::BuiltIn::ClipDistance);
}

Id VertexModule::WriteFuncSanitizeVertex() {
    const Id func_type = TypeFunction(ids.vec.Get(4), ids.vec.Get(4));
    const Id func = Name(OpFunction(ids.vec.Get(4), spv::FunctionControlMask::MaskNone, func_type),
                         "SanitizeVertex");
    const Id arg_pos = OpFunctionParameter(ids.vec.Get(4));

    AddLabel(OpLabel());

    const Id result = AddLocalVariable(TypePointer(spv::StorageClass::Function, ids.vec.Get(4)),
                                       spv::StorageClass::Function);
    OpStore(result, arg_pos);

    const Id pos_z = OpCompositeExtract(ids.f32, arg_pos, 2);
    const Id pos_w = OpCompositeExtract(ids.f32, arg_pos, 3);

    const Id ndc_z = OpFDiv(ids.f32, pos_z, pos_w);

    // if (ndc_z > 0.f && ndc_z < 0.000001f)
    const Id test_1 =
        OpLogicalAnd(ids.bool_, OpFOrdGreaterThan(ids.bool_, ndc_z, Constant(ids.f32, 0.0f)),
                     OpFOrdLessThan(ids.bool_, ndc_z, Constant(ids.f32, 0.000001f)));

    {
        const Id true_label = OpLabel();
        const Id end_label = OpLabel();

        OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(test_1, true_label, end_label);
        AddLabel(true_label);

        // .z = 0.0f;
        OpStore(result, OpCompositeInsert(ids.vec.Get(4), ConstantNull(ids.f32), arg_pos, 2));

        OpBranch(end_label);
        AddLabel(end_label);
    }

    // if (ndc_z < -1.f && ndc_z > -1.00001f)
    const Id test_2 =
        OpLogicalAnd(ids.bool_, OpFOrdLessThan(ids.bool_, ndc_z, Constant(ids.f32, -1.0f)),
                     OpFOrdGreaterThan(ids.bool_, ndc_z, Constant(ids.f32, -1.00001f)));
    {
        const Id true_label = OpLabel();
        const Id end_label = OpLabel();

        OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(test_2, true_label, end_label);
        AddLabel(true_label);

        // .z = -.w;
        const Id neg_w = OpFNegate(ids.f32, OpCompositeExtract(ids.f32, arg_pos, 3));
        OpStore(result, OpCompositeInsert(ids.vec.Get(4), neg_w, arg_pos, 2));

        OpBranch(end_label);
        AddLabel(end_label);
    }

    OpReturnValue(OpLoad(ids.vec.Get(4), result));
    OpFunctionEnd();
    return func;
}

void VertexModule::Generate(Common::UniqueFunction<void, Sirit::Module&, const ModuleIds&> proc) {
    AddLabel(OpLabel());

    ids.ptr_enable_clip1 = OpAccessChain(TypePointer(spv::StorageClass::Uniform, ids.u32),
                                         ids.ptr_vs_data, Constant(ids.u32, 0));

    ids.ptr_clip_coef = OpAccessChain(TypePointer(spv::StorageClass::Uniform, ids.vec.Get(4)),
                                      ids.ptr_vs_data, Constant(ids.u32, 1));

    proc(*this, ids);
    OpReturn();
    OpFunctionEnd();
}

namespace {
// vDIRA v146 (ROOT CAUSE FIX, BORKED3DS_V3DV_NEG_ZERO_FIX=1): OpenGL clips depth against
// -w <= z <= w while Vulkan uses 0 <= z <= w, and this shader emits gl_Position.z = -z (and
// gl_ClipDistance[0] = -z) with no conversion between the two conventions. Flat 2D overlay
// geometry -- dialogue glyphs -- reaches this point with z EXACTLY 0.0 and w 1.0 (measured:
// pos0=(1.666760e-02,-3.499923e-01,0.000000e+00,1.000000e+00)), so both derived values land on
// exactly zero: dead centre of the GL volume, but precisely on the near boundary of the Vulkan
// one. V3DV rejects that boundary and the primitive never rasterizes, which is why this text
// renders under OpenGL but not Vulkan. Real 3D geometry (z=-497, w=497) maps to the far boundary
// and is accepted, so only flat overlay draws vanish.
// Measured proof (Sonic Lost World): biasing the vertex DATA by z -= 0.001 or 0.5 restores the
// text; +0.5 does not; unbiased does not. Occlusion queries read 0 samples for those draws while
// 3D draws sharing the same path, pipeline, viewport and scissor read 36..392.
// Earlier attempts were placed in glsl_shader_gen.cpp and had no effect for a simple reason: the
// Vulkan software path does not use the GLSL generator at all. PipelineCache builds its trivial
// vertex shader from SPIRV::GenerateTrivialVertexShader (vk_pipeline_cache.cpp:203), i.e. THIS
// function, so the GLSL fixup could never reach these draws -- and the GLSL trivial-VS dump file
// was never even created, which confirmed it.
// The fix clamps z to a small negative value scaled by |w| before the negation, so it propagates
// to gl_Position.z and gl_ClipDistance[0] alike, exactly like the validated data-side bias. Only
// geometry sitting on or past the boundary is moved; genuine depths are orders of magnitude larger
// and untouched. Note this shader is built once in the PipelineCache constructor, so the flag is
// read at startup and a shader-cache purge alone does not regenerate it.
// Enabled by default: this is a correctness fix for the GL/Vulkan clip-convention mismatch, not a
// tuning knob. BORKED3DS_V3DV_NO_NEG_ZERO_FIX=1 restores the old behaviour for diagnostics.
bool IsNegZeroFixEnabled() {
    static const bool disabled = std::getenv("BORKED3DS_V3DV_NO_NEG_ZERO_FIX") != nullptr;
    return !disabled;
}

// v150 : restaure le fixup unilateral v146 (bornage d'un seul cote) pour
// comparaison directe sans recompilation. Voir GenerateTrivialVertexShader.
bool IsLegacyNegZeroFixEnabled() {
    static const bool legacy =
        std::getenv("BORKED3DS_V3DV_LEGACY_NEG_ZERO_FIX") != nullptr;
    return legacy;
}
} // Anonymous namespace

std::vector<u32> GenerateTrivialVertexShader(bool use_clip_planes) {
    VertexModule module;
    module.Generate([use_clip_planes](Sirit::Module& spv,
                                      const VertexModule::ModuleIds& ids) -> void {
        const Id pos_raw = spv.OpFunctionCall(
            ids.vec.Get(4), ids.sanitize_vertex, spv.OpLoad(ids.vec.Get(4), ids.vert_in_position));

        // vDIRA v146: pull z just inside the Vulkan clip volume before it is negated, so both
        // gl_Position.z and gl_ClipDistance[0] become strictly positive for flat 2D geometry
        // (z == 0) instead of landing exactly on the rejected near boundary. |w| is computed with
        // a select rather than OpFAbs to avoid pulling in the GLSL.std.450 extended instruction
        // set, and OpSelect avoids introducing a function-scope variable (SPIR-V requires those in
        // the entry block).
        // vDIRA v150 (BANDE SYMETRIQUE -- corrige la regression du plancher).
        // Le fixup v146 forcait a -zlim tout vtx_pos.z superieur a -zlim. En
        // convention PICA (gl_Position.z = -vtx_pos.z), la geometrie 3D devant le
        // plan de clip a vtx_pos.z positif, donc TOUTE cette geometrie -- dont le
        // plancher de Sonic (~+200) -- etait ecrasee a -zlim et aplatie sur la
        // borne near, la rendant transparente. On n'agit desormais que dans une
        // bande [-zband, +zband] autour de zero : le texte plat (z==0) est pousse
        // a +zband (donc strictement a l'interieur apres negation), tandis que le
        // plancher et la 3D lointaine, hors bande, passent intacts.
        // BORKED3DS_V3DV_LEGACY_NEG_ZERO_FIX=1 restaure le comportement v146.
        Id pos_sanitized = pos_raw;
        if (IsNegZeroFixEnabled()) {
            const Id pos_z = spv.OpCompositeExtract(ids.f32, pos_raw, 2);
            const Id pos_w = spv.OpCompositeExtract(ids.f32, pos_raw, 3);
            const Id w_is_negative =
                spv.OpFOrdLessThan(ids.bool_, pos_w, spv.Constant(ids.f32, 0.0f));
            const Id abs_w =
                spv.OpSelect(ids.f32, w_is_negative, spv.OpFNegate(ids.f32, pos_w), pos_w);
            if (IsLegacyNegZeroFixEnabled()) {
                // Comportement v146 : bornage unilateral (regression plancher).
                const Id z_limit = spv.OpFMul(ids.f32, spv.Constant(ids.f32, -1.0e-3f), abs_w);
                const Id z_too_high = spv.OpFOrdGreaterThan(ids.bool_, pos_z, z_limit);
                const Id z_fixed = spv.OpSelect(ids.f32, z_too_high, z_limit, pos_z);
                pos_sanitized = spv.OpCompositeInsert(ids.vec.Get(4), z_fixed, pos_raw, 2);
            } else {
                // Bande symetrique [-zband, +zband]. |z| est calcule par select
                // (evite GLSL.std.450), puis compare a zband. Hors bande, pos_z
                // est garde inchange.
                // v150b : pousser vers le NEGATIF. Mesure du recap -- le texte de
                // Sonic s'affiche a z = -0.001 et -0.5, mais pas a z = 0 ni +0.5.
                // La cible est un vtx_pos.z negatif : apres negation PICA,
                // gl_Position.z devient positif, a l'interieur du volume Vulkan
                // [0, w], ce qui affiche le texte plat. La v150 initiale poussait
                // a +z_band (signe inverse) et rejetait le texte.
                const Id z_band = spv.OpFMul(ids.f32, spv.Constant(ids.f32, 1.0e-3f), abs_w);
                const Id neg_z_band = spv.OpFNegate(ids.f32, z_band);
                const Id z_is_negative =
                    spv.OpFOrdLessThan(ids.bool_, pos_z, spv.Constant(ids.f32, 0.0f));
                const Id abs_z =
                    spv.OpSelect(ids.f32, z_is_negative, spv.OpFNegate(ids.f32, pos_z), pos_z);
                const Id in_band = spv.OpFOrdLessThanEqual(ids.bool_, abs_z, z_band);
                const Id z_fixed = spv.OpSelect(ids.f32, in_band, neg_z_band, pos_z);
                pos_sanitized = spv.OpCompositeInsert(ids.vec.Get(4), z_fixed, pos_raw, 2);
            }
        }

        // Negate Z
        const Id neg_z = spv.OpFNegate(ids.f32, spv.OpCompositeExtract(ids.f32, pos_sanitized, 2));
        const Id negated_z = spv.OpCompositeInsert(ids.vec.Get(4), neg_z, pos_sanitized, 2);

        spv.OpStore(ids.gl_position, negated_z);

        // Pass-through
        spv.OpStore(ids.vert_out_color, spv.OpLoad(ids.vec.Get(4), ids.vert_in_color));
        spv.OpStore(ids.vert_out_texcoord0, spv.OpLoad(ids.vec.Get(2), ids.vert_in_texcoord0));
        spv.OpStore(ids.vert_out_texcoord1, spv.OpLoad(ids.vec.Get(2), ids.vert_in_texcoord1));
        spv.OpStore(ids.vert_out_texcoord2, spv.OpLoad(ids.vec.Get(2), ids.vert_in_texcoord2));
        spv.OpStore(ids.vert_out_texcoord0_w, spv.OpLoad(ids.f32, ids.vert_in_texcoord0_w));
        spv.OpStore(ids.vert_out_normquat, spv.OpLoad(ids.vec.Get(4), ids.vert_in_normquat));
        spv.OpStore(ids.vert_out_view, spv.OpLoad(ids.vec.Get(3), ids.vert_in_view));

        if (use_clip_planes) {
            spv.OpStore(spv.OpAccessChain(spv.TypePointer(spv::StorageClass::Output, ids.f32),
                                          ids.gl_clip_distance, spv.Constant(ids.u32, 0)),
                        neg_z);

            const Id enable_clip1 = spv.OpINotEqual(
                ids.bool_, spv.OpLoad(ids.u32, ids.ptr_enable_clip1), spv.Constant(ids.u32, 0));

            {
                const Id true_label = spv.OpLabel();
                const Id false_label = spv.OpLabel();
                const Id end_label = spv.OpLabel();

                spv.OpSelectionMerge(end_label, spv::SelectionControlMask::MaskNone);
                spv.OpBranchConditional(enable_clip1, true_label, false_label);
                {
                    spv.AddLabel(true_label);

                    spv.OpStore(
                        spv.OpAccessChain(spv.TypePointer(spv::StorageClass::Output, ids.f32),
                                          ids.gl_clip_distance, spv.Constant(ids.u32, 1)),
                        spv.OpDot(ids.f32, spv.OpLoad(ids.vec.Get(4), ids.ptr_clip_coef),
                                  pos_sanitized));

                    spv.OpBranch(end_label);
                }
                {
                    spv.AddLabel(false_label);

                    spv.OpStore(
                        spv.OpAccessChain(spv.TypePointer(spv::StorageClass::Output, ids.f32),
                                          ids.gl_clip_distance, spv.Constant(ids.u32, 1)),
                        spv.ConstantNull(ids.f32));

                    spv.OpBranch(end_label);
                }
                spv.AddLabel(end_label);
            }
        }
    });

    // Run through SPIR-V Optimizer
    if (Settings::values.optimize_spirv_output.GetValue() == Settings::OptimizeSpirv::Disabled) {
        return module.Assemble();
    } else {
        std::vector<u32> result;
        result = Vulkan::OptimizeSPIRV(module.Assemble());
        return result;
    }
}

} // namespace Pica::Shader::Generator::SPIRV
