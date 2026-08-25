// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/pica/pica_core.h"
#include "video_core/rasterizer_accelerated.h"

namespace VideoCore {

using Pica::f24;

// TG09 / TG10 : declarations anticipees. Les definitions vivent dans le namespace anonyme en
// bas de ce fichier, avec le bloc de commentaires qui explique la sonde.
namespace {
u32 TG09Level();
} // Anonymous namespace

static Common::Vec4f ColorRGBA8(const u32 color) {
    const auto rgba =
        Common::Vec4u{color >> 0 & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF, color >> 24 & 0xFF};
    return rgba / 255.0f;
}

static Common::Vec3f LightColor(const Pica::LightingRegs::LightColor& color) {
    return Common::Vec3u{color.r, color.g, color.b} / 255.0f;
}

RasterizerAccelerated::HardwareVertex::HardwareVertex(const Pica::OutputVertex& v,
                                                      bool flip_quaternion) {
    // Get the vector components first
    auto pos = v.pos();
    auto col = v.color();
    auto tc0 = v.tc0();
    auto tc1 = v.tc1();
    auto tc2 = v.tc2();
    auto q = v.quat();
    auto view_vec = v.view();

    // Now assign to the float arrays
    position =
        Common::Vec4f{pos.x.ToFloat32(), pos.y.ToFloat32(), pos.z.ToFloat32(), pos.w.ToFloat32()};

    color =
        Common::Vec4f{col.x.ToFloat32(), col.y.ToFloat32(), col.z.ToFloat32(), col.w.ToFloat32()};

    tex_coord0 = Common::Vec2f{tc0.x.ToFloat32(), tc0.y.ToFloat32()};

    tex_coord1 = Common::Vec2f{tc1.x.ToFloat32(), tc1.y.ToFloat32()};

    tex_coord2 = Common::Vec2f{tc2.x.ToFloat32(), tc2.y.ToFloat32()};

    tex_coord0_w = v.tc0_w.ToFloat32();

    normquat = Common::Vec4f{q.x.ToFloat32(), q.y.ToFloat32(), q.z.ToFloat32(), q.w.ToFloat32()};

    view = Common::Vec3f{view_vec.x.ToFloat32(), view_vec.y.ToFloat32(), view_vec.z.ToFloat32()};

    if (flip_quaternion) {
        normquat = -normquat;
    }
}

RasterizerAccelerated::RasterizerAccelerated(Memory::MemorySystem& memory_, Pica::PicaCore& pica_)
    : memory{memory_}, pica{pica_}, regs{pica.regs.internal} {
    fs_uniform_block_data.lighting_lut_dirty.fill(true);
}

/**
 * This is a helper function to resolve an issue when interpolating opposite quaternions. See below
 * for a detailed description of this issue (yuriks):
 *
 * For any rotation, there are two quaternions Q, and -Q, that represent the same rotation. If you
 * interpolate two quaternions that are opposite, instead of going from one rotation to another
 * using the shortest path, you'll go around the longest path. You can test if two quaternions are
 * opposite by checking if Dot(Q1, Q2) < 0. In that case, you can flip either of them, therefore
 * making Dot(Q1, -Q2) positive.
 *
 * This solution corrects this issue per-vertex before passing the quaternions to OpenGL. This is
 * correct for most cases but can still rotate around the long way sometimes. An implementation
 * which did `lerp(lerp(Q1, Q2), Q3)` (with proper weighting), applying the dot product check
 * between each step would work for those cases at the cost of being more complex to implement.
 *
 * Fortunately however, the 3DS hardware happens to also use this exact same logic to work around
 * these issues, making this basic implementation actually more accurate to the hardware.
 */
static bool AreQuaternionsOpposite(const Pica::OutputVertex& v1, const Pica::OutputVertex& v2) {
    auto q1 = v1.quat();
    auto q2 = v2.quat();

    Common::Vec4f a{q1.x.ToFloat32(), q1.y.ToFloat32(), q1.z.ToFloat32(), q1.w.ToFloat32()};
    Common::Vec4f b{q2.x.ToFloat32(), q2.y.ToFloat32(), q2.z.ToFloat32(), q2.w.ToFloat32()};

    return (Common::Dot(a, b) < 0.f);
}

// vFACET trace (BORKED3DS_TRACE_QUAT=1) : mesure, pour un echantillon de triangles, l'ecart L1
// entre les quaternions de normale des 3 sommets. Ecart ~0 => les 3 sommets partagent le meme
// quaternion = normales de FACE (ombrage plat, cause du facettage). Ecart reel (>~0.01) => les
// normales sont distinctes par sommet et le bandage vient d'ailleurs (LUT d'eclairage). Journalise
// des stats cumulatives + quelques echantillons bruts pour lecture directe. Aucun cout si la
// variable d'env est absente.
static float QuatSpreadL1(const Pica::OutputVertex& a, const Pica::OutputVertex& b) {
    const auto qa = a.quat();
    const auto qb = b.quat();
    return std::abs(qa.x.ToFloat32() - qb.x.ToFloat32()) +
           std::abs(qa.y.ToFloat32() - qb.y.ToFloat32()) +
           std::abs(qa.z.ToFloat32() - qb.z.ToFloat32()) +
           std::abs(qa.w.ToFloat32() - qb.w.ToFloat32());
}

void RasterizerAccelerated::AddTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                        const Pica::OutputVertex& v2) {
    static const bool s_trace_quat = (std::getenv("BORKED3DS_TRACE_QUAT") != nullptr);
    if (s_trace_quat) {
        // Separe les draws ECLAIRES (lighting.disable == 0) des NON eclaires. La question decisive :
        // les draws eclaires (la boule) recoivent-ils un quaternion NUL (0,0,0,0) ? Un quaternion
        // nul sur un objet eclaire => normale degeneree => facettage. Sur un objet non eclaire il
        // est inoffensif (jamais lu). q0len = norme L1 de q0 ; < 0.01 => quaternion nul.
        const bool lit = (regs.lighting.disable == 0);
        const auto q0 = v0.quat();
        const float q0len = std::abs(q0.x.ToFloat32()) + std::abs(q0.y.ToFloat32()) +
                            std::abs(q0.z.ToFloat32()) + std::abs(q0.w.ToFloat32());
        const bool zero = (q0len < 0.01f);
        const float spread = std::max(QuatSpreadL1(v0, v1), QuatSpreadL1(v0, v2));

        static std::atomic<u64> lit_total{0}, lit_zero{0}, unlit_total{0}, unlit_zero{0};
        if (lit) {
            const u64 t = ++lit_total;
            if (zero) {
                ++lit_zero;
            }
            if ((t % 200) == 0) {
                LOG_INFO(Render,
                         "TRACE_QUAT LIT sample tri={} q0len={:.3f} spread={:.3f} "
                         "q0=({:.3f},{:.3f},{:.3f},{:.3f})",
                         t, q0len, spread, q0.x.ToFloat32(), q0.y.ToFloat32(), q0.z.ToFloat32(),
                         q0.w.ToFloat32());
            }
            if ((t % 2000) == 0) {
                LOG_INFO(Render,
                         "TRACE_QUAT LIT stats total={} zero_quat={} frac_zero={:.3f} "
                         "(frac_zero eleve => objets ECLAIRES avec quaternion NUL = la cause du "
                         "facettage)",
                         t, lit_zero.load(),
                         static_cast<double>(lit_zero.load()) / static_cast<double>(t));
            }
        } else {
            const u64 t = ++unlit_total;
            if (zero) {
                ++unlit_zero;
            }
            if ((t % 5000) == 0) {
                LOG_INFO(Render,
                         "TRACE_QUAT UNLIT stats total={} zero_quat={} frac_zero={:.3f} (non "
                         "eclaire : un quaternion nul y est inoffensif)",
                         t, unlit_zero.load(),
                         static_cast<double>(unlit_zero.load()) / static_cast<double>(t));
            }
        }
    }
    vertex_batch.emplace_back(v0, false);
    vertex_batch.emplace_back(v1, AreQuaternionsOpposite(v0, v1));
    vertex_batch.emplace_back(v2, AreQuaternionsOpposite(v0, v2));
}

RasterizerAccelerated::VertexArrayInfo RasterizerAccelerated::AnalyzeVertexArray(
    bool is_indexed, u32 stride_alignment) {
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;

    u32 vertex_min;
    u32 vertex_max;
    if (is_indexed) {
        const auto& index_info = regs.pipeline.index_array;
        const PAddr address = vertex_attributes.GetPhysicalBaseAddress() + index_info.offset;
        const u8* index_address_8 = memory.GetPhysicalPointer(address);
        const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
        const bool index_u16 = index_info.format != 0;

        vertex_min = 0xFFFF;
        vertex_max = 0;
        const u32 size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);
        FlushRegion(address, size);
        for (u32 index = 0; index < regs.pipeline.num_vertices; ++index) {
            const u32 vertex = index_u16 ? index_address_16[index] : index_address_8[index];
            vertex_min = std::min(vertex_min, vertex);
            vertex_max = std::max(vertex_max, vertex);
        }
    } else {
        vertex_min = regs.pipeline.vertex_offset;
        vertex_max = regs.pipeline.vertex_offset + regs.pipeline.num_vertices - 1;
    }

    const u32 vertex_num = vertex_max - vertex_min + 1;
    u32 vs_input_size = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count != 0) {
            const u32 aligned_stride =
                Common::AlignUp(static_cast<u32>(loader.byte_count), stride_alignment);
            vs_input_size += Common::AlignUp(aligned_stride * vertex_num, 4);
        }
    }

    return {vertex_min, vertex_max, vs_input_size};
}

void RasterizerAccelerated::SyncEntireState() {
    // Sync renderer-specific fixed-function state
    SyncFixedState();

    // Sync uniforms
    SyncClipPlane();
    SyncDepthScale();
    SyncDepthOffset();
    SyncAlphaTest();
    SyncCombinerColor();
    auto& tev_stages = regs.texturing.GetTevStages();
    for (std::size_t index = 0; index < tev_stages.size(); ++index) {
        SyncTevConstColor(index, tev_stages[index]);
    }

    SyncGlobalAmbient();
    for (u32 light_index = 0; light_index < 8; light_index++) {
        SyncLightSpecular0(light_index);
        SyncLightSpecular1(light_index);
        SyncLightDiffuse(light_index);
        SyncLightAmbient(light_index);
        SyncLightPosition(light_index);
        SyncLightDistanceAttenuationBias(light_index);
        SyncLightDistanceAttenuationScale(light_index);
    }

    SyncFogColor();
    SyncProcTexNoise();
    SyncProcTexBias();
    SyncShadowBias();
    SyncShadowTextureBias();

    for (u32 tex_index = 0; tex_index < 3; tex_index++) {
        SyncTextureLodBias(tex_index);
    }
}

void RasterizerAccelerated::NotifyPicaRegisterChanged(u32 id) {
    switch (id) {
    // Depth modifiers
    case PICA_REG_INDEX(rasterizer.viewport_depth_range):
        SyncDepthScale();
        break;
    case PICA_REG_INDEX(rasterizer.viewport_depth_near_plane):
        SyncDepthOffset();
        break;

    // Depth buffering
    case PICA_REG_INDEX(rasterizer.depthmap_enable):
        shader_dirty = true;
        break;

    // Shadow texture
    case PICA_REG_INDEX(texturing.shadow):
        SyncShadowTextureBias();
        break;

    // Fog state
    case PICA_REG_INDEX(texturing.fog_color):
        SyncFogColor();
        break;
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        fs_uniform_block_data.fog_lut_dirty = true;
        break;

    // ProcTex state
    case PICA_REG_INDEX(texturing.proctex):
    case PICA_REG_INDEX(texturing.proctex_lut):
    case PICA_REG_INDEX(texturing.proctex_lut_offset):
        SyncProcTexBias();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.proctex_noise_u):
    case PICA_REG_INDEX(texturing.proctex_noise_v):
    case PICA_REG_INDEX(texturing.proctex_noise_frequency):
        SyncProcTexNoise();
        break;

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]):
        using Pica::TexturingRegs;
        switch (regs.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            fs_uniform_block_data.proctex_noise_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            fs_uniform_block_data.proctex_color_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            fs_uniform_block_data.proctex_alpha_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            fs_uniform_block_data.proctex_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            fs_uniform_block_data.proctex_diff_lut_dirty = true;
            break;
        }
        break;

    // Fragment operation mode
    case PICA_REG_INDEX(framebuffer.output_merger.fragment_operation_mode):
        shader_dirty = true;
        break;

    // Alpha test
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_test):
        SyncAlphaTest();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(framebuffer.shadow):
        SyncShadowBias();
        break;

    // Scissor test
    case PICA_REG_INDEX(rasterizer.scissor_test.mode):
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.main_config):
        shader_dirty = true;
        break;

    // Texture 0 type
    case PICA_REG_INDEX(texturing.texture0.type):
        shader_dirty = true;
        break;

    // TEV stages
    // (This also syncs fog_mode and fog_flip which are part of tev_combiner_buffer_input)
    case PICA_REG_INDEX(texturing.tev_stage0.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_op):
    case PICA_REG_INDEX(texturing.tev_stage0.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage1.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_op):
    case PICA_REG_INDEX(texturing.tev_stage1.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage2.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_op):
    case PICA_REG_INDEX(texturing.tev_stage2.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage3.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_op):
    case PICA_REG_INDEX(texturing.tev_stage3.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage4.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_op):
    case PICA_REG_INDEX(texturing.tev_stage4.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage5.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_op):
    case PICA_REG_INDEX(texturing.tev_stage5.color_scale):
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_input):
        shader_dirty = true;
        break;
    case PICA_REG_INDEX(texturing.tev_stage0.const_r):
        SyncTevConstColor(0, regs.texturing.tev_stage0);
        break;
    case PICA_REG_INDEX(texturing.tev_stage1.const_r):
        SyncTevConstColor(1, regs.texturing.tev_stage1);
        break;
    case PICA_REG_INDEX(texturing.tev_stage2.const_r):
        SyncTevConstColor(2, regs.texturing.tev_stage2);
        break;
    case PICA_REG_INDEX(texturing.tev_stage3.const_r):
        SyncTevConstColor(3, regs.texturing.tev_stage3);
        break;
    case PICA_REG_INDEX(texturing.tev_stage4.const_r):
        SyncTevConstColor(4, regs.texturing.tev_stage4);
        break;
    case PICA_REG_INDEX(texturing.tev_stage5.const_r):
        SyncTevConstColor(5, regs.texturing.tev_stage5);
        break;

    // TEV combiner buffer color
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_color):
        SyncCombinerColor();
        break;

    // Fragment lighting switches
    case PICA_REG_INDEX(lighting.disable):
    case PICA_REG_INDEX(lighting.max_light_index):
    case PICA_REG_INDEX(lighting.config0):
    case PICA_REG_INDEX(lighting.config1):
    case PICA_REG_INDEX(lighting.abs_lut_input):
    case PICA_REG_INDEX(lighting.lut_input):
    case PICA_REG_INDEX(lighting.lut_scale):
    case PICA_REG_INDEX(lighting.light_enable):
        break;

    // Fragment lighting specular 0 color
    case PICA_REG_INDEX(lighting.light[0].specular_0):
        SyncLightSpecular0(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_0):
        SyncLightSpecular0(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_0):
        SyncLightSpecular0(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_0):
        SyncLightSpecular0(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_0):
        SyncLightSpecular0(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_0):
        SyncLightSpecular0(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_0):
        SyncLightSpecular0(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_0):
        SyncLightSpecular0(7);
        break;

    // Fragment lighting specular 1 color
    case PICA_REG_INDEX(lighting.light[0].specular_1):
        SyncLightSpecular1(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_1):
        SyncLightSpecular1(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_1):
        SyncLightSpecular1(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_1):
        SyncLightSpecular1(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_1):
        SyncLightSpecular1(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_1):
        SyncLightSpecular1(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_1):
        SyncLightSpecular1(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_1):
        SyncLightSpecular1(7);
        break;

    // Fragment lighting diffuse color
    case PICA_REG_INDEX(lighting.light[0].diffuse):
        SyncLightDiffuse(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].diffuse):
        SyncLightDiffuse(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].diffuse):
        SyncLightDiffuse(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].diffuse):
        SyncLightDiffuse(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].diffuse):
        SyncLightDiffuse(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].diffuse):
        SyncLightDiffuse(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].diffuse):
        SyncLightDiffuse(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].diffuse):
        SyncLightDiffuse(7);
        break;

    // Fragment lighting ambient color
    case PICA_REG_INDEX(lighting.light[0].ambient):
        SyncLightAmbient(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].ambient):
        SyncLightAmbient(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].ambient):
        SyncLightAmbient(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].ambient):
        SyncLightAmbient(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].ambient):
        SyncLightAmbient(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].ambient):
        SyncLightAmbient(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].ambient):
        SyncLightAmbient(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].ambient):
        SyncLightAmbient(7);
        break;

    // Fragment lighting position
    case PICA_REG_INDEX(lighting.light[0].x):
    case PICA_REG_INDEX(lighting.light[0].z):
        SyncLightPosition(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].x):
    case PICA_REG_INDEX(lighting.light[1].z):
        SyncLightPosition(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].x):
    case PICA_REG_INDEX(lighting.light[2].z):
        SyncLightPosition(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].x):
    case PICA_REG_INDEX(lighting.light[3].z):
        SyncLightPosition(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].x):
    case PICA_REG_INDEX(lighting.light[4].z):
        SyncLightPosition(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].x):
    case PICA_REG_INDEX(lighting.light[5].z):
        SyncLightPosition(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].x):
    case PICA_REG_INDEX(lighting.light[6].z):
        SyncLightPosition(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].x):
    case PICA_REG_INDEX(lighting.light[7].z):
        SyncLightPosition(7);
        break;

    // Fragment spot lighting direction
    case PICA_REG_INDEX(lighting.light[0].spot_x):
    case PICA_REG_INDEX(lighting.light[0].spot_z):
        SyncLightSpotDirection(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].spot_x):
    case PICA_REG_INDEX(lighting.light[1].spot_z):
        SyncLightSpotDirection(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].spot_x):
    case PICA_REG_INDEX(lighting.light[2].spot_z):
        SyncLightSpotDirection(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].spot_x):
    case PICA_REG_INDEX(lighting.light[3].spot_z):
        SyncLightSpotDirection(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].spot_x):
    case PICA_REG_INDEX(lighting.light[4].spot_z):
        SyncLightSpotDirection(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].spot_x):
    case PICA_REG_INDEX(lighting.light[5].spot_z):
        SyncLightSpotDirection(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].spot_x):
    case PICA_REG_INDEX(lighting.light[6].spot_z):
        SyncLightSpotDirection(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].spot_x):
    case PICA_REG_INDEX(lighting.light[7].spot_z):
        SyncLightSpotDirection(7);
        break;

    // Fragment lighting light source config
    case PICA_REG_INDEX(lighting.light[0].config):
    case PICA_REG_INDEX(lighting.light[1].config):
    case PICA_REG_INDEX(lighting.light[2].config):
    case PICA_REG_INDEX(lighting.light[3].config):
    case PICA_REG_INDEX(lighting.light[4].config):
    case PICA_REG_INDEX(lighting.light[5].config):
    case PICA_REG_INDEX(lighting.light[6].config):
    case PICA_REG_INDEX(lighting.light[7].config):
        shader_dirty = true;
        break;

    // Fragment lighting distance attenuation bias
    case PICA_REG_INDEX(lighting.light[0].dist_atten_bias):
        SyncLightDistanceAttenuationBias(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_bias):
        SyncLightDistanceAttenuationBias(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_bias):
        SyncLightDistanceAttenuationBias(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_bias):
        SyncLightDistanceAttenuationBias(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_bias):
        SyncLightDistanceAttenuationBias(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_bias):
        SyncLightDistanceAttenuationBias(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_bias):
        SyncLightDistanceAttenuationBias(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_bias):
        SyncLightDistanceAttenuationBias(7);
        break;

    // Fragment lighting distance attenuation scale
    case PICA_REG_INDEX(lighting.light[0].dist_atten_scale):
        SyncLightDistanceAttenuationScale(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_scale):
        SyncLightDistanceAttenuationScale(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_scale):
        SyncLightDistanceAttenuationScale(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_scale):
        SyncLightDistanceAttenuationScale(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_scale):
        SyncLightDistanceAttenuationScale(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_scale):
        SyncLightDistanceAttenuationScale(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_scale):
        SyncLightDistanceAttenuationScale(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_scale):
        SyncLightDistanceAttenuationScale(7);
        break;

    // Fragment lighting global ambient color (emission + ambient * ambient)
    case PICA_REG_INDEX(lighting.global_ambient):
        SyncGlobalAmbient();
        break;

    // Fragment lighting lookup tables
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        const auto& lut_config = regs.lighting.lut_config;
        fs_uniform_block_data.lighting_lut_dirty[lut_config.type] = true;
        fs_uniform_block_data.lighting_lut_dirty_any = true;
        // TG10 : compte les ecritures du jeu dans chaque LUT. Inerte hors sonde TG09.
        if (TG09Level() != 0) {
            const u32 lut_type = lut_config.type.Value();
            if (lut_type < Pica::LightingRegs::NumLightingSampler) {
                ++tg10_lut_writes[lut_type];
            }
        }
        break;
    }

    // Texture LOD biases
    case PICA_REG_INDEX(texturing.texture0.lod.bias):
        SyncTextureLodBias(0);
        break;
    case PICA_REG_INDEX(texturing.texture1.lod.bias):
        SyncTextureLodBias(1);
        break;
    case PICA_REG_INDEX(texturing.texture2.lod.bias):
        SyncTextureLodBias(2);
        break;

    // Texture borders
    case PICA_REG_INDEX(texturing.texture0.border_color):
        SyncTextureBorderColor(0);
        break;
    case PICA_REG_INDEX(texturing.texture1.border_color):
        SyncTextureBorderColor(1);
        break;
    case PICA_REG_INDEX(texturing.texture2.border_color):
        SyncTextureBorderColor(2);
        break;

    // Clipping plane
    case PICA_REG_INDEX(rasterizer.clip_enable):
    case PICA_REG_INDEX(rasterizer.clip_coef[0]):
    case PICA_REG_INDEX(rasterizer.clip_coef[1]):
    case PICA_REG_INDEX(rasterizer.clip_coef[2]):
    case PICA_REG_INDEX(rasterizer.clip_coef[3]):
        SyncClipPlane();
        break;
    }

    // Forward registers that map to fixed function API features to the video backend
    NotifyFixedFunctionPicaRegisterChanged(id);
}

void RasterizerAccelerated::SyncDepthScale() {
    const f32 depth_scale = f24::FromRaw(regs.rasterizer.viewport_depth_range).ToFloat32();

    if (depth_scale != fs_uniform_block_data.data.depth_scale) {
        fs_uniform_block_data.data.depth_scale = depth_scale;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncDepthOffset() {
    const f32 depth_offset = f24::FromRaw(regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    if (depth_offset != fs_uniform_block_data.data.depth_offset) {
        fs_uniform_block_data.data.depth_offset = depth_offset;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncFogColor() {
    const auto& fog_color_regs = regs.texturing.fog_color;
    const Common::Vec3f fog_color = {
        fog_color_regs.r.Value() / 255.0f,
        fog_color_regs.g.Value() / 255.0f,
        fog_color_regs.b.Value() / 255.0f,
    };

    if (fog_color != fs_uniform_block_data.data.fog_color) {
        fs_uniform_block_data.data.fog_color = fog_color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncProcTexNoise() {
    const Common::Vec2f proctex_noise_f = {
        Pica::f16::FromRaw(regs.texturing.proctex_noise_frequency.u).ToFloat32(),
        Pica::f16::FromRaw(regs.texturing.proctex_noise_frequency.v).ToFloat32(),
    };
    const Common::Vec2f proctex_noise_a = {
        regs.texturing.proctex_noise_u.amplitude / 4095.0f,
        regs.texturing.proctex_noise_v.amplitude / 4095.0f,
    };
    const Common::Vec2f proctex_noise_p = {
        Pica::f16::FromRaw(regs.texturing.proctex_noise_u.phase).ToFloat32(),
        Pica::f16::FromRaw(regs.texturing.proctex_noise_v.phase).ToFloat32(),
    };

    if (proctex_noise_f != fs_uniform_block_data.data.proctex_noise_f ||
        proctex_noise_a != fs_uniform_block_data.data.proctex_noise_a ||
        proctex_noise_p != fs_uniform_block_data.data.proctex_noise_p) {
        fs_uniform_block_data.data.proctex_noise_f = proctex_noise_f;
        fs_uniform_block_data.data.proctex_noise_a = proctex_noise_a;
        fs_uniform_block_data.data.proctex_noise_p = proctex_noise_p;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncProcTexBias() {
    const auto proctex_bias = Pica::f16::FromRaw(regs.texturing.proctex.bias_low |
                                                 (regs.texturing.proctex_lut.bias_high << 8))
                                  .ToFloat32();
    if (proctex_bias != fs_uniform_block_data.data.proctex_bias) {
        fs_uniform_block_data.data.proctex_bias = proctex_bias;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncAlphaTest() {
    if (regs.framebuffer.output_merger.alpha_test.ref !=
        static_cast<u32>(fs_uniform_block_data.data.alphatest_ref)) {
        fs_uniform_block_data.data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncCombinerColor() {
    const auto combiner_color = ColorRGBA8(regs.texturing.tev_combiner_buffer_color.raw);
    if (combiner_color != fs_uniform_block_data.data.tev_combiner_buffer_color) {
        fs_uniform_block_data.data.tev_combiner_buffer_color = combiner_color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTevConstColor(
    const std::size_t stage_index, const Pica::TexturingRegs::TevStageConfig& tev_stage) {
    const auto const_color = ColorRGBA8(tev_stage.const_color);

    if (const_color == fs_uniform_block_data.data.const_color[stage_index]) {
        return;
    }

    fs_uniform_block_data.data.const_color[stage_index] = const_color;
    fs_uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncGlobalAmbient() {
    const auto color = LightColor(regs.lighting.global_ambient);
    if (color != fs_uniform_block_data.data.lighting_global_ambient) {
        fs_uniform_block_data.data.lighting_global_ambient = color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular0(int light_index) {
    const auto color = LightColor(regs.lighting.light[light_index].specular_0);
    if (color != fs_uniform_block_data.data.light_src[light_index].specular_0) {
        fs_uniform_block_data.data.light_src[light_index].specular_0 = color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular1(int light_index) {
    const auto color = LightColor(regs.lighting.light[light_index].specular_1);
    if (color != fs_uniform_block_data.data.light_src[light_index].specular_1) {
        fs_uniform_block_data.data.light_src[light_index].specular_1 = color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDiffuse(int light_index) {
    const auto color = LightColor(regs.lighting.light[light_index].diffuse);
    if (color != fs_uniform_block_data.data.light_src[light_index].diffuse) {
        fs_uniform_block_data.data.light_src[light_index].diffuse = color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightAmbient(int light_index) {
    const auto color = LightColor(regs.lighting.light[light_index].ambient);
    if (color != fs_uniform_block_data.data.light_src[light_index].ambient) {
        fs_uniform_block_data.data.light_src[light_index].ambient = color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightPosition(int light_index) {
    const Common::Vec3f position = {
        Pica::f16::FromRaw(regs.lighting.light[light_index].x).ToFloat32(),
        Pica::f16::FromRaw(regs.lighting.light[light_index].y).ToFloat32(),
        Pica::f16::FromRaw(regs.lighting.light[light_index].z).ToFloat32(),
    };

    if (position != fs_uniform_block_data.data.light_src[light_index].position) {
        fs_uniform_block_data.data.light_src[light_index].position = position;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpotDirection(int light_index) {
    const auto& light = regs.lighting.light[light_index];
    const auto spot_direction =
        Common::Vec3f{light.spot_x / 2047.0f, light.spot_y / 2047.0f, light.spot_z / 2047.0f};

    if (spot_direction != fs_uniform_block_data.data.light_src[light_index].spot_direction) {
        fs_uniform_block_data.data.light_src[light_index].spot_direction = spot_direction;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationBias(int light_index) {
    const f32 dist_atten_bias =
        Pica::f20::FromRaw(regs.lighting.light[light_index].dist_atten_bias).ToFloat32();

    if (dist_atten_bias != fs_uniform_block_data.data.light_src[light_index].dist_atten_bias) {
        fs_uniform_block_data.data.light_src[light_index].dist_atten_bias = dist_atten_bias;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationScale(int light_index) {
    const f32 dist_atten_scale =
        Pica::f20::FromRaw(regs.lighting.light[light_index].dist_atten_scale).ToFloat32();

    if (dist_atten_scale != fs_uniform_block_data.data.light_src[light_index].dist_atten_scale) {
        fs_uniform_block_data.data.light_src[light_index].dist_atten_scale = dist_atten_scale;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowBias() {
    const auto& shadow = regs.framebuffer.shadow;
    const f32 constant = Pica::f16::FromRaw(shadow.constant).ToFloat32();
    const f32 linear = Pica::f16::FromRaw(shadow.linear).ToFloat32();

    if (constant != fs_uniform_block_data.data.shadow_bias_constant ||
        linear != fs_uniform_block_data.data.shadow_bias_linear) {
        fs_uniform_block_data.data.shadow_bias_constant = constant;
        fs_uniform_block_data.data.shadow_bias_linear = linear;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowTextureBias() {
    const s32 bias = regs.texturing.shadow.bias << 1;
    if (bias != fs_uniform_block_data.data.shadow_texture_bias) {
        fs_uniform_block_data.data.shadow_texture_bias = bias;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTextureLodBias(int tex_index) {
    const auto pica_textures = regs.texturing.GetTextures();
    const f32 bias = pica_textures[tex_index].config.lod.bias / 256.0f;
    if (bias != fs_uniform_block_data.data.tex_lod_bias[tex_index]) {
        fs_uniform_block_data.data.tex_lod_bias[tex_index] = bias;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTextureBorderColor(int tex_index) {
    const auto pica_textures = regs.texturing.GetTextures();
    const auto params = pica_textures[tex_index].config;
    const Common::Vec4f border_color = ColorRGBA8(params.border_color.raw);
    if (border_color != fs_uniform_block_data.data.tex_border_color[tex_index]) {
        fs_uniform_block_data.data.tex_border_color[tex_index] = border_color;
        fs_uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncClipPlane() {
    const u32 enable_clip1 = regs.rasterizer.clip_enable != 0;
    const auto raw_clip_coef = regs.rasterizer.GetClipCoef();
    const Common::Vec4f new_clip_coef = {raw_clip_coef.x.ToFloat32(), raw_clip_coef.y.ToFloat32(),
                                         raw_clip_coef.z.ToFloat32(), raw_clip_coef.w.ToFloat32()};
    if (enable_clip1 != vs_uniform_block_data.data.enable_clip1 ||
        new_clip_coef != vs_uniform_block_data.data.clip_coef) {
        vs_uniform_block_data.data.enable_clip1 = enable_clip1;
        vs_uniform_block_data.data.clip_coef = new_clip_coef;
        vs_uniform_block_data.dirty = true;
    }
}

// ---------------------------------------------------------------------------------------------
// TG09 (BORKED3DS_TG09_LIGHT_DUMP=1|2|3) -- sonde de MESURE, inerte hors variable d'environnement.
//
// Motif (RECAP v163, section 6, point 2). Apres quatorze eliminations, il ne reste qu'une seule
// famille d'explications non instrumentee pour le defaut de reflexion du vaisseau de Metroid :
//
//   - la SOURCE des shaders est commune aux deux backends (audit v162bis, faits F1 a F3) ;
//   - la CONFIGURATION de shader est identique entre save state et partie depuis le debut
//     (sonde TG05, section 2.5 du RECAP v163) ;
//   - il reste donc les DONNEES : contenu reel des LUT de reflexion RR / RG / RB, parametres des
//     sources de lumiere, ambiante globale, et leur chemin de televersement.
//
// La sonde vit dans la classe de base COMMUNE aux deux backends (RasterizerAccelerated), donc le
// formatage est rigoureusement identique en OpenGL/GLES et en Vulkan : un `diff` de deux logs
// repond directement a la question. Elle est appelee depuis SyncAndUploadLUTsLF() de chaque
// backend, c'est-a-dire au moment exact du televersement, et lit :
//
//   - lighting_lut_data[]                     = ce qui a REELLEMENT ete televerse (pas la source) ;
//   - fs_uniform_block_data.data.light_src[]  = ce qui atteint REELLEMENT le fragment shader ;
//   - fs_uniform_block_data.data.lighting_lut_offset[] = l'adressage, seul element structurellement
//     divergent entre les deux backends (TBO Vulkan vs TBO/texture 2D OpenGL).
//
// Trois usages prevus, dans cet ordre :
//
//   1. Vulkan vs OpenGL/GLES, meme scene : si les hachages de LUT et les light_src sont identiques,
//      les donnees sont innocentees et le mur se deplace vers l'adressage (champ `off=`) ou vers
//      l'echantillonnage cote shader. S'ils different, la cause racine est trouvee.
//   2. Save state vs partie depuis le debut, MEME backend : repond au point 2 de la section 6 du
//      RECAP v163 (les ~50 % de triangles en moins du save state).
//   3. Attribution du gain 60-70 % : rejouer deux binaires avec la meme sonde et diffter.
//
// Dedupliquee par signature de contenu (configuration + lumieres actives + hachage des LUT
// pertinentes) : quelques lignes par session, pas un flot. Plafond dur de TG09_MAX_SIGNATURES
// signatures distinctes pour borner la memoire et le log.
//
// Niveaux :
//   1 = configuration + lumieres + une ligne de resume par LUT pertinente (hachage, min/max/moyenne,
//       cinq points d'echantillonnage, offset d'adressage) ;
//   2 = niveau 1 + dump integral des 256 entrees des LUT de reflexion RR / RG / RB ;
//   3 = niveau 2 + dump integral de TOUTES les LUT pertinentes (verbeux, diagnostic ponctuel).
//
// Cout par defaut : une lecture de bool statique. Aucun effet sur le rendu, a aucun niveau.
// ---------------------------------------------------------------------------------------------
namespace {

u32 TG09Level() {
    static const u32 level = []() -> u32 {
        const char* const value = std::getenv("BORKED3DS_TG09_LIGHT_DUMP");
        if (value == nullptr) {
            return 0u;
        }
        const int parsed = std::atoi(value);
        return parsed <= 0 ? 0u : static_cast<u32>(parsed);
    }();
    return level;
}

constexpr u64 TG09_FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr u64 TG09_FNV_PRIME = 0x100000001b3ULL;
constexpr std::size_t TG09_MAX_SIGNATURES = 64;
constexpr u64 TG09_OFFSET_PERIOD = 1024;

/// FNV-1a par mots de 32 bits (assez rapide pour tourner sur le chemin de televersement).
u64 TG09HashWords(const void* data, std::size_t size_bytes, u64 seed) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    const std::size_t count = size_bytes / sizeof(u32);
    u64 hash = seed;
    for (std::size_t i = 0; i < count; ++i) {
        u32 word = 0;
        std::memcpy(&word, bytes + i * sizeof(u32), sizeof(word));
        hash ^= static_cast<u64>(word);
        hash *= TG09_FNV_PRIME;
    }
    return hash;
}

u64 TG09HashU64(u64 value, u64 seed) {
    u64 hash = seed;
    hash ^= value;
    hash *= TG09_FNV_PRIME;
    return hash;
}

/// Retourne true une seule fois par signature de contenu, et jamais au-dela du plafond.
bool TG09ShouldLog(u64 key) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    std::scoped_lock lock{mutex};
    if (seen.size() >= TG09_MAX_SIGNATURES) {
        return false;
    }
    return seen.insert(key).second;
}

const char* TG09SamplerName(u32 index) {
    using Sampler = Pica::LightingRegs::LightingSampler;
    if (index >= static_cast<u32>(Sampler::DistanceAttenuation)) {
        return "DA";
    }
    if (index >= static_cast<u32>(Sampler::SpotlightAttenuation)) {
        return "SP";
    }
    switch (static_cast<Sampler>(index)) {
    case Sampler::Distribution0:
        return "D0";
    case Sampler::Distribution1:
        return "D1";
    case Sampler::Fresnel:
        return "FR";
    case Sampler::ReflectBlue:
        return "RB";
    case Sampler::ReflectGreen:
        return "RG";
    case Sampler::ReflectRed:
        return "RR";
    default:
        return "??";
    }
}

const char* TG09LutInputName(Pica::LightingRegs::LightingLutInput input) {
    using Input = Pica::LightingRegs::LightingLutInput;
    switch (input) {
    case Input::NH:
        return "NH";
    case Input::VH:
        return "VH";
    case Input::NV:
        return "NV";
    case Input::LN:
        return "LN";
    case Input::SP:
        return "SP";
    case Input::CP:
        return "CP";
    default:
        return "??";
    }
}

} // Anonymous namespace

u32 RasterizerAccelerated::TG10ForceLutUploadLevel() {
    static const u32 level = []() -> u32 {
        const char* const value = std::getenv("BORKED3DS_TG10_FORCE_LUT_UPLOAD");
        if (value == nullptr) {
            return 0u;
        }
        const int parsed = std::atoi(value);
        return parsed <= 0 ? 0u : static_cast<u32>(parsed);
    }();
    return level;
}

void RasterizerAccelerated::TG09LogLightingState(const char* backend, const char* path,
                                                 u64 map_offset, u64 bytes_used) {
    const u32 level = TG09Level();
    if (level == 0) {
        return;
    }

    using LightingRegs = Pica::LightingRegs;
    using Sampler = LightingRegs::LightingSampler;

    const auto& lighting = regs.lighting;
    const auto config = lighting.config0.config.Value();
    const u32 num_lights = lighting.max_light_index.Value() + 1u;

    // --- 1. Quelles LUT comptent pour CETTE configuration ? ------------------------------------
    // On ne hache et n'affiche que celles-la : hacher les 24 a chaque televersement couterait
    // trop cher sur le chemin chaud, et les LUT non selectionnees ne sont jamais lues par le FS.
    std::array<bool, LightingRegs::NumLightingSampler> relevant{};
    const std::array<Sampler, 6> fixed_samplers{
        Sampler::Distribution0, Sampler::Distribution1, Sampler::Fresnel,
        Sampler::ReflectBlue,   Sampler::ReflectGreen,  Sampler::ReflectRed,
    };
    for (const auto sampler : fixed_samplers) {
        if (LightingRegs::IsLightingSamplerSupported(config, sampler)) {
            relevant[static_cast<std::size_t>(sampler)] = true;
        }
    }
    const bool spot_supported =
        LightingRegs::IsLightingSamplerSupported(config, Sampler::SpotlightAttenuation);
    for (u32 slot = 0; slot < num_lights; ++slot) {
        const u32 light_index = lighting.light_enable.GetNum(slot);
        if (spot_supported && !lighting.IsSpotAttenDisabled(light_index)) {
            relevant[static_cast<std::size_t>(Sampler::SpotlightAttenuation) + light_index] = true;
        }
        if (!lighting.IsDistAttenDisabled(light_index)) {
            relevant[static_cast<std::size_t>(Sampler::DistanceAttenuation) + light_index] = true;
        }
    }

    // --- 2. Signature de contenu : registres + lumieres actives + contenu reel des LUT ----------
    // Volontairement SANS les offsets de televersement : ils bougent a chaque frame (buffer en
    // flot) et noieraient la deduplication. Ils sont echantillonnes separement (TG09_OFFSETS).
    u64 key = TG09HashU64(static_cast<u64>(static_cast<unsigned char>(backend[0])),
                          TG09_FNV_OFFSET);
    key = TG09HashU64(lighting.disable.Value(), key);
    key = TG09HashWords(&lighting.config0, sizeof(lighting.config0), key);
    key = TG09HashWords(&lighting.config1, sizeof(lighting.config1), key);
    key = TG09HashWords(&lighting.abs_lut_input, sizeof(lighting.abs_lut_input), key);
    key = TG09HashWords(&lighting.lut_input, sizeof(lighting.lut_input), key);
    key = TG09HashWords(&lighting.lut_scale, sizeof(lighting.lut_scale), key);
    key = TG09HashWords(&lighting.light_enable, sizeof(lighting.light_enable), key);
    key = TG09HashWords(&lighting.global_ambient, sizeof(lighting.global_ambient), key);
    key = TG09HashU64(num_lights, key);
    for (u32 slot = 0; slot < num_lights; ++slot) {
        const u32 light_index = lighting.light_enable.GetNum(slot);
        key = TG09HashWords(&lighting.light[light_index], sizeof(lighting.light[light_index]), key);
        key = TG09HashWords(&fs_uniform_block_data.data.light_src[light_index],
                            sizeof(fs_uniform_block_data.data.light_src[light_index]), key);
    }

    std::array<u64, LightingRegs::NumLightingSampler> lut_hash{};
    for (u32 index = 0; index < LightingRegs::NumLightingSampler; ++index) {
        if (!relevant[index]) {
            continue;
        }
        lut_hash[index] = TG09HashWords(lighting_lut_data[index].data(),
                                        lighting_lut_data[index].size() * sizeof(Common::Vec2f),
                                        TG09_FNV_OFFSET);
        key = TG09HashU64(lut_hash[index], key);
    }

    // --- 3. Echantillonnage periodique des offsets, independant de la deduplication -------------
    // Seul point structurellement divergent entre les backends : Vulkan adresse un texel buffer,
    // OpenGL adresse soit un texel buffer, soit une texture 2D (repli GLES sans
    // GL_OES_texture_buffer, ou l'offset devient x + y * 1024). Une divergence d'adressage ferait
    // lire la BONNE LUT au MAUVAIS endroit -- exactement le genre de defaut qui epargne l'ombre et
    // casse le reflet.
    {
        static std::atomic<u64> counter{0};
        const u64 n = counter.fetch_add(1, std::memory_order_relaxed);
        if ((n % TG09_OFFSET_PERIOD) == 0) {
            std::string offsets;
            offsets.reserve(256);
            for (u32 index = 0; index < LightingRegs::NumLightingSampler; ++index) {
                if (!relevant[index]) {
                    continue;
                }
                offsets += fmt::format(
                    "{}:{} ", TG09SamplerName(index),
                    fs_uniform_block_data.data.lighting_lut_offset[index / 4][index % 4]);
            }
            LOG_INFO(Render,
                     "TG09_OFFSETS [{}/{}] n={} map_off={} bytes={} use_tex2d_lut={} off({})",
                     backend, path, n, map_offset, bytes_used,
                     fs_uniform_block_data.data.use_texture2d_lut, offsets);
        }
    }

    if (!TG09ShouldLog(key)) {
        return;
    }

    // --- 4. Configuration d'eclairage ------------------------------------------------------------
    std::string slots;
    slots.reserve(32);
    for (u32 slot = 0; slot < num_lights; ++slot) {
        slots += fmt::format("{}{}", slot == 0 ? "" : ",", lighting.light_enable.GetNum(slot));
    }
    const auto& ambient = fs_uniform_block_data.data.lighting_global_ambient;

    LOG_INFO(Render,
             "TG09_LIGHT [{}/{}] sig={:#018x} disable={} config={} lights={} slots=({}) "
             "bump_mode={} bump_sel={} renorm_off={} clamp_hl={} shadow={} prim_alpha={} "
             "sec_alpha={} global_ambient=({:.6f},{:.6f},{:.6f}) map_off={} bytes={} "
             "use_tex2d_lut={}",
             backend, path, key, lighting.disable.Value(), static_cast<u32>(config), num_lights,
             slots, static_cast<u32>(lighting.config0.bump_mode.Value()),
             lighting.config0.bump_selector.Value(),
             lighting.config0.disable_bump_renorm.Value(),
             lighting.config0.clamp_highlights.Value(), lighting.config0.enable_shadow.Value(),
             lighting.config0.enable_primary_alpha.Value(),
             lighting.config0.enable_secondary_alpha.Value(), ambient.x, ambient.y, ambient.z,
             map_offset, bytes_used, fs_uniform_block_data.data.use_texture2d_lut);

    LOG_INFO(Render,
             "TG09_LUTCFG [{}] in(d0={} d1={} fr={} rr={} rg={} rb={} sp={}) "
             "abs(d0={} d1={} fr={} rr={} rg={} rb={} sp={}) "
             "scale(d0={:.4f} d1={:.4f} fr={:.4f} rr={:.4f} rg={:.4f} rb={:.4f} sp={:.4f}) "
             "lut_off(d0={} d1={} fr={} rr={} rg={} rb={})",
             backend, TG09LutInputName(lighting.lut_input.d0.Value()),
             TG09LutInputName(lighting.lut_input.d1.Value()),
             TG09LutInputName(lighting.lut_input.fr.Value()),
             TG09LutInputName(lighting.lut_input.rr.Value()),
             TG09LutInputName(lighting.lut_input.rg.Value()),
             TG09LutInputName(lighting.lut_input.rb.Value()),
             TG09LutInputName(lighting.lut_input.sp.Value()),
             lighting.abs_lut_input.disable_d0.Value(), lighting.abs_lut_input.disable_d1.Value(),
             lighting.abs_lut_input.disable_fr.Value(), lighting.abs_lut_input.disable_rr.Value(),
             lighting.abs_lut_input.disable_rg.Value(), lighting.abs_lut_input.disable_rb.Value(),
             lighting.abs_lut_input.disable_sp.Value(),
             lighting.lut_scale.GetScale(lighting.lut_scale.d0.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.d1.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.fr.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.rr.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.rg.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.rb.Value()),
             lighting.lut_scale.GetScale(lighting.lut_scale.sp.Value()),
             lighting.config1.disable_lut_d0.Value(), lighting.config1.disable_lut_d1.Value(),
             lighting.config1.disable_lut_fr.Value(), lighting.config1.disable_lut_rr.Value(),
             lighting.config1.disable_lut_rg.Value(), lighting.config1.disable_lut_rb.Value());

    // --- 5. Sources de lumiere, telles qu'elles atteignent le fragment shader ---------------------
    for (u32 slot = 0; slot < num_lights; ++slot) {
        const u32 light_index = lighting.light_enable.GetNum(slot);
        const auto& src = fs_uniform_block_data.data.light_src[light_index];
        const auto& raw = lighting.light[light_index];
        LOG_INFO(Render,
                 "TG09_LIGHTSRC [{}] slot={} light={} spec0=({:.6f},{:.6f},{:.6f}) "
                 "spec1=({:.6f},{:.6f},{:.6f}) diff=({:.6f},{:.6f},{:.6f}) "
                 "amb=({:.6f},{:.6f},{:.6f}) pos=({:.6f},{:.6f},{:.6f}) "
                 "spot_dir=({:.6f},{:.6f},{:.6f}) datten(bias={:.6f} scale={:.6f}) "
                 "directional={} two_sided={} geo0={} geo1={} spot_off={} datten_off={} "
                 "shadow_off={}",
                 backend, slot, light_index, src.specular_0.x, src.specular_0.y, src.specular_0.z,
                 src.specular_1.x, src.specular_1.y, src.specular_1.z, src.diffuse.x,
                 src.diffuse.y, src.diffuse.z, src.ambient.x, src.ambient.y, src.ambient.z,
                 src.position.x, src.position.y, src.position.z, src.spot_direction.x,
                 src.spot_direction.y, src.spot_direction.z, src.dist_atten_bias,
                 src.dist_atten_scale, raw.config.directional.Value(),
                 raw.config.two_sided_diffuse.Value(), raw.config.geometric_factor_0.Value(),
                 raw.config.geometric_factor_1.Value(),
                 lighting.IsSpotAttenDisabled(light_index) ? 1 : 0,
                 lighting.IsDistAttenDisabled(light_index) ? 1 : 0,
                 lighting.IsShadowDisabled(light_index) ? 1 : 0);
    }

    // --- 6. Contenu reel des LUT televersees ------------------------------------------------------
    for (u32 index = 0; index < LightingRegs::NumLightingSampler; ++index) {
        if (!relevant[index]) {
            continue;
        }
        const auto& lut = lighting_lut_data[index];

        f32 min_value = lut[0].x;
        f32 max_value = lut[0].x;
        double sum_value = 0.0;
        f32 max_abs_diff = 0.0f;
        u32 nan_count = 0;
        for (const auto& entry : lut) {
            min_value = std::min(min_value, entry.x);
            max_value = std::max(max_value, entry.x);
            sum_value += static_cast<double>(entry.x);
            max_abs_diff = std::max(max_abs_diff, std::fabs(entry.y));
            if (!std::isfinite(entry.x) || !std::isfinite(entry.y)) {
                ++nan_count;
            }
        }

        // --- TG10 : la SOURCE PICA, a comparer au cache televerse ci-dessus ---------------------
        // TG09 lit lighting_lut_data[] = ce qui a ete TELEVERSE. Si la source pica.lighting.luts[]
        // est non nulle alors que le cache est nul, le bug est dans la synchro. Si les deux sont
        // nuls, le jeu n'a jamais rempli cette LUT -- et writes= le dit directement.
        const auto& source_lut = pica.lighting.luts[index];
        std::array<Common::Vec2f, 256> source_data;
        std::transform(source_lut.begin(), source_lut.end(), source_data.begin(),
                       [](const auto& entry) {
                           return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                       });
        const u64 source_hash = TG09HashWords(
            source_data.data(), source_data.size() * sizeof(Common::Vec2f), TG09_FNV_OFFSET);
        f32 source_min = source_data[0].x;
        f32 source_max = source_data[0].x;
        u32 source_nonzero = 0;
        for (const auto& entry : source_data) {
            source_min = std::min(source_min, entry.x);
            source_max = std::max(source_max, entry.x);
            if (entry.x != 0.0f || entry.y != 0.0f) {
                ++source_nonzero;
            }
        }

        LOG_INFO(Render,
                 "TG09_LUT [{}] idx={:2d} name={} off={} hash={:#018x} min={:.6f} max={:.6f} "
                 "mean={:.6f} max_abs_diff={:.6f} non_finite={} "
                 "s[0]=({:.6f},{:.6f}) s[64]=({:.6f},{:.6f}) s[128]=({:.6f},{:.6f}) "
                 "s[192]=({:.6f},{:.6f}) s[255]=({:.6f},{:.6f}) | "
                 "src_hash={:#018x} src_min={:.6f} src_max={:.6f} src_nonzero={}/256 "
                 "writes={} src_eq_cache={}",
                 backend, index, TG09SamplerName(index),
                 fs_uniform_block_data.data.lighting_lut_offset[index / 4][index % 4],
                 lut_hash[index], min_value, max_value,
                 static_cast<f32>(sum_value / static_cast<double>(lut.size())), max_abs_diff,
                 nan_count, lut[0].x, lut[0].y, lut[64].x, lut[64].y, lut[128].x, lut[128].y,
                 lut[192].x, lut[192].y, lut[255].x, lut[255].y, source_hash, source_min,
                 source_max, source_nonzero, tg10_lut_writes[index],
                 source_hash == lut_hash[index] ? 1 : 0);

        if (level < 2) {
            continue;
        }
        const bool is_reflection = index == static_cast<u32>(Sampler::ReflectRed) ||
                                   index == static_cast<u32>(Sampler::ReflectGreen) ||
                                   index == static_cast<u32>(Sampler::ReflectBlue);
        if (level < 3 && !is_reflection) {
            continue;
        }
        for (std::size_t base = 0; base < lut.size(); base += 8) {
            std::string values;
            values.reserve(160);
            for (std::size_t k = 0; k < 8; ++k) {
                values += fmt::format("{:.6f}/{:.6f} ", lut[base + k].x, lut[base + k].y);
            }
            LOG_INFO(Render, "TG09_LUTDUMP [{}] idx={:2d} name={} [{:3d}] {}", backend, index,
                     TG09SamplerName(index), base, values);
        }
    }
}

} // namespace VideoCore
