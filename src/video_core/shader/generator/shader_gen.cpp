// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#ifndef __APPLE__
#include <glad/gl.h>
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_vars.h"
#endif

#include <cstdlib>
#include <mutex>
#include <unordered_set>

#include "common/bit_set.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "video_core/pica/regs_internal.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/shader/generator/shader_gen.h"

namespace Pica::Shader::Generator {

// ---------------------------------------------------------------------------------------------
// TG05 (BORKED3DS_TG05_SEMANTIC_MAP=1) -- sonde de MESURE, inerte hors variable d'environnement.
//
// Motif : le desassemblage SPIR-V de la passe du 17/08 a montre que 2 des 3 geometry shaders
// compiles ecrivent  OpStore %normquat %76  avec  %76 = vec4(1.0, 1.0, 1.0, 1.0)  -- soit un
// quaternion de normale CONSTANT, sur des draws ECLAIRES dont le fragment shader lit et
// normalise ce meme normquat. Normale constante par draw = ombrage plat = les triangles
// observes sur le vaisseau de Metroid.
//
// Le "1.0" vient du repli de la lambda semantic() de glsl_shader_gen.cpp :
//
//     if (attrib < state.gs_output_attributes) { return "vtx.attributes[attrib].comp"; }
//     return "1.0";
//
// Or les deux bornes en presence ne sont PAS calculees a partir du meme registre PICA :
//
//     gs_output_attributes  =  popcount(regs.vs.output_mask)
//     boucle de remplissage :  attrib < regs.rasterizer.vs_output_total
//
// Si vs_output_total > popcount(output_mask), toute semantique logee dans l'intervalle
// [popcount, vs_output_total) est ecrasee par la constante 1.0 -- y compris les quatre
// composantes QUATERNION_X..W.
//
// Cette sonde mesure exactement ca, sans rien corriger :
//   vs_total  = regs.rasterizer.vs_output_total
//   popcount  = popcount(regs.vs.output_mask)   (= gs_output_attributes)
//   quat_attr = index d'attribut resolu pour QUATERNION_X..W (16 = jamais mappe)
//   quat_ok   = 1 si les 4 composantes tombent DANS la borne, 0 si au moins une est perdue
//
// Le meme code sert aux deux backends : lancer une fois en Vulkan, une fois en OpenGL/GLES, et
// comparer. Si quat_ok=0 apparait des deux cotes, le repli n'est pas le differenciateur
// GL/Vulkan et il faudra chercher ailleurs. S'il n'apparait qu'en Vulkan, c'est la cause racine.
//
// Deduplique par cle (vs_total, popcount, quat_ok) : quelques lignes par session, pas un flot.
// ---------------------------------------------------------------------------------------------
namespace {

bool IsTG05Enabled() {
    static const bool enabled = std::getenv("BORKED3DS_TG05_SEMANTIC_MAP") != nullptr;
    return enabled;
}

bool TG05ShouldLog(u64 key) {
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    std::scoped_lock lock{mutex};
    return seen.insert(key).second;
}

} // Anonymous namespace

void PicaGSConfigState::Init(const Pica::RegsInternal& regs, bool use_clip_planes_) {
    use_clip_planes = use_clip_planes_;

    vs_output_attributes = Common::BitSet<u32>(regs.vs.output_mask).Count();
    gs_output_attributes = vs_output_attributes;

    semantic_maps.fill({16, 0});
    for (u32 attrib = 0; attrib < regs.rasterizer.vs_output_total; ++attrib) {
        const std::array semantics{
            regs.rasterizer.vs_output_attributes[attrib].map_x.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_y.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_z.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_w.Value(),
        };
        for (u32 comp = 0; comp < 4; ++comp) {
            const auto semantic = semantics[comp];
            if (static_cast<std::size_t>(semantic) < 24) {
                semantic_maps[static_cast<std::size_t>(semantic)] = {attrib, comp};
            } else if (semantic != Pica::RasterizerRegs::VSOutputAttributes::INVALID) {
                LOG_ERROR(Render, "Invalid/unknown semantic id: {}", semantic);
            }
        }
    }

    // --- TG05 : mesure seule, aucune modification du comportement -----------------------------
    if (IsTG05Enabled()) {
        using Semantic = Pica::RasterizerRegs::VSOutputAttributes::Semantic;
        const u32 qx = semantic_maps[static_cast<std::size_t>(Semantic::QUATERNION_X)].attribute_index;
        const u32 qy = semantic_maps[static_cast<std::size_t>(Semantic::QUATERNION_Y)].attribute_index;
        const u32 qz = semantic_maps[static_cast<std::size_t>(Semantic::QUATERNION_Z)].attribute_index;
        const u32 qw = semantic_maps[static_cast<std::size_t>(Semantic::QUATERNION_W)].attribute_index;

        const bool quat_ok = (qx < gs_output_attributes) && (qy < gs_output_attributes) &&
                             (qz < gs_output_attributes) && (qw < gs_output_attributes);
        const bool quat_never_mapped = (qx == 16) && (qy == 16) && (qz == 16) && (qw == 16);

        const u32 vs_total = regs.rasterizer.vs_output_total;

        // --- TG05b : etat d'eclairage associe a CETTE configuration de sortie ------------------
        // Motif : le defaut observe sur le vaisseau de Metroid est SPECULAIRE seulement -- correct
        // a l'ombre, faux sur le reflet. La question decisive est donc : ces draws sans quaternion
        // demandent-ils vraiment un terme speculaire / une reflexion (LUT D1, RR, RG, RB) ?
        const u32 lighting_disable = regs.lighting.disable.Value();
        const u32 light_config = static_cast<u32>(regs.lighting.config0.config.Value());
        const u32 bump_mode = static_cast<u32>(regs.lighting.config0.bump_mode.Value());
        const u32 bump_selector = regs.lighting.config0.bump_selector.Value();
        const u32 bump_renorm_off = regs.lighting.config0.disable_bump_renorm.Value();
        const u32 clamp_highlights = regs.lighting.config0.clamp_highlights.Value();
        const u32 enable_shadow = regs.lighting.config0.enable_shadow.Value();
        const u32 num_lights = regs.lighting.max_light_index.Value() + 1u;
        const u32 lut_d0 = regs.lighting.config1.disable_lut_d0.Value();
        const u32 lut_d1 = regs.lighting.config1.disable_lut_d1.Value();
        const u32 lut_fr = regs.lighting.config1.disable_lut_fr.Value();
        const u32 lut_rr = regs.lighting.config1.disable_lut_rr.Value();
        const u32 lut_rg = regs.lighting.config1.disable_lut_rg.Value();
        const u32 lut_rb = regs.lighting.config1.disable_lut_rb.Value();

        const u64 key = (static_cast<u64>(vs_total) << 40) |
                        (static_cast<u64>(gs_output_attributes) << 32) |
                        (static_cast<u64>(quat_ok ? 1u : 0u) << 31) |
                        (static_cast<u64>(quat_never_mapped ? 1u : 0u) << 30) |
                        (static_cast<u64>(lighting_disable) << 29) |
                        (static_cast<u64>(light_config & 0xF) << 25) |
                        (static_cast<u64>(bump_mode & 0x3) << 23) |
                        (static_cast<u64>(lut_d1) << 22) | (static_cast<u64>(lut_rr) << 21) |
                        (static_cast<u64>(lut_rg) << 20) | (static_cast<u64>(lut_rb) << 19) |
                        (static_cast<u64>(lut_d0) << 18) | (static_cast<u64>(lut_fr) << 17) |
                        (static_cast<u64>(num_lights & 0xF) << 13);

        if (TG05ShouldLog(key)) {
            LOG_INFO(Render,
                     "TG05_SEMANTIC vs_total={} popcount={} quat_attr=({},{},{},{}) quat_ok={} "
                     "quat_never_mapped={} | light_disable={} config={} lights={} bump_mode={} "
                     "bump_sel={} bump_renorm_off={} clamp_hl={} shadow={} | LUT_off d0={} d1={} "
                     "fr={} rr={} rg={} rb={} (d1/rr/rg/rb = speculaire et reflexion : 0 = LUT "
                     "ACTIF)",
                     vs_total, gs_output_attributes, qx, qy, qz, qw, quat_ok ? 1 : 0,
                     quat_never_mapped ? 1 : 0, lighting_disable, light_config, num_lights,
                     bump_mode, bump_selector, bump_renorm_off, clamp_highlights, enable_shadow,
                     lut_d0, lut_d1, lut_fr, lut_rr, lut_rg, lut_rb);
        }
    }
    // ------------------------------------------------------------------------------------------
}

void PicaVSConfigState::Init(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
                             bool use_clip_planes_, bool use_geometry_shader_, bool accurate_mul_) {

    use_clip_planes = use_clip_planes_;
    use_geometry_shader = use_geometry_shader_;
    sanitize_mul = accurate_mul_;

    program_hash = setup.GetProgramCodeHash();
    swizzle_hash = setup.GetSwizzleDataHash();
    main_offset = regs.vs.main_offset;

    num_outputs = 0;
    load_flags.fill(AttribLoadFlags::Float);
    output_map.fill(16);

    for (u32 reg : Common::BitSet<u32>(regs.vs.output_mask)) {
        output_map[reg] = num_outputs++;
    }

    if (!use_geometry_shader_) {
        gs_state.Init(regs, use_clip_planes_);
    }
}

PicaVSConfig::PicaVSConfig(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
                           bool use_clip_planes_, bool use_geometry_shader_, bool accurate_mul_) {
    state.Init(regs, setup, use_clip_planes_, use_geometry_shader_, accurate_mul_);
}

PicaFixedGSConfig::PicaFixedGSConfig(const Pica::RegsInternal& regs, bool use_clip_planes_) {
    state.Init(regs, use_clip_planes_);
}

} // namespace Pica::Shader::Generator
