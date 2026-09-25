// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/hash.h"

namespace Pica {
struct RegsInternal;
struct ShaderSetup;
} // namespace Pica

namespace Pica::Shader::Generator {

// NOTE: Changing the order impacts shader transferable and precompiled cache loading.
enum ProgramType : u32 {
    VS = 0,
    FS = 1,
    GS = 2,
};

enum Attributes {
    ATTRIBUTE_POSITION,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_TEXCOORD0,
    ATTRIBUTE_TEXCOORD1,
    ATTRIBUTE_TEXCOORD2,
    ATTRIBUTE_TEXCOORD0_W,
    ATTRIBUTE_NORMQUAT,
    ATTRIBUTE_VIEW,
};

enum class AttribLoadFlags {
    Float = 1 << 0,
    Sint = 1 << 1,
    Uint = 1 << 2,
    ZeroW = 1 << 3,
};
DECLARE_ENUM_FLAG_OPERATORS(AttribLoadFlags)

/**
 * This struct contains common information to identify a GLSL geometry shader generated from
 * PICA geometry shader.
 */
struct PicaGSConfigState {
    void Init(const Pica::RegsInternal& regs, bool use_clip_planes_);

    bool use_clip_planes;

    u32 vs_output_attributes;
    u32 gs_output_attributes;

    struct SemanticMap {
        u32 attribute_index;
        u32 component_index;
    };

    // semantic_maps[semantic name] -> GS output attribute index + component index
    std::array<SemanticMap, 24> semantic_maps;
};

/**
 * This struct contains common information to identify a GLSL vertex shader generated from
 * PICA vertex shader.
 */
struct PicaVSConfigState {
    void Init(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup, bool use_clip_planes_,
              bool use_geometry_shader_, bool accurate_mul_);

    bool use_clip_planes;
    bool use_geometry_shader;

    u64 program_hash;
    u64 swizzle_hash;
    u32 main_offset;
    bool sanitize_mul;

    u32 num_outputs;
    // Load operations to apply to the input vertex data
    std::array<AttribLoadFlags, 16> load_flags;

    // v380 : booleens uniformes figes dans le shader genere. Seuls ceux qui controlent des JMPU
    // formant une boucle statique (CyclicJumpBoolMaskCached) ; 0 pour tous les autres shaders.
    // Font partie de la cle : une autre valeur de ces booleens = une autre variante du shader.
    u16 jmpu_spec_mask;
    u16 jmpu_spec_values;

    // v385 : compteurs de boucle (i[0..3]) figes dans le shader genere. Bits 0-3 du masque ;
    // valeurs x | y << 8 | z << 16, celles du draw courant. Partie de la cle, comme ci-dessus.
    // v385 etend aussi jmpu_spec_mask / values a tous les booleens lus par IFU / CALLU / JMPU.
    u8 loop_spec_mask;
    std::array<u32, 4> loop_spec_values;

    // output_map[output register index] -> output attribute index
    std::array<u32, 16> output_map;

    PicaGSConfigState gs_state;
};

/**
 * This struct contains information to identify a GL vertex shader generated from PICA vertex
 * shader.
 */
struct PicaVSConfig : Common::HashableStruct<PicaVSConfigState> {
    explicit PicaVSConfig(const Pica::RegsInternal& regs, Pica::ShaderSetup& setup,
                          bool use_clip_planes_, bool use_geometry_shader_, bool accurate_mul_);
};

/**
 * This struct contains information to identify a GL geometry shader generated from PICA no-geometry
 * shader pipeline
 */
struct PicaFixedGSConfig : Common::HashableStruct<PicaGSConfigState> {
    explicit PicaFixedGSConfig(const Pica::RegsInternal& regs, bool use_clip_planes_);
};

} // namespace Pica::Shader::Generator

namespace std {
template <>
struct hash<Pica::Shader::Generator::PicaVSConfig> {
    std::size_t operator()(const Pica::Shader::Generator::PicaVSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<Pica::Shader::Generator::PicaFixedGSConfig> {
    std::size_t operator()(const Pica::Shader::Generator::PicaFixedGSConfig& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std
