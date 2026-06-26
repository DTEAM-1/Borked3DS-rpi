// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <string>
#include "video_core/pica/shader_setup.h"

namespace Pica::Shader::Generator::GLSL {

using RegGetter = std::function<std::string(u32)>;

std::string DecompileProgram(const Pica::ProgramCode& program_code,
                             const Pica::SwizzleData& swizzle_data, u32 main_offset,
                             const RegGetter& inputreg_getter, const RegGetter& outputreg_getter,
                             bool sanitize_mul);

/// v117c-MIRROR (Plan A gating): true if this VS should receive the V3DV low-bank mirror, i.e. it
/// reads f[64..95] via an address-register (dynamic) index AND never reads any uniform f[<32]. The
/// Vulkan rasterizer uses this per draw so the mirror (which overwrites f[0..31]) is applied only to
/// draws that don't read the low bank -- the dialogue glyphs -- and never to matrix-reading 3D draws.
bool VertexShaderWantsLowMirror(const Pica::ProgramCode& program_code, u32 main_offset);

} // namespace Pica::Shader::Generator::GLSL
