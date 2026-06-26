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

/// v117b-MIRROR (Plan A gating): true if any instruction reachable from main_offset reads the upper
/// float-uniform bank f[64..95] via an address-register (dynamic) index. The Vulkan rasterizer uses
/// this to apply the V3DV low-bank mirror only to the draws that need it (e.g. dialogue glyphs).
bool ProgramReadsHighIndexedUniform(const Pica::ProgramCode& program_code, u32 main_offset);

} // namespace Pica::Shader::Generator::GLSL
