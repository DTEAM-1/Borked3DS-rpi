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

/// v118-MIRROR (Plan A, per-VS base): plan describing how to mirror the V3DV-miscompiled upper float
/// uniform bank into a conflict-free low window for one VS.
///   ok    : the mirror should be applied to this draw
///   base  : low slot where f[64] is mirrored (= highest f[<32] index the VS reads, + 1; 0 if none)
///   count : number of contiguous upper-bank slots mirrored: f[64..64+count) -> f[base..base+count)
/// The generated get_offset_register_sw re-fetches the texcoord via a dynamic LOW index at the same
/// base, clamped into [0, count). For a pure upper-bank VS this is base=0/count=32 (the original
/// f[0..31] <- f[64..95] mirror). Unlike VertexShaderWantsLowMirror it also accepts hybrid VSs that
/// read low constants (e.g. the Sonic Lost World glyph VS: low f[0..6] + upper-bank texcoord), by
/// placing the window above their low reads instead of excluding them.
struct LowMirrorPlan {
    bool ok;
    u32 base;
    u32 count;
};

LowMirrorPlan VertexShaderLowMirrorPlan(const Pica::ProgramCode& program_code, u32 main_offset);

/// vDIRA (Direction A, v119): true if this VS must be routed to the per-draw SOFTWARE vertex
/// shader fallback on V3DV. That is the case when it reads the upper float-uniform bank f[64..95]
/// through a dynamic (address-register) index -- the pattern V3D 7.1 miscompiles into a constant
/// index -- AND it also reads the low bank f[<32], which makes the v118 low-bank mirror
/// inapplicable (hybrid text+3D VS, e.g. the Sonic Lost World glyph VS: low f[0..6] + upper
/// f[64..69]). Pure upper-bank VSs keep the cheaper hardware mirror (LowMirrorPlan.ok == true);
/// VSs without any dynamic upper-bank read stay fully hardware. Returns false when control-flow
/// analysis fails (conservative: same behaviour as the mirror gate).
bool VertexShaderNeedsSoftwareVSFallback(const Pica::ProgramCode& program_code, u32 main_offset);

} // namespace Pica::Shader::Generator::GLSL
