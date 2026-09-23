// Copyright 2014 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <boost/circular_buffer.hpp>
#include <boost/container/static_vector.hpp>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "common/profiling.h"
#include "common/vector_math.h"
#include "video_core/pica/shader_setup.h"
#include "video_core/pica/shader_unit.h"
#include "video_core/pica_types.h"
#include "video_core/shader/shader_interpreter.h"

using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

namespace Pica::Shader {

struct IfStackElement {
    u32 else_address;
    u32 end_address;
};

struct CallStackElement {
    u32 end_address;
    u32 return_address;
};

struct LoopStackElement {
    u32 entry_address;
    u32 end_address;
    u8 loop_downcounter;
    u8 address_increment;
    u8 previous_aL;
};

template <bool Debug>
static void RunInterpreter(const ShaderSetup& setup, ShaderUnit& state,
                           DebugData<Debug>& debug_data, unsigned entry_point) {
    boost::circular_buffer<IfStackElement> if_stack(8);
    boost::circular_buffer<CallStackElement> call_stack(4);
    boost::circular_buffer<LoopStackElement> loop_stack(4);
    u32 program_counter = entry_point;

    const auto do_if = [&](Instruction instr, bool condition) {
        if (condition) {
            if_stack.push_back({
                .else_address = instr.flow_control.dest_offset,
                .end_address = instr.flow_control.dest_offset + instr.flow_control.num_instructions,
            });
        } else {
            program_counter = instr.flow_control.dest_offset - 1;
        }
    };

    const auto do_call = [&](Instruction instr) {
        call_stack.push_back({
            .end_address = instr.flow_control.dest_offset + instr.flow_control.num_instructions,
            .return_address = program_counter + 1,
        });
        program_counter = instr.flow_control.dest_offset - 1;
    };

    const auto do_loop = [&](Instruction instr, const Common::Vec4<u8>& loop_param) {
        const u8 previous_aL = static_cast<u8>(state.address_registers[2]);
        loop_stack.push_back({
            .entry_address = program_counter + 1,
            .end_address = instr.flow_control.dest_offset + 1,
            .loop_downcounter = loop_param.x,
            .address_increment = loop_param.z,
            .previous_aL = previous_aL,
        });
        state.address_registers[2] = loop_param.y;
    };

    auto evaluate_condition = [&state](Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;

        bool result_x = flow_control.refx.Value() == state.conditional_code[0];
        bool result_y = flow_control.refy.Value() == state.conditional_code[1];

        switch (flow_control.op) {
        case Op::Or:
            return result_x || result_y;
        case Op::And:
            return result_x && result_y;
        case Op::JustX:
            return result_x;
        case Op::JustY:
            return result_y;
        default:
            UNREACHABLE();
            return false;
        }
    };

    const auto& uniforms = setup.uniforms;
    const auto& swizzle_data = setup.swizzle_data;
    const auto& program_code = setup.program_code;

    // Constants for handling invalid inputs
    static f24 dummy_vec4_float24_zeros[4] = {f24::Zero(), f24::Zero(), f24::Zero(), f24::Zero()};
    static f24 dummy_vec4_float24_ones[4] = {f24::One(), f24::One(), f24::One(), f24::One()};

    u32 iteration = 0;
    bool should_stop = false;

    // v367 -- SONDE-GARDE-FOU : passage de shader sans fin.
    //
    // Runs X et Y (Kid Icarus, savestate) : apres le correctif v366 la memoire ne s'emballe plus,
    // mais EmuThread reste pris dans un seul draw qui produit des triangles sans fin (8,3 M/s en
    // JIT, 1,9 M/s en interpreteur) et plus aucune frame ne se termine. Pile gdb du run Y
    // (interpreteur, entierement symbolisee) : InterpreterEngine::Run <- PicaCore::LoadVertices
    // <- DrawArrays. Le JIT et l'interpreteur bouclent tous deux : ce n'est pas un bug du JIT,
    // c'est le programme PICA qui, avec l'etat que lui donne l'emulateur, ne rencontre jamais END.
    // JMPC/JMPU peuvent sauter en arriere sans limite, et rien ne borne program_counter a
    // MAX_PROGRAM_CODE_LENGTH.
    //
    // Au-dela de BORKED3DS_V3DV_SHADER_MAX_STEPS instructions (defaut 4 194 304 ; 0 = illimite)
    // dans UN passage, on releve les 48 adresses suivantes, puis on journalise : point d'entree,
    // nombre d'EMIT, adresses et instructions de la boucle, uniformes booleens et entiers,
    // conditional_code, profondeur des piles -- et on arrete ce passage. Actif en interpreteur
    // seulement (use_shader_jit=false) : c'est un instrument de diagnostic.
    static const u64 v367_cap = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_SHADER_MAX_STEPS");
        if (v == nullptr || v[0] == '\0') {
            return u64{1} << 22;
        }
        return static_cast<u64>(std::strtoull(v, nullptr, 10));
    }();
    u64 v367_steps = 0;
    u64 v367_emits = 0;
    std::array<u32, 48> v367_trace{};
    std::size_t v367_trace_n = 0;

    while (!should_stop) {
        bool is_break = false;
        const u32 old_program_counter = program_counter;

        if (program_counter >= MAX_PROGRAM_CODE_LENGTH) [[unlikely]] {
            static std::atomic<u64> v367_oob{0};
            const u64 k = ++v367_oob;
            if (k <= 8 || (k % 256) == 0) {
                LOG_ERROR(HW_GPU,
                          "V367_SHADER_PC_HORS_PROGRAMME #{} pc={:#x} entree={:#x} pas={} emits={}",
                          k, program_counter, entry_point, v367_steps, v367_emits);
            }
            break;
        }
        if (v367_cap != 0 && ++v367_steps > v367_cap) [[unlikely]] {
            if (v367_trace_n < v367_trace.size()) {
                v367_trace[v367_trace_n++] = program_counter;
            } else {
                static std::atomic<u64> v367_runaway{0};
                const u64 k = ++v367_runaway;
                if (k <= 8 || (k % 256) == 0) {
                    std::string pcs;
                    std::string ops;
                    for (std::size_t t = 0; t < v367_trace_n; ++t) {
                        pcs += fmt::format("{:03x} ", v367_trace[t]);
                        ops += fmt::format("{:08x} ", program_code[v367_trace[t]]);
                    }
                    std::string bools;
                    for (std::size_t bi = 0; bi < uniforms.b.size(); ++bi) {
                        bools += uniforms.b[bi] ? '1' : '0';
                    }
                    LOG_ERROR(
                        HW_GPU,
                        "V367_SHADER_SANS_FIN #{} pas={} emits={} entree={:#x} "
                        "b[0..15]={} i0=({},{},{}) i1=({},{},{}) i2=({},{},{}) i3=({},{},{}) "
                        "cc=({},{}) piles call={} loop={} if={} | pc: {}| instr: {}",
                        k, v367_steps, v367_emits, entry_point, bools,
                        static_cast<u32>(uniforms.i[0].x), static_cast<u32>(uniforms.i[0].y),
                        static_cast<u32>(uniforms.i[0].z), static_cast<u32>(uniforms.i[1].x),
                        static_cast<u32>(uniforms.i[1].y), static_cast<u32>(uniforms.i[1].z),
                        static_cast<u32>(uniforms.i[2].x), static_cast<u32>(uniforms.i[2].y),
                        static_cast<u32>(uniforms.i[2].z), static_cast<u32>(uniforms.i[3].x),
                        static_cast<u32>(uniforms.i[3].y), static_cast<u32>(uniforms.i[3].z),
                        state.conditional_code[0], state.conditional_code[1], call_stack.size(),
                        loop_stack.size(), if_stack.size(), pcs, ops);

                    // v368 -- les VALEURS. La boucle du run Z est : 062 CMP c90 < r0 ;
                    // 063 JMPC sortie si (c90.x < r0.x) OU NON(c90.y < r0.y) ; 0b3 ADD r0 = r0 + r1 ;
                    // 0b5 JMPU b0 -> 062. Elle ne finit que si r0 depasse la limite c90. Il faut donc
                    // c90, r0, r1 (le pas), r2, les entrees v0..v7 du geometry shader, et le code
                    // d'initialisation (du point d'entree jusqu'a la boucle) pour savoir d'ou vient r1.
                    // Chaque composante : valeur decimale et bits du float32 (0x...) pour voir une
                    // absorption de precision f24 (r0 + r1 == r0).
                    const auto v4 = [](const auto& v) {
                        return fmt::format("({:.9g}/{:08x} {:.9g}/{:08x} {:.9g}/{:08x} {:.9g}/{:08x})",
                                           v.x.ToFloat32(), std::bit_cast<u32>(v.x.ToFloat32()),
                                           v.y.ToFloat32(), std::bit_cast<u32>(v.y.ToFloat32()),
                                           v.z.ToFloat32(), std::bit_cast<u32>(v.z.ToFloat32()),
                                           v.w.ToFloat32(), std::bit_cast<u32>(v.w.ToFloat32()));
                    };
                    std::string ins;
                    for (u32 k2 = 0; k2 < 8; ++k2) {
                        ins += fmt::format(" v{}={}", k2, v4(state.input[k2]));
                    }
                    LOG_ERROR(HW_GPU,
                              "V368_VALEURS #{} c90={} r0={} r1={} r2={} r3={} |{}", k,
                              v4(uniforms.f[90]), v4(state.temporary[0]), v4(state.temporary[1]),
                              v4(state.temporary[2]), v4(state.temporary[3]), ins);
                    std::string prog;
                    for (u32 a = entry_point; a < entry_point + 96 && a < MAX_PROGRAM_CODE_LENGTH;
                         ++a) {
                        prog += fmt::format("{:08x} ", program_code[a]);
                    }
                    LOG_ERROR(HW_GPU, "V368_PROGRAMME #{} entree={:#x} mots: {}", k, entry_point,
                              prog);
                }
                break;
            }
        }

        const Instruction instr(program_code[program_counter]);
        const SwizzlePattern swizzle(swizzle_data[instr.common.operand_desc_id]);

        Record<DebugDataRecord::CUR_INSTR>(debug_data, iteration, program_counter);
        if (iteration > 0)
            Record<DebugDataRecord::NEXT_INSTR>(debug_data, iteration - 1, program_counter);

        debug_data.max_offset = std::max<u32>(debug_data.max_offset, 1 + program_counter);

        auto LookupSourceRegister = [&](const SourceRegister& source_reg,
                                        int address_register_index) -> const f24* {
            int index = source_reg.GetIndex();
            switch (source_reg.GetRegisterType()) {
            case RegisterType::Input:
                return &state.input[index].x;

            case RegisterType::Temporary:
                return &state.temporary[index].x;

            case RegisterType::FloatUniform:
                if (address_register_index != 0) {
                    int offset = state.address_registers[address_register_index - 1];
                    if (offset < std::numeric_limits<s8>::min() ||
                        offset > std::numeric_limits<s8>::max()) [[unlikely]] {
                        offset = 0;
                    }
                    index = (index + offset) & 0x7F;
                    // If the index is above 96, the result is all one.
                    if (index >= 96) [[unlikely]] {
                        return dummy_vec4_float24_ones;
                    }
                }
                return &uniforms.f[index].x;

            default:
                return dummy_vec4_float24_zeros;
            }
        };

        switch (instr.opcode.Value().GetInfo().type) {
        case OpCode::Type::Arithmetic: {
            const bool is_inverted =
                (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

            const f24* src1_ =
                LookupSourceRegister(instr.common.GetSrc1(is_inverted),
                                     !is_inverted * instr.common.address_register_index);
            const f24* src2_ =
                LookupSourceRegister(instr.common.GetSrc2(is_inverted),
                                     is_inverted * instr.common.address_register_index);

            const bool negate_src1 = swizzle.negate_src1.Value() != 0;
            const bool negate_src2 = swizzle.negate_src2.Value() != 0;

            f24 src1[4] = {
                src1_[(int)swizzle.src1_selector_0.Value()],
                src1_[(int)swizzle.src1_selector_1.Value()],
                src1_[(int)swizzle.src1_selector_2.Value()],
                src1_[(int)swizzle.src1_selector_3.Value()],
            };
            if (negate_src1) {
                src1[0] = -src1[0];
                src1[1] = -src1[1];
                src1[2] = -src1[2];
                src1[3] = -src1[3];
            }
            f24 src2[4] = {
                src2_[(int)swizzle.src2_selector_0.Value()],
                src2_[(int)swizzle.src2_selector_1.Value()],
                src2_[(int)swizzle.src2_selector_2.Value()],
                src2_[(int)swizzle.src2_selector_3.Value()],
            };
            if (negate_src2) {
                src2[0] = -src2[0];
                src2[1] = -src2[1];
                src2[2] = -src2[2];
                src2[3] = -src2[3];
            }

            f24* dest = (instr.common.dest.Value() < 0x10)
                            ? &state.output[instr.common.dest.Value().GetIndex()][0]
                        : (instr.common.dest.Value() < 0x20)
                            ? &state.temporary[instr.common.dest.Value().GetIndex()][0]
                            : dummy_vec4_float24_zeros;

            debug_data.max_opdesc_id =
                std::max<u32>(debug_data.max_opdesc_id, 1 + instr.common.operand_desc_id);

            switch (instr.opcode.Value().EffectiveOpCode()) {
            case OpCode::Id::ADD: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] + src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::MUL: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] * src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::FLR:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = f24::FromFloat32(std::floor(src1[i].ToFloat32()));
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::MAX:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // NOTE: Exact form required to match NaN semantics to hardware:
                    //   max(0, NaN) -> NaN
                    //   max(NaN, 0) -> 0
                    dest[i] = (src1[i] > src2[i]) ? src1[i] : src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::MIN:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // NOTE: Exact form required to match NaN semantics to hardware:
                    //   min(0, NaN) -> NaN
                    //   min(NaN, 0) -> 0
                    dest[i] = (src1[i] < src2[i]) ? src1[i] : src2[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::DP3:
            case OpCode::Id::DP4:
            case OpCode::Id::DPH:
            case OpCode::Id::DPHI: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                OpCode::Id opcode = instr.opcode.Value().EffectiveOpCode();
                if (opcode == OpCode::Id::DPH || opcode == OpCode::Id::DPHI)
                    src1[3] = f24::One();

                int num_components = (opcode == OpCode::Id::DP3) ? 3 : 4;
                f24 dot = std::inner_product(src1, src1 + num_components, src2, f24::Zero());

                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = dot;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            // Reciprocal
            case OpCode::Id::RCP: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                f24 rcp_res = f24::FromFloat32(1.0f / src1[0].ToFloat32());
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = rcp_res;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            // Reciprocal Square Root
            case OpCode::Id::RSQ: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                f24 rsq_res = f24::FromFloat32(1.0f / std::sqrt(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = rsq_res;
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::MOVA: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                for (int i = 0; i < 2; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    // TODO: Figure out how the rounding is done on hardware
                    state.address_registers[i] = static_cast<s32>(src1[i].ToFloat32());
                }
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::MOV: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::SGE:
            case OpCode::Id::SGEI:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = (src1[i] >= src2[i]) ? f24::One() : f24::Zero();
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::SLT:
            case OpCode::Id::SLTI:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = (src1[i] < src2[i]) ? f24::One() : f24::Zero();
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;

            case OpCode::Id::CMP:
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                for (int i = 0; i < 2; ++i) {
                    // TODO: Can you restrict to one compare via dest masking?

                    auto compare_op = instr.common.compare_op;
                    auto op = (i == 0) ? compare_op.x.Value() : compare_op.y.Value();

                    switch (op) {
                    case Instruction::Common::CompareOpType::Equal:
                        state.conditional_code[i] = (src1[i] == src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::NotEqual:
                        state.conditional_code[i] = (src1[i] != src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::LessThan:
                        state.conditional_code[i] = (src1[i] < src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::LessEqual:
                        state.conditional_code[i] = (src1[i] <= src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::GreaterThan:
                        state.conditional_code[i] = (src1[i] > src2[i]);
                        break;

                    case Instruction::Common::CompareOpType::GreaterEqual:
                        state.conditional_code[i] = (src1[i] >= src2[i]);
                        break;

                    default:
                        LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", static_cast<int>(op));
                        break;
                    }
                }
                Record<DebugDataRecord::CMP_RESULT>(debug_data, iteration, state.conditional_code);
                break;

            case OpCode::Id::EX2: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                // EX2 only takes first component exp2 and writes it to all dest components
                f24 ex2_res = f24::FromFloat32(std::exp2(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = ex2_res;
                }

                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            case OpCode::Id::LG2: {
                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);

                // LG2 only takes the first component log2 and writes it to all dest components
                f24 lg2_res = f24::FromFloat32(std::log2(src1[0].ToFloat32()));
                for (int i = 0; i < 4; ++i) {
                    if (!swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = lg2_res;
                }

                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
                break;
            }

            default:
                LOG_ERROR(HW_GPU, "Unhandled arithmetic instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                DEBUG_ASSERT(false);
                break;
            }

            break;
        }

        case OpCode::Type::MultiplyAdd: {
            if ((instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD) ||
                (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI)) {
                const SwizzlePattern& mad_swizzle = *reinterpret_cast<const SwizzlePattern*>(
                    &swizzle_data[instr.mad.operand_desc_id]);

                bool is_inverted = (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI);

                const f24* src1_ = LookupSourceRegister(instr.mad.GetSrc1(is_inverted), 0);
                const f24* src2_ =
                    LookupSourceRegister(instr.mad.GetSrc2(is_inverted),
                                         !is_inverted * instr.mad.address_register_index);
                const f24* src3_ = LookupSourceRegister(
                    instr.mad.GetSrc3(is_inverted), is_inverted * instr.mad.address_register_index);

                const bool negate_src1 = mad_swizzle.negate_src1.Value() != 0;
                const bool negate_src2 = mad_swizzle.negate_src2.Value() != 0;
                const bool negate_src3 = mad_swizzle.negate_src3.Value() != 0;

                f24 src1[4] = {
                    src1_[(int)mad_swizzle.src1_selector_0.Value()],
                    src1_[(int)mad_swizzle.src1_selector_1.Value()],
                    src1_[(int)mad_swizzle.src1_selector_2.Value()],
                    src1_[(int)mad_swizzle.src1_selector_3.Value()],
                };
                if (negate_src1) {
                    src1[0] = -src1[0];
                    src1[1] = -src1[1];
                    src1[2] = -src1[2];
                    src1[3] = -src1[3];
                }
                f24 src2[4] = {
                    src2_[(int)mad_swizzle.src2_selector_0.Value()],
                    src2_[(int)mad_swizzle.src2_selector_1.Value()],
                    src2_[(int)mad_swizzle.src2_selector_2.Value()],
                    src2_[(int)mad_swizzle.src2_selector_3.Value()],
                };
                if (negate_src2) {
                    src2[0] = -src2[0];
                    src2[1] = -src2[1];
                    src2[2] = -src2[2];
                    src2[3] = -src2[3];
                }
                f24 src3[4] = {
                    src3_[(int)mad_swizzle.src3_selector_0.Value()],
                    src3_[(int)mad_swizzle.src3_selector_1.Value()],
                    src3_[(int)mad_swizzle.src3_selector_2.Value()],
                    src3_[(int)mad_swizzle.src3_selector_3.Value()],
                };
                if (negate_src3) {
                    src3[0] = -src3[0];
                    src3[1] = -src3[1];
                    src3[2] = -src3[2];
                    src3[3] = -src3[3];
                }

                f24* dest = (instr.mad.dest.Value() < 0x10)
                                ? &state.output[instr.mad.dest.Value().GetIndex()][0]
                            : (instr.mad.dest.Value() < 0x20)
                                ? &state.temporary[instr.mad.dest.Value().GetIndex()][0]
                                : dummy_vec4_float24_zeros;

                Record<DebugDataRecord::SRC1>(debug_data, iteration, src1);
                Record<DebugDataRecord::SRC2>(debug_data, iteration, src2);
                Record<DebugDataRecord::SRC3>(debug_data, iteration, src3);
                Record<DebugDataRecord::DEST_IN>(debug_data, iteration, dest);
                for (int i = 0; i < 4; ++i) {
                    if (!mad_swizzle.DestComponentEnabled(i))
                        continue;

                    dest[i] = src1[i] * src2[i] + src3[i];
                }
                Record<DebugDataRecord::DEST_OUT>(debug_data, iteration, dest);
            } else {
                LOG_ERROR(HW_GPU, "Unhandled multiply-add instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
            }
            break;
        }

        default: {
            // Handle each instruction on its own
            switch (instr.opcode.Value()) {
            case OpCode::Id::END:
                should_stop = true;
                break;

            case OpCode::Id::JMPC:
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    program_counter = instr.flow_control.dest_offset - 1;
                }
                break;

            case OpCode::Id::JMPU:
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);

                if (uniforms.b[instr.flow_control.bool_uniform_id] ==
                    !(instr.flow_control.num_instructions & 1)) {
                    program_counter = instr.flow_control.dest_offset - 1;
                }
                break;

            case OpCode::Id::CALL:
                do_call(instr);
                break;

            case OpCode::Id::CALLU:
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);
                if (uniforms.b[instr.flow_control.bool_uniform_id]) {
                    do_call(instr);
                }
                break;

            case OpCode::Id::CALLC:
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    do_call(instr);
                }
                break;

            case OpCode::Id::NOP:
                break;

            case OpCode::Id::IFU: {
                Record<DebugDataRecord::COND_BOOL_IN>(
                    debug_data, iteration, uniforms.b[instr.flow_control.bool_uniform_id]);
                const bool cond = uniforms.b[instr.flow_control.bool_uniform_id];
                do_if(instr, cond);
                break;
            }

            case OpCode::Id::IFC: {
                // TODO: Do we need to consider swizzlers here?
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                const bool cond = evaluate_condition(instr.flow_control);
                do_if(instr, cond);
                break;
            }

            case OpCode::Id::LOOP: {
                const Common::Vec4<u8>& loop_param = uniforms.i[instr.flow_control.int_uniform_id];
                state.address_registers[2] = loop_param.y;

                Record<DebugDataRecord::LOOP_INT_IN>(debug_data, iteration, loop_param);
                do_loop(instr, loop_param);
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::BREAK: {
                is_break = true;
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::BREAKC: {
                Record<DebugDataRecord::COND_CMP_IN>(debug_data, iteration, state.conditional_code);
                if (evaluate_condition(instr.flow_control)) {
                    is_break = true;
                }
                Record<DebugDataRecord::ADDR_REG_OUT>(debug_data, iteration,
                                                      state.address_registers);
                break;
            }

            case OpCode::Id::EMIT: {
                auto* emitter = state.emitter_ptr;
                ASSERT_MSG(emitter, "Execute EMIT on VS");
                emitter->Emit(state.output);
                ++v367_emits; // v367 : compte des EMIT du passage (GS)
                break;
            }

            case OpCode::Id::SETEMIT: {
                auto* emitter = state.emitter_ptr;
                ASSERT_MSG(emitter, "Execute SETEMIT on VS");
                emitter->vertex_id = instr.setemit.vertex_id;
                emitter->prim_emit = instr.setemit.prim_emit != 0;
                emitter->winding = instr.setemit.winding != 0;
                break;
            }

            default:
                LOG_ERROR(HW_GPU, "Unhandled instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                break;
            }

            break;
        }
        }

        ++program_counter;
        ++iteration;

        // Stacks are checked in the order CALL -> IF -> LOOP. The CALL stack
        // can be popped multiple times per instruction. A JMP at the end of a
        // scope is never taken, this is why we compare against
        // old_program_counter + 1 here.
        u32 next_program_counter = old_program_counter + 1;
        for (u32 i = 0; i < 4; i++) {
            if (call_stack.empty() || call_stack.back().end_address != next_program_counter)
                break;
            // Hardware bug: when popping four CALL scopes at once, the last
            // one doesn't update the program counter
            if (i < 3) {
                program_counter = call_stack.back().return_address;
                next_program_counter = program_counter;
            }
            call_stack.pop_back();
        }

        // The other two stacks can only pop one entry per instruction. They
        // are checked against the original program counter before any CALL
        // scopes were closed and they overwrite any previous program counter
        // updates.
        if (!if_stack.empty() && if_stack.back().else_address == old_program_counter + 1) {
            program_counter = if_stack.back().end_address;
            if_stack.pop_back();
        }

        if (!loop_stack.empty() &&
            (loop_stack.back().end_address == old_program_counter + 1 || is_break)) {
            auto& loop = loop_stack.back();
            state.address_registers[2] += loop.address_increment;
            if (!is_break && loop.loop_downcounter--) {
                program_counter = loop.entry_address;
            } else {
                program_counter = loop.end_address;
                // Only restore previous value if there is a surrounding LOOP scope.
                if (loop_stack.size() > 1)
                    state.address_registers[2] = loop.previous_aL;
                loop_stack.pop_back();
            }
        }
    }
}

void InterpreterEngine::SetupBatch(ShaderSetup& setup, unsigned int entry_point) {
    ASSERT(entry_point < MAX_PROGRAM_CODE_LENGTH);
    setup.entry_point = entry_point;
}

void InterpreterEngine::Run(const ShaderSetup& setup, ShaderUnit& state) const {
    BORKED3DS_PROFILE("Shader", "Shader Interpreter");

    DebugData<false> dummy_debug_data;
    RunInterpreter(setup, state, dummy_debug_data, setup.entry_point);
}

DebugData<true> InterpreterEngine::ProduceDebugInfo(const ShaderSetup& setup,
                                                    const AttributeBuffer& input,
                                                    const ShaderRegs& config) const {
    ShaderUnit state;
    DebugData<true> debug_data;

    // Setup input register table
    state.input.fill(Common::Vec4<f24>::AssignToAll(f24::Zero()));
    state.LoadInput(config, input);
    RunInterpreter(setup, state, debug_data, setup.entry_point);
    return debug_data;
}

} // namespace Pica::Shader
