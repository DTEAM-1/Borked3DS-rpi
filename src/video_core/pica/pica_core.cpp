// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdlib>
#include "common/arch.h"
#include "common/archives.h"
#include "common/profiling.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"

namespace Pica {

using namespace DebugUtils;

union CommandHeader {
    u32 hex;
    BitField<0, 16, u32> cmd_id;
    BitField<16, 4, u32> parameter_mask;
    BitField<20, 8, u32> extra_data_length;
    BitField<31, 1, u32> group_commands;
};
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

[[nodiscard]] bool IsEnvEnabled(const char* name) {
    if (const char* value = std::getenv(name)) {
        return value[0] != '\0' && value[0] != '0';
    }
    return false;
}

[[nodiscard]] bool IsInterestingPicaStateReg(u32 id) {
    return id == 0x203 || id == 0x206 || (id >= 0x1C8 && id <= 0x1CF);
}

[[nodiscard]] bool IsPicaHotpathTraceEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_TRACE_PICA_STATE");
}

[[nodiscard]] bool IsPicaDrawTraceEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_TRACE_DRAW");
}

[[nodiscard]] bool IsStrictCompatEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_STRICT_COMPAT");
}

[[nodiscard]] bool ArePrimaryTexturesDisabled(const RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        if (textures[i].enabled) {
            return false;
        }
    }
    return true;
}

void LogPicaTextureState(const RegsInternal& regs, const char* tag) {
    if (!IsPicaDrawTraceEnabled()) {
        return;
    }

    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        const auto& texture = textures[i];
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA {} tex{} enabled={} type={} format={} addr=0x{:08X}",
                 tag, i, texture.enabled, static_cast<u32>(texture.config.type.Value()),
                 static_cast<u32>(texture.format), texture.config.GetPhysicalAddress());
    }
}

void LogMainConfigTransition(const RegsInternal& regs, u32 id, u32 old_value, u32 new_value, u32 mask) {
    if (!IsPicaDrawTraceEnabled() || id != PICA_REG_INDEX(texturing.main_config)) {
        return;
    }

    const u32 old_t0 = old_value & 1u;
    const u32 new_t0 = new_value & 1u;
    const u32 old_t1 = (old_value >> 1) & 1u;
    const u32 new_t1 = (new_value >> 1) & 1u;
    const u32 old_t2 = (old_value >> 2) & 1u;
    const u32 new_t2 = (new_value >> 2) & 1u;

    LOG_INFO(HW_GPU,
             "TRACE_DRAW_PICA main_config_transition old=0x{:08X} new=0x{:08X} mask=0x{:X} old_t0={} new_t0={} old_t1={} new_t1={} old_t2={} new_t2={}",
             old_value, new_value, mask, old_t0, new_t0, old_t1, new_t1, old_t2, new_t2);
}

std::atomic<u64> g_pica_draw_counter{0};
std::atomic<u64> g_fragile_startup_draw_counter{0};
std::atomic<u64> g_large_textured_startup_draw_counter{0};
std::atomic<u64> g_tiny_textured_startup_draw_counter{0};
std::atomic<u64> g_pica_cmdlist_counter{0};
std::atomic<bool> g_logged_first_non_fragile_draw{false};
std::atomic<bool> g_logged_first_non_fragile_textured_draw{false};
std::atomic<bool> g_logged_first_non_fragile_software_draw{false};
std::atomic<bool> g_logged_first_non_fragile_accel_candidate{false};
std::atomic<bool> g_logged_first_non_fragile_accel_attempt{false};
std::atomic<bool> g_logged_first_non_fragile_accel_failed{false};

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())} {
    InitializeRegs();

    const auto submit_vertex = [this](const AttributeBuffer& buffer) {
        const auto add_triangle = [this](const OutputVertex& v0, const OutputVertex& v1,
                                         const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
        const auto vertex = OutputVertex(regs.internal.rasterizer, buffer);
        primitive_assembler.SubmitVertex(vertex, add_triangle);
    };

    gs_unit.SetVertexHandlers(submit_vertex, [this]() { primitive_assembler.SetWinding(); });
    geometry_pipeline.SetVertexHandler(submit_vertex);

    primitive_assembler.Reconfigure(PipelineRegs::TriangleTopology::List);
}

PicaCore::~PicaCore() = default;

void PicaCore::InitializeRegs() {
    auto& framebuffer_top = regs.framebuffer_config[0];
    auto& framebuffer_sub = regs.framebuffer_config[1];

    // Set framebuffer defaults from nn::gx::Initialize
    framebuffer_top.address_left1 = 0x181E6000;
    framebuffer_top.address_left2 = 0x1822C800;
    framebuffer_top.address_right1 = 0x18273000;
    framebuffer_top.address_right2 = 0x182B9800;
    framebuffer_sub.address_left1 = 0x1848F000;
    framebuffer_sub.address_left2 = 0x184C7800;

    framebuffer_top.width.Assign(240);
    framebuffer_top.height.Assign(400);
    framebuffer_top.stride = 3 * 240;
    framebuffer_top.color_format.Assign(PixelFormat::RGB8);
    framebuffer_top.active_fb = 0;

    framebuffer_sub.width.Assign(240);
    framebuffer_sub.height.Assign(320);
    framebuffer_sub.stride = 3 * 240;
    framebuffer_sub.color_format.Assign(PixelFormat::RGB8);
    framebuffer_sub.active_fb = 0;

    // Tales of Abyss expects this register to have the following default values.
    auto& gs = regs.internal.gs;
    gs.max_input_attribute_index.Assign(1);
    gs.shader_mode.Assign(ShaderRegs::ShaderMode::VS);
}

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer) {
    this->rasterizer = rasterizer;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt) {
    this->signal_interrupt = signal_interrupt;
}

void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) {
    const u64 cmdlist_index = ++g_pica_cmdlist_counter;
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::ProcessCmdList begin cmdlist_index={} list={:#010X} size={} ignore_list={}",
                  cmdlist_index, list, size, ignore_list);
    }

    if (ignore_list) {
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList ignored list={:#010X}", list);
        }
        signal_interrupt(Service::GSP::InterruptId::P3D);
        return;
    }

    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    while (cmd_list.current_index < cmd_list.length) {
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::ProcessCmdList cmdlist_index={} cmd id=0x{:03X} value=0x{:08X} mask=0x{:X} extra_len={} grouped={}",
                      cmdlist_index, header.cmd_id.Value(), value, header.parameter_mask.Value(),
                      header.extra_data_length.Value(), header.group_commands.Value());
        }

        WriteInternalReg(header.cmd_id, value, header.parameter_mask);

        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];

            if (trace_hotpath) {
                LOG_DEBUG(HW_GPU,
                          "PicaCore::ProcessCmdList cmdlist_index={} extra cmd id=0x{:03X} value=0x{:08X} mask=0x{:X}",
                          cmdlist_index, cmd, extra_value, header.parameter_mask.Value());
            }

            WriteInternalReg(cmd, extra_value, header.parameter_mask);
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::ProcessCmdList end cmdlist_index={} list={:#010X} processed_words={} length={}",
                  cmdlist_index, list, cmd_list.current_index, cmd_list.length);
    }
}

void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask) {
    if (id >= RegsInternal::NUM_REGS) {
        LOG_ERROR(
            HW_GPU,
            "Commandlist tried to write to invalid register 0x{:03X} (value: {:08X}, mask: {:X})",
            id, value, mask);
        return;
    }

    // Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF
    constexpr std::array<u32, 16> ExpandBitsToBytes = {
        0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff,
        0x00ffff00, 0x00ffffff, 0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff,
        0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
    };

    // TODO: Figure out how register masking acts on e.g. vs.uniform_setup.set_value
    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    regs.internal.reg_array[id] = (old_value & ~write_mask) | (value & write_mask);

    LogMainConfigTransition(regs.internal, id, old_value, regs.internal.reg_array[id], mask);

    // Track register write.
    DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);

    // Track events.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::PicaCommandLoaded, &id);
        SCOPE_EXIT({ debug_context->OnEvent(DebugContext::Event::PicaCommandProcessed, &id); });
    }

    if (IsPicaHotpathTraceEnabled() && IsInterestingPicaStateReg(id)) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::WriteInternalReg interesting_state_reg id=0x{:03X} value=0x{:08X} mask=0x{:X}",
                  id, regs.internal.reg_array[id], mask);
    }

    switch (id) {
    // Trigger IRQ
    case PICA_REG_INDEX(trigger_irq):
        signal_interrupt(Service::GSP::InterruptId::P3D);
        break;

    case PICA_REG_INDEX(pipeline.triangle_topology):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg triangle_topology={}",
                      static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        }
        primitive_assembler.Reconfigure(regs.internal.pipeline.triangle_topology);
        break;

    case PICA_REG_INDEX(pipeline.restart_primitive):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg restart_primitive value=0x{:08X}", value);
        }
        primitive_assembler.Reset();
        break;

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index):
        immediate.Reset();
        break;

    // Load default vertex input attributes
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]):
        SubmitImmediate(value);
        break;

    case PICA_REG_INDEX(pipeline.gpu_mode):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg gpu_mode value=0x{:08X}", value);
        }
        // This register likely just enables vertex processing and doesn't need any special handling
        break;

    case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
    case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
        const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
        const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::WriteInternalReg command_buffer.trigger index={} addr={:#010X} size={}",
                      index, addr, size);
        }
        const u8* head = memory.GetPhysicalPointer(addr);
        cmd_list.Reset(addr, head, size);
        break;
    }

    // It seems like these trigger vertex rendering
    case PICA_REG_INDEX(pipeline.trigger_draw):
    case PICA_REG_INDEX(pipeline.trigger_draw_indexed): {
        const bool is_indexed = (id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::WriteInternalReg trigger_draw id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={}",
                      id, is_indexed, regs.internal.pipeline.num_vertices,
                      regs.internal.pipeline.vertex_offset,
                      static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                      static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
        }
        if (IsPicaDrawTraceEnabled()) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA trigger_draw id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                     id, is_indexed, regs.internal.pipeline.num_vertices,
                     regs.internal.pipeline.vertex_offset,
                     static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                     static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                     regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            LogPicaTextureState(regs.internal, "pre_draw");
        }
        DrawArrays(is_indexed);
        break;
    }

    case PICA_REG_INDEX(gs.bool_uniforms):
        gs_setup.WriteUniformBoolReg(regs.internal.gs.bool_uniforms.Value());
        break;

    case PICA_REG_INDEX(gs.int_uniforms[0]):
    case PICA_REG_INDEX(gs.int_uniforms[1]):
    case PICA_REG_INDEX(gs.int_uniforms[2]):
    case PICA_REG_INDEX(gs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(gs.int_uniforms[0]));
        gs_setup.WriteUniformIntReg(index, regs.internal.gs.GetIntUniform(index));
        break;
    }

    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]): {
        gs_setup.WriteUniformFloatReg(regs.internal.gs, value);
        break;
    }

    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset >= 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {}", offset);
        } else {
            gs_setup.program_code[offset] = value;
            gs_setup.MarkProgramCodeDirty();
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset >= gs_setup.swizzle_data.size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {}", offset);
        } else {
            gs_setup.swizzle_data[offset] = value;
            gs_setup.MarkSwizzleDataDirty();
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.output_mask):
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            regs.internal.gs.output_mask.Assign(value);
        }
        break;

    case PICA_REG_INDEX(vs.bool_uniforms):
        vs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        }
        break;

    case PICA_REG_INDEX(vs.int_uniforms[0]):
    case PICA_REG_INDEX(vs.int_uniforms[1]):
    case PICA_REG_INDEX(vs.int_uniforms[2]):
    case PICA_REG_INDEX(vs.int_uniforms[3]): {
        const u32 index = (id - PICA_REG_INDEX(vs.int_uniforms[0]));
        vs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        }
        break;
    }

    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto index = vs_setup.WriteUniformFloatReg(regs.internal.vs, value);
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No && index) {
            gs_setup.uniforms.f[index.value()] = vs_setup.uniforms.f[index.value()];
        }
        break;
    }

    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset >= 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {}", offset);
        } else {
            vs_setup.program_code[offset] = value;
            vs_setup.MarkProgramCodeDirty();
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration) {
                gs_setup.program_code[offset] = value;
                gs_setup.MarkProgramCodeDirty();
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset >= vs_setup.swizzle_data.size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {}", offset);
        } else {
            vs_setup.swizzle_data[offset] = value;
            vs_setup.MarkSwizzleDataDirty();
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration) {
                gs_setup.swizzle_data[offset] = value;
                gs_setup.MarkSwizzleDataDirty();
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;
        ASSERT_MSG(lut_config.index < 256, "lut_config.index exceeded maximum value of 255!");

        lighting.luts[lut_config.type][lut_config.index].raw = value;
        lut_config.index.Assign(lut_config.index + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]): {
        fog.lut[regs.internal.texturing.fog_lut_offset % 128].raw = value;
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;

        switch (regs.internal.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            proctex.noise_table[index % proctex.noise_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            proctex.color_map_table[index % proctex.color_map_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            proctex.alpha_map_table[index % proctex.alpha_map_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            proctex.color_table[index % proctex.color_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            proctex.color_diff_table[index % proctex.color_diff_table.size()].raw = value;
            break;
        }
        index.Assign(index + 1);
        break;
    }
    default:
        break;
    }

    // Notify the rasterizer an internal register was updated.
    rasterizer->NotifyPicaRegisterChanged(id);
}

void PicaCore::SubmitImmediate(u32 value) {
    // Push to word to the queue. This returns true when a full attribute is formed.
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t IMMEDIATE_MODE_INDEX = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > IMMEDIATE_MODE_INDEX) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    // Retrieve the attribute and place it in the default attribute buffer.
    const auto attribute = immediate.queue.Get();
    if (setup.index < IMMEDIATE_MODE_INDEX) {
        input_default_attributes[setup.index] = attribute;
        setup.index++;
        return;
    }

    // When index is 0xF the attribute is used for immediate mode drawing.
    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    // We formed a vertex, flush.
    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    BORKED3DS_PROFILE("PicaCore", "Draw Immediate");
    if (IsPicaHotpathTraceEnabled()) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawImmediate invoked");
    }
    if (IsPicaDrawTraceEnabled()) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA immediate_draw topology={} current_attr={} max_attr={}",
                 static_cast<u32>(primitive_assembler.GetTopology()), immediate.current_attribute,
                 regs.internal.pipeline.max_input_attrib_index.Value());
    }

    // Compile the vertex shader.
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    // Invoke the vertex shader for the vertex.
    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    // Reconfigure geometry pipeline if needed.
    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    // Send to geometry pipeline.
    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

    // Flush the immediate triangle.
    if (IsPicaDrawTraceEnabled()) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA immediate -> rasterizer->DrawTriangles()");
    }
    rasterizer->DrawTriangles();
    immediate.current_attribute = 0;
}

void PicaCore::DrawArrays(bool is_indexed) {
    BORKED3DS_PROFILE("PicaCore", "Draw Arrays");
    const u64 draw_index = ++g_pica_draw_counter;
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();
    const bool trace_draw = IsPicaDrawTraceEnabled();
    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::DrawArrays begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={}",
                  draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                  regs.internal.pipeline.vertex_offset,
                  Settings::values.use_hw_shader.GetValue(), Settings::values.skip_slow_draw.GetValue(),
                  static_cast<u32>(primitive_assembler.GetTopology()),
                  static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={} primitive_empty={}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 regs.internal.pipeline.vertex_offset,
                 Settings::values.use_hw_shader.GetValue(), Settings::values.skip_slow_draw.GetValue(),
                 static_cast<u32>(primitive_assembler.GetTopology()),
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()), primitive_assembler.IsEmpty());
        LogPicaTextureState(regs.internal, "drawarrays_begin");
    }

    if (IsEnvEnabled("BORKED3DS_V3DV_BYPASS_FIRST_DRAW") && draw_index == 1) {
        LOG_WARNING(HW_GPU,
                    "V3DV test bypass: skipping first PICA draw because BORKED3DS_V3DV_BYPASS_FIRST_DRAW=1");
        return;
    }

    // Track vertex in the debug recorder.
    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    bool accelerate_draw = [this] {
        // Geometry shaders cannot be accelerated due to register preservation.
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        // TODO (wwylele): for Strip/Fan topology, if the primitive assember is not restarted
        // after this draw call, the buffered vertex from this draw should "leak" to the next
        // draw, in which case we should buffer the vertex into the software primitive assember,
        // or disable accelerate draw completely. However, there is not game found yet that does
        // this, so this is left unimplemented for now. Revisit this when an issue is found in
        // games.

        bool accelerate_draw = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            accelerate_draw = accelerate_draw && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return accelerate_draw;
    }();

    // Pi 5 / V3DV strict-compat startup workaround:
    // The fragile startup draws are not contiguous in the global draw stream, so track them with
    // a dedicated counter instead of the global draw_index. Skip the first fragile draws entirely,
    // then keep a short software-fallback tail for the next fragile draws before exposing the next
    // real failure beyond this startup family.
    const bool is_fragile_startup_draw = accelerate_draw && IsStrictCompatEnabled() && is_indexed &&
                                         regs.internal.pipeline.num_vertices == 6 &&
                                         primitive_assembler.IsEmpty() &&
                                         ArePrimaryTexturesDisabled(regs.internal);
    const u32 textures_disabled = ArePrimaryTexturesDisabled(regs.internal) ? 1u : 0u;

    if (is_fragile_startup_draw) {
        const u64 fragile_startup_index = ++g_fragile_startup_draw_counter;

        if (fragile_startup_index <= 16) {
            if (trace_draw) {
                LOG_INFO(HW_GPU,
                         "TRACE_DRAW_PICA strict_compat skipping fragile startup draw draw_index={} fragile_startup_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled=1 startup_skip_window=4",
                         draw_index, fragile_startup_index, is_indexed,
                         regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty());
            }
            return;
        }

        if (fragile_startup_index <= 24) {
            if (trace_draw) {
                LOG_INFO(HW_GPU,
                         "TRACE_DRAW_PICA strict_compat forcing software fallback draw_index={} fragile_startup_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled=1 extended_startup_window=8",
                         draw_index, fragile_startup_index, is_indexed,
                         regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty());
            }
            accelerate_draw = false;
        }
    }

    const bool is_large_textured_startup_draw =
        IsStrictCompatEnabled() && accelerate_draw && !is_fragile_startup_draw && is_indexed &&
        primitive_assembler.IsEmpty() && textures_disabled == 0 &&
        regs.internal.pipeline.num_vertices > 12 &&
        regs.internal.pipeline.num_vertices <= 48;

    if (is_large_textured_startup_draw) {
        const u64 large_textured_startup_index = ++g_large_textured_startup_draw_counter;
        if (large_textured_startup_index <= 2) {
            if (trace_draw) {
                LOG_INFO(HW_GPU,
                         "TRACE_DRAW_PICA strict_compat forcing software fallback for large textured startup draw draw_index={} large_textured_startup_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} large_textured_startup_window=2",
                         draw_index, large_textured_startup_index, is_indexed,
                         regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty(),
                         textures_disabled);
                LogPicaTextureState(regs.internal, "large_textured_startup_fallback");
            }
            accelerate_draw = false;
        }
    }

    const bool tiny_textured_startup_candidate =
        IsStrictCompatEnabled() && !is_fragile_startup_draw && is_indexed &&
        primitive_assembler.IsEmpty() && textures_disabled == 0 &&
        regs.internal.pipeline.num_vertices == 6;

    if (tiny_textured_startup_candidate && !accelerate_draw &&
        Settings::values.use_hw_shader.GetValue()) {
        if (trace_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA strict_compat forcing tiny textured startup draw acceleration draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} tiny_textured_startup_force_v2=1",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled);
            LogPicaTextureState(regs.internal, "tiny_textured_startup_force_v2");
        }
        accelerate_draw = true;
    }

    const bool is_tiny_textured_startup_draw = tiny_textured_startup_candidate && accelerate_draw;

    if (is_tiny_textured_startup_draw) {
        const u64 tiny_textured_startup_index = ++g_tiny_textured_startup_draw_counter;
        if (trace_draw && tiny_textured_startup_index <= 16) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA strict_compat allowing_tiny_textured_startup_draw_v2 draw_index={} tiny_textured_startup_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} tiny_textured_startup_window_v2=1",
                     draw_index, tiny_textured_startup_index, is_indexed,
                     regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty(),
                     textures_disabled);
            LogPicaTextureState(regs.internal, "tiny_textured_startup_accel_v2");
        }
        if (trace_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA tiny_textured_pica_step_1_after_allow_v3 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} accelerate_draw={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled, accelerate_draw);
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays accelerate_draw={}", accelerate_draw);
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA accelerate_draw={} gs_mode={} topology={}",
                 accelerate_draw, static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(primitive_assembler.GetTopology()));
        if (is_tiny_textured_startup_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA tiny_textured_pica_step_2_before_accel_state_v3 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} accelerate_draw={} gs_mode={} topology={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled, accelerate_draw,
                     static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                     static_cast<u32>(primitive_assembler.GetTopology()));
        }
    }

    const bool reason_use_gs = regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes;
    const bool reason_primitive_not_empty = !primitive_assembler.IsEmpty();
    const auto topology = primitive_assembler.GetTopology();
    const bool topology_requires_multiple_of_3 =
        topology == PipelineRegs::TriangleTopology::Shader ||
        topology == PipelineRegs::TriangleTopology::List;
    const bool reason_topology_not_multiple_of_3 =
        topology_requires_multiple_of_3 && (regs.internal.pipeline.num_vertices % 3) != 0;
    const bool reason_hw_shader_disabled = !Settings::values.use_hw_shader.GetValue();

    if (!is_fragile_startup_draw && !g_logged_first_non_fragile_draw.exchange(true)) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA first_non_fragile_draw draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} use_gs={} topology={} accelerate_draw={} reason_hw_shader_disabled={} reason_use_gs={} reason_primitive_not_empty={} reason_topology_not_multiple_of_3={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled,
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(topology), accelerate_draw, reason_hw_shader_disabled,
                 reason_use_gs, reason_primitive_not_empty, reason_topology_not_multiple_of_3,
                 regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        LogPicaTextureState(regs.internal, "first_non_fragile_draw");
    }

    if (!is_fragile_startup_draw && textures_disabled == 0 &&
        !g_logged_first_non_fragile_textured_draw.exchange(true)) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA first_non_fragile_textured_draw draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} use_gs={} topology={} accelerate_draw={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled,
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(topology), accelerate_draw,
                 regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        LogPicaTextureState(regs.internal, "first_non_fragile_textured_draw");
    }

    if (!is_fragile_startup_draw && !accelerate_draw &&
        !g_logged_first_non_fragile_software_draw.exchange(true)) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA first_non_fragile_software_draw draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} use_gs={} topology={} reason_hw_shader_disabled={} reason_use_gs={} reason_primitive_not_empty={} reason_topology_not_multiple_of_3={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled,
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(topology), reason_hw_shader_disabled, reason_use_gs,
                 reason_primitive_not_empty, reason_topology_not_multiple_of_3,
                 regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        LogPicaTextureState(regs.internal, "first_non_fragile_software_draw");
    }

    if (trace_draw && is_tiny_textured_startup_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA tiny_textured_pica_step_2b_before_first_accel_candidate_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} accelerate_draw={} topology={}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled, accelerate_draw,
                 static_cast<u32>(primitive_assembler.GetTopology()));
    }

    if (accelerate_draw && !is_fragile_startup_draw &&
        !g_logged_first_non_fragile_accel_candidate.exchange(true)) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA first_non_fragile_accelerated_draw draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled,
                 static_cast<u32>(primitive_assembler.GetTopology()),
                 regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                 regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        LogPicaTextureState(regs.internal, "first_non_fragile_accelerated_draw");
    }

    if (trace_draw && is_tiny_textured_startup_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA tiny_textured_pica_step_2c_after_first_accel_candidate_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} accelerate_draw={} topology={}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 primitive_assembler.IsEmpty(), textures_disabled, accelerate_draw,
                 static_cast<u32>(primitive_assembler.GetTopology()));
    }

    // Attempt to use hardware vertex shaders if possible.
    if (accelerate_draw) {
        if (trace_draw && is_tiny_textured_startup_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA tiny_textured_pica_step_2d_before_first_accel_attempt_flag_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()));
        }

        const bool first_non_fragile_accel_attempt_log =
            !is_fragile_startup_draw && !g_logged_first_non_fragile_accel_attempt.exchange(true);

        if (trace_draw && is_tiny_textured_startup_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA tiny_textured_pica_step_2e_after_first_accel_attempt_flag_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} first_attempt_log={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(first_non_fragile_accel_attempt_log));
        }

        if (first_non_fragile_accel_attempt_log) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA first_non_fragile_accel_attempt draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()));
        }
        if (trace_draw && is_tiny_textured_startup_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA tiny_textured_pica_step_3_before_accel_call_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()));
            LogPicaTextureState(regs.internal, "tiny_textured_before_accel_call_v4");
        }
        if (trace_draw) {
            LOG_INFO(HW_GPU, "TRACE_DRAW_PICA calling AccelerateDrawBatch indexed={}", is_indexed);
        }
        const bool accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays AccelerateDrawBatch returned {}", accelerated);
        }
        if (trace_draw) {
            LOG_INFO(HW_GPU, "TRACE_DRAW_PICA AccelerateDrawBatch returned {}", accelerated);
            if (is_tiny_textured_startup_draw) {
                LOG_INFO(HW_GPU,
                         "TRACE_DRAW_PICA tiny_textured_pica_step_4_after_accel_call_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} accelerated={}",
                         draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                         primitive_assembler.IsEmpty(), textures_disabled, accelerated);
            }
        }
        if (!accelerated && !is_fragile_startup_draw &&
            !g_logged_first_non_fragile_accel_failed.exchange(true)) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA first_non_fragile_accel_failed draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()),
                     regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            LogPicaTextureState(regs.internal, "first_non_fragile_accel_failed");
        }
        if (accelerated) {
            if (trace_draw) {
                if (is_tiny_textured_startup_draw) {
                    LOG_INFO(HW_GPU,
                             "TRACE_DRAW_PICA tiny_textured_pica_step_5_returning_early_v4 draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={}",
                             draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                             primitive_assembler.IsEmpty(), textures_disabled);
                }
                LOG_INFO(HW_GPU, "TRACE_DRAW_PICA returning early after accelerated draw");
            }
            return;
        }
    } else if (Settings::values.skip_slow_draw) {
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays skipping slow draw");
        }
        if (trace_draw) {
            LOG_INFO(HW_GPU, "TRACE_DRAW_PICA skip_slow_draw prevented software fallback");
            LogPicaTextureState(regs.internal, "skip_slow_draw");
        }
        return;
    }

    // We cannot accelerate the draw, so load and execute the vertex shader for each vertex.
    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays falling back to software vertex path");
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA falling back to software vertex path");
    }
    LoadVertices(is_indexed);

    // Draw emitted triangles.
    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays calling rasterizer->DrawTriangles()");
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA software path -> rasterizer->DrawTriangles()");
        LogPicaTextureState(regs.internal, "before_draw_triangles");
    }
    rasterizer->DrawTriangles();
}

void PicaCore::LoadVertices(bool is_indexed) {
    // Read and validate vertex information from the loaders
    const auto& pipeline = regs.internal.pipeline;
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();
    const bool trace_draw = IsPicaDrawTraceEnabled();
    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::LoadVertices begin indexed={} num_vertices={} vertex_offset={} base_address={:#010X}",
                  is_indexed, pipeline.num_vertices, pipeline.vertex_offset,
                  pipeline.vertex_attributes.GetPhysicalBaseAddress());
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA load_vertices begin indexed={} num_vertices={} vertex_offset={} base_address={:#010X}",
                 is_indexed, pipeline.num_vertices, pipeline.vertex_offset,
                 pipeline.vertex_attributes.GetPhysicalBaseAddress());
    }
    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto loader = VertexLoader(memory, pipeline);
    regs.internal.rasterizer.ValidateSemantics();

    // Locate index buffer.
    const auto& index_info = pipeline.index_array;
    const bool index_u16 = index_info.format != 0;
    const PAddr index_base = base_address + index_info.offset;
    const u32 index_stride = index_u16 ? sizeof(u16) : sizeof(u8);
    const u32 needed_index_bytes = pipeline.num_vertices * index_stride;
    const MemoryRef index_ref = memory.GetPhysicalRef(index_base);
    const u8* index_bytes = index_ref.GetPtr();
    const u32 available_index_bytes =
        static_cast<u32>(std::min<std::size_t>(index_ref.GetSize(), needed_index_bytes));
    const u32 available_indices = index_stride != 0 ? (available_index_bytes / index_stride) : 0;

    if (is_indexed && available_index_bytes < needed_index_bytes) {
        LOG_ERROR(HW_GPU,
                  "PicaCore::LoadVertices index buffer truncated addr={:#010X} need={} have={} format={}",
                  index_base, needed_index_bytes, available_index_bytes,
                  static_cast<u32>(index_info.format));
    }

    const auto read_index = [index_bytes, available_indices, index_u16](u32 index) -> u32 {
        if (index >= available_indices || index_bytes == nullptr) {
            return 0;
        }
        if (!index_u16) {
            return index_bytes[index];
        }
        u16 value = 0;
        std::memcpy(&value, index_bytes + index * sizeof(u16), sizeof(u16));
        return value;
    };

    // Simple circular-replacement vertex cache
    const std::size_t VERTEX_CACHE_SIZE = 64;
    std::array<bool, VERTEX_CACHE_SIZE> vertex_cache_valid{};
    std::array<u16, VERTEX_CACHE_SIZE> vertex_cache_ids;
    std::array<AttributeBuffer, VERTEX_CACHE_SIZE> vertex_cache;
    u32 vertex_cache_pos = 0;

    // Compile the vertex shader for this batch.
    ShaderUnit shader_unit;
    AttributeBuffer vs_output;
    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    // Setup geometry pipeline in case we are using a geometry shader.
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        // Indexed rendering doesn't use the start offset
        const u32 vertex = is_indexed ? read_index(index) : (index + pipeline.vertex_offset);

        if (trace_hotpath && index < 4) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::LoadVertices vertex_index={} source_vertex={} indexed={}",
                      index, vertex, is_indexed);
        }
        if (trace_draw && is_indexed && index >= available_indices && index < 4) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA load_vertices missing_index index={} available_indices={} substituting_vertex=0",
                     index, available_indices);
        }

        bool vertex_cache_hit = false;
        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            for (u32 i = 0; i < VERTEX_CACHE_SIZE; ++i) {
                if (vertex_cache_valid[i] && vertex == vertex_cache_ids[i]) {
                    vs_output = vertex_cache[i];
                    vertex_cache_hit = true;
                    break;
                }
            }
        }

        if (!vertex_cache_hit) {
            // Initialize data for the current vertex
            AttributeBuffer input;
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            // Record vertex processing to the debugger.
            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            // Invoke the vertex shader for this vertex.
            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            // Cache the vertex when doing indexed rendering.
            if (is_indexed) {
                vertex_cache[vertex_cache_pos] = vs_output;
                vertex_cache_valid[vertex_cache_pos] = true;
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache_pos = (vertex_cache_pos + 1) % VERTEX_CACHE_SIZE;
            }
        }

        // Send to geometry pipeline
        geometry_pipeline.SubmitVertex(vs_output);
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::LoadVertices end indexed={} num_vertices={}", is_indexed,
                  pipeline.num_vertices);
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA load_vertices end indexed={} num_vertices={}",
                 is_indexed, pipeline.num_vertices);
    }
}

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
    // Initialize command list tracking.
    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    constexpr size_t max_iterations = 0x100;

    RenderPropertiesGuess find_info{};

    find_info.vp_height = regs.internal.rasterizer.viewport_size_y.Value();
    find_info.paddr = regs.internal.framebuffer.framebuffer.color_buffer_address.Value() * 8;

    auto process_write = [this, &find_info](u32 cmd_id, u32 value) {
        switch (cmd_id) {
        case PICA_REG_INDEX(rasterizer.viewport_size_y):
            find_info.vp_height = value;
            find_info.vp_heigh_found = true;
            break;
        case PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address):
            find_info.paddr = value * 8;
            find_info.paddr_found = true;
            break;
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
            const u32 index =
                static_cast<u32>(cmd_id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
            const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
            const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
            const u8* head = memory.GetPhysicalPointer(addr);
            cmd_list.Reset(addr, head, size);
            break;
        }
        default:
            break;
        }
        return find_info.vp_heigh_found && find_info.paddr_found;
    };

    size_t iterations = 0;
    while (cmd_list.current_index < cmd_list.length && iterations < max_iterations) {
        // Align read pointer to 8 bytes
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        // Read the header and the value to write.
        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        // Write to the requested PICA register.
        if (process_write(header.cmd_id, value))
            break;

        // Write any extra paramters as well.
        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];
            if (process_write(cmd, extra_value))
                break;
        }

        iterations++;
    }

    return find_info;
}

template <class Archive>
void PicaCore::CommandList::serialize(Archive& ar, const u32 file_version) {
    ar & addr;
    ar & length;
    ar & current_index;
    if (Archive::is_loading::value) {
        const u8* ptr = Core::System::GetInstance().Memory().GetPhysicalPointer(addr);
        head = reinterpret_cast<const u32*>(ptr);
    }
}

SERIALIZE_IMPL(PicaCore::CommandList)

} // namespace Pica
