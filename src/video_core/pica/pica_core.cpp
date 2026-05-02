// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <utility>

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

[[nodiscard]] bool IsPicaAccelAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_PICA_ACCEL");
}

[[nodiscard]] bool IsPicaAccelForcedOff() {
    return IsEnvEnabled("BORKED3DS_V3DV_FORCE_PICA_SOFTWARE");
}

[[nodiscard]] u32 GetEnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return fallback;
    }

    constexpr unsigned long max_u32 = 0xFFFFFFFFul;
    return parsed > max_u32 ? 0xFFFFFFFFu : static_cast<u32>(parsed);
}


[[nodiscard]] bool IsV114C6PicaGateFileTraceEnabled() {
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE");
}

void V114C6PicaGateFileTraceRaw(const char* message) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v114c7_pica_gate.log", "a")) {
        std::fputs(message, fp);
        std::fputc('\n', fp);
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceU32(const char* key, u32 value) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v114c7_pica_gate.log", "a")) {
        std::fprintf(fp, "%s=0x%08X\n", key, value);
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceU64(const char* key, u64 value) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v114c7_pica_gate.log", "a")) {
        std::fprintf(fp, "%s=%llu\n", key, static_cast<unsigned long long>(value));
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceReset() {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v114c7_pica_gate.log", "w")) {
        std::fputs("v114c7 pica_gate_file_trace_reset\n", fp);
        std::fclose(fp);
    }
}

[[nodiscard]] bool IsSafePicaHwDrawAllowed() {
    // v114 follows plan de travail 1, with the result from the v110 runtime log:
    // v110 proved the backend can emit raw_enter_noargs and continue until hotkey exit.
    // Therefore v114 keeps pica_core pre/post handoff logs suppressed, disables raw-enter-only
    // return, emits raw_enter_noargs + raw_enter_simple in vk_rasterizer, and returns before
    // stage=1/shader work.
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_SAFE_PICA_HW_DRAWS") &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_DRAWS");
}

[[nodiscard]] bool IsStagedPicaHwPreflightRequested() {
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) > 0;
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffEnabled() {
    // v114 diagnostic:
    // v108 reached the safe micro-HW candidate with enter=1, then the log began a new
    // "HW.GPU <" line but never completed pre_call_direct_noargs and no TRACE_ACCEL_STAGE line
    // appeared. Keep direct handoff enabled, but let the normal v114 test suppress the pica_core
    // pre-call log and make the backend return silently before any raw_enter logging.
    return IsStrictCompatEnabled() &&
           (IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF") ||
            (IsStagedPicaHwPreflightRequested() &&
             IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffNoPrelogEnabled() {
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG");
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffPostlogEnabled() {
    // Disabled by default for v114. Enable only if the silent call-boundary probe already proved
    // stable and we want a pica_core post-call marker.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_POSTLOG");
}

[[nodiscard]] bool IsSafePicaHwEnterAllowed() {
    // v114 diagnostic guard:
    // v110 survived raw_enter_noargs. Keep staged preflight entry active and suppress
    // pica_core pre/post call logs while the backend emits raw_enter_simple and returns.
    //
    // This still does NOT enable broad PICA acceleration:
    //   - the caller must already have strict_safe_pica_hw_draw=true;
    //   - the per-run budget and max-vertex filters still apply;
    //   - the backend must still stop at BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER before stage=7.
    const bool explicit_enter = IsEnvEnabled("BORKED3DS_V3DV_ENTER_SAFE_PICA_HW_DRAWS") ||
                                IsEnvEnabled("BORKED3DS_V3DV_EXECUTE_SAFE_PICA_HW_DRAWS");

    return (explicit_enter || IsStagedPicaHwPreflightRequested()) &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_ENTER");
}

[[nodiscard]] bool IsSafePicaHwDryRunEnabled() {
    return !IsSafePicaHwEnterAllowed() &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_DRY_RUN");
}

[[nodiscard]] bool IsVerbosePicaMicroTextureTraceEnabled() {
    // v87 appeared to terminate during the verbose texture-state block around the first
    // micro candidate. Keep the candidate log minimal by default; this opt-in exists only
    // for diagnosis once the handoff is stable.
    return IsEnvEnabled("BORKED3DS_V3DV_VERBOSE_PICA_MICRO_TEXTURE_TRACE");
}

[[nodiscard]] u32 GetSafePicaHwDrawBudget() {
    // Keep this tiny. If this path is correct, one or a few hardware draws should already
    // change the render target from pure black. Higher budgets belong in later passes.
    return GetEnvU32("BORKED3DS_V3DV_SAFE_PICA_HW_DRAW_BUDGET", 1);
}

[[nodiscard]] u32 GetSafePicaHwMaxVertices() {
    return GetEnvU32("BORKED3DS_V3DV_SAFE_PICA_HW_MAX_VERTICES", 6);
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

    V114C6PicaGateFileTraceRaw("v114c7 main_config_transition");
    V114C6PicaGateFileTraceU32("v114c7 main_config_old", old_value);
    V114C6PicaGateFileTraceU32("v114c7 main_config_new", new_value);
    V114C6PicaGateFileTraceU32("v114c7 main_config_mask", mask);
}


std::atomic<u64> g_pica_draw_counter{0};
std::atomic<u64> g_pica_safe_hw_draw_counter{0};
std::atomic<bool> g_logged_strict_accel_gate{false};

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())} {
    InitializeRegs();

    V114C6PicaGateFileTraceReset();
    V114C6PicaGateFileTraceRaw("v114c7 pica_core_constructor_predecision_marker");

    if (IsStrictCompatEnabled()) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v114c7 pica_core_constructor_predecision_marker direct_handoff={} no_prelog={} file_trace={} silent_stages={} generate_guarded_probe={} stage_stop_after={}",
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_SILENT_STAGES")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")),
                    GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0));
    }

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

    // Set framebuffer defaults from nn::gx::Initialize.
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

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt_) {
    signal_interrupt = signal_interrupt_;
}

void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) {
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList begin list={:#010X} size={} ignore_list={}",
                  list, size, ignore_list);
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
                      "PicaCore::ProcessCmdList cmd id=0x{:03X} value=0x{:08X} mask=0x{:X} extra_len={} grouped={}",
                      header.cmd_id.Value(), value, header.parameter_mask.Value(),
                      header.extra_data_length.Value(), header.group_commands.Value());
        }

        WriteInternalReg(header.cmd_id, value, header.parameter_mask);

        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];

            if (trace_hotpath) {
                LOG_DEBUG(HW_GPU,
                          "PicaCore::ProcessCmdList extra cmd id=0x{:03X} value=0x{:08X} mask=0x{:X}",
                          cmd, extra_value, header.parameter_mask.Value());
            }

            WriteInternalReg(cmd, extra_value, header.parameter_mask);
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList end list={:#010X} processed_words={} length={}",
                  list, cmd_list.current_index, cmd_list.length);
    }
}

void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask) {
    if (id >= RegsInternal::NUM_REGS) {
        LOG_ERROR(HW_GPU,
                  "Commandlist tried to write to invalid register 0x{:03X} (value: {:08X}, mask: {:X})",
                  id, value, mask);
        return;
    }

    // Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF.
    constexpr std::array<u32, 16> ExpandBitsToBytes = {
        0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff,
        0x00ffff00, 0x00ffffff, 0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff,
        0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
    };

    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    regs.internal.reg_array[id] = (old_value & ~write_mask) | (value & write_mask);

    LogMainConfigTransition(regs.internal, id, old_value, regs.internal.reg_array[id], mask);
    DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);

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

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]):
        SubmitImmediate(value);
        break;

    case PICA_REG_INDEX(pipeline.gpu_mode):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg gpu_mode value=0x{:08X}", value);
        }
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
        V114C6PicaGateFileTraceRaw("v114c7 trigger_draw_before_drawarrays");
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_id", id);
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_indexed", static_cast<u32>(is_indexed));
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_num_vertices", regs.internal.pipeline.num_vertices);
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_topology", static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_color_addr", regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
        V114C6PicaGateFileTraceU32("v114c7 trigger_draw_depth_addr", regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());

        if (IsStrictCompatEnabled()) {
            static std::atomic<u64> trigger_console_counter{0};
            const u64 trigger_console_index = ++trigger_console_counter;
            if (trigger_console_index <= 32 || trigger_console_index == 64 ||
                trigger_console_index == 128 || trigger_console_index == 256 ||
                trigger_console_index == 512) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v114c7 trigger_draw_before_drawarrays console_index={} id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                            trigger_console_index, id, is_indexed,
                            regs.internal.pipeline.num_vertices, regs.internal.pipeline.vertex_offset,
                            static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                            static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                            regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                            regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
        }

        DrawArrays(is_indexed);
        V114C6PicaGateFileTraceRaw("v114c7 trigger_draw_after_drawarrays");

        if (IsStrictCompatEnabled()) {
            static std::atomic<u64> trigger_after_console_counter{0};
            const u64 trigger_after_console_index = ++trigger_after_console_counter;
            if (trigger_after_console_index <= 32 || trigger_after_console_index == 64 ||
                trigger_after_console_index == 128 || trigger_after_console_index == 256 ||
                trigger_after_console_index == 512) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v114c7 trigger_draw_after_drawarrays console_index={} id=0x{:03X}",
                            trigger_after_console_index, id);
            }
        }
        break;
    }

    case PICA_REG_INDEX(gs.bool_uniforms):
        gs_setup.WriteUniformBoolReg(regs.internal.gs.bool_uniforms.Value());
        break;

    case PICA_REG_INDEX(gs.int_uniforms[0]):
    case PICA_REG_INDEX(gs.int_uniforms[1]):
    case PICA_REG_INDEX(gs.int_uniforms[2]):
    case PICA_REG_INDEX(gs.int_uniforms[3]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(gs.int_uniforms[0]));
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
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]):
        gs_setup.WriteUniformFloatReg(regs.internal.gs, value);
        break;

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
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(vs.int_uniforms[0]));
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
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        fog.lut[regs.internal.texturing.fog_lut_offset % 128].raw = value;
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset + 1);
        break;

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

    rasterizer->NotifyPicaRegisterChanged(id);
}

void PicaCore::SubmitImmediate(u32 value) {
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t ImmediateModeIndex = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > ImmediateModeIndex) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    const auto attribute = immediate.queue.Get();
    if (setup.index < ImmediateModeIndex) {
        input_default_attributes[setup.index] = attribute;
        setup.index++;
        return;
    }

    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    BORKED3DS_PROFILE("PicaCore", "Draw Immediate");

    if (IsPicaDrawTraceEnabled()) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA immediate_draw topology={} current_attr={} max_attr={}",
                 static_cast<u32>(primitive_assembler.GetTopology()), immediate.current_attribute,
                 regs.internal.pipeline.max_input_attrib_index.Value());
    }

    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

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

    V114C6PicaGateFileTraceRaw("v114c7 drawarrays_enter");
    V114C6PicaGateFileTraceU64("v114c7 draw_index", draw_index);
    V114C6PicaGateFileTraceU32("v114c7 draw_indexed", static_cast<u32>(is_indexed));
    V114C6PicaGateFileTraceU32("v114c7 draw_num_vertices", regs.internal.pipeline.num_vertices);
    V114C6PicaGateFileTraceU32("v114c7 draw_topology", static_cast<u32>(primitive_assembler.GetTopology()));
    V114C6PicaGateFileTraceU32("v114c7 draw_use_gs", static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
    V114C6PicaGateFileTraceU32("v114c7 draw_primitive_empty", static_cast<u32>(primitive_assembler.IsEmpty()));

    if (IsStrictCompatEnabled()) {
        static std::atomic<u64> drawarrays_console_counter{0};
        const u64 drawarrays_console_index = ++drawarrays_console_counter;
        if (drawarrays_console_index <= 32 || drawarrays_console_index == 64 ||
            drawarrays_console_index == 128 || drawarrays_console_index == 256 ||
            drawarrays_console_index == 512) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114c7 drawarrays_enter console_index={} draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} topology={} use_gs={} primitive_empty={}",
                        drawarrays_console_index, draw_index, is_indexed,
                        regs.internal.pipeline.num_vertices, regs.internal.pipeline.vertex_offset,
                        static_cast<u32>(Settings::values.use_hw_shader.GetValue()),
                        static_cast<u32>(primitive_assembler.GetTopology()),
                        static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                        static_cast<u32>(primitive_assembler.IsEmpty()));
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::DrawArrays begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={}",
                  draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                  regs.internal.pipeline.vertex_offset, Settings::values.use_hw_shader.GetValue(),
                  Settings::values.skip_slow_draw.GetValue(),
                  static_cast<u32>(primitive_assembler.GetTopology()),
                  static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={} primitive_empty={}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 regs.internal.pipeline.vertex_offset, Settings::values.use_hw_shader.GetValue(),
                 Settings::values.skip_slow_draw.GetValue(),
                 static_cast<u32>(primitive_assembler.GetTopology()),
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()), primitive_assembler.IsEmpty());

        if (draw_index == 1) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114c7 pica_core_shader_multiplex_predecision_marker stage_stop_after={} force_stage_trace={} entry_only_probe={} enter_safe_hw={} safe_budget={} safe_max_vertices={}",
                        GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE")),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE")),
                        static_cast<u32>(IsSafePicaHwEnterAllowed()),
                        GetSafePicaHwDrawBudget(), GetSafePicaHwMaxVertices());
        }

        LogPicaTextureState(regs.internal, "drawarrays_begin");
    }

    // v114-C7: the v114-C6 log reached trigger_draw_before_drawarrays, drawarrays_enter,
    // and the multiplex marker, but not safe_hw_decision. Move a tiny, untextured
    // predecision path before debug_context / broad fallback logic so the next log tells us
    // whether the crash is in the PICA decision block or inside AccelerateDrawBatch stage 1..7.
    if (IsStrictCompatEnabled() && IsSafePicaHwDrawAllowed() && IsSafePicaHwEnterAllowed() &&
        IsDirectSafePicaHwHandoffEnabled() && !IsPicaAccelAllowed() && !IsPicaAccelForcedOff()) {
        V114C6PicaGateFileTraceRaw("v114c7 early_predecision_begin");
        const bool v114c7_early_hw_shader = Settings::values.use_hw_shader.GetValue();
        const bool v114c7_early_primitive_empty = primitive_assembler.IsEmpty();
        const u32 v114c7_early_topology = static_cast<u32>(primitive_assembler.GetTopology());
        const u32 v114c7_early_use_gs = static_cast<u32>(regs.internal.pipeline.use_gs.Value());
        const u32 v114c7_early_textures_disabled = ArePrimaryTexturesDisabled(regs.internal) ? 1u : 0u;
        const u32 v114c7_early_max_vertices = GetSafePicaHwMaxVertices();
        const bool v114c7_early_accelerate_shape =
            v114c7_early_hw_shader && v114c7_early_primitive_empty &&
            regs.internal.pipeline.use_gs != PipelineRegs::UseGS::Yes &&
            regs.internal.pipeline.num_vertices > 0 &&
            regs.internal.pipeline.num_vertices <= v114c7_early_max_vertices &&
            (primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::Shader ||
             primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::List
                 ? ((regs.internal.pipeline.num_vertices % 3) == 0)
                 : true);
        const bool v114c7_early_safe_candidate =
            v114c7_early_accelerate_shape && v114c7_early_textures_disabled != 0;

        V114C6PicaGateFileTraceU32("v114c7 early_hw_shader", static_cast<u32>(v114c7_early_hw_shader));
        V114C6PicaGateFileTraceU32("v114c7 early_primitive_empty", static_cast<u32>(v114c7_early_primitive_empty));
        V114C6PicaGateFileTraceU32("v114c7 early_topology", v114c7_early_topology);
        V114C6PicaGateFileTraceU32("v114c7 early_use_gs", v114c7_early_use_gs);
        V114C6PicaGateFileTraceU32("v114c7 early_textures_disabled", v114c7_early_textures_disabled);
        V114C6PicaGateFileTraceU32("v114c7 early_safe_candidate", static_cast<u32>(v114c7_early_safe_candidate));

        static std::atomic<u64> v114c7_predecision_console_counter{0};
        const u64 v114c7_predecision_console_index = ++v114c7_predecision_console_counter;
        if (v114c7_predecision_console_index <= 16) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114c7 early_predecision console_index={} draw_index={} indexed={} num_vertices={} hw_shader={} primitive_empty={} textures_disabled={} topology={} use_gs={} candidate={} budget={} max_vertices={}",
                        v114c7_predecision_console_index, draw_index, is_indexed,
                        regs.internal.pipeline.num_vertices,
                        static_cast<u32>(v114c7_early_hw_shader),
                        static_cast<u32>(v114c7_early_primitive_empty),
                        v114c7_early_textures_disabled, v114c7_early_topology,
                        v114c7_early_use_gs, static_cast<u32>(v114c7_early_safe_candidate),
                        GetSafePicaHwDrawBudget(), v114c7_early_max_vertices);
        }

        if (v114c7_early_safe_candidate) {
            const u64 v114c7_early_hw_index = ++g_pica_safe_hw_draw_counter;
            const bool v114c7_early_budget_ok =
                GetSafePicaHwDrawBudget() != 0 &&
                v114c7_early_hw_index <= GetSafePicaHwDrawBudget();
            V114C6PicaGateFileTraceU64("v114c7 early_safe_hw_index", v114c7_early_hw_index);
            V114C6PicaGateFileTraceU32("v114c7 early_budget_ok", static_cast<u32>(v114c7_early_budget_ok));

            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114c7 early_safe_hw_decision draw_index={} indexed={} num_vertices={} safe_hw_index={} budget_ok={} stage_stop_after={} generate_guarded_probe={}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        v114c7_early_hw_index, static_cast<u32>(v114c7_early_budget_ok),
                        GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));

            if (v114c7_early_budget_ok) {
                if (IsSafePicaHwDryRunEnabled()) {
                    LOG_WARNING(HW_GPU,
                                "TRACE_DRAW_PICA strict_compat v114c7 early_dry_run_consumed draw_index={} safe_hw_index={}",
                                draw_index, v114c7_early_hw_index);
                    return;
                }

                V114C6PicaGateFileTraceRaw("v114c7 early_before_accelerate_draw_batch");
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v114c7 early_before_accelerate_draw_batch draw_index={} indexed={} num_vertices={} safe_hw_index={} stage_stop_after={}",
                            draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                            v114c7_early_hw_index,
                            GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0));
                const bool v114c7_early_accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
                V114C6PicaGateFileTraceRaw("v114c7 early_after_accelerate_draw_batch");
                V114C6PicaGateFileTraceU32("v114c7 early_accelerate_draw_batch_result",
                                           static_cast<u32>(v114c7_early_accelerated));
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v114c7 early_after_accelerate_draw_batch draw_index={} result={}",
                            draw_index, static_cast<u32>(v114c7_early_accelerated));
                return;
            }
        }
    }

    if (IsEnvEnabled("BORKED3DS_V3DV_BYPASS_FIRST_DRAW") && draw_index == 1) {
        LOG_WARNING(HW_GPU,
                    "V3DV test bypass: skipping first PICA draw because BORKED3DS_V3DV_BYPASS_FIRST_DRAW=1");
        return;
    }

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    bool accelerate_draw = [this] {
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        bool can_accelerate = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            can_accelerate = can_accelerate && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return can_accelerate;
    }();

    const u32 textures_disabled = ArePrimaryTexturesDisabled(regs.internal) ? 1u : 0u;
    const u32 safe_hw_max_vertices = GetSafePicaHwMaxVertices();
    const bool strict_safe_pica_hw_candidate =
        accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !IsPicaAccelForcedOff() && IsSafePicaHwDrawAllowed() &&
        Settings::values.use_hw_shader.GetValue() && primitive_assembler.IsEmpty() &&
        regs.internal.pipeline.use_gs != PipelineRegs::UseGS::Yes && textures_disabled != 0 &&
        regs.internal.pipeline.num_vertices > 0 &&
        regs.internal.pipeline.num_vertices <= safe_hw_max_vertices;

    u64 strict_safe_pica_hw_index = 0;
    const bool strict_safe_pica_hw_draw = [&] {
        if (!strict_safe_pica_hw_candidate) {
            return false;
        }
        strict_safe_pica_hw_index = ++g_pica_safe_hw_draw_counter;
        const u32 budget = GetSafePicaHwDrawBudget();
        return budget != 0 && strict_safe_pica_hw_index <= budget;
    }();

    const bool strict_safe_pica_hw_enter =
        strict_safe_pica_hw_draw && IsSafePicaHwEnterAllowed();
    const bool strict_safe_pica_hw_dry_run =
        strict_safe_pica_hw_draw && IsSafePicaHwDryRunEnabled();

    // v114: do not copy GVX64's final software-style behavior. Keep broad PICA acceleration
    // blocked. A tiny HW-shader candidate may enter AccelerateDrawBatch only as a grouped
    // stage-6 preflight. The backend may pass geometry-shader gate, topology, vertex analysis,
    // and SetupVertexArray(), but must return before shader setup, pipeline binding,
    // descriptors, or vkCmdDraw/vkCmdDrawIndexed.
    if (accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !strict_safe_pica_hw_draw) {
        if (trace_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA strict_compat v114 forcing software path before broad PICA acceleration draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} allow_pica_accel=0 safe_hw_candidate={} safe_hw_allowed={} safe_hw_enter={} safe_hw_dry_run={} safe_hw_index={} safe_hw_budget={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()),
                     static_cast<u32>(strict_safe_pica_hw_candidate),
                     static_cast<u32>(IsSafePicaHwDrawAllowed()),
                     static_cast<u32>(strict_safe_pica_hw_enter),
                     static_cast<u32>(strict_safe_pica_hw_dry_run), strict_safe_pica_hw_index,
                     GetSafePicaHwDrawBudget());
            LogPicaTextureState(regs.internal, "v114_force_software_before_broad_accel");
        }
        if (!g_logged_strict_accel_gate.exchange(true)) {
            LOG_WARNING(HW_GPU,
                        "Pi5/V3DV strict compatibility v114: broad PICA AccelerateDrawBatch remains disabled by default; tiny untextured HW candidates may enter AccelerateDrawBatch only with the v114 raw-enter-simple handoff probe enabled. Keep BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF=1, BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG=1, BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_POSTLOG=0, BORKED3DS_V3DV_ACCEL_SILENT_ENTRY_RETURN=0, BORKED3DS_V3DV_ACCEL_RAW_ENTER_RETURN=0, BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN=0, BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE=0, BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=1, BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE=1, BORKED3DS_V3DV_ENTER_SAFE_PICA_HW_DRAWS=1, BORKED3DS_V3DV_USE_TRIVIAL_VERTEX_SHADER_PROBE=0, BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_CONFIG_ONLY=0, BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_ONLY=0, and BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY=1 for normal v114 tests");
        }
        accelerate_draw = false;
    } else if (strict_safe_pica_hw_draw && trace_draw) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v114 safe micro PICA HW candidate hw_index={} budget={} draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} use_hw_shader={} enter={} dry_run={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                    strict_safe_pica_hw_index, GetSafePicaHwDrawBudget(), draw_index, is_indexed,
                    regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty(),
                    textures_disabled, static_cast<u32>(primitive_assembler.GetTopology()),
                    static_cast<u32>(Settings::values.use_hw_shader.GetValue()),
                    static_cast<u32>(strict_safe_pica_hw_enter),
                    static_cast<u32>(strict_safe_pica_hw_dry_run),
                    regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                    regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        if (IsVerbosePicaMicroTextureTraceEnabled()) {
            LogPicaTextureState(regs.internal, "v114_safe_micro_hw_candidate");
        }
    }

    V114C6PicaGateFileTraceRaw("v114c7 safe_hw_decision");
    V114C6PicaGateFileTraceU32("v114c7 accelerate_draw", static_cast<u32>(accelerate_draw));
    V114C6PicaGateFileTraceU32("v114c7 textures_disabled", textures_disabled);
    V114C6PicaGateFileTraceU32("v114c7 safe_hw_candidate", static_cast<u32>(strict_safe_pica_hw_candidate));
    V114C6PicaGateFileTraceU32("v114c7 safe_hw_draw", static_cast<u32>(strict_safe_pica_hw_draw));
    V114C6PicaGateFileTraceU32("v114c7 safe_hw_enter", static_cast<u32>(strict_safe_pica_hw_enter));
    V114C6PicaGateFileTraceU32("v114c7 safe_hw_dry_run", static_cast<u32>(strict_safe_pica_hw_dry_run));
    V114C6PicaGateFileTraceU64("v114c7 safe_hw_index", strict_safe_pica_hw_index);

    if (IsStrictCompatEnabled()) {
        static std::atomic<u64> safe_decision_console_counter{0};
        const u64 safe_decision_console_index = ++safe_decision_console_counter;
        if (safe_decision_console_index <= 32 || safe_decision_console_index == 64 ||
            safe_decision_console_index == 128 || safe_decision_console_index == 256 ||
            safe_decision_console_index == 512) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114c7 safe_hw_decision console_index={} draw_index={} accelerate_draw={} textures_disabled={} safe_hw_candidate={} safe_hw_draw={} safe_hw_enter={} safe_hw_dry_run={} safe_hw_index={} safe_hw_budget={} safe_hw_max_vertices={}",
                        safe_decision_console_index, draw_index, static_cast<u32>(accelerate_draw),
                        textures_disabled, static_cast<u32>(strict_safe_pica_hw_candidate),
                        static_cast<u32>(strict_safe_pica_hw_draw),
                        static_cast<u32>(strict_safe_pica_hw_enter),
                        static_cast<u32>(strict_safe_pica_hw_dry_run), strict_safe_pica_hw_index,
                        GetSafePicaHwDrawBudget(), GetSafePicaHwMaxVertices());
        }
    }

    const bool strict_direct_safe_hw_handoff =
        accelerate_draw && strict_safe_pica_hw_draw && strict_safe_pica_hw_enter &&
        IsDirectSafePicaHwHandoffEnabled();

    if (strict_direct_safe_hw_handoff) {
        V114C6PicaGateFileTraceRaw("v114c7 before_accelerate_draw_batch");
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v114c7 before_accelerate_draw_batch draw_index={} indexed={} num_vertices={} safe_hw_index={} stage_stop_after={} generate_guarded_probe={}",
                    draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                    strict_safe_pica_hw_index,
                    GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));
        // v114:
        // v110 proved raw_enter_noargs survives. Keep pica_core handoff logging suppressed
        // and let the backend emit TRACE_ACCEL_STAGE v114 raw_enter_noargs plus raw_enter_simple
        // before returning with BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN=0.
        if (trace_draw && !IsDirectSafePicaHwHandoffNoPrelogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 pre_call_direct_noargs AccelerateDrawBatch");
        }

        const bool accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
        V114C6PicaGateFileTraceRaw("v114c7 after_accelerate_draw_batch");
        V114C6PicaGateFileTraceU32("v114c7 accelerate_draw_batch_result", static_cast<u32>(accelerated));
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v114c7 after_accelerate_draw_batch draw_index={} result={}",
                    draw_index, static_cast<u32>(accelerated));

        if (trace_draw && IsDirectSafePicaHwHandoffPostlogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 post_call_direct AccelerateDrawBatch returned {}",
                        static_cast<u32>(accelerated));
        }

        if (accelerated) {
            return;
        }

        // Do not call the backend twice if the direct diagnostic path returns false.
        accelerate_draw = false;
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays accelerate_draw={}", accelerate_draw);
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA accelerate_draw={} gs_mode={} topology={}",
                 accelerate_draw, static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(primitive_assembler.GetTopology()));
    }

    if (strict_safe_pica_hw_dry_run) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114 handoff dry-run consumed safe micro PICA HW candidate before AccelerateDrawBatch draw_index={} indexed={} num_vertices={} hw_index={} budget={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        strict_safe_pica_hw_index, GetSafePicaHwDrawBudget(),
                        regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            if (IsVerbosePicaMicroTextureTraceEnabled()) {
                LogPicaTextureState(regs.internal, "v114_safe_micro_hw_dry_run_consumed");
            }
        }
        return;
    }

    // v114 last-chance guard: only explicit full diagnosis or explicitly executed tiny HW
    // probes may enter AccelerateDrawBatch in strict mode.
    if (accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !strict_safe_pica_hw_enter) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114 late guard blocked AccelerateDrawBatch draw_index={} indexed={} num_vertices={} textures_disabled={} topology={} safe_hw_candidate={} safe_hw_allowed={} safe_hw_enter={} safe_hw_dry_run={}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        textures_disabled, static_cast<u32>(primitive_assembler.GetTopology()),
                        static_cast<u32>(strict_safe_pica_hw_candidate),
                        static_cast<u32>(IsSafePicaHwDrawAllowed()),
                        static_cast<u32>(strict_safe_pica_hw_enter),
                        static_cast<u32>(strict_safe_pica_hw_dry_run));
            LogPicaTextureState(regs.internal, "v114_late_guard_force_software");
        }
        accelerate_draw = false;
    }

    if (accelerate_draw) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 pre_call AccelerateDrawBatch indexed={} draw_index={} num_vertices={} hw_index={} stop_after={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        is_indexed, draw_index, regs.internal.pipeline.num_vertices,
                        strict_safe_pica_hw_index, GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                        regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }
        const bool accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays AccelerateDrawBatch returned {}", accelerated);
        }
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                     "TRACE_DRAW_PICA v114 post_call AccelerateDrawBatch returned {} indexed={} draw_index={} num_vertices={} hw_index={}",
                     accelerated, is_indexed, draw_index, regs.internal.pipeline.num_vertices,
                     strict_safe_pica_hw_index);
        }
        if (accelerated) {
            if (trace_draw) {
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

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays falling back to software vertex path");
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA falling back to software vertex path");
    }
    LoadVertices(is_indexed);

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

    const auto& index_info = pipeline.index_array;
    const bool index_u16 = index_info.format != 0;
    const PAddr index_base = base_address + index_info.offset;
    const u32 index_stride = index_u16 ? sizeof(u16) : sizeof(u8);
    const u32 needed_index_bytes = pipeline.num_vertices * index_stride;

    const auto index_ref = memory.GetPhysicalRef(index_base);
    const auto index_ref_size = index_ref.GetSize();
    const u32 available_index_bytes = static_cast<u32>(
        std::min<std::size_t>(static_cast<std::size_t>(index_ref_size),
                              static_cast<std::size_t>(needed_index_bytes)));
    const u32 available_indices = index_stride != 0 ? (available_index_bytes / index_stride) : 0;
    const u8* index_bytes = index_ref.GetPtr();

    if (is_indexed && available_index_bytes < needed_index_bytes) {
        LOG_ERROR(HW_GPU,
                  "PicaCore::LoadVertices index array truncated base={:#010X} needed={} available={} num_vertices={} u16={}",
                  index_base, needed_index_bytes, available_index_bytes, pipeline.num_vertices,
                  index_u16);
    }

    const auto read_index = [&](u32 index) -> u32 {
        if (!is_indexed) {
            return index + pipeline.vertex_offset;
        }
        if (available_indices == 0) {
            return pipeline.vertex_offset;
        }
        const u32 clamped_index = std::min(index, available_indices - 1);
        if (index_u16) {
            u16 value{};
            std::memcpy(&value, index_bytes + clamped_index * sizeof(u16), sizeof(u16));
            return static_cast<u32>(value) + pipeline.vertex_offset;
        }
        return static_cast<u32>(index_bytes[clamped_index]) + pipeline.vertex_offset;
    };

    constexpr std::size_t VertexCacheSize = 64;
    std::array<bool, VertexCacheSize> vertex_cache_valid{};
    std::array<u32, VertexCacheSize> vertex_cache_ids{};
    std::array<AttributeBuffer, VertexCacheSize> vertex_cache{};
    u32 vertex_cache_pos = 0;

    ShaderUnit shader_unit;
    AttributeBuffer vs_output{};

    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        const u32 vertex = read_index(index);
        bool vertex_cache_hit = false;

        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            for (std::size_t i = 0; i < VertexCacheSize; ++i) {
                if (vertex_cache_valid[i] && vertex_cache_ids[i] == vertex) {
                    vs_output = vertex_cache[i];
                    vertex_cache_hit = true;
                    break;
                }
            }
        }

        if (!vertex_cache_hit) {
            AttributeBuffer input{};
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            if (is_indexed) {
                vertex_cache_valid[vertex_cache_pos] = true;
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache[vertex_cache_pos] = vs_output;
                vertex_cache_pos = (vertex_cache_pos + 1) % VertexCacheSize;
            }
        }

        geometry_pipeline.SubmitVertex(vs_output);
    }

    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA load_vertices end indexed={} num_vertices={}", is_indexed,
                 pipeline.num_vertices);
    }
}

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
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
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        if (process_write(header.cmd_id, value))
            break;

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
