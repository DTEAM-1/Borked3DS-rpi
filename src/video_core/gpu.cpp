// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdlib>

#include "common/archives.h"
#include "common/common_types.h"
#include "common/hacks/hack_manager.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/service/gsp/gsp_gpu.h"
#include "core/hle/service/plgldr/plgldr.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/gpu.h"
#include "video_core/gpu_debugger.h"
#include "video_core/gpu_impl.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/regs_lcd.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_software/sw_blitter.h"
#include "video_core/right_eye_disabler.h"
#include "video_core/video_core.h"

namespace VideoCore {

constexpr VAddr VADDR_LCD = 0x1ED02000;
constexpr VAddr VADDR_GPU = 0x1EF00000;

// Frame skip
/// True if the current frame was skipped
bool g_skip_frame;
/// True if the last frame was skipped
static bool last_skip_frame;
/// Total number of frames drawn
static u64 frame_count;

namespace {

bool IsA7Z68GpuExecuteSubmitTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BORKED3DS_V3DV_A7Z68_GPU_EXECUTE_SUBMIT_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool IsV405SkipRightEyeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BORKED3DS_V3DV_V405_SKIP_RIGHT_EYE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool IsV408OeilDroitEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BORKED3DS_V3DV_V408_OEIL_DROIT");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

} // namespace

// V408 : classe de la destination du display transfer en cours, lue par le rasterizer Vulkan
// (vk_rasterizer.cpp, AccelerateDisplayTransfer) juste apres : 0 = autre, 1 = ecran du haut
// gauche, 2 = ecran du haut droite. Une adresse droite egale a une adresse gauche (configuration
// mono) est classee gauche.
std::atomic<u32> g_v408_classe{0};

// V405 (MESURE, inactif par defaut) : BORKED3DS_V3DV_V405_SKIP_RIGHT_EYE=1.
// TB115 (Castlevania) : le jeu dessine chaque image deux fois, oeil gauche et oeil droit, draw par
// draw en alternance, meme avec le curseur 3D annonce a 0 ; a chaque image, le tampon de l'oeil
// droit est copie (display transfer) vers le framebuffer droit de l'ecran du haut. 3D coupee,
// l'emulateur n'affiche que l'oeil gauche : tout le travail de l'oeil droit est perdu, et
// l'alternance ferme et rouvre le render pass Vulkan ~240 fois par image (GPU a tuiles).
// Ici on repere le tampon SOURCE d'un display transfer dont la destination est exactement l'une
// des deux adresses droites de l'ecran du haut (et differente des adresses gauches). Le
// rasterizer Vulkan saute ensuite les draws dont le tampon couleur est exactement ce tampon
// (voir vk_rasterizer.cpp, Draw), seulement si la 3D est coupee, l'oeil affiche est le gauche,
// et le reperage a ete confirme dans les 12 derniers transferts.
// Ce n'est PAS le mecanisme d'Azahar (right_eye_disabler, qui bloque des listes de commandes
// entieres et coupait la moitie de l'ecran de Kid Icarus) : ici, seuls les draws vers le tampon
// exact de l'oeil droit sont sautes.
std::atomic<u32> g_v405_right_src{0};
std::atomic<u64> g_v405_transfer_seq{0};
std::atomic<u64> g_v405_confirm_seq{0};

GPU::GPU(Core::System& system, Frontend::EmuWindow& emu_window,
         Frontend::EmuWindow* secondary_window)
    : right_eye_disabler{std::make_unique<RightEyeDisabler>(*this)},
      impl{std::make_unique<Impl>(system, emu_window, secondary_window)} {
    frame_count = 0;
    last_skip_frame = false;
    g_skip_frame = false;
    impl->vblank_event = impl->timing.RegisterEvent(
        "GPU::VBlankCallback",
        [this](uintptr_t user_data, s64 cycles_late) { VBlankCallback(user_data, cycles_late); });
    impl->timing.ScheduleEvent(FRAME_TICKS, impl->vblank_event);

    // Bind the rasterizer to the PICA GPU
    impl->pica.BindRasterizer(impl->rasterizer);

    if (IsA7Z68GpuExecuteSubmitTraceEnabled()) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "constructor_gpu_execute_submit_trace active=1");
    }
}

GPU::~GPU() = default;

PAddr GPU::VirtualToPhysicalAddress(VAddr addr) {
    if (addr == 0) {
        return 0;
    }

    if (addr >= Memory::VRAM_VADDR && addr <= Memory::VRAM_VADDR_END) {
        return addr - Memory::VRAM_VADDR + Memory::VRAM_PADDR;
    }
    if (addr >= Memory::LINEAR_HEAP_VADDR && addr <= Memory::LINEAR_HEAP_VADDR_END) {
        return addr - Memory::LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::NEW_LINEAR_HEAP_VADDR && addr <= Memory::NEW_LINEAR_HEAP_VADDR_END) {
        return addr - Memory::NEW_LINEAR_HEAP_VADDR + Memory::FCRAM_PADDR;
    }
    if (addr >= Memory::PLUGIN_3GX_FB_VADDR && addr <= Memory::PLUGIN_3GX_FB_VADDR_END) {
        auto plg_ldr = Service::PLGLDR::GetService(impl->system);
        if (plg_ldr) {
            return addr - Memory::PLUGIN_3GX_FB_VADDR + plg_ldr->GetPluginFBAddr();
        }
    }

    LOG_ERROR(HW_Memory, "Unknown virtual address @ 0x{:08X}", addr);
    return addr;
}

void GPU::SetInterruptHandler(Service::GSP::InterruptHandler handler) {
    impl->signal_interrupt = handler;
    impl->pica.SetInterruptHandler(handler);
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    impl->rasterizer->FlushRegion(addr, size);
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    impl->rasterizer->InvalidateRegion(addr, size);
}

void GPU::ClearAll(bool flush) {
    impl->rasterizer->ClearAll(flush);
}

void GPU::Execute(const Service::GSP::Command& command) {
    using Service::GSP::CommandId;
    auto& regs = impl->pica.regs;
    const bool a7z68_trace = IsA7Z68GpuExecuteSubmitTraceEnabled();

    if (a7z68_trace) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "execute_enter cmd_id={} is_submit={}",
                    command.id.Value(), command.id == CommandId::SubmitCmdList ? 1 : 0);
    }

    switch (command.id) {
    case CommandId::RequestDma: {
        impl->system.Memory().RasterizerFlushVirtualRegion(
            command.dma_request.source_address, command.dma_request.size, Memory::FlushMode::Flush);
        impl->system.Memory().RasterizerFlushVirtualRegion(command.dma_request.dest_address,
                                                           command.dma_request.size,
                                                           Memory::FlushMode::Invalidate);

        // TODO(Subv): These memory accesses should not go through the application's memory mapping.
        // They should go through the GSP module's memory mapping.
        const auto process = impl->system.Kernel().GetCurrentProcess();
        impl->memory.CopyBlock(*process, command.dma_request.dest_address,
                               command.dma_request.source_address, command.dma_request.size);
        impl->signal_interrupt(Service::GSP::InterruptId::DMA);
        break;
    }
    case CommandId::SubmitCmdList: {
        auto& params = command.submit_gpu_cmdlist;
        auto& cmdbuffer = regs.internal.pipeline.command_buffer;
        const PAddr physical_address = VirtualToPhysicalAddress(params.address);
        const u32 command_list_size = params.size;

        if (a7z68_trace) {
            LOG_WARNING(HW_GPU,
                        "TRACE_GPU strict_compat v115d_a7z68 "
                        "submit_case_enter vaddr={:#010X} paddr={:#010X} size={:#010X} "
                        "size_words={}",
                        params.address, physical_address, command_list_size, command_list_size >> 3);
        }

        // Write to the command buffer GPU registers
        cmdbuffer.addr[0].Assign(physical_address >> 3);
        cmdbuffer.size[0].Assign(command_list_size >> 3);
        cmdbuffer.trigger[0] = 1;

        if (a7z68_trace) {
            LOG_WARNING(HW_GPU,
                        "TRACE_GPU strict_compat v115d_a7z68 "
                        "submit_before_submitcmdlist index=0 paddr={:#010X} size={:#010X}",
                        physical_address, command_list_size);
        }

        // Trigger processing of the command list
        SubmitCmdList(0);

        if (a7z68_trace) {
            LOG_WARNING(HW_GPU,
                        "TRACE_GPU strict_compat v115d_a7z68 "
                        "submit_after_submitcmdlist index=0 paddr={:#010X} size={:#010X}",
                        physical_address, command_list_size);
        }
        break;
    }
    case CommandId::MemoryFill: {
        auto& params = command.memory_fill;
        auto& memfill = regs.memory_fill_config;

        // Write to the memory fill GPU registers.
        if (params.start1 != 0) {
            memfill[0].address_start = VirtualToPhysicalAddress(params.start1) >> 3;
            memfill[0].address_end = VirtualToPhysicalAddress(params.end1) >> 3;
            memfill[0].value_32bit = params.value1;
            memfill[0].control = params.control1;
            MemoryFill(0);
        }
        if (params.start2 != 0) {
            memfill[1].address_start = VirtualToPhysicalAddress(params.start2) >> 3;
            memfill[1].address_end = VirtualToPhysicalAddress(params.end2) >> 3;
            memfill[1].value_32bit = params.value2;
            memfill[1].control = params.control2;
            MemoryFill(1);
        }
        break;
    }
    case CommandId::DisplayTransfer: {
        auto& params = command.display_transfer;
        auto& display_transfer = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        display_transfer.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        display_transfer.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        display_transfer.input_size = params.in_buffer_size;
        display_transfer.output_size = params.out_buffer_size;
        display_transfer.flags = params.flags;
        display_transfer.trigger.Assign(1);

        // Trigger the display transfer.
        MemoryTransfer();
        break;
    }
    case CommandId::TextureCopy: {
        auto& params = command.texture_copy;
        auto& texture_copy = regs.display_transfer_config;

        // Write to the transfer engine GPU registers.
        texture_copy.input_address = VirtualToPhysicalAddress(params.in_buffer_address) >> 3;
        texture_copy.output_address = VirtualToPhysicalAddress(params.out_buffer_address) >> 3;
        texture_copy.texture_copy.size = params.size;
        texture_copy.texture_copy.input_size = params.in_width_gap;
        texture_copy.texture_copy.output_size = params.out_width_gap;
        texture_copy.flags = params.flags;
        texture_copy.trigger.Assign(1);

        // Trigger the texture copy.
        MemoryTransfer();
        break;
    }
    case CommandId::CacheFlush: {
        // Rasterizer flushing handled elsewhere in CPU read/write and other GPU handlers
        // Use command.cache_flush.regions to implement this handler
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Unknown command {:#08X}", command.id.Value());
    }

    if (a7z68_trace) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "execute_leave cmd_id={} is_submit={}",
                    command.id.Value(), command.id == CommandId::SubmitCmdList ? 1 : 0);
    }

    // Notify debugger that a GSP command was processed.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::GSPCommandProcessed, &command);
    }
}

void GPU::SetBufferSwap(u32 screen_id, const Service::GSP::FrameBufferInfo& info) {
    const PAddr phys_address_left = VirtualToPhysicalAddress(info.address_left);
    const PAddr phys_address_right = VirtualToPhysicalAddress(info.address_right);

    // Update framebuffer properties.
    auto& framebuffer = impl->pica.regs.framebuffer_config[screen_id];
    if (info.active_fb == 0) {
        framebuffer.address_left1 = phys_address_left;
        framebuffer.address_right1 = phys_address_right;
    } else {
        framebuffer.address_left2 = phys_address_left;
        framebuffer.address_right2 = phys_address_right;
    }

    framebuffer.stride = info.stride;
    framebuffer.format = info.format;
    framebuffer.active_fb = info.shown_fb;

    // Notify debugger about the buffer swap.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::BufferSwapped, nullptr);
    }

    if (screen_id == 0) {
        impl->system.perf_stats->EndGameFrame();
        right_eye_disabler->ReportEndFrame();
    }
}

void GPU::SetColorFill(const Pica::ColorFill& fill) {
    impl->pica.regs_lcd.color_fill_top = fill;
    impl->pica.regs_lcd.color_fill_bottom = fill;
}

u32 GPU::ReadReg(VAddr addr) {
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        return impl->pica.regs_lcd[index];
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        return impl->pica.regs.reg_array[index];
    }
    default:
        UNREACHABLE_MSG("Read from unknown GPU address {:#08X}", addr);
    }
}

void GPU::WriteReg(VAddr addr, u32 data) {
    switch (addr & 0xFFFFF000) {
    case VADDR_LCD: {
        const u32 offset = addr - VADDR_LCD;
        const u32 index = offset / sizeof(u32);
        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::RegsLcd::NumIds());
        impl->pica.regs_lcd[index] = data;
        break;
    }
    case VADDR_GPU:
    case VADDR_GPU + 0x1000: {
        const u32 offset = addr - VADDR_GPU;
        const u32 index = offset / sizeof(u32);

        ASSERT(addr % sizeof(u32) == 0);
        ASSERT(index < Pica::PicaCore::Regs::NUM_REGS);
        impl->pica.regs.reg_array[index] = data;

        // Handle registers that trigger GPU actions
        switch (index) {
        case GPU_REG_INDEX(memory_fill_config[0].trigger):
            MemoryFill(0);
            break;
        case GPU_REG_INDEX(memory_fill_config[1].trigger):
            MemoryFill(1);
            break;
        case GPU_REG_INDEX(display_transfer_config.trigger):
            MemoryTransfer();
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[0]):
            SubmitCmdList(0);
            break;
        case GPU_REG_INDEX(internal.pipeline.command_buffer.trigger[1]):
            SubmitCmdList(1);
            break;
        default:
            break;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("Write to unknown GPU address {:#08X}", addr);
    }
}

void GPU::Sync() {
    impl->renderer->Sync();
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

Pica::PicaCore& GPU::PicaCore() {
    return impl->pica;
}

const Pica::PicaCore& GPU::PicaCore() const {
    return impl->pica;
}

Pica::DebugContext& GPU::DebugContext() {
    return *Pica::g_debug_context;
}

GraphicsDebugger& GPU::Debugger() {
    return impl->gpu_debugger;
}

void GPU::ReportLoadingProgramID(u64 program_ID) {
    auto hack = Common::Hacks::hack_manager.GetHack(
        Common::Hacks::HackType::ACCURATE_MULTIPLICATION, program_ID);
    bool use_accurate_mul = Settings::values.shaders_accurate_mul.GetValue();
    if (hack) {
        switch (hack->mode) {
        case Common::Hacks::HackAllowMode::DISALLOW:
            use_accurate_mul = false;
            break;
        case Common::Hacks::HackAllowMode::FORCE:
            use_accurate_mul = true;
            break;
        case Common::Hacks::HackAllowMode::ALLOW:
        default:
            break;
        }
    }
    impl->rasterizer->SetAccurateMul(use_accurate_mul);
}

void GPU::SubmitCmdList(u32 index) {
    // Check if a command list was triggered.
    auto& config = impl->pica.regs.internal.pipeline.command_buffer;
    const bool a7z68_trace = IsA7Z68GpuExecuteSubmitTraceEnabled();
    if (!config.trigger[index]) {
        if (a7z68_trace) {
            LOG_WARNING(HW_GPU,
                        "TRACE_GPU strict_compat v115d_a7z68 "
                        "submitcmdlist_no_trigger index={}",
                        index);
        }
        return;
    }

    BORKED3DS_PROFILE("GPU", "Command List Processing");

    // Forward command list processing to the PICA core.
    const PAddr addr = config.GetPhysicalAddress(index);
    const u32 size = config.GetSize(index);
    const bool skip_cmd_queue = !right_eye_disabler->ShouldAllowCmdQueueTrigger(addr, size);

    if (a7z68_trace) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "submitcmdlist_enter index={} paddr={:#010X} size={:#010X} "
                    "skip_cmd_queue={}",
                    index, addr, size, skip_cmd_queue ? 1 : 0);
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "before_process_cmd_list index={} paddr={:#010X} size={:#010X}",
                    index, addr, size);
    }

    impl->pica.ProcessCmdList(addr, size, skip_cmd_queue);

    if (a7z68_trace) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "after_process_cmd_list index={} paddr={:#010X} size={:#010X}",
                    index, addr, size);
    }

    config.trigger[index] = 0;

    if (a7z68_trace) {
        LOG_WARNING(HW_GPU,
                    "TRACE_GPU strict_compat v115d_a7z68 "
                    "submitcmdlist_trigger_reset index={} paddr={:#010X} size={:#010X}",
                    index, addr, size);
    }
}

void GPU::MemoryFill(u32 index) {
    // Check if a memory fill was triggered.
    auto& config = impl->pica.regs.memory_fill_config[index];
    if (!config.trigger) {
        return;
    }

    // Perform memory fill.
    if (!impl->rasterizer->AccelerateFill(config)) {
        impl->sw_blitter->MemoryFill(config);
    }

    // It seems that it won't signal interrupt if "address_start" is zero.
    // TODO: hwtest this
    if (config.GetStartAddress() != 0) {
        if (!index) {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC0);
        } else {
            impl->signal_interrupt(Service::GSP::InterruptId::PSC1);
        }
    }

    // Reset "trigger" flag and set the "finish" flag
    // This was confirmed to happen on hardware even if "address_start" is zero.
    config.trigger.Assign(0);
    config.finished.Assign(1);
}

void GPU::MemoryTransfer() {
    // Check if a transfer was triggered.
    auto& config = impl->pica.regs.display_transfer_config;
    if (!config.trigger.Value()) {
        return;
    }

    // Notify debugger about the display transfer.
    if (impl->debug_context) {
        impl->debug_context->OnEvent(Pica::DebugContext::Event::IncomingDisplayTransfer, nullptr);
    }

    // Perform memory transfer
    if (config.is_texture_copy) {
        BORKED3DS_PROFILE("GPU", "Texture Copy");
        if (!impl->rasterizer->AccelerateTextureCopy(config)) {
            impl->sw_blitter->TextureCopy(config);
        }
    } else {
        BORKED3DS_PROFILE("GPU", "Display Transfer");
        if (IsV405SkipRightEyeEnabled()) {
            const u64 v405_seq = g_v405_transfer_seq.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto& top = impl->pica.regs.framebuffer_config[0];
            const PAddr v405_src = config.GetPhysicalInputAddress();
            const PAddr v405_dst = config.GetPhysicalOutputAddress();
            const bool v405_right1 = v405_dst != 0 && v405_dst == top.address_right1 &&
                                     top.address_right1 != top.address_left1 &&
                                     top.address_right1 != top.address_left2;
            const bool v405_right2 = v405_dst != 0 && v405_dst == top.address_right2 &&
                                     top.address_right2 != top.address_left1 &&
                                     top.address_right2 != top.address_left2;
            // Adresses de l'ecran du haut journalisees a chaque changement (8 fois au plus) :
            // si le reperage echoue, le log dit pourquoi (droite = gauche, ou zero).
            static u32 v405_lcd_last[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
            static u32 v405_lcd_logs = 0;
            const u32 v405_lcd_now[4] = {top.address_left1, top.address_left2,
                                         top.address_right1, top.address_right2};
            if (v405_lcd_logs < 8 &&
                (v405_lcd_now[0] != v405_lcd_last[0] || v405_lcd_now[1] != v405_lcd_last[1] ||
                 v405_lcd_now[2] != v405_lcd_last[2] || v405_lcd_now[3] != v405_lcd_last[3])) {
                ++v405_lcd_logs;
                for (int i = 0; i < 4; ++i) {
                    v405_lcd_last[i] = v405_lcd_now[i];
                }
                LOG_WARNING(HW_GPU,
                            "V405_ECRAN_HAUT actif=1 gauche1=0x{:08x} gauche2=0x{:08x} "
                            "droite1=0x{:08x} droite2=0x{:08x} transfert=0x{:08x}->0x{:08x}",
                            v405_lcd_now[0], v405_lcd_now[1], v405_lcd_now[2], v405_lcd_now[3],
                            v405_src, v405_dst);
            }
            if (v405_right1 || v405_right2) {
                const u32 v405_old = g_v405_right_src.exchange(v405_src, std::memory_order_relaxed);
                g_v405_confirm_seq.store(v405_seq, std::memory_order_relaxed);
                if (v405_old != v405_src) {
                    LOG_WARNING(HW_GPU, "V405_OEIL_DROIT reperage src=0x{:08x} dst=0x{:08x}",
                                v405_src, v405_dst);
                }
            }
        }
        if (IsV408OeilDroitEnabled()) {
            const auto& v408_top = impl->pica.regs.framebuffer_config[0];
            const PAddr v408_dst = config.GetPhysicalOutputAddress();
            u32 v408_classe = 0;
            if (v408_dst != 0 &&
                (v408_dst == v408_top.address_left1 || v408_dst == v408_top.address_left2)) {
                v408_classe = 1;
            } else if (v408_dst != 0 && (v408_dst == v408_top.address_right1 ||
                                         v408_dst == v408_top.address_right2)) {
                v408_classe = 2;
            }
            g_v408_classe.store(v408_classe, std::memory_order_relaxed);
        }
        if (right_eye_disabler->ShouldAllowDisplayTransfer(config.GetPhysicalInputAddress(),
                                                           config.input_height)) {
            if (!impl->rasterizer->AccelerateDisplayTransfer(config)) {
                impl->sw_blitter->DisplayTransfer(config);
            }
        }
    }

    // Complete transfer.
    config.trigger.Assign(0);
    impl->signal_interrupt(Service::GSP::InterruptId::PPF);
}

void GPU::VBlankCallback(std::uintptr_t user_data, s64 cycles_late) {
    /// Frame Skip
    frame_count++;
    last_skip_frame = g_skip_frame;
    g_skip_frame = (frame_count & Settings::values.frame_skip.GetValue()) != 0;

    // Swap buffers based on the frameskip mode, which is a little bit tricky. When
    // a frame is being skipped, nothing is being rendered to the internal framebuffer(s).
    // So, we should only swap frames if the last frame was rendered. The rules are:
    //  - If frameskip == 0 (disabled), always swap buffers
    //  - If frameskip == 1, swap buffers every other frame (starting from the first frame)
    //  - If frameskip > 1, swap buffers every frameskip^n frames (starting from the second frame)
    if ((((Settings::values.frame_skip.GetValue() != 1) ^ last_skip_frame) &&
         last_skip_frame != g_skip_frame) ||
        Settings::values.frame_skip.GetValue() == 0) {

        // Present renderered frame.
        impl->renderer->SwapBuffers();
    }

    // Signal to GSP that GPU interrupt has occurred
    impl->signal_interrupt(Service::GSP::InterruptId::PDC0);
    impl->signal_interrupt(Service::GSP::InterruptId::PDC1);

    // Reschedule recurrent event
    impl->timing.ScheduleEvent(FRAME_TICKS - cycles_late, impl->vblank_event);
}

template <class Archive>
void GPU::serialize(Archive& ar, const u32 file_version) {
    ar & impl->pica;
}

SERIALIZE_IMPL(GPU)

} // namespace VideoCore
