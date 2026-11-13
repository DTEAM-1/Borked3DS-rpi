// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
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

namespace Pica {
// V413 (MESURE) : compteurs definis dans video_core/pica/pica_core.cpp.
extern u64 g_v413_listes;
extern u64 g_v413_octets;
extern u64 g_v413_draws;
extern u64 g_v413_draws_indexes;
extern u64 g_v413_sommets;
extern u64 g_v413_eclaires;
extern u64 g_v413_lumieres;
extern u64 g_v413_irq;
} // namespace Pica

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

// V408 (v409) : adresses deja vues comme ecran du haut gauche / droit (8 par cote, anneau).
std::array<PAddr, 8> g_v408_gauche{};
std::size_t g_v408_ngauche = 0;
std::array<PAddr, 8> g_v408_droite{};
std::size_t g_v408_ndroite = 0;

bool V408Connu(const std::array<PAddr, 8>& liste, PAddr a) {
    if (a == 0) {
        return false;
    }
    for (const PAddr x : liste) {
        if (x == a) {
            return true;
        }
    }
    return false;
}

void V408Retenir(std::array<PAddr, 8>& liste, std::size_t& n, PAddr a) {
    if (a == 0 || V408Connu(liste, a)) {
        return;
    }
    liste[n % liste.size()] = a;
    ++n;
}

} // namespace

// V413 (MESURE) : BORKED3DS_V3DV_V413_TRAVAIL=1. Une ligne V413_TRAVAIL par seconde de temps
// EMULE : nombre d'images du jeu (echanges d'images de l'ecran du haut), temps emule moyen par
// image, et travail GPU moyen par image (listes, draws, sommets, eclairage, interruptions P3D,
// transferts, copies, effacements) avec les maximums par image. Aucun effet sur le rendu.
// But : calibrer une estimation du temps GPU de la vraie console (etape 2 du chantier
// « temps GPU emule »).
namespace {
bool IsV413TravailEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BORKED3DS_V3DV_V413_TRAVAIL");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

u64 g_v413_transferts = 0;
u64 g_v413_transfert_pixels = 0;
u64 g_v413_copies = 0;
u64 g_v413_copie_octets = 0;
u64 g_v413_effacements = 0;
u64 g_v413_effacement_octets = 0;
// v414 : images CALCULEES par le jeu = display transfers vers l'ecran du haut GAUCHE (registres
// actuels, puis historique : meme classement que V408). TB134/TB135 : sans delai, Castlevania
// calcule ~2 images par image AFFICHEE (20 listes, 6,75 transferts par echange contre 9 et 3 a
// 2590 us) ; compter par echange melangeait donc plusieurs images du jeu.
u64 g_v413_rendus_gauche = 0;
u64 g_v413_rendus_droite = 0;

struct V413Releve {
    u64 listes, octets, draws, indexes, sommets, eclaires, lumieres, irq, transferts, pixels,
        copies, copie_octets, effacements, effacement_octets, rendus_g, rendus_d;
};

V413Releve V413Lire() {
    return V413Releve{Pica::g_v413_listes,    Pica::g_v413_octets,   Pica::g_v413_draws,
                      Pica::g_v413_draws_indexes, Pica::g_v413_sommets, Pica::g_v413_eclaires,
                      Pica::g_v413_lumieres,  Pica::g_v413_irq,      g_v413_transferts,
                      g_v413_transfert_pixels, g_v413_copies,        g_v413_copie_octets,
                      g_v413_effacements,     g_v413_effacement_octets,
                      g_v413_rendus_gauche,   g_v413_rendus_droite};
}

// Appelee a chaque fin d'image du jeu (echange de l'ecran du haut), sur l'EmuThread.
void V413FinImage(s64 ticks) {
    static bool demarre = false;
    static s64 debut = 0;
    static u64 images = 0;
    static V413Releve base{};
    static V413Releve precedent{};
    static u64 max_draws = 0, max_sommets = 0, max_listes = 0;
    const V413Releve r = V413Lire();
    if (!demarre) {
        demarre = true;
        debut = ticks;
        base = r;
        precedent = r;
        return;
    }
    ++images;
    max_draws = std::max<u64>(max_draws, r.draws - precedent.draws);
    max_sommets = std::max<u64>(max_sommets, r.sommets - precedent.sommets);
    max_listes = std::max<u64>(max_listes, r.listes - precedent.listes);
    precedent = r;
    const s64 duree = ticks - debut;
    if (duree < static_cast<s64>(BASE_CLOCK_RATE_ARM11)) {
        return;
    }
    const double n = static_cast<double>(images);
    const auto par = [n](u64 a, u64 b) { return static_cast<double>(a - b) / n; };
    const u64 rendus = r.rendus_g - base.rendus_g;
    const double ms_rendu = rendus == 0 ? 0.0
                                        : static_cast<double>(duree) * 1000.0 /
                                              BASE_CLOCK_RATE_ARM11 / static_cast<double>(rendus);
    // Travail par image CALCULEE (rendus_g) : la base de calibrage (v414).
    const double nr = rendus == 0 ? 1.0 : static_cast<double>(rendus);
    const auto parr = [nr](u64 a, u64 b) { return static_cast<double>(a - b) / nr; };
    LOG_WARNING(HW_GPU,
                "V414_RENDU rendus={} ms_emule_rendu={:.2f} rendus_d={} listes={:.1f} draws={:.1f} "
                "sommets={:.0f} eclaires={:.1f} lumieres={:.1f} transferts={:.2f} pixels={:.0f} "
                "copies={:.2f} copie_octets={:.0f} effacements={:.2f} effacement_octets={:.0f}",
                rendus, ms_rendu, r.rendus_d - base.rendus_d, parr(r.listes, base.listes),
                parr(r.draws, base.draws), parr(r.sommets, base.sommets),
                parr(r.eclaires, base.eclaires), parr(r.lumieres, base.lumieres),
                parr(r.transferts, base.transferts), parr(r.pixels, base.pixels),
                parr(r.copies, base.copies), parr(r.copie_octets, base.copie_octets),
                parr(r.effacements, base.effacements),
                parr(r.effacement_octets, base.effacement_octets));
    LOG_WARNING(HW_GPU,
                "V413_TRAVAIL images={} ms_emule_image={:.2f} listes={:.1f} octets={:.0f} "
                "draws={:.1f} indexes={:.1f} sommets={:.0f} eclaires={:.1f} lumieres={:.1f} "
                "irq={:.2f} transferts={:.2f} pixels={:.0f} copies={:.2f} copie_octets={:.0f} "
                "effacements={:.2f} effacement_octets={:.0f} max_draws={} max_sommets={} "
                "max_listes={}",
                images, static_cast<double>(duree) * 1000.0 / BASE_CLOCK_RATE_ARM11 / n,
                par(r.listes, base.listes), par(r.octets, base.octets), par(r.draws, base.draws),
                par(r.indexes, base.indexes), par(r.sommets, base.sommets),
                par(r.eclaires, base.eclaires), par(r.lumieres, base.lumieres),
                par(r.irq, base.irq), par(r.transferts, base.transferts),
                par(r.pixels, base.pixels), par(r.copies, base.copies),
                par(r.copie_octets, base.copie_octets), par(r.effacements, base.effacements),
                par(r.effacement_octets, base.effacement_octets), max_draws, max_sommets,
                max_listes);
    debut = ticks;
    base = r;
    images = 0;
    max_draws = max_sommets = max_listes = 0;
}
} // namespace

// V408 : classe de la destination du display transfer en cours, lue par le rasterizer Vulkan
// (vk_rasterizer.cpp, AccelerateDisplayTransfer) juste apres : 0 = autre, 1 = ecran du haut
// gauche, 2 = ecran du haut droite, 3 = ecran du bas (v412). Une adresse droite egale a une
// adresse gauche (configuration mono) est classee gauche.
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
        if (IsV413TravailEnabled()) {
            V413FinImage(impl->timing.GetGlobalTicks());
        }
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
    ++g_v413_effacements; // V413 (mesure)
    g_v413_effacement_octets += config.GetEndAddress() - config.GetStartAddress();

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

    // V413 (mesure)
    if (config.is_texture_copy) {
        ++g_v413_copies;
        g_v413_copie_octets += config.texture_copy.size;
    } else {
        ++g_v413_transferts;
        g_v413_transfert_pixels += static_cast<u64>(config.output_width.Value()) *
                                   config.output_height.Value();
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
        if (IsV408OeilDroitEnabled() || IsV413TravailEnabled()) {
            // V408 (v409) : classement par HISTORIQUE. TB123 (Metroid) : le premier transfert vers
            // le second tampon de l'ecran droit (0x18370800, double tampon avec 0x183b6d00, TB118)
            // a eu lieu avant que cette adresse paraisse dans les registres ; un classement sur les
            // seuls registres du moment l'a pris pour une autre destination. Une adresse vue une
            // fois comme ecran gauche (ou droit) le reste. Vue des deux cotes : classee gauche (le
            // cas prudent, qui bloque si l'oeil droit y est envoye).
            const auto& v408_top = impl->pica.regs.framebuffer_config[0];
            V408Retenir(g_v408_gauche, g_v408_ngauche, v408_top.address_left1);
            V408Retenir(g_v408_gauche, g_v408_ngauche, v408_top.address_left2);
            V408Retenir(g_v408_droite, g_v408_ndroite, v408_top.address_right1);
            V408Retenir(g_v408_droite, g_v408_ndroite, v408_top.address_right2);
            const PAddr v408_dst = config.GetPhysicalOutputAddress();
            // v412 : les registres ACTUELS d'abord (haut gauche, haut droite, bas), l'historique
            // seulement pour une adresse absente des registres actuels. TB125-TB127 : 0x18300000,
            // vu dans les registres du haut au demarrage, restait classe gauche alors que c'est
            // l'ecran du bas (TB118). 3 = ecran du bas.
            const auto& v408_bas = impl->pica.regs.framebuffer_config[1];
            u32 v408_classe = 0;
            if (v408_dst == 0) {
                v408_classe = 0;
            } else if (v408_dst == v408_top.address_left1 || v408_dst == v408_top.address_left2) {
                v408_classe = 1;
            } else if (v408_dst == v408_top.address_right1 ||
                       v408_dst == v408_top.address_right2) {
                v408_classe = 2;
            } else if (v408_dst == v408_bas.address_left1 || v408_dst == v408_bas.address_left2) {
                v408_classe = 3;
            } else if (V408Connu(g_v408_gauche, v408_dst)) {
                v408_classe = 1;
            } else if (V408Connu(g_v408_droite, v408_dst)) {
                v408_classe = 2;
            }
            g_v408_classe.store(v408_classe, std::memory_order_relaxed);
            g_v413_rendus_gauche += v408_classe == 1 ? 1 : 0; // v414
            g_v413_rendus_droite += v408_classe == 2 ? 1 : 0;
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
