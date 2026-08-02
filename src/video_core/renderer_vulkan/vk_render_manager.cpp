// Copyright 2024 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include "common/assert.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Vulkan {

constexpr u32 MinDrawsToFlush = 20;

// --- TB14 : compteurs d'instrumentation des render pass (voir vk_render_manager.h) ---
std::atomic<u32> g_tb14_rp_begin{0};
std::atomic<u32> g_tb14_rp_switch{0};
std::atomic<u32> g_tb14_rp_switch_area_only{0};
std::atomic<u32> g_tb14_rp_end{0};
std::atomic<u32> g_tb14_rp_flush{0};

// --- TB26 : decomposition des causes de bascule (voir vk_render_manager.h) ---
std::atomic<u32> g_tb26_diff_fb{0};
std::atomic<u32> g_tb26_diff_rp{0};
std::atomic<u32> g_tb26_diff_area{0};
std::atomic<u32> g_tb26_diff_clear{0};
std::atomic<u32> g_tb26_first_fb{0};
std::atomic<u32> g_tb26_first_rp{0};
std::atomic<u32> g_tb26_first_area{0};
std::atomic<u32> g_tb26_first_clear{0};
std::atomic<u32> g_tb26_fb_distinct{0};

// --- TB27 : longueur des runs consecutifs par cible + histogramme par framebuffer ---
std::atomic<u32> g_tb27_seq_count{0};
std::atomic<u32> g_tb27_seq_draws{0};
std::atomic<u32> g_tb27_fb_draws[6] = {};

// --- TB28a : identite des cibles de rendu (voir vk_render_manager.h) ---
Tb28aTarget g_tb28a_targets[6] = {};
std::atomic<u32> g_tb28a_count{0};

namespace {
/// Compte les framebuffers distincts rencontres. Table minuscule, mono-thread
/// (BeginRendering n'est appele que depuis l'EmuThread) : pas de verrou.
/// Si le jeu depasse la capacite, on cesse simplement de compter -- la valeur
/// devient un plancher, ce qui suffit a distinguer "quelques cibles" de "des dizaines".
constexpr std::size_t Tb26MaxTracked = 64;
std::array<VkFramebuffer, Tb26MaxTracked> tb26_seen_fb{};
std::size_t tb26_seen_count = 0;

// TB27 : segmentation en runs. tb27_run_fb retient la cible du run courant ;
// tb27_have_run distingue "aucun draw encore vu cette frame" du framebuffer nul.
// Mono-thread (EmuThread), meme justification que la table tb26 ci-dessus.
VkFramebuffer tb27_run_fb = VK_NULL_HANDLE;
bool tb27_have_run = false;

// TB28b : adresses physiques 3DS du draw en cours, posees par le rasterizer juste
// avant BeginRendering et lues a la premiere apparition de chaque cible.
// Mono-thread (EmuThread), meme justification que la table tb26 ci-dessus.
u32 tb28b_pending_color_addr = 0;
u32 tb28b_pending_depth_addr = 0;

/// Enregistre le framebuffer et renvoie son index d'apparition dans la frame
/// (0 = premier vu). Renvoie Tb26MaxTracked si la table est pleine (jamais atteint
/// avec ~5 cibles), auquel cas l'appelant n'incremente aucun bucket d'histogramme.
[[nodiscard]] std::size_t Tb26TrackFramebuffer(vk::Framebuffer fb) noexcept {
    const VkFramebuffer raw = static_cast<VkFramebuffer>(fb);
    for (std::size_t i = 0; i < tb26_seen_count; ++i) {
        if (tb26_seen_fb[i] == raw) {
            return i;
        }
    }
    if (tb26_seen_count < Tb26MaxTracked) {
        const std::size_t idx = tb26_seen_count;
        tb26_seen_fb[tb26_seen_count++] = raw;
        g_tb26_fb_distinct.fetch_add(1, std::memory_order_relaxed);
        return idx;
    }
    return Tb26MaxTracked;
}

/// Convertit un handle Vulkan natif en entier pour le log. Les handles non
/// dispatchables sont un pointeur sur cible 64 bits (cas du Pi5) et un u64 ailleurs :
/// le if constexpr couvre les deux sans avertissement.
template <typename H>
[[nodiscard]] u64 RawHandleU64(H raw) noexcept {
    if constexpr (std::is_pointer_v<H>) {
        return static_cast<u64>(reinterpret_cast<std::uintptr_t>(raw));
    } else {
        return static_cast<u64>(raw);
    }
}

/// Remet le suivi a zero -- appele par le census a chaque tick via
/// RenderManager::Tb26ResetFrame(). Reinitialise aussi le run TB27 pour que le
/// premier draw de la frame suivante ouvre toujours un nouveau run, et le compte
/// TB28a pour que les identites soient recapturees a chaque frame.
void Tb26ResetTracking() noexcept {
    tb26_seen_count = 0;
    tb27_run_fb = VK_NULL_HANDLE;
    tb27_have_run = false;
    g_tb28a_count.store(0, std::memory_order_relaxed);
}
} // Anonymous namespace

void RenderManager::Tb26ResetFrame() noexcept {
    Tb26ResetTracking();
}

void RenderManager::Tb28bNoteAddresses(u32 color_addr, u32 depth_addr) noexcept {
    tb28b_pending_color_addr = color_addr;
    tb28b_pending_depth_addr = depth_addr;
}


namespace {

// Helpers env memoises des le depart (lecon A1 : un mutex + une allocation par appel
// sur un chemin par-draw coute plus cher que le getenv qu'ils evitaient).
[[nodiscard]] bool ReadEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] u32 ReadEnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return parsed > 0xFFFFFFFFul ? 0xFFFFFFFFu : static_cast<u32>(parsed);
}

// TB15 : le seuil de flush vient du guide Mali amont et n'a jamais ete mesure sur
// V3DV. ShouldFlush() retourne vrai pour eMesaV3Dv, donc chaque fin de render pass
// au-dela du seuil declenche une soumission GPU complete.
//   BORKED3DS_V3DV_MIN_DRAWS_TO_FLUSH=<n>   (defaut 20)
//   BORKED3DS_V3DV_DISABLE_RENDERPASS_FLUSH=1  supprime le flush periodique
[[nodiscard]] u32 GetMinDrawsToFlush() {
    static const u32 cached =
        ReadEnvU32("BORKED3DS_V3DV_MIN_DRAWS_TO_FLUSH", MinDrawsToFlush);
    return cached;
}

[[nodiscard]] bool IsRenderpassFlushDisabled() {
    static const bool cached = ReadEnvFlag("BORKED3DS_V3DV_DISABLE_RENDERPASS_FLUSH");
    return cached;
}

} // Anonymous namespace

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

RenderManager::RenderManager(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {}

RenderManager::~RenderManager() = default;

void RenderManager::BeginRendering(const Framebuffer* framebuffer,
                                   Common::Rectangle<u32> draw_rect) {
    g_tb14_rp_begin.fetch_add(1, std::memory_order_relaxed);
    if (framebuffer) {
        const VkFramebuffer raw = static_cast<VkFramebuffer>(framebuffer->Handle());
        const std::size_t fb_idx = Tb26TrackFramebuffer(framebuffer->Handle());

        // TB27 : un run s'ouvre au premier draw de la frame ou des que la cible change
        // par rapport au draw precedent. seq_draws compte chaque draw segmente.
        if (!tb27_have_run || raw != tb27_run_fb) {
            g_tb27_seq_count.fetch_add(1, std::memory_order_relaxed);
            tb27_run_fb = raw;
            tb27_have_run = true;
        }
        g_tb27_seq_draws.fetch_add(1, std::memory_order_relaxed);
        if (fb_idx < 6) {
            g_tb27_fb_draws[fb_idx].fetch_add(1, std::memory_order_relaxed);

            // TB28a : les index d'apparition sont attribues dans l'ordre croissant,
            // donc fb_idx >= count identifie exactement la premiere apparition de
            // cette cible dans la frame. Capture une seule fois par cible et par
            // frame : aucun cout sur les ~320 draws suivants.
            if (fb_idx >= g_tb28a_count.load(std::memory_order_relaxed)) {
                const std::array<vk::Image, 2> imgs = framebuffer->Images();
                Tb28aTarget& t = g_tb28a_targets[fb_idx];
                t.fb = RawHandleU64(raw);
                t.render_pass =
                    RawHandleU64(static_cast<VkRenderPass>(framebuffer->RenderPass()));
                t.img_color = RawHandleU64(static_cast<VkImage>(imgs[0]));
                t.img_depth = RawHandleU64(static_cast<VkImage>(imgs[1]));
                t.color_id = framebuffer->color_id.index;
                t.depth_id = framebuffer->depth_id.index;
                t.color_level = framebuffer->color_level;
                t.depth_level = framebuffer->depth_level;
                t.width = framebuffer->Width();
                t.height = framebuffer->Height();
                t.scale = framebuffer->Scale();
                t.color_fmt = static_cast<u32>(framebuffer->Format(SurfaceType::Color));
                t.depth_fmt = static_cast<u32>(framebuffer->Format(SurfaceType::DepthStencil));
                t.shadow = framebuffer->shadow_rendering ? 1u : 0u;
                t.color_addr = tb28b_pending_color_addr;
                t.depth_addr = tb28b_pending_depth_addr;
                g_tb28a_count.store(static_cast<u32>(fb_idx) + 1u, std::memory_order_relaxed);
            }
        }
    }
    const vk::Rect2D render_area = {
        .offset{
            .x = static_cast<s32>(draw_rect.left),
            .y = static_cast<s32>(draw_rect.bottom),
        },
        .extent{
            .width = draw_rect.GetWidth(),
            .height = draw_rect.GetHeight(),
        },
    };
    const RenderPass new_pass = {
        .framebuffer = framebuffer->Handle(),
        .render_pass = framebuffer->RenderPass(),
        .render_area = render_area,
        .clear = {},
        .do_clear = false,
    };
    images = framebuffer->Images();
    aspects = framebuffer->Aspects();
    BeginRendering(new_pass);
}

void RenderManager::BeginRendering(const RenderPass& new_pass) {
    if (pass == new_pass) [[likely]] {
        num_draws++;
        return;
    }

    // TB14 : une bascule est sur le point d'avoir lieu. On distingue le cas ou seule
    // la zone de rendu change (meme framebuffer, meme render pass) : c'est celui qui
    // serait evitable en unifiant le draw_rect, donc le chiffre a regarder en premier.
    g_tb14_rp_switch.fetch_add(1, std::memory_order_relaxed);
    if (pass.render_pass && pass.framebuffer == new_pass.framebuffer &&
        pass.render_pass == new_pass.render_pass) {
        g_tb14_rp_switch_area_only.fetch_add(1, std::memory_order_relaxed);
    }

    // TB26 : quel champ a change ? Un draw peut en faire varier plusieurs, donc on
    // compte a la fois les differences individuelles et une attribution exclusive
    // (first_*) au premier champ different, du plus structurel au plus anodin.
    {
        const bool d_fb = pass.framebuffer != new_pass.framebuffer;
        const bool d_rp = pass.render_pass != new_pass.render_pass;
        const bool d_area = pass.render_area != new_pass.render_area;
        const bool d_clear = pass.do_clear != new_pass.do_clear ||
                             std::memcmp(&pass.clear, &new_pass.clear, sizeof(vk::ClearValue)) != 0;

        if (d_fb) {
            g_tb26_diff_fb.fetch_add(1, std::memory_order_relaxed);
        }
        if (d_rp) {
            g_tb26_diff_rp.fetch_add(1, std::memory_order_relaxed);
        }
        if (d_area) {
            g_tb26_diff_area.fetch_add(1, std::memory_order_relaxed);
        }
        if (d_clear) {
            g_tb26_diff_clear.fetch_add(1, std::memory_order_relaxed);
        }

        if (d_fb) {
            g_tb26_first_fb.fetch_add(1, std::memory_order_relaxed);
        } else if (d_rp) {
            g_tb26_first_rp.fetch_add(1, std::memory_order_relaxed);
        } else if (d_area) {
            g_tb26_first_area.fetch_add(1, std::memory_order_relaxed);
        } else if (d_clear) {
            g_tb26_first_clear.fetch_add(1, std::memory_order_relaxed);
        }
    }

    EndRendering();
    scheduler.Record([info = new_pass](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = info.render_pass,
            .framebuffer = info.framebuffer,
            .renderArea = info.render_area,
            .clearValueCount = info.do_clear ? 1u : 0u,
            .pClearValues = &info.clear,
        };
        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    pass = new_pass;
}

void RenderManager::EndRendering() {
    if (!pass.render_pass) {
        return;
    }

    g_tb14_rp_end.fetch_add(1, std::memory_order_relaxed);

    scheduler.Record([images = images, aspects = aspects](vk::CommandBuffer cmdbuf) {
        u32 num_barriers = 0;
        vk::PipelineStageFlags pipeline_flags{};
        std::array<vk::ImageMemoryBarrier, 2> barriers;
        for (u32 i = 0; i < images.size(); i++) {
            if (!images[i]) {
                continue;
            }
            const bool is_color = static_cast<bool>(aspects[i] & vk::ImageAspectFlagBits::eColor);
            if (is_color) {
                pipeline_flags |= vk::PipelineStageFlagBits::eColorAttachmentOutput;
            } else {
                pipeline_flags |= vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                  vk::PipelineStageFlagBits::eLateFragmentTests;
            }
            barriers[num_barriers++] = vk::ImageMemoryBarrier{
                .srcAccessMask = is_color ? vk::AccessFlagBits::eColorAttachmentWrite
                                          : vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                .dstAccessMask =
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
                .oldLayout = is_color ? vk::ImageLayout::eColorAttachmentOptimal
                                     : vk::ImageLayout::eDepthStencilAttachmentOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i],
                .subresourceRange{
                    .aspectMask = aspects[i],
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
        }
        cmdbuf.endRenderPass();
        if (num_barriers == 0) {
            return;
        }
        cmdbuf.pipelineBarrier(pipeline_flags,
                               vk::PipelineStageFlagBits::eFragmentShader |
                                   vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr,
                               num_barriers, barriers.data());
    });

    // Reset state.
    pass.render_pass = VK_NULL_HANDLE;
    images = {};
    aspects = {};

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    //
    // TB15 : seuil reglable, et desactivation possible. Le reglage amont (20) vise
    // Mali ; ShouldFlush() est vrai pour eMesaV3Dv, donc ce flush s'applique au Pi5
    // sans avoir jamais ete mesure dessus.
    if (!IsRenderpassFlushDisabled() && num_draws > GetMinDrawsToFlush() &&
        instance.ShouldFlush()) {
        g_tb14_rp_flush.fetch_add(1, std::memory_order_relaxed);
        scheduler.Flush();
        num_draws = 0;
    }
}

vk::RenderPass RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                            VideoCore::PixelFormat depth, bool is_clear) {
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid ? NumDepthFormats : (static_cast<u32>(depth) - 14);

    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    vk::UniqueRenderPass& renderpass = cached_renderpasses[color_index][depth_index][is_clear];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const vk::AttachmentLoadOp load_op =
            is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        renderpass = CreateRenderPass(color_format, depth_format, load_op);
    }

    return *renderpass;
}

vk::UniqueRenderPass RenderManager::CreateRenderPass(vk::Format color, vk::Format depth,
                                                     vk::AttachmentLoadOp load_op) const {
    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        use_depth = true;
    }

    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    const std::array<vk::SubpassDependency, 2> dependencies = {{
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eFragmentShader |
                            vk::PipelineStageFlagBits::eTransfer,
            .dstStageMask = use_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                      : vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                            vk::PipelineStageFlagBits::eLateFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eShaderRead |
                             vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = use_color
                                 ? vk::AccessFlagBits::eColorAttachmentRead |
                                       vk::AccessFlagBits::eColorAttachmentWrite
                                 : vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                       vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
        {
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = use_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                      : vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                            vk::PipelineStageFlagBits::eLateFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader |
                            vk::PipelineStageFlagBits::eTransfer,
            .srcAccessMask = use_color
                                 ? vk::AccessFlagBits::eColorAttachmentWrite
                                 : vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead |
                             vk::AccessFlagBits::eTransferRead,
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        },
    }};

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 2,
        .pDependencies = dependencies.data(),
    };

    return instance.GetDevice().createRenderPassUnique(renderpass_info);
}

} // namespace Vulkan
