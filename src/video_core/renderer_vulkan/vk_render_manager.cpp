// Copyright 2024 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <cstdlib>
#include <limits>
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
