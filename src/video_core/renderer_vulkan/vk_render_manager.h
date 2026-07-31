// Copyright 2024 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <mutex>

#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
enum class PixelFormat : u32;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Framebuffer;

// ---------------------------------------------------------------------------
// TB14 -- instrumentation des render pass (axe B).
//
// Sur un GPU tile-based comme le V3D du Pi5, fermer un render pass force un store
// du tile buffer et un reload au suivant. RenderManager::BeginRendering ne
// court-circuite que si le pass est STRICTEMENT identique, et render_area (donc le
// draw_rect) fait partie de la comparaison : tout changement de rectangle ferme le
// pass, pose deux barrieres d'image, et peut declencher un scheduler.Flush().
//
// Ces compteurs sont lus et remis a zero par le census A7Z12 (vk_rasterizer.cpp).
// Cout : un fetch_add relaxed par draw, negligeable devant le reste du chemin.
//
//   g_tb14_rp_begin            appels a BeginRendering (~= draws)
//   g_tb14_rp_switch           bascules reelles (pass != new_pass)
//   g_tb14_rp_switch_area_only bascules ou SEUL le rectangle change (meme cible)
//   g_tb14_rp_end              EndRendering effectifs
//   g_tb14_rp_flush            scheduler.Flush() declenches par le seuil
// ---------------------------------------------------------------------------
extern std::atomic<u32> g_tb14_rp_begin;
extern std::atomic<u32> g_tb14_rp_switch;
extern std::atomic<u32> g_tb14_rp_switch_area_only;
extern std::atomic<u32> g_tb14_rp_end;
extern std::atomic<u32> g_tb14_rp_flush;

struct RenderPass {
    vk::Framebuffer framebuffer;
    vk::RenderPass render_pass;
    vk::Rect2D render_area;
    vk::ClearValue clear;
    u32 do_clear;

    bool operator==(const RenderPass& other) const noexcept {
        return std::tie(framebuffer, render_pass, render_area, do_clear) ==
                   std::tie(other.framebuffer, other.render_pass, other.render_area,
                            other.do_clear) &&
               std::memcmp(&clear, &other.clear, sizeof(vk::ClearValue)) == 0;
    }
};

class RenderManager {
    static constexpr u32 NumColorFormats = 13;
    static constexpr u32 NumDepthFormats = 4;

public:
    explicit RenderManager(const Instance& instance, Scheduler& scheduler);
    ~RenderManager();

    /// Begins a new renderpass with the provided framebuffer as render target.
    void BeginRendering(const Framebuffer* framebuffer, Common::Rectangle<u32> draw_rect);

    /// Begins a new renderpass with the provided render state.
    void BeginRendering(const RenderPass& new_pass);

    /// Exits from any currently active renderpass instance
    void EndRendering();

    /// Returns the renderpass associated with the color-depth format pair
    vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                 bool is_clear);

private:
    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPass(vk::Format color, vk::Format depth,
                                          vk::AttachmentLoadOp load_op) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueRenderPass cached_renderpasses[NumColorFormats + 1][NumDepthFormats + 1][2];
    std::mutex cache_mutex;
    std::array<vk::Image, 2> images;
    std::array<vk::ImageAspectFlags, 2> aspects;
    RenderPass pass{};
    u32 num_draws{};
};

} // namespace Vulkan
