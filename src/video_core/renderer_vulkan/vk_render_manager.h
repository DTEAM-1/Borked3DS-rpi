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

// ---------------------------------------------------------------------------
// TB26 -- POURQUOI le render pass bascule-t-il a chaque draw ?
//
// Mesure TB24b (log brut, valeurs PAR FRAME) : entered=320, rp_begin=320,
// rp_switch=315, rp_end=315. Autrement dit 315 render passes par frame, alors que
// le V3D est TILE-BASED : chaque fermeture force un store complet du tile buffer,
// chaque ouverture un reload. 67 ms / 315 ~ 213 us par cycle -- l'ordre de grandeur
// colle exactement au temps mesure.
//
// rp_area vaut 0, donc ce n'est PAS la zone de rendu seule qui change. Ces compteurs
// isolent le champ responsable, en les testant du plus structurel au plus anodin.
// Un draw peut faire varier plusieurs champs a la fois : les compteurs ne
// s'excluent pas, sauf cause_first_* qui attribue la bascule au premier champ
// different dans l'ordre framebuffer > render_pass > area > clear.
//
//   g_tb26_diff_fb        le framebuffer differe (changement de cible de rendu)
//   g_tb26_diff_rp        l'objet render pass differe (format/attachements)
//   g_tb26_diff_area      la zone de rendu differe
//   g_tb26_diff_clear     do_clear ou la valeur de clear differe
//   g_tb26_first_fb/rp/area/clear  attribution exclusive de la bascule
//   g_tb26_fb_distinct    nombre de framebuffers DISTINCTS vus dans la frame
// ---------------------------------------------------------------------------
extern std::atomic<u32> g_tb26_diff_fb;
extern std::atomic<u32> g_tb26_diff_rp;
extern std::atomic<u32> g_tb26_diff_area;
extern std::atomic<u32> g_tb26_diff_clear;
extern std::atomic<u32> g_tb26_first_fb;
extern std::atomic<u32> g_tb26_first_rp;
extern std::atomic<u32> g_tb26_first_area;
extern std::atomic<u32> g_tb26_first_clear;
extern std::atomic<u32> g_tb26_fb_distinct;

// ---------------------------------------------------------------------------
// TB27 -- FAISABILITE du regroupement des draws par cible de rendu.
//
// TB26 a etabli QUE la bascule vient du framebuffer (f_fb=148/frame pour fbn=5).
// TB27 mesure la GEOMETRIE de ces bascules : les draws d'une meme cible arrivent-ils
// deja groupes, ou entrelaces ?
//
//   runs LONGS  (ex. 60 draws/run) -> fusionner les sequences adjacentes suffit,
//                                      correctif simple et sur ;
//   runs COURTS (2-3 draws/run)    -> il faut un vrai tri avec analyse de
//                                      dependances lecture/ecriture, chantier lourd.
//
// Un "run" est une sequence maximale de draws CONSECUTIFS visant le meme framebuffer
// (meme cible = meme tile buffer sur le V3D). Segmentation par frame, comme fbn.
//
//   g_tb27_seq_count   nombre de runs dans la frame
//   g_tb27_seq_draws   total des draws segmentes (== rp_begin ; compte local pour que
//                      seq_len_avg = seq_draws / seq_count soit auto-coherent)
//   g_tb27_fb_draws[i] draws vers le i-eme framebuffer dans l'ordre de premiere
//                      apparition dans la frame (publie en fbh0..fbh5 ; au-dela de 6
//                      cibles, le surplus n'est pas ventile -- fbn en donne le total)
// ---------------------------------------------------------------------------
extern std::atomic<u32> g_tb27_seq_count;
extern std::atomic<u32> g_tb27_seq_draws;
extern std::atomic<u32> g_tb27_fb_draws[6];

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

    /// TB26 : remet a zero le suivi des framebuffers distincts (appele par le census).
    void Tb26ResetFrame() noexcept;

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
