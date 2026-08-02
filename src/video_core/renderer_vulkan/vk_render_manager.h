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

// ---------------------------------------------------------------------------
// TB28a -- IDENTITE des cibles de rendu.
//
// TB27 a montre 5 cibles et un entrelacement serre (seq_len_avg = 2,16), avec un
// histogramme en PAIRES : 130 / 130 / 12 / 12 / 36. Avant de trier ou de fusionner
// quoi que ce soit, il faut savoir CE QUE SONT ces cibles -- on ne reordonne pas des
// cibles qu'on n'a pas identifiees.
//
// L'hypothese stereo (oeil gauche / oeil droit) est CONTREDITE par la config :
// render_3d=0 et factor_3d=0. Les champs ci-dessous tranchent :
//
//   color_id / depth_id  identifiants de surface dans la rasterizer cache. Deux
//                        cibles partageant le meme color_id visent la MEME surface
//                        couleur -> doublon (deduplication possible, correctif simple)
//                        plutot que deux cibles logiques distinctes (tri necessaire).
//   shadow_rendering     distingue une passe d'ombre d'une passe principale : cause
//                        candidate directe des paires observees.
//   color_level/depth_level  niveau de mip -- discrimine un rendu multi-niveau.
//   width/height/scale/formats  geometrie et format : deux cibles "jumelles" doivent
//                        coincider sur tout cela pour etre reellement equivalentes.
//
// Purement descriptif : aucune decision de rendu ne lit ces valeurs. Capture a la
// PREMIERE apparition de chaque cible dans la frame, publiee par le census.
// ---------------------------------------------------------------------------
struct Tb28aTarget {
    u64 fb;
    u64 render_pass;
    u64 img_color;
    u64 img_depth;
    u32 color_id;
    u32 depth_id;
    u32 color_level;
    u32 depth_level;
    u32 width;
    u32 height;
    u32 scale;
    u32 color_fmt;
    u32 depth_fmt;
    u32 shadow;
};

/// Renseignees dans l'ordre d'apparition dans la frame, comme fbh0..fbh5 de TB27.
/// Ecrites et lues depuis le seul EmuThread (meme justification que la table tb26) ;
/// g_tb28a_count est atomique car le census la lit pour savoir combien sont valides.
extern Tb28aTarget g_tb28a_targets[6];
extern std::atomic<u32> g_tb28a_count;

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
