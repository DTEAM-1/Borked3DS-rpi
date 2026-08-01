// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <queue>
#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

// ---------------------------------------------------------------------------
// TB24 -- instrumentation des soumissions GPU (axe B).
//
// Etabli par TB13/TB17/TB18 et les profils threads : le temps par frame est cloue
// a ~68 ms alors qu'on peut retirer >20 ms de travail CPU sans rien gagner, et
// qu'aucun thread n'est sature. Le profil perf place ~34 % du temps dans le noyau,
// [drm] et libvulkan_broadcom -- des symboles de VALIDATION DE SOUMISSION
// (objects_lookup, dma_resv_*, drm_gem_lock_reservations), pas de rendu.
//
// Question tranchee ici : le GPU calcule-t-il reellement 68 ms, ou paie-t-on un
// cout fixe par vkQueueSubmit ?
//
//   g_tb24_submits        nombre de vkQueueSubmit
//   g_tb24_submit_ns      temps mur passe DANS l'appel submit() (cout d'emission)
//   g_tb24_submit_max_ns  pire appel unique
//   g_tb24_gpu_lag        somme des (CurrentTick - KnownGpuTick) au moment du submit,
//                         soit la profondeur de pipeline : 0 = le GPU suit en temps
//                         reel (donc on ne l'attend pas), eleve = le GPU est en retard
//
// Cout : un fetch_add relaxed par soumission (~5/frame), negligeable.
// Lus et remis a zero par le census A7Z12 dans vk_rasterizer.cpp.
// ---------------------------------------------------------------------------
extern std::atomic<u64> g_tb24_submits;
extern std::atomic<u64> g_tb24_submit_ns;
extern std::atomic<u64> g_tb24_submit_max_ns;
extern std::atomic<u64> g_tb24_gpu_lag;


class Instance;
class Scheduler;

class MasterSemaphore {
public:
    virtual ~MasterSemaphore() = default;

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    /// Refresh the known GPU tick
    virtual void Refresh() = 0;

    /// Waits for a tick to be hit on the GPU
    virtual void Wait(u64 tick) = 0;

    /// Submits the provided command buffer for execution
    virtual void SubmitWork(vk::CommandBuffer cmdbuf, vk::Semaphore wait, vk::Semaphore signal,
                            u64 signal_value) = 0;

protected:
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
};

class MasterSemaphoreTimeline : public MasterSemaphore {
public:
    explicit MasterSemaphoreTimeline(const Instance& instance);
    ~MasterSemaphoreTimeline() override;

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    void Refresh() override;

    void Wait(u64 tick) override;

    void SubmitWork(vk::CommandBuffer cmdbuf, vk::Semaphore wait, vk::Semaphore signal,
                    u64 signal_value) override;

private:
    const Instance& instance;
    vk::UniqueSemaphore semaphore; ///< Timeline semaphore.
};

class MasterSemaphoreFence : public MasterSemaphore {
    using Waitable = std::pair<vk::Fence, u64>;

public:
    explicit MasterSemaphoreFence(const Instance& instance);
    ~MasterSemaphoreFence() override;

    void Refresh() override;

    void Wait(u64 tick) override;

    void SubmitWork(vk::CommandBuffer cmdbuf, vk::Semaphore wait, vk::Semaphore signal,
                    u64 signal_value) override;

private:
    void WaitThread(std::stop_token token);

    vk::Fence GetFreeFence();

private:
    const Instance& instance;
    std::deque<vk::Fence> free_queue;
    std::queue<Waitable> wait_queue;
    std::mutex free_mutex;
    std::mutex wait_mutex;
    std::condition_variable free_cv;
    std::condition_variable_any wait_cv;
    std::jthread wait_thread;
};

} // namespace Vulkan
