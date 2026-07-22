// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "common/logging/log.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

std::unique_ptr<MasterSemaphore> MakeMasterSemaphore(const Instance& instance) {
    if (instance.IsTimelineSemaphoreSupported()) {
        return std::make_unique<MasterSemaphoreTimeline>(instance);
    } else {
        return std::make_unique<MasterSemaphoreFence>(instance);
    }
}

/// Un site d'appel de Finish() : cumul du nombre d'appels et du temps bloque.
struct SyncSiteEntry {
    u64 count = 0;
    u64 total_ns = 0;
};

/// Cle : (fichier, ligne). Le pointeur de fichier vient de source_location et
/// pointe une chaine litterale statique -- comparable et stable.
using SyncSiteKey = std::pair<const char*, u32>;

std::mutex sync_sites_mutex;
std::map<SyncSiteKey, SyncSiteEntry> sync_sites;

/// Ne garde que le nom de fichier, sans le chemin, pour que la ligne de log
/// reste lisible.
const char* BaseName(const char* path) {
    if (path == nullptr) {
        return "?";
    }
    const char* slash = std::strrchr(path, '/');
    return slash != nullptr ? slash + 1 : path;
}

} // Anonymous namespace

// ---------------------------------------------------------------------------
// Sonde de synchronisation CPU/GPU -- BORKED3DS_V3DV_TRACE_SYNC
// ---------------------------------------------------------------------------

std::atomic<u64> SyncStats::finish_calls{0};
std::atomic<u64> SyncStats::finish_ns{0};
std::atomic<u64> SyncStats::flush_calls{0};
std::atomic<u64> SyncStats::dispatch_calls{0};
std::atomic<u64> SyncStats::waitworker_calls{0};
std::atomic<u64> SyncStats::waitworker_ns{0};
std::atomic<u64> SyncStats::site_surface_dtor{0};
std::atomic<u64> SyncStats::site_surface_download{0};
std::atomic<u64> SyncStats::site_runtime_finish{0};
std::atomic<u64> SyncStats::streambuf_wait_calls{0};
std::atomic<u64> SyncStats::streambuf_wait_ns{0};
std::atomic<u64> SyncStats::streambuf_wait_max_ns{0};
std::atomic<u64> SyncStats::streambuf_wait_ticks{0};
std::atomic<u64> SyncStats::streambuf_gap_max{0};
std::atomic<u64> SyncStats::streambuf_wait_by_type[3] = {};

bool SyncStats::Enabled() noexcept {
    // getenv evalue UNE seule fois. Antipattern connu du projet : un getenv par
    // draw coute plus cher que ce qu'il mesure.
    static const bool enabled = (std::getenv("BORKED3DS_V3DV_TRACE_SYNC") != nullptr);
    return enabled;
}

void SyncStats::RecordFinish(const char* file, u32 line, u64 elapsed_ns) {
    std::scoped_lock lock{sync_sites_mutex};
    SyncSiteEntry& e = sync_sites[SyncSiteKey{file, line}];
    e.count++;
    e.total_ns += elapsed_ns;
}

void SyncStats::RecordStreamWait(u32 type_index, u64 elapsed_ns, u64 ticks_waited,
                                 u64 tick_gap) noexcept {
    streambuf_wait_calls.fetch_add(1, std::memory_order_relaxed);
    streambuf_wait_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    streambuf_wait_ticks.fetch_add(ticks_waited, std::memory_order_relaxed);
    if (type_index < 3) {
        streambuf_wait_by_type[type_index].fetch_add(1, std::memory_order_relaxed);
    }

    // Max monotone du blocage unique, sans verrou (CAS).
    u64 prev_max = streambuf_wait_max_ns.load(std::memory_order_relaxed);
    while (elapsed_ns > prev_max &&
           !streambuf_wait_max_ns.compare_exchange_weak(prev_max, elapsed_ns,
                                                        std::memory_order_relaxed)) {
    }

    // Max monotone de l'avance CPU.
    u64 prev_gap = streambuf_gap_max.load(std::memory_order_relaxed);
    while (tick_gap > prev_gap &&
           !streambuf_gap_max.compare_exchange_weak(prev_gap, tick_gap,
                                                    std::memory_order_relaxed)) {
    }
}

void SyncStats::ReportIfDue() {
    using clock = std::chrono::steady_clock;

    static std::mutex report_mutex;
    static clock::time_point last = clock::now();

    const clock::time_point now = clock::now();
    if (now - last < std::chrono::seconds(1)) {
        return;
    }

    std::unique_lock lock{report_mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
        return;
    }
    // Re-verification sous verrou : un autre thread a pu emettre entre-temps.
    if (now - last < std::chrono::seconds(1)) {
        return;
    }

    const double window_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(now - last).count() / 1000.0;
    last = now;

    const u64 n_finish = finish_calls.exchange(0, std::memory_order_relaxed);
    const u64 t_finish = finish_ns.exchange(0, std::memory_order_relaxed);
    const u64 n_flush = flush_calls.exchange(0, std::memory_order_relaxed);
    const u64 n_dispatch = dispatch_calls.exchange(0, std::memory_order_relaxed);
    const u64 n_ww = waitworker_calls.exchange(0, std::memory_order_relaxed);
    const u64 t_ww = waitworker_ns.exchange(0, std::memory_order_relaxed);
    const u64 s_dtor = site_surface_dtor.exchange(0, std::memory_order_relaxed);
    const u64 s_dl = site_surface_download.exchange(0, std::memory_order_relaxed);
    const u64 s_rt = site_runtime_finish.exchange(0, std::memory_order_relaxed);
    const u64 n_sbw = streambuf_wait_calls.exchange(0, std::memory_order_relaxed);
    const u64 t_sbw = streambuf_wait_ns.exchange(0, std::memory_order_relaxed);
    const u64 t_sbw_max = streambuf_wait_max_ns.exchange(0, std::memory_order_relaxed);
    const u64 n_sbticks = streambuf_wait_ticks.exchange(0, std::memory_order_relaxed);
    const u64 gap_max = streambuf_gap_max.exchange(0, std::memory_order_relaxed);
    const u64 sb_up = streambuf_wait_by_type[0].exchange(0, std::memory_order_relaxed);
    const u64 sb_dl = streambuf_wait_by_type[1].exchange(0, std::memory_order_relaxed);
    const u64 sb_st = streambuf_wait_by_type[2].exchange(0, std::memory_order_relaxed);

    const double finish_ms = t_finish / 1.0e6;
    const double ww_ms = t_ww / 1.0e6;
    const double sbw_ms = t_sbw / 1.0e6;
    const double sbw_max_ms = t_sbw_max / 1.0e6;
    // blocked_pct inclut desormais le rebouclage du stream buffer : c'est LUI le
    // suspect du figement, il doit peser dans la fraction de temps bloque.
    const double blocked_pct =
        window_ms > 0.0 ? (finish_ms + ww_ms + sbw_ms) * 100.0 / window_ms : 0.0;
    const double finish_avg_us = n_finish > 0 ? (t_finish / 1000.0) / double(n_finish) : 0.0;

    LOG_INFO(Render_Vulkan,
             "TRACE_SYNC finish={} finish_ms={:.2f} finish_avg_us={:.1f} flush={} dispatch={} "
             "waitworker={} waitworker_ms={:.2f} "
             "streambuf_wait={} streambuf_wait_ms={:.1f} streambuf_wait_max_ms={:.1f} "
             "streambuf_ticks={} gap_max={} sb_up={} sb_dl={} sb_st={} "
             "blocked_pct={:.1f} "
             "site_surf_dtor={} site_surf_dl={} site_rt_finish={} window_ms={:.1f}",
             n_finish, finish_ms, finish_avg_us, n_flush, n_dispatch, n_ww, ww_ms,
             n_sbw, sbw_ms, sbw_max_ms, n_sbticks, gap_max, sb_up, sb_dl, sb_st,
             blocked_pct,
             s_dtor, s_dl, s_rt, window_ms);

    // Seconde ligne : classement des sites d'appel par temps bloque decroissant.
    std::vector<std::pair<SyncSiteKey, SyncSiteEntry>> ranked;
    {
        std::scoped_lock sl{sync_sites_mutex};
        ranked.assign(sync_sites.begin(), sync_sites.end());
        sync_sites.clear();
    }

    if (ranked.empty()) {
        return;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.second.total_ns > b.second.total_ns; });

    std::string sites;
    const std::size_t shown = std::min<std::size_t>(ranked.size(), 8);
    for (std::size_t i = 0; i < shown; i++) {
        const auto& key = ranked[i].first;
        const auto& entry = ranked[i].second;
        if (!sites.empty()) {
            sites += ' ';
        }
        sites += fmt::format("{}:{}/n={}/ms={:.1f}", BaseName(key.first), key.second, entry.count,
                             entry.total_ns / 1.0e6);
    }

    LOG_INFO(Render_Vulkan, "TRACE_SYNC sites distinct={} {}", ranked.size(), sites);
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Instance& instance)
    : master_semaphore{MakeMasterSemaphore(instance)},
      command_pool{instance, master_semaphore.get()}, use_worker_thread{true} {
    AllocateWorkerCommandBuffers();
    if (use_worker_thread) {
        AcquireNewChunk();
        worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
    }
}

Scheduler::~Scheduler() = default;

void Scheduler::Flush(vk::Semaphore signal, vk::Semaphore wait) {
    if (SyncStats::Enabled()) {
        SyncStats::flush_calls.fetch_add(1, std::memory_order_relaxed);
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    SubmitExecution(signal, wait);
}

void Scheduler::Finish(vk::Semaphore signal, vk::Semaphore wait, std::source_location loc) {
    // When finishing, we need to wait for the submission to have executed on the device.
    //
    // Note de mesure : SubmitExecution() n'emet PAS le vkQueueSubmit ; elle
    // l'enregistre dans le chunk courant et le confie a VulkanWorker. Le Wait()
    // qui suit bloque donc sur : reveil du worker + submit + execution GPU +
    // reveil. C'est ce cumul que la sonde chiffre, et qu'elle attribue
    // maintenant au site d'appel via loc.
    if (!SyncStats::Enabled()) {
        const u64 presubmit_tick = CurrentTick();
        SubmitExecution(signal, wait);
        Wait(presubmit_tick);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal, wait);
    Wait(presubmit_tick);
    const auto t1 = std::chrono::steady_clock::now();

    const u64 elapsed_ns =
        u64(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    SyncStats::finish_calls.fetch_add(1, std::memory_order_relaxed);
    SyncStats::finish_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    SyncStats::RecordFinish(loc.file_name(), u32(loc.line()), elapsed_ns);
    SyncStats::ReportIfDue();
}

void Scheduler::WaitWorker() {
    if (!use_worker_thread) {
        return;
    }

    BORKED3DS_PROFILE("Vulkan", "Vulkan WaitWorker");

    const bool trace = SyncStats::Enabled();
    const auto t0 =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    // This needs to be done in the same order as WorkerThread.
    {
        std::scoped_lock el{execution_mutex};
    }

    if (trace) {
        const auto t1 = std::chrono::steady_clock::now();
        SyncStats::waitworker_calls.fetch_add(1, std::memory_order_relaxed);
        SyncStats::waitworker_ns.fetch_add(
            u64(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
            std::memory_order_relaxed);
    }
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore->CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        Flush();
    }
    master_semaphore->Wait(tick);
}

void Scheduler::DispatchWork() {
    if (!use_worker_thread || chunk->Empty()) {
        return;
    }

    if (SyncStats::Enabled()) {
        SyncStats::dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    }

    on_dispatch();

    {
        std::scoped_lock ql{queue_mutex};
        work_queue.push(std::move(chunk));
    }

    event_cv.notify_all();
    AcquireNewChunk();
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            Common::CondvarWait(event_cv, lk, stop_token, [&] { return TryPopQueue(work); });

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            std::exchange(lk, std::unique_lock{execution_mutex});

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffers();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    current_cmdbuf.begin(begin_info);
}

void Scheduler::SubmitExecution(vk::Semaphore signal_semaphore, vk::Semaphore wait_semaphore) {
    state = StateFlags::AllDirty;
    const u64 signal_value = master_semaphore->NextTick();

    on_submit();

    Record([signal_semaphore, wait_semaphore, signal_value, this](vk::CommandBuffer cmdbuf) {
        BORKED3DS_PROFILE("Vulkan", "Vulkan Submit");
        std::scoped_lock lock{submit_mutex};
        master_semaphore->SubmitWork(cmdbuf, wait_semaphore, signal_semaphore, signal_value);
    });

    master_semaphore->Refresh();

    if (!use_worker_thread) {
        AllocateWorkerCommandBuffers();
    } else {
        chunk->MarkSubmit();
        DispatchWork();
    }
}

void Scheduler::AcquireNewChunk() {
    std::scoped_lock lock{reserve_mutex};
    if (chunk_reserve.empty()) {
        chunk = std::make_unique<CommandChunk>();
        return;
    }

    chunk = std::move(chunk_reserve.back());
    chunk_reserve.pop_back();
}

} // namespace Vulkan
