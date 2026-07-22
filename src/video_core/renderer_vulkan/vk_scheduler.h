// Copyright 2019 yuzu Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <source_location>
#include <utility>
#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/polyfill_thread.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

enum class StateFlags {
    AllDirty = 0,
    Pipeline = 1 << 0,
    DescriptorSets = 1 << 1,
};
DECLARE_ENUM_FLAG_OPERATORS(StateFlags)

class Instance;

/// ---------------------------------------------------------------------------
/// Sonde de synchronisation CPU/GPU -- BORKED3DS_V3DV_TRACE_SYNC
///
/// Mesure v149 (Metroid, zone 3D) : ~30 Finish() par trame, ~900 us chacun,
/// blocked_pct 60-69 %. Aucun des trois sites de vk_texture_runtime.cpp n'y
/// contribue en regime etabli. Cette version attribue donc CHAQUE Finish() a
/// son site d'appel reel via std::source_location, sans toucher aux appelants :
/// l'argument par defaut est evalue chez l'APPELANT, pas ici.
///
/// Entierement inerte tant que la variable d'environnement est absente : le
/// getenv n'est lu qu'UNE fois (static local), jamais par draw.
///
/// Deux lignes par seconde en LOG_INFO, toutes deux prefixees "TRACE_SYNC".
/// ---------------------------------------------------------------------------
struct SyncStats {
    /// Vrai si BORKED3DS_V3DV_TRACE_SYNC est presente dans l'environnement.
    /// Le getenv n'est evalue qu'au premier appel.
    [[nodiscard]] static bool Enabled() noexcept;

    /// Enregistre un Finish() attribue a son site d'appel.
    static void RecordFinish(const char* file, u32 line, u64 elapsed_ns);

    /// Enregistre un blocage au rebouclage du stream buffer (site reel du
    /// figement periodique ~5 s a 0 %). Appele depuis
    /// StreamBuffer::WaitPendingOperations, sur l'EmuThread.
    ///   type_index : 0=Upload 1=Download 2=Stream
    ///   elapsed_ns : temps mur pendant lequel l'EmuThread est reste bloque.
    ///   ticks_waited : nombre de watches attendues dans ce rebouclage.
    ///   tick_gap : CurrentTick - tick de la plus vieille watche attendue
    ///              (mesure de l'avance prise par le CPU en unites de soumission).
    static void RecordStreamWait(u32 type_index, u64 elapsed_ns, u64 ticks_waited,
                                 u64 tick_gap) noexcept;

    /// Emet les lignes de releve si une seconde s'est ecoulee depuis la derniere.
    /// Remet tous les compteurs a zero apres emission.
    static void ReportIfDue();

    // Compteurs generaux du scheduler
    static std::atomic<u64> finish_calls;
    static std::atomic<u64> finish_ns;
    static std::atomic<u64> flush_calls;
    static std::atomic<u64> dispatch_calls;
    static std::atomic<u64> waitworker_calls;
    static std::atomic<u64> waitworker_ns;

    // Attribution manuelle conservee (incrementee depuis vk_texture_runtime.cpp).
    // Utile pour distinguer les rafales de chargement du regime etabli.
    static std::atomic<u64> site_surface_dtor;
    static std::atomic<u64> site_surface_download;
    static std::atomic<u64> site_runtime_finish;

    // -----------------------------------------------------------------------
    // Figement periodique -- rebouclage du stream buffer (v150c).
    //
    // La cause probable du figement ~5 s a 0 % n'est ni Finish() ni WaitWorker()
    // mais StreamBuffer::WaitPendingOperations : en mode C, plus rien n'est
    // soumis par draw, le CPU accumule un lot enorme, et au rebouclage la
    // premiere scheduler.Wait() force la soumission puis bloque l'EmuThread le
    // temps que le GPU draine TOUT le lot. Ces compteurs chiffrent ce blocage
    // la ou il se produit reellement.
    static std::atomic<u64> streambuf_wait_calls; // rebouclages ayant bloque
    static std::atomic<u64> streambuf_wait_ns;    // temps bloque cumule
    static std::atomic<u64> streambuf_wait_max_ns; // pire blocage unique (max)
    static std::atomic<u64> streambuf_wait_ticks; // watches attendues cumulees
    static std::atomic<u64> streambuf_gap_max;    // pire avance CPU (max)
    // Rebouclages bloquants par type : [0]=Upload [1]=Download [2]=Stream.
    static std::atomic<u64> streambuf_wait_by_type[3];
};

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    void Flush(vk::Semaphore signal = nullptr, vk::Semaphore wait = nullptr);

    /// Sends the current execution context to the GPU and waits for it to complete.
    ///
    /// Le dernier parametre n'est jamais a fournir : il capture automatiquement
    /// le fichier et la ligne de l'APPELANT pour la sonde TRACE_SYNC.
    void Finish(vk::Semaphore signal = nullptr, vk::Semaphore wait = nullptr,
                std::source_location loc = std::source_location::current());

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);

    /// Sends currently recorded work to the worker thread.
    void DispatchWork();

    /// Records the command to the current chunk.
    template <typename T>
    void Record(T&& command) {
        if (chunk->Record(command)) {
            return;
        }
        DispatchWork();
        (void)chunk->Record(command);
    }

    /// Marks the provided state as non dirty
    void MarkStateNonDirty(StateFlags flag) noexcept {
        state |= flag;
    }

    /// Marks the provided state as dirty
    void MakeDirty(StateFlags flag) noexcept {
        state &= ~flag;
    }

    /// Returns true if the state is dirty
    [[nodiscard]] bool IsStateDirty(StateFlags flag) const noexcept {
        return False(state & flag);
    }

    /// Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// Registers a callback to perform on queue submission.
    void RegisterOnDispatch(std::function<void()>&& func) {
        on_dispatch = std::move(func);
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return master_semaphore.get();
    }

    std::mutex submit_mutex;

private:
    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf) const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf) const override {
            command(cmdbuf);
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        void ExecuteAll(vk::CommandBuffer cmdbuf);

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

            recorded_counts++;
            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        bool Empty() const {
            return recorded_counts == 0;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        std::size_t recorded_counts = 0;
        std::size_t command_offset = 0;
        bool submit = false;
        alignas(std::max_align_t) std::array<u8, 0x8000> data{};
    };

private:
    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffers();

    void SubmitExecution(vk::Semaphore signal_semaphore, vk::Semaphore wait_semaphore);

    void AcquireNewChunk();

private:
    std::unique_ptr<MasterSemaphore> master_semaphore;
    CommandPool command_pool;
    std::unique_ptr<CommandChunk> chunk;
    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    vk::CommandBuffer current_cmdbuf;
    StateFlags state{};
    std::function<void()> on_submit;
    std::function<void()> on_dispatch;
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable_any event_cv;
    std::jthread worker_thread;
    bool use_worker_thread;
};

} // namespace Vulkan
