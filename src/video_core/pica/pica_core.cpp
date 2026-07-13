// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "common/arch.h"
#include "common/archives.h"
#include "common/profiling.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/pica/pica_core.h"
#include "video_core/pica/vertex_loader.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/shader/shader.h"

namespace Pica {

using namespace DebugUtils;

union CommandHeader {
    u32 hex;
    BitField<0, 16, u32> cmd_id;
    BitField<16, 4, u32> parameter_mask;
    BitField<20, 8, u32> extra_data_length;
    BitField<31, 1, u32> group_commands;
};
static_assert(sizeof(CommandHeader) == sizeof(u32), "CommandHeader has incorrect size!");

[[nodiscard]] bool IsEnvEnabled(const char* name) {
    // Perf: les variables d'environnement sont fixees au lancement et ne changent jamais
    // en cours d'execution. On met le resultat en cache pour eviter un getenv() par-draw
    // sur les predicats Is*Enabled() du chemin chaud partage GL+Vulkan. Comportement
    // identique : les sondes repondent toujours a emulators.cfg.
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, bool> cache;
    std::scoped_lock lock(cache_mutex);
    if (const auto it = cache.find(name); it != cache.end()) {
        return it->second;
    }
    const char* value = std::getenv(name);
    const bool enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
    cache.emplace(name, enabled);
    return enabled;
}

[[nodiscard]] bool IsInterestingPicaStateReg(u32 id) {
    return id == 0x203 || id == 0x206 || (id >= 0x1C8 && id <= 0x1CF);
}

[[nodiscard]] bool IsPicaHotpathTraceEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_TRACE_PICA_STATE");
}

[[nodiscard]] bool IsPicaDrawTraceEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_TRACE_DRAW");
}

[[nodiscard]] bool IsStrictCompatEnabled() {
    return IsEnvEnabled("BORKED3DS_V3DV_STRICT_COMPAT");
}

[[nodiscard]] bool IsPicaAccelAllowed() {
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_PICA_ACCEL");
}

[[nodiscard]] bool IsPicaAccelForcedOff() {
    return IsEnvEnabled("BORKED3DS_V3DV_FORCE_PICA_SOFTWARE");
}

[[nodiscard]] u32 GetEnvU32(const char* name, u32 fallback) {
    // Perf: meme logique de cache que IsEnvEnabled. Chaque nom est utilise avec un
    // fallback constant dans ce code, donc la mise en cache par nom est sans effet de bord.
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, u32> cache;
    std::scoped_lock lock(cache_mutex);
    if (const auto it = cache.find(name); it != cache.end()) {
        return it->second;
    }

    u32 result = fallback;
    const char* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end != value) {
            constexpr unsigned long max_u32 = 0xFFFFFFFFul;
            result = parsed > max_u32 ? 0xFFFFFFFFu : static_cast<u32>(parsed);
        }
    }
    cache.emplace(name, result);
    return result;
}


[[nodiscard]] bool IsV114C6PicaGateFileTraceEnabled() {
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE");
}

void V114C6PicaGateFileTraceRaw(const char* message) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_pica_gate.log", "a")) {
        std::fputs(message, fp);
        std::fputc('\n', fp);
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceU32(const char* key, u32 value) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_pica_gate.log", "a")) {
        std::fprintf(fp, "%s=0x%08X\n", key, value);
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceU64(const char* key, u64 value) {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_pica_gate.log", "a")) {
        std::fprintf(fp, "%s=%llu\n", key, static_cast<unsigned long long>(value));
        std::fclose(fp);
    }
}

void V114C6PicaGateFileTraceReset() {
    if (!IsV114C6PicaGateFileTraceEnabled()) {
        return;
    }
    if (std::FILE* fp = std::fopen("/tmp/borked3ds_v115d_mux_pica_gate.log", "w")) {
        std::fputs("v115d_mux pica_gate_file_trace_reset\n", fp);
        std::fputs("v115d_a7x pica_gate_file_trace_reset\n", fp);
        std::fclose(fp);
    }
}

[[nodiscard]] bool IsV115DA7XTraceExpected() {
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT") &&
           IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) == 7;
}

void V115DA7XPicaTraceRaw(const char* message) {
    if (!IsV115DA7XTraceExpected()) {
        return;
    }
    V114C6PicaGateFileTraceRaw(message);
}

void V115DA7XPicaTraceU32(const char* key, u32 value) {
    if (!IsV115DA7XTraceExpected()) {
        return;
    }
    V114C6PicaGateFileTraceU32(key, value);
}

void V115DA7XPicaTraceU64(const char* key, u64 value) {
    if (!IsV115DA7XTraceExpected()) {
        return;
    }
    V114C6PicaGateFileTraceU64(key, value);
}

[[nodiscard]] bool IsV115DA7Z19PicaCallBoundaryProbeEnabled() {
    // v115-D-E-A7Z19: caller-side boundary probe. The backend A7Z18 can return false before
    // pipeline bind, but previous logs did not show the PICA caller regaining control. This
    // flag keeps the probe in pica_core.cpp and writes only sidecar markers around the call.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z19_PICA_CALL_BOUNDARY_RETURN_PROBE");
}

[[nodiscard]] bool IsV115DA7Z19PicaSkipBackendCallEnabled() {
    // Optional emergency sanity check: prove the PICA caller-side return path without entering
    // the Vulkan backend. Keep disabled for the normal A7Z19 test.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z19_PICA_SKIP_BACKEND_CALL_RETURN_FALSE");
}

[[nodiscard]] bool IsV115DA7Z20PicaAfterBackendControlledReturnEnabled() {
    // v115-D-E-A7Z20: after-backend controlled return. A7Z19 proved that PICA can
    // skip the backend and return cleanly; this probe calls the backend, records the
    // result immediately after the call, then returns from DrawArrays without executing
    // the normal post-call path.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z20_PICA_AFTER_BACKEND_CONTROLLED_RETURN");
}

[[nodiscard]] bool IsV115DA7Z21PicaAfterBackendUltraCleanReturnEnabled() {
    // v115-D-E-A7Z21: A7Z20 proved that PICA reaches the first marker immediately after
    // a real backend call, but cuts before writing the result value. Return immediately
    // after the after-call breadcrumb, without formatting/logging the bool result and
    // without writing an additional final return breadcrumb.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z21_PICA_AFTER_BACKEND_ULTRA_CLEAN_RETURN");
}

[[nodiscard]] bool IsV115DA7Z22TriggerDrawArraysCallBoundaryProbeEnabled() {
    // v115-D-E-A7Z22: A7Z21 reaches the DrawArrays-side after-call breadcrumb after a
    // real backend call, but the outer trigger_draw_after_drawarrays marker is still absent.
    // This caller-side probe wraps DrawArrays() itself from the register-trigger path and
    // returns immediately after the call if control reaches this boundary.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z22_TRIGGER_DRAWARRAYS_CALL_BOUNDARY_PROBE");
}

[[nodiscard]] bool IsV115DA7Z62PicaPredrawLivenessEnabled() {
    // v115-D-E-A7Z62: lightweight PICA command-stream liveness. Keep this independent
    // from Vulkan so it can be used as a recovery breadcrumb when the draw corridor is
    // intermittent.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z62_PICA_PREDRAW_LIVENESS");
}

[[nodiscard]] bool IsV115DA7Z64PicaDrawArraysUltraQuietBoundaryEnabled() {
    // v115-D-E-A7Z64: replace the fragile dynamic DrawArrays console log with short
    // boundary breadcrumbs around trigger -> DrawArrays -> early direct decision.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z64_PICA_DRAWARRAYS_ULTRA_QUIET_BOUNDARY");
}

[[nodiscard]] bool IsV115DA7Z65PicaEarlyDirectNoPrebackendLogEnabled() {
    // v115-D-E-A7Z65: after the safe early-direct candidate is true and budget is accepted,
    // enter the Vulkan backend without a final PICA pre-backend log.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z65_PICA_EARLY_DIRECT_NO_PREBACKEND_LOG");
}

[[nodiscard]] bool IsV115DA7Z69PicaProcessCmdListUltraEarlyEnabled() {
    // v115-D-E-A7Z69: ultra-early ProcessCmdList boundary trace. A7Z68 proved that
    // GPU::SubmitCmdList calls and returns from ProcessCmdList, but the A7Z62 marker may
    // not appear. Emit the first breadcrumb before memory lookup/reset/dynamic command logs.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z69_PICA_PROCESS_CMDLIST_ULTRA_EARLY");
}

[[nodiscard]] bool IsV115DA7Z70PicaTriggerDirectDrawArraysEnabled() {
    // v115-D-E-A7Z70: A7Z69 proved that ProcessCmdList reaches trigger 0x22F and
    // WriteInternalReg reaches the trigger case, then the main log cuts while emitting
    // the legacy trigger/pre-draw trace. Skip all formatted/sidecar trigger breadcrumbs
    // between trigger_case_enter and DrawArrays(), then return immediately after the
    // DrawArrays() call so the next visible marker must come from DrawArrays/A7Z65 or
    // the Vulkan backend.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z70_PICA_TRIGGER_DIRECT_DRAWARRAYS");
}

[[nodiscard]] bool IsV115DA7Z71PicaTriggerSilentDrawArraysEnabled() {
    // v115-D-E-A7Z71: A7Z70 proved that even a fixed pre-DrawArrays breadcrumb can be
    // the last visible output (the log cuts at TRACE_DRAW_). For this micro-pass, do
    // not emit any trigger-case breadcrumb at all. Once WriteInternalReg has updated
    // the trigger register, jump directly to DrawArrays() and return.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z71_PICA_TRIGGER_SILENT_DRAWARRAYS");
}


[[nodiscard]] bool IsV115DA7Z72PicaDrawArraysSilentEarlyBackendEnabled() {
    // v115-D-E-A7Z72: A7Z71 proved that WriteInternalReg(0x22F) reaches DrawArrays(),
    // then the main log cuts at the next fixed DrawArrays/early-direct breadcrumb.
    // This mode performs the same strict safe-candidate gate, but with zero console/file
    // breadcrumbs inside DrawArrays before the backend call. The next visible proof must
    // therefore come from vk_rasterizer::AccelerateDrawBatch().
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z72_PICA_DRAWARRAYS_SILENT_EARLY_BACKEND");
}

[[nodiscard]] bool IsV115DA7Z76PicaSingleBackendCallMarkerEnabled() {
    // v115-D-E-A7Z76: A7Z75 proved the Vulkan-side internal-boundary marker is armed,
    // but the silent run did not show it. Emit exactly one fixed, no-argument marker
    // from the PICA side immediately before calling rasterizer->AccelerateDrawBatch().
    // This distinguishes a missed/false safe-candidate gate from a crash/stop inside
    // the backend call transition itself, without re-enabling the heavy GSP/PICA logs.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z76_PICA_SINGLE_BACKEND_CALL_MARKER");
}

[[nodiscard]] bool IsV115DA7Z77PicaDrawArraysSingleEntryMarkerEnabled() {
    // v115-D-E-A7Z77: A7Z76 proved the PICA-side backend-call marker is armed, but
    // it does not appear. Add exactly one fixed DrawArrays entry breadcrumb, before
    // the A7Z72 safe-candidate filters, to determine whether the silent trigger path
    // reaches DrawArrays at all without re-enabling broad GSP/GPU/PICA logging.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z77_PICA_DRAWARRAYS_SINGLE_ENTRY_MARKER");
}

[[nodiscard]] bool IsV115DA7Z78PicaTriggerSinglePreDrawArraysMarkerEnabled() {
    // v115-D-E-A7Z78: A7Z77 proved the DrawArrays entry marker is armed, but it does
    // not appear. Add exactly one fixed trigger-side breadcrumb immediately before
    // the A7Z71 silent DrawArrays() call. This determines whether the silent trigger
    // branch reaches the DrawArrays call site without re-enabling broad GSP/GPU/PICA
    // logging.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z78_PICA_TRIGGER_SINGLE_PRE_DRAWARRAYS_MARKER");
}

[[nodiscard]] bool IsV115DA7Z79PicaProcessSingleDrawTriggerMarkerEnabled() {
    // v115-D-E-A7Z79: A7Z78 proved the trigger-side pre-DrawArrays marker is armed,
    // but it does not appear in the quiet run. Emit exactly one fixed breadcrumb from
    // ProcessCmdList when a draw trigger command is parsed, immediately before the
    // command is handed to WriteInternalReg. This distinguishes "no draw trigger in
    // this quiet pass" from "trigger reaches WriteInternalReg but the silent branch is
    // not reached", without restoring broad A7Z66/A7Z69 logging.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z79_PICA_PROCESS_SINGLE_DRAW_TRIGGER_MARKER");
}

[[nodiscard]] bool IsV115DA7Z80PicaProcessSingleEntryMarkerEnabled() {
    // v115-D-E-A7Z80: A7Z79 is armed, but no draw-trigger breadcrumb appears in the
    // quiet run. Emit exactly one fixed marker at the very beginning of ProcessCmdList,
    // before memory lookup, reset, command parsing, or draw-trigger filtering. This
    // distinguishes "PICA command-list entry is not reached" from "PICA is entered,
    // but the submitted list does not contain the useful draw trigger" without
    // re-enabling broad GSP/GPU/PICA logging.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z80_PICA_PROCESS_SINGLE_ENTRY_MARKER");
}

[[nodiscard]] bool IsV115DA7Z81PicaProcessSingleLeaveMarkerEnabled() {
    // v115-D-E-A7Z81: A7Z80 proves ProcessCmdList is entered once in the quiet run,
    // but A7Z79 does not see a draw trigger. Emit exactly one fixed marker when
    // ProcessCmdList returns, including the ignore-list return path, to distinguish
    // "first quiet command list completed without draw" from "parser stalls/cuts
    // inside the first command list before reaching a draw trigger".
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z81_PICA_PROCESS_SINGLE_LEAVE_MARKER");
}

[[nodiscard]] bool IsV115DA7Z82PicaProcessEntryLeaveWindowEnabled() {
    // v115-D-E-A7Z82: A7Z80/A7Z81 proved the first quiet ProcessCmdList enters and
    // leaves, but A7Z79 does not see a draw trigger. Emit a tiny fixed window for the
    // next few ProcessCmdList calls so we can determine whether later command lists
    // are reached before the crash, without restoring the broad A7Z66/A7Z69 logging.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z82_PICA_PROCESS_ENTRY_LEAVE_WINDOW");
}

[[nodiscard]] bool IsV115DA7Z83PicaProcessSingleCommandBufferTriggerMarkerEnabled() {
    // v115-D-E-A7Z83: A7Z82 proved several quiet ProcessCmdList calls enter and leave
    // cleanly without a direct draw trigger. Emit exactly one fixed breadcrumb when a
    // PICA command-buffer trigger register is parsed. This checks whether the quiet
    // lists are scheduling/arming later GPU command buffers rather than issuing
    // trigger_draw/trigger_draw_indexed directly, without returning to broad logs.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z83_PICA_PROCESS_SINGLE_CMDBUF_TRIGGER_MARKER");
}

[[nodiscard]] bool IsV115DA7Z84PicaProcessExtendedWindowEnabled() {
    // v115-D-E-A7Z84 run3: run2 showed 64 lists (16 frames at 30Hz), all clean, no draw
    // trigger, no command_buffer trigger. Session alive after list 64 (APT/FS at t=2.76s).
    // The 64 lists are pure PICA register config. Sonic's loading phase has no draws at all.
    // Window limit REMOVED for run3 — emit all lists until A7Z79/A7Z83 fires, so we know
    // the exact list number of the first draw trigger without bounding the window.
    // Disabled in emulators.cfg for run3: A7Z84 spam is no longer useful now that we know
    // the loading phase is all config. Only A7Z79 and A7Z83 matter going forward.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z84_PICA_PROCESS_EXTENDED_WINDOW");
}

[[nodiscard]] bool IsV115DA7Z85PicaDrawArraysBackendOnceEnabled() {
    // v115-D-E-A7Z85: emit exactly one fixed breadcrumb the first time DrawArrays
    // reaches the AccelerateDrawBatch() call site in the active mux path.
    // A7Z84 run2 exhausted 64 lists without A7Z79/A7Z85. For run3, A7Z84 is disabled
    // and A7Z85 stays armed. Combined with A7Z79/A7Z83, three outcomes remain:
    // (a) A7Z79/A7Z83 + A7Z85 appear: backend reached, crash is inside AccelerateDrawBatch
    //     or immediately after in the Vulkan pipeline.
    // (b) A7Z79/A7Z83 appear, A7Z85 absent: trigger parsed but safe-candidate gate blocks
    //     the backend call — budget=0 or vertex count mismatch.
    // (c) Neither A7Z79 nor A7Z83 appear before crash: Sonic crashes before any draw, the
    //     problem is upstream of PICA drawing (APT/FS/init crash, not a Vulkan draw issue).
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z85_PICA_DRAWARRAYS_BACKEND_ONCE");
}

[[nodiscard]] bool IsV115DA7Z86PicaDrawArraysAfterBackendOnceEnabled() {
    // v115-D-E-A7Z86: emit exactly one fixed breadcrumb immediately AFTER the
    // AccelerateDrawBatch() call returns in the mux early-direct path. This fires only
    // if AccelerateDrawBatch() does not crash and returns control to DrawArrays.
    // Combined with A7Z85: if A7Z85 fires but A7Z86 does not, the crash is inside
    // AccelerateDrawBatch() itself (Vulkan pipeline / vk_rasterizer). If both fire,
    // the crash is after the draw call returns — in the post-draw path or next frame.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z86_PICA_DRAWARRAYS_AFTER_BACKEND_ONCE");
}

[[nodiscard]] bool IsV115DA7Z87PicaDrawTriggerCounterEnabled() {
    // v115-D-E-A7Z87: emit a breadcrumb every N-th draw trigger (configurable via env),
    // without the safe-candidate gate. Counts ALL trigger_draw and trigger_draw_indexed
    // commands regardless of vertex count or hw_shader state. Used to measure how many
    // draws Sonic emits before the crash, to distinguish "first draw crashes" from
    // "later draw crashes". Controlled by BORKED3DS_V3DV_A7Z87_DRAW_TRIGGER_EVERY_N.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z87_PICA_DRAW_TRIGGER_COUNTER");
}

[[nodiscard]] u32 GetV115DA7Z87DrawTriggerEveryN() {
    return GetEnvU32("BORKED3DS_V3DV_A7Z87_DRAW_TRIGGER_EVERY_N", 1);
}

[[nodiscard]] bool IsV115DA7Z88PicaProcessCmdListCountEnabled() {
    // v115-D-E-A7Z88: emit a compact breadcrumb every 64 command lists (no entry/leave
    // spam). Format: "cmd_list_count_N" where N is a multiple of 64. Replaces A7Z84 for
    // long-running sessions where we only need to know how far we've gone, not the exact
    // list number. Lets us detect if Sonic reaches 128, 256, or 512 lists before the crash.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z88_PICA_PROCESS_CMDLIST_COUNT");
}

[[nodiscard]] bool IsV115DA7Z89PicaDrawArraysUltraEarlyProbeEnabled() {
    // v115-D-E-A7Z89: ultra-early DrawArrays entry probe. Emitted at the very first line
    // of DrawArrays(), before any safe-candidate gate, before use_hw_shader checks, before
    // AccelerateDrawBatch decisions. Every call is logged (no once-guard).
    //
    // Context: the software run (use_hw_shader=false) reaches 658 draw triggers and
    // completes cleanly. The HW shader run (use_hw_shader=true) crashes silently at
    // t≈2.549 after OpenFile /network_id.dat, before the first draw trigger at t≈2.944.
    // A7Z85 (before AccelerateDrawBatch) has never appeared in any HW run.
    //
    // This probe answers: does DrawArrays() get called at all in HW mode before the crash?
    // If A7Z89 never appears → the crash is before DrawArrays (ARM code, HLE, or GPU thread).
    // If A7Z89 appears → DrawArrays is reached, crash is inside DrawArrays or AccelerateDrawBatch.
    // Combined with A7Z85: if A7Z89 fires but A7Z85 does not → crash in the safe-candidate
    // gate path between the two probes.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_A7Z89_PICA_DRAWARRAYS_ULTRA_EARLY_PROBE");
}

void V115DA7Z80EmitProcessEntryOnce() {
    if (!IsV115DA7Z80PicaProcessSingleEntryMarkerEnabled()) {
        return;
    }
    static std::atomic<u64> a7z80_process_entry_counter{0};
    if (++a7z80_process_entry_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z80 process_cmd_list_entry_once");
    }
}

void V115DA7Z81EmitProcessLeaveOnce() {
    if (!IsV115DA7Z81PicaProcessSingleLeaveMarkerEnabled()) {
        return;
    }
    static std::atomic<u64> a7z81_process_leave_counter{0};
    if (++a7z81_process_leave_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z81 process_cmd_list_leave_once");
    }
}

void V115DA7Z82EmitProcessEntryWindow() {
    if (!IsV115DA7Z82PicaProcessEntryLeaveWindowEnabled()) {
        return;
    }
    static std::atomic<u64> a7z82_process_entry_counter{0};
    const u64 index = ++a7z82_process_entry_counter;
    switch (index) {
    case 1:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_1");
        break;
    case 2:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_2");
        break;
    case 3:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_3");
        break;
    case 4:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_4");
        break;
    case 5:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_5");
        break;
    case 6:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_6");
        break;
    case 7:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_7");
        break;
    case 8:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_entry_8");
        break;
    default:
        break;
    }
}

void V115DA7Z82EmitProcessLeaveWindow() {
    if (!IsV115DA7Z82PicaProcessEntryLeaveWindowEnabled()) {
        return;
    }
    static std::atomic<u64> a7z82_process_leave_counter{0};
    const u64 index = ++a7z82_process_leave_counter;
    switch (index) {
    case 1:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_1");
        break;
    case 2:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_2");
        break;
    case 3:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_3");
        break;
    case 4:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_4");
        break;
    case 5:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_5");
        break;
    case 6:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_6");
        break;
    case 7:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_7");
        break;
    case 8:
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z82 process_cmd_list_leave_8");
        break;
    default:
        break;
    }
}

void V115DA7Z79EmitProcessDrawTriggerOnce() {
    if (!IsV115DA7Z79PicaProcessSingleDrawTriggerMarkerEnabled()) {
        return;
    }
    static std::atomic<u64> a7z79_process_draw_trigger_counter{0};
    if (++a7z79_process_draw_trigger_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z79 process_draw_trigger_once");
    }
}

void V115DA7Z83EmitProcessCommandBufferTriggerOnce() {
    if (!IsV115DA7Z83PicaProcessSingleCommandBufferTriggerMarkerEnabled()) {
        return;
    }
    static std::atomic<u64> a7z83_process_command_buffer_trigger_counter{0};
    if (++a7z83_process_command_buffer_trigger_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z83 process_command_buffer_trigger_once");
    }
}

void V115DA7Z84EmitProcessEntryWindow() {
    if (!IsV115DA7Z84PicaProcessExtendedWindowEnabled()) {
        return;
    }
    static std::atomic<u64> a7z84_process_entry_counter{0};
    const u64 index = ++a7z84_process_entry_counter;
    if (index <= 64) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z84 process_cmd_list_entry_{}", index);
    }
}

void V115DA7Z84EmitProcessLeaveWindow() {
    if (!IsV115DA7Z84PicaProcessExtendedWindowEnabled()) {
        return;
    }
    static std::atomic<u64> a7z84_process_leave_counter{0};
    const u64 index = ++a7z84_process_leave_counter;
    if (index <= 64) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z84 process_cmd_list_leave_{}", index);
    }
}

void V115DA7Z85EmitDrawArraysBackendOnce() {
    if (!IsV115DA7Z85PicaDrawArraysBackendOnceEnabled()) {
        return;
    }
    static std::atomic<u64> a7z85_backend_counter{0};
    if (++a7z85_backend_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z85 drawarrays_backend_once");
    }
}

void V115DA7Z86EmitDrawArraysAfterBackendOnce() {
    if (!IsV115DA7Z86PicaDrawArraysAfterBackendOnceEnabled()) {
        return;
    }
    static std::atomic<u64> a7z86_after_backend_counter{0};
    if (++a7z86_after_backend_counter == 1) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z86 drawarrays_after_backend_once");
    }
}

void V115DA7Z87EmitDrawTriggerCounter() {
    if (!IsV115DA7Z87PicaDrawTriggerCounterEnabled()) {
        return;
    }
    static std::atomic<u64> a7z87_draw_trigger_counter{0};
    const u64 count = ++a7z87_draw_trigger_counter;
    const u32 every_n = GetV115DA7Z87DrawTriggerEveryN();
    if (every_n == 0) {
        return;
    }
    if (count == 1 || count % every_n == 0) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z87 draw_trigger_count_{}", count);
    }
}

void V115DA7Z88EmitProcessCmdListCount() {
    if (!IsV115DA7Z88PicaProcessCmdListCountEnabled()) {
        return;
    }
    static std::atomic<u64> a7z88_cmdlist_counter{0};
    const u64 count = ++a7z88_cmdlist_counter;
    if (count % 64 == 0) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z88 cmd_list_count_{}", count);
    }
}

void V115DA7Z89EmitDrawArraysUltraEarly(u64 draw_index, bool is_indexed,
                                         u32 num_vertices, u32 topology,
                                         bool primitive_empty, bool use_hw_shader) {
    if (!IsV115DA7Z89PicaDrawArraysUltraEarlyProbeEnabled()) {
        return;
    }
    // Every call logged — no once-guard. This tells us if DrawArrays is reached
    // at all in HW mode and exactly what draw parameters arrive.
    LOG_WARNING(HW_GPU,
                "TRACE_DRAW_PICA strict_compat v115d_a7z89 drawarrays_ultra_early"
                " draw_index={} indexed={} num_vertices={} topology={}"
                " primitive_empty={} use_hw_shader={}",
                draw_index, static_cast<u32>(is_indexed), num_vertices, topology,
                static_cast<u32>(primitive_empty), static_cast<u32>(use_hw_shader));
}

[[nodiscard]] bool IsSafePicaHwDrawAllowed() {
    // v114 follows plan de travail 1, with the result from the v110 runtime log:
    // v110 proved the backend can emit raw_enter_noargs and continue until hotkey exit.
    // Therefore v114 keeps pica_core pre/post handoff logs suppressed, disables raw-enter-only
    // return, emits raw_enter_noargs + raw_enter_simple in vk_rasterizer, and returns before
    // stage=1/shader work.
    return IsEnvEnabled("BORKED3DS_V3DV_ALLOW_SAFE_PICA_HW_DRAWS") &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_DRAWS");
}

[[nodiscard]] bool IsStagedPicaHwPreflightRequested() {
    return IsStrictCompatEnabled() && IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE") &&
           GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0) > 0;
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffEnabled() {
    // v114 diagnostic:
    // v108 reached the safe micro-HW candidate with enter=1, then the log began a new
    // "HW.GPU <" line but never completed pre_call_direct_noargs and no TRACE_ACCEL_STAGE line
    // appeared. Keep direct handoff enabled, but let the normal v114 test suppress the pica_core
    // pre-call log and make the backend return silently before any raw_enter logging.
    return IsStrictCompatEnabled() &&
           (IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF") ||
            (IsStagedPicaHwPreflightRequested() &&
             IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffNoPrelogEnabled() {
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG");
}

[[nodiscard]] bool IsDirectSafePicaHwHandoffPostlogEnabled() {
    // Disabled by default for v114. Enable only if the silent call-boundary probe already proved
    // stable and we want a pica_core post-call marker.
    return IsStrictCompatEnabled() &&
           IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_POSTLOG");
}

[[nodiscard]] bool IsSafePicaHwEnterAllowed() {
    // v114 diagnostic guard:
    // v110 survived raw_enter_noargs. Keep staged preflight entry active and suppress
    // pica_core pre/post call logs while the backend emits raw_enter_simple and returns.
    //
    // This still does NOT enable broad PICA acceleration:
    //   - the caller must already have strict_safe_pica_hw_draw=true;
    //   - the per-run budget and max-vertex filters still apply;
    //   - the backend must still stop at BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER before stage=7.
    const bool explicit_enter = IsEnvEnabled("BORKED3DS_V3DV_ENTER_SAFE_PICA_HW_DRAWS") ||
                                IsEnvEnabled("BORKED3DS_V3DV_EXECUTE_SAFE_PICA_HW_DRAWS");

    return (explicit_enter || IsStagedPicaHwPreflightRequested()) &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_ENTER");
}

[[nodiscard]] bool IsSafePicaHwDryRunEnabled() {
    return !IsSafePicaHwEnterAllowed() &&
           !IsEnvEnabled("BORKED3DS_V3DV_DISABLE_SAFE_PICA_HW_DRY_RUN");
}

[[nodiscard]] bool IsVerbosePicaMicroTextureTraceEnabled() {
    // v87 appeared to terminate during the verbose texture-state block around the first
    // micro candidate. Keep the candidate log minimal by default; this opt-in exists only
    // for diagnosis once the handoff is stable.
    return IsEnvEnabled("BORKED3DS_V3DV_VERBOSE_PICA_MICRO_TEXTURE_TRACE");
}

[[nodiscard]] u32 GetSafePicaHwDrawBudget() {
    // Keep this tiny. If this path is correct, one or a few hardware draws should already
    // change the render target from pure black. Higher budgets belong in later passes.
    return GetEnvU32("BORKED3DS_V3DV_SAFE_PICA_HW_DRAW_BUDGET", 1);
}

[[nodiscard]] u32 GetSafePicaHwMaxVertices() {
    return GetEnvU32("BORKED3DS_V3DV_SAFE_PICA_HW_MAX_VERTICES", 6);
}

[[nodiscard]] bool ArePrimaryTexturesDisabled(const RegsInternal& regs) {
    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        if (textures[i].enabled) {
            return false;
        }
    }
    return true;
}

void LogPicaTextureState(const RegsInternal& regs, const char* tag) {
    if (!IsPicaDrawTraceEnabled()) {
        return;
    }

    const auto& textures = regs.texturing.GetTextures();
    for (u32 i = 0; i < 3; ++i) {
        const auto& texture = textures[i];
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA {} tex{} enabled={} type={} format={} addr=0x{:08X}",
                 tag, i, texture.enabled, static_cast<u32>(texture.config.type.Value()),
                 static_cast<u32>(texture.format), texture.config.GetPhysicalAddress());
    }
}

void LogMainConfigTransition(const RegsInternal& regs, u32 id, u32 old_value, u32 new_value, u32 mask) {
    if (!IsPicaDrawTraceEnabled() || id != PICA_REG_INDEX(texturing.main_config)) {
        return;
    }

    const u32 old_t0 = old_value & 1u;
    const u32 new_t0 = new_value & 1u;
    const u32 old_t1 = (old_value >> 1) & 1u;
    const u32 new_t1 = (new_value >> 1) & 1u;
    const u32 old_t2 = (old_value >> 2) & 1u;
    const u32 new_t2 = (new_value >> 2) & 1u;

    LOG_INFO(HW_GPU,
             "TRACE_DRAW_PICA main_config_transition old=0x{:08X} new=0x{:08X} mask=0x{:X} old_t0={} new_t0={} old_t1={} new_t1={} old_t2={} new_t2={}",
             old_value, new_value, mask, old_t0, new_t0, old_t1, new_t1, old_t2, new_t2);

    V114C6PicaGateFileTraceRaw("v115d_mux main_config_transition");
    V114C6PicaGateFileTraceU32("v115d_mux main_config_old", old_value);
    V114C6PicaGateFileTraceU32("v115d_mux main_config_new", new_value);
    V114C6PicaGateFileTraceU32("v115d_mux main_config_mask", mask);
}


std::atomic<u64> g_pica_draw_counter{0};
std::atomic<u64> g_pica_safe_hw_draw_counter{0};
std::atomic<bool> g_logged_strict_accel_gate{false};

PicaCore::PicaCore(Memory::MemorySystem& memory_, std::shared_ptr<DebugContext> debug_context_)
    : memory{memory_}, debug_context{std::move(debug_context_)},
      geometry_pipeline{regs.internal, gs_unit, gs_setup},
      shader_engine{CreateEngine(Settings::values.use_shader_jit.GetValue())} {
    InitializeRegs();

    V114C6PicaGateFileTraceReset();
    V114C6PicaGateFileTraceRaw("v115d_mux pica_core_constructor_first_vkcmd_draw_zero_count_real_vertex_bind_ultra_quiet_marker");
    V114C6PicaGateFileTraceU32("v115d_mux constructor_pipeline_bind_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PIPELINE_BIND_ONLY")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_first_vkcmd_draw_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ONLY")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_first_vkcmd_draw_zero_count_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_ONLY")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_first_vkcmd_draw_zero_count_real_vertex_bind_ultra_quiet_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_REAL_VERTEX_BIND_ULTRA_QUIET_ONLY")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_d_a_draw0_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_d_b_draw3_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_B_REAL_VERTEX_BIND_DRAWCMD_3")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_d_c_draw6_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_C_REAL_VERTEX_BIND_DRAWCMD_6")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_d_d_drawindexed0_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_D_INDEXED_SETUP_DRAWINDEXED_ZEROCOUNT")));
    V114C6PicaGateFileTraceU32("v115d_mux constructor_d_e_drawindexed3_probe",
                                static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_E_INDEXED_SETUP_DRAWINDEXED_3")));
    V115DA7XPicaTraceRaw("v115d_a7x pica_core_constructor_marker");
    V115DA7XPicaTraceU32("v115d_a7x constructor_d_a_draw0",
                         static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT")));
    V115DA7XPicaTraceU32("v115d_a7x constructor_generate_guarded",
                         static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));
    V115DA7XPicaTraceU32("v115d_a7x constructor_stage_stop_after",
                         GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0));

    if (IsStrictCompatEnabled()) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_mux pica_core_constructor_draw_command_mux_marker direct_handoff={} no_prelog={} file_trace={} silent_stages={} shader_module_probe={} pipeline_bind_probe={} first_vkcmd_draw_probe={} first_vkcmd_draw_zero_count_probe={} first_vkcmd_draw_zero_count_real_vertex_bind_ultra_quiet_probe={} d_a_draw0={} d_b_draw3={} d_c_draw6={} d_d_drawindexed0={} d_e_drawindexed3={} stage_stop_after={}",
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_FILE_TRACE")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_SHADER_MULTIPLEX_SILENT_STAGES")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_SHADER_MODULE_ONLY")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PIPELINE_BIND_ONLY")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ONLY")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_ONLY")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_REAL_VERTEX_BIND_ULTRA_QUIET_ONLY")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_B_REAL_VERTEX_BIND_DRAWCMD_3")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_C_REAL_VERTEX_BIND_DRAWCMD_6")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_D_INDEXED_SETUP_DRAWINDEXED_ZEROCOUNT")),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_E_INDEXED_SETUP_DRAWINDEXED_3")),
                    GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0));
        if (IsV115DA7XTraceExpected()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7x pica_core_constructor_marker d_a_draw0={} generate_guarded={} stage_stop_after={}",
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_A_REAL_VERTEX_BIND_DRAWCMD_ZEROCOUNT")),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")),
                        GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0));
        }
        if (IsV115DA7Z62PicaPredrawLivenessEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z62 constructor_predraw_liveness active=1 d_e_drawindexed3={} direct_handoff={} no_prelog={} safe_enter={} safe_budget={} safe_max_vertices={}",
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_V115_D_E_INDEXED_SETUP_DRAWINDEXED_3")),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF")),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG")),
                        static_cast<u32>(IsSafePicaHwEnterAllowed()), GetSafePicaHwDrawBudget(),
                        GetSafePicaHwMaxVertices());
        }
        if (IsV115DA7Z64PicaDrawArraysUltraQuietBoundaryEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z64 constructor_drawarrays_ultra_quiet active=1");
        }
        if (IsV115DA7Z65PicaEarlyDirectNoPrebackendLogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z65 constructor_early_direct_no_prebackend active=1");
        }
        if (IsV115DA7Z69PicaProcessCmdListUltraEarlyEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 constructor_process_cmdlist_ultra_early active=1");
        }
        if (IsV115DA7Z70PicaTriggerDirectDrawArraysEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z70 constructor_trigger_direct_drawarrays active=1");
        }
        if (IsV115DA7Z71PicaTriggerSilentDrawArraysEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z71 constructor_trigger_silent_drawarrays active=1");
        }
        if (IsV115DA7Z72PicaDrawArraysSilentEarlyBackendEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z72 constructor_drawarrays_silent_early_backend active=1");
        }
        if (IsV115DA7Z76PicaSingleBackendCallMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z76 constructor_single_backend_call_marker active=1");
        }
        if (IsV115DA7Z77PicaDrawArraysSingleEntryMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z77 constructor_drawarrays_single_entry_marker active=1");
        }
        if (IsV115DA7Z78PicaTriggerSinglePreDrawArraysMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z78 constructor_trigger_single_pre_drawarrays_marker active=1");
        }
        if (IsV115DA7Z79PicaProcessSingleDrawTriggerMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z79 constructor_process_single_draw_trigger_marker active=1");
        }
        if (IsV115DA7Z80PicaProcessSingleEntryMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z80 constructor_process_single_entry_marker active=1");
        }
        if (IsV115DA7Z81PicaProcessSingleLeaveMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z81 constructor_process_single_leave_marker active=1");
        }
        if (IsV115DA7Z82PicaProcessEntryLeaveWindowEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z82 constructor_process_entry_leave_window active=1");
        }
        if (IsV115DA7Z83PicaProcessSingleCommandBufferTriggerMarkerEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z83 constructor_process_single_cmdbuf_trigger_marker active=1");
        }
        if (IsV115DA7Z84PicaProcessExtendedWindowEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z84 constructor_process_extended_window active=1 window_size=unlimited");
        }
        if (IsV115DA7Z85PicaDrawArraysBackendOnceEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z85 constructor_drawarrays_backend_once active=1");
        }
        if (IsV115DA7Z86PicaDrawArraysAfterBackendOnceEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z86 constructor_drawarrays_after_backend_once active=1");
        }
        if (IsV115DA7Z87PicaDrawTriggerCounterEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z87 constructor_draw_trigger_counter active=1 every_n={}",
                        GetV115DA7Z87DrawTriggerEveryN());
        }
        if (IsV115DA7Z88PicaProcessCmdListCountEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z88 constructor_process_cmdlist_count active=1");
        }
        if (IsV115DA7Z89PicaDrawArraysUltraEarlyProbeEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z89 constructor_drawarrays_ultra_early active=1");
        }
    }

    const auto submit_vertex = [this](const AttributeBuffer& buffer) {
        const auto add_triangle = [this](const OutputVertex& v0, const OutputVertex& v1,
                                         const OutputVertex& v2) {
            rasterizer->AddTriangle(v0, v1, v2);
        };
        const auto vertex = OutputVertex(regs.internal.rasterizer, buffer);
        primitive_assembler.SubmitVertex(vertex, add_triangle);
    };

    gs_unit.SetVertexHandlers(submit_vertex, [this]() { primitive_assembler.SetWinding(); });
    geometry_pipeline.SetVertexHandler(submit_vertex);

    primitive_assembler.Reconfigure(PipelineRegs::TriangleTopology::List);
}

PicaCore::~PicaCore() = default;

void PicaCore::InitializeRegs() {
    auto& framebuffer_top = regs.framebuffer_config[0];
    auto& framebuffer_sub = regs.framebuffer_config[1];

    // Set framebuffer defaults from nn::gx::Initialize.
    framebuffer_top.address_left1 = 0x181E6000;
    framebuffer_top.address_left2 = 0x1822C800;
    framebuffer_top.address_right1 = 0x18273000;
    framebuffer_top.address_right2 = 0x182B9800;
    framebuffer_sub.address_left1 = 0x1848F000;
    framebuffer_sub.address_left2 = 0x184C7800;

    framebuffer_top.width.Assign(240);
    framebuffer_top.height.Assign(400);
    framebuffer_top.stride = 3 * 240;
    framebuffer_top.color_format.Assign(PixelFormat::RGB8);
    framebuffer_top.active_fb = 0;

    framebuffer_sub.width.Assign(240);
    framebuffer_sub.height.Assign(320);
    framebuffer_sub.stride = 3 * 240;
    framebuffer_sub.color_format.Assign(PixelFormat::RGB8);
    framebuffer_sub.active_fb = 0;

    // Tales of Abyss expects this register to have the following default values.
    auto& gs = regs.internal.gs;
    gs.max_input_attribute_index.Assign(1);
    gs.shader_mode.Assign(ShaderRegs::ShaderMode::VS);
}

void PicaCore::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

void PicaCore::SetInterruptHandler(Service::GSP::InterruptHandler& signal_interrupt_) {
    signal_interrupt = signal_interrupt_;
}

void PicaCore::ProcessCmdList(PAddr list, u32 size, bool ignore_list) {
    V115DA7Z80EmitProcessEntryOnce();
    V115DA7Z82EmitProcessEntryWindow();
    V115DA7Z84EmitProcessEntryWindow();
    V115DA7Z88EmitProcessCmdListCount();

    const bool trace_hotpath = IsPicaHotpathTraceEnabled();
    const bool trace_a7z62 = IsV115DA7Z62PicaPredrawLivenessEnabled();
    const bool trace_a7z69 = IsV115DA7Z69PicaProcessCmdListUltraEarlyEnabled();

    if (trace_a7z69) {
        static std::atomic<u64> a7z69_process_enter_counter{0};
        const u64 a7z69_process_enter_index = ++a7z69_process_enter_counter;
        if (a7z69_process_enter_index <= 256 || a7z69_process_enter_index == 512 ||
            a7z69_process_enter_index == 1024) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd_list_ultra_early_enter seq={} list=0x{:08X} size=0x{:08X} ignore_list={}",
                        a7z69_process_enter_index, list, size, static_cast<u32>(ignore_list));
        }
        V114C6PicaGateFileTraceRaw("v115d_a7z69 process_cmd_list_ultra_early_enter");
        V114C6PicaGateFileTraceU32("v115d_a7z69 process_cmd_list_list", list);
        V114C6PicaGateFileTraceU32("v115d_a7z69 process_cmd_list_size", size);
        V114C6PicaGateFileTraceU32("v115d_a7z69 process_cmd_list_ignore", static_cast<u32>(ignore_list));
    }

    if (trace_a7z62) {
        static std::atomic<u64> a7z62_process_begin_counter{0};
        const u64 a7z62_process_begin_index = ++a7z62_process_begin_counter;
        if (a7z62_process_begin_index <= 128 || a7z62_process_begin_index == 256 ||
            a7z62_process_begin_index == 512) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z62 process_cmd_list_begin seq={} list=0x{:08X} size=0x{:08X} ignore_list={}",
                        a7z62_process_begin_index, list, size, static_cast<u32>(ignore_list));
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList begin list={:#010X} size={} ignore_list={}",
                  list, size, ignore_list);
    }

    if (ignore_list) {
        if (trace_a7z69) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd_list_ignore_return list=0x{:08X} size=0x{:08X}",
                        list, size);
        }
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList ignored list={:#010X}", list);
        }
        V115DA7Z81EmitProcessLeaveOnce();
        V115DA7Z82EmitProcessLeaveWindow();
        V115DA7Z84EmitProcessLeaveWindow();
        signal_interrupt(Service::GSP::InterruptId::P3D);
        return;
    }

    const u8* head = memory.GetPhysicalPointer(list);
    if (trace_a7z69) {
        static std::atomic<u64> a7z69_after_pointer_counter{0};
        const u64 a7z69_after_pointer_index = ++a7z69_after_pointer_counter;
        if (a7z69_after_pointer_index <= 256 || a7z69_after_pointer_index == 512 ||
            a7z69_after_pointer_index == 1024) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd_list_after_pointer seq={} list=0x{:08X} size=0x{:08X} head_valid={}",
                        a7z69_after_pointer_index, list, size, static_cast<u32>(head != nullptr));
        }
        V114C6PicaGateFileTraceRaw("v115d_a7z69 process_cmd_list_after_pointer");
        V114C6PicaGateFileTraceU32("v115d_a7z69 process_cmd_list_head_valid", static_cast<u32>(head != nullptr));
    }
    cmd_list.Reset(list, head, size);
    if (trace_a7z69) {
        static std::atomic<u64> a7z69_after_reset_counter{0};
        const u64 a7z69_after_reset_index = ++a7z69_after_reset_counter;
        if (a7z69_after_reset_index <= 256 || a7z69_after_reset_index == 512 ||
            a7z69_after_reset_index == 1024) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd_list_after_reset seq={} current_index={} length={}",
                        a7z69_after_reset_index, cmd_list.current_index, cmd_list.length);
        }
    }

    while (cmd_list.current_index < cmd_list.length) {
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};
        const u32 header_cmd_id = header.cmd_id.Value();

        if (trace_a7z69) {
            static std::atomic<u64> a7z69_cmd_counter{0};
            const u64 a7z69_cmd_index = ++a7z69_cmd_counter;
            const bool a7z69_draw_trigger =
                header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw) ||
                header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw_indexed);
            if (a7z69_cmd_index <= 96 || a7z69_draw_trigger) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd seq={} cmd_id=0x{:03X} value=0x{:08X} mask=0x{:X} extra_len={} grouped={} draw_trigger={}",
                            a7z69_cmd_index, header_cmd_id, value, header.parameter_mask.Value(),
                            header.extra_data_length.Value(), header.group_commands.Value(),
                            static_cast<u32>(a7z69_draw_trigger));
            }
        }

        if (trace_a7z62) {
            static std::atomic<u64> a7z62_cmd_counter{0};
            const u64 a7z62_cmd_index = ++a7z62_cmd_counter;
            const bool a7z62_draw_trigger =
                header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw) ||
                header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw_indexed);
            if (a7z62_cmd_index <= 64 || a7z62_draw_trigger) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_a7z62 process_cmd seq={} id=0x{:03X} draw_trigger={} value=0x{:08X} extra_len={}",
                            a7z62_cmd_index, header_cmd_id, static_cast<u32>(a7z62_draw_trigger),
                            value, header.extra_data_length.Value());
            }
        }

        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::ProcessCmdList cmd id=0x{:03X} value=0x{:08X} mask=0x{:X} extra_len={} grouped={}",
                      header.cmd_id.Value(), value, header.parameter_mask.Value(),
                      header.extra_data_length.Value(), header.group_commands.Value());
        }

        if (header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw) ||
            header_cmd_id == PICA_REG_INDEX(pipeline.trigger_draw_indexed)) {
            V115DA7Z79EmitProcessDrawTriggerOnce();
            V115DA7Z87EmitDrawTriggerCounter();
        }
        if (header_cmd_id == PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) ||
            header_cmd_id == PICA_REG_INDEX(pipeline.command_buffer.trigger[1])) {
            V115DA7Z83EmitProcessCommandBufferTriggerOnce();
        }

        WriteInternalReg(header.cmd_id, value, header.parameter_mask);

        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];

            if (trace_a7z69) {
                const bool a7z69_extra_draw_trigger =
                    cmd == PICA_REG_INDEX(pipeline.trigger_draw) ||
                    cmd == PICA_REG_INDEX(pipeline.trigger_draw_indexed);
                if (a7z69_extra_draw_trigger) {
                    LOG_WARNING(HW_GPU,
                                "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_extra_draw_trigger cmd_id=0x{:03X} value=0x{:08X} mask=0x{:X} extra_index={}",
                                cmd, extra_value, header.parameter_mask.Value(), i);
                }
            }

            if (trace_hotpath) {
                LOG_DEBUG(HW_GPU,
                          "PicaCore::ProcessCmdList extra cmd id=0x{:03X} value=0x{:08X} mask=0x{:X}",
                          cmd, extra_value, header.parameter_mask.Value());
            }

            if (cmd == PICA_REG_INDEX(pipeline.trigger_draw) ||
                cmd == PICA_REG_INDEX(pipeline.trigger_draw_indexed)) {
                V115DA7Z79EmitProcessDrawTriggerOnce();
                V115DA7Z87EmitDrawTriggerCounter();
            }
            if (cmd == PICA_REG_INDEX(pipeline.command_buffer.trigger[0]) ||
                cmd == PICA_REG_INDEX(pipeline.command_buffer.trigger[1])) {
                V115DA7Z83EmitProcessCommandBufferTriggerOnce();
            }

            WriteInternalReg(cmd, extra_value, header.parameter_mask);
        }
    }

    if (trace_a7z69) {
        static std::atomic<u64> a7z69_process_leave_counter{0};
        const u64 a7z69_process_leave_index = ++a7z69_process_leave_counter;
        if (a7z69_process_leave_index <= 256 || a7z69_process_leave_index == 512 ||
            a7z69_process_leave_index == 1024) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z69 process_cmd_list_leave seq={} list=0x{:08X} processed_words={} length={}",
                        a7z69_process_leave_index, list, cmd_list.current_index, cmd_list.length);
        }
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::ProcessCmdList end list={:#010X} processed_words={} length={}",
                  list, cmd_list.current_index, cmd_list.length);
    }

    V115DA7Z81EmitProcessLeaveOnce();
    V115DA7Z82EmitProcessLeaveWindow();
    V115DA7Z84EmitProcessLeaveWindow();
}

void PicaCore::WriteInternalReg(u32 id, u32 value, u32 mask) {
    if (id >= RegsInternal::NUM_REGS) {
        LOG_ERROR(HW_GPU,
                  "Commandlist tried to write to invalid register 0x{:03X} (value: {:08X}, mask: {:X})",
                  id, value, mask);
        return;
    }

    // Expand a 4-bit mask to 4-byte mask, e.g. 0b0101 -> 0x00FF00FF.
    constexpr std::array<u32, 16> ExpandBitsToBytes = {
        0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff,
        0x00ffff00, 0x00ffffff, 0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff,
        0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
    };

    const u32 old_value = regs.internal.reg_array[id];
    const u32 write_mask = ExpandBitsToBytes[mask];
    regs.internal.reg_array[id] = (old_value & ~write_mask) | (value & write_mask);

    LogMainConfigTransition(regs.internal, id, old_value, regs.internal.reg_array[id], mask);
    DebugUtils::OnPicaRegWrite(id, mask, regs.internal.reg_array[id]);

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::PicaCommandLoaded, &id);
        SCOPE_EXIT({ debug_context->OnEvent(DebugContext::Event::PicaCommandProcessed, &id); });
    }

    if (IsPicaHotpathTraceEnabled() && IsInterestingPicaStateReg(id)) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::WriteInternalReg interesting_state_reg id=0x{:03X} value=0x{:08X} mask=0x{:X}",
                  id, regs.internal.reg_array[id], mask);
    }

    switch (id) {
    case PICA_REG_INDEX(trigger_irq):
        signal_interrupt(Service::GSP::InterruptId::P3D);
        break;

    case PICA_REG_INDEX(pipeline.triangle_topology):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg triangle_topology={}",
                      static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        }
        primitive_assembler.Reconfigure(regs.internal.pipeline.triangle_topology);
        break;

    case PICA_REG_INDEX(pipeline.restart_primitive):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg restart_primitive value=0x{:08X}", value);
        }
        primitive_assembler.Reset();
        break;

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.index):
        immediate.Reset();
        break;

    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[0]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[1]):
    case PICA_REG_INDEX(pipeline.vs_default_attributes_setup.set_value[2]):
        SubmitImmediate(value);
        break;

    case PICA_REG_INDEX(pipeline.gpu_mode):
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU, "PicaCore::WriteInternalReg gpu_mode value=0x{:08X}", value);
        }
        break;

    case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
    case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
        const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
        const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::WriteInternalReg command_buffer.trigger index={} addr={:#010X} size={}",
                      index, addr, size);
        }
        const u8* head = memory.GetPhysicalPointer(addr);
        cmd_list.Reset(addr, head, size);
        break;
    }

    case PICA_REG_INDEX(pipeline.trigger_draw):
    case PICA_REG_INDEX(pipeline.trigger_draw_indexed): {
        const bool is_indexed = (id == PICA_REG_INDEX(pipeline.trigger_draw_indexed));
        if (IsV115DA7Z71PicaTriggerSilentDrawArraysEnabled()) {
            if (IsV115DA7Z78PicaTriggerSinglePreDrawArraysMarkerEnabled()) {
                static std::atomic<u64> a7z78_trigger_pre_drawarrays_counter{0};
                if (++a7z78_trigger_pre_drawarrays_counter == 1) {
                    LOG_WARNING(HW_GPU,
                                "TRACE_DRAW_PICA strict_compat v115d_a7z78 trigger_pre_drawarrays_once");
                }
            }
            DrawArrays(is_indexed);
            return;
        }
        if (IsV115DA7Z64PicaDrawArraysUltraQuietBoundaryEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z64 trigger_case_enter id=0x{:03X} indexed={} num_vertices={} vertex_offset={}",
                        id, static_cast<u32>(is_indexed), regs.internal.pipeline.num_vertices,
                        regs.internal.pipeline.vertex_offset);
        }
        if (IsV115DA7Z65PicaEarlyDirectNoPrebackendLogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z65 trigger_case_enter id=0x{:03X} indexed={} num_vertices={} vertex_offset={}",
                        id, static_cast<u32>(is_indexed), regs.internal.pipeline.num_vertices,
                        regs.internal.pipeline.vertex_offset);
        }
        if (IsV115DA7Z70PicaTriggerDirectDrawArraysEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z70 trigger_direct_before_drawarrays_call");
            DrawArrays(is_indexed);
            return;
        }
        if (IsPicaHotpathTraceEnabled()) {
            LOG_DEBUG(HW_GPU,
                      "PicaCore::WriteInternalReg trigger_draw id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={}",
                      id, is_indexed, regs.internal.pipeline.num_vertices,
                      regs.internal.pipeline.vertex_offset,
                      static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                      static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
        }
        // v115-D-MUX rollback: v114-C8 proved the build is active but the console log can die before the
        // normal pre_draw texture dump and before DrawArrays. Emit a very early trigger gate
        // marker before LogPicaTextureState(pre_draw), so the next log tells us if the stop is
        // inside the texture-state trace or before the PICA draw call itself.
        V114C6PicaGateFileTraceRaw("v115d_mux trigger_draw_pre_predraw_trace");
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_id", id);
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_indexed", static_cast<u32>(is_indexed));
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_num_vertices", regs.internal.pipeline.num_vertices);
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_topology", static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_color_addr", regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_pre_predraw_depth_addr", regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        V115DA7XPicaTraceRaw("v115d_a7x pica_trigger_pre_predraw_trace");
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_id", id);
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_indexed", static_cast<u32>(is_indexed));
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_num_vertices", regs.internal.pipeline.num_vertices);
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_topology", static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_color_addr", regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
        V115DA7XPicaTraceU32("v115d_a7x trigger_draw_depth_addr", regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());

        if (IsStrictCompatEnabled()) {
            static std::atomic<u64> trigger_pre_predraw_console_counter{0};
            const u64 trigger_pre_predraw_console_index = ++trigger_pre_predraw_console_counter;
            if (trigger_pre_predraw_console_index <= 64 ||
                trigger_pre_predraw_console_index == 128 ||
                trigger_pre_predraw_console_index == 256 ||
                trigger_pre_predraw_console_index == 512) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_mux trigger_draw_pre_predraw_trace console_index={} id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                            trigger_pre_predraw_console_index, id, is_indexed,
                            regs.internal.pipeline.num_vertices, regs.internal.pipeline.vertex_offset,
                            static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                            static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                            regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                            regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
        }

        if (IsPicaDrawTraceEnabled()) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA trigger_draw id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                     id, is_indexed, regs.internal.pipeline.num_vertices,
                     regs.internal.pipeline.vertex_offset,
                     static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                     static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                     regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                     regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            LogPicaTextureState(regs.internal, "pre_draw");
        }
        V114C6PicaGateFileTraceRaw("v115d_mux trigger_draw_before_drawarrays");
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_id", id);
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_indexed", static_cast<u32>(is_indexed));
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_num_vertices", regs.internal.pipeline.num_vertices);
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_topology", static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()));
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_color_addr", regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress());
        V114C6PicaGateFileTraceU32("v115d_mux trigger_draw_depth_addr", regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());

        if (IsStrictCompatEnabled()) {
            static std::atomic<u64> trigger_console_counter{0};
            const u64 trigger_console_index = ++trigger_console_counter;
            if (trigger_console_index <= 32 || trigger_console_index == 64 ||
                trigger_console_index == 128 || trigger_console_index == 256 ||
                trigger_console_index == 512) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_mux trigger_draw_before_drawarrays console_index={} id=0x{:03X} indexed={} num_vertices={} vertex_offset={} topology={} use_gs={} color_addr=0x{:08X} depth_addr=0x{:08X}",
                            trigger_console_index, id, is_indexed,
                            regs.internal.pipeline.num_vertices, regs.internal.pipeline.vertex_offset,
                            static_cast<u32>(regs.internal.pipeline.triangle_topology.Value()),
                            static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                            regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                            regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            }
        }

        V115DA7XPicaTraceRaw("v115d_a7x pica_trigger_before_drawarrays");
        if (IsV115DA7Z64PicaDrawArraysUltraQuietBoundaryEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z64 trigger_before_drawarrays_call id=0x{:03X}",
                        id);
        }
        if (IsV115DA7Z65PicaEarlyDirectNoPrebackendLogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z65 trigger_before_drawarrays_call id=0x{:03X}",
                        id);
        }
        if (IsV115DA7Z22TriggerDrawArraysCallBoundaryProbeEnabled()) {
            V114C6PicaGateFileTraceRaw("v115d_a7z22 trigger_drawarrays_call_boundary_begin");
            V114C6PicaGateFileTraceU32("v115d_a7z22 trigger_draw_id", id);
            V114C6PicaGateFileTraceU32("v115d_a7z22 trigger_draw_indexed", static_cast<u32>(is_indexed));
            V114C6PicaGateFileTraceU32("v115d_a7z22 trigger_draw_num_vertices", regs.internal.pipeline.num_vertices);
            DrawArrays(is_indexed);
            V114C6PicaGateFileTraceRaw("v115d_a7z22 trigger_drawarrays_call_boundary_after_call");
            V114C6PicaGateFileTraceRaw("v115d_a7z22 trigger_drawarrays_call_boundary_return");
            return;
        }
        DrawArrays(is_indexed);
        V115DA7XPicaTraceRaw("v115d_a7x pica_trigger_after_drawarrays");
        V114C6PicaGateFileTraceRaw("v115d_mux trigger_draw_after_drawarrays");

        if (IsStrictCompatEnabled()) {
            static std::atomic<u64> trigger_after_console_counter{0};
            const u64 trigger_after_console_index = ++trigger_after_console_counter;
            if (trigger_after_console_index <= 32 || trigger_after_console_index == 64 ||
                trigger_after_console_index == 128 || trigger_after_console_index == 256 ||
                trigger_after_console_index == 512) {
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_mux trigger_draw_after_drawarrays console_index={} id=0x{:03X}",
                            trigger_after_console_index, id);
            }
        }
        break;
    }

    case PICA_REG_INDEX(gs.bool_uniforms):
        gs_setup.WriteUniformBoolReg(regs.internal.gs.bool_uniforms.Value());
        break;

    case PICA_REG_INDEX(gs.int_uniforms[0]):
    case PICA_REG_INDEX(gs.int_uniforms[1]):
    case PICA_REG_INDEX(gs.int_uniforms[2]):
    case PICA_REG_INDEX(gs.int_uniforms[3]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(gs.int_uniforms[0]));
        gs_setup.WriteUniformIntReg(index, regs.internal.gs.GetIntUniform(index));
        break;
    }

    case PICA_REG_INDEX(gs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(gs.uniform_setup.set_value[7]):
        gs_setup.WriteUniformFloatReg(regs.internal.gs, value);
        break;

    case PICA_REG_INDEX(gs.program.set_word[0]):
    case PICA_REG_INDEX(gs.program.set_word[1]):
    case PICA_REG_INDEX(gs.program.set_word[2]):
    case PICA_REG_INDEX(gs.program.set_word[3]):
    case PICA_REG_INDEX(gs.program.set_word[4]):
    case PICA_REG_INDEX(gs.program.set_word[5]):
    case PICA_REG_INDEX(gs.program.set_word[6]):
    case PICA_REG_INDEX(gs.program.set_word[7]): {
        u32& offset = regs.internal.gs.program.offset;
        if (offset >= 4096) {
            LOG_ERROR(HW_GPU, "Invalid GS program offset {}", offset);
        } else {
            gs_setup.program_code[offset] = value;
            gs_setup.MarkProgramCodeDirty();
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(gs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.gs.swizzle_patterns.offset;
        if (offset >= gs_setup.swizzle_data.size()) {
            LOG_ERROR(HW_GPU, "Invalid GS swizzle pattern offset {}", offset);
        } else {
            gs_setup.swizzle_data[offset] = value;
            gs_setup.MarkSwizzleDataDirty();
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.output_mask):
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            regs.internal.gs.output_mask.Assign(value);
        }
        break;

    case PICA_REG_INDEX(vs.bool_uniforms):
        vs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformBoolReg(regs.internal.vs.bool_uniforms.Value());
        }
        break;

    case PICA_REG_INDEX(vs.int_uniforms[0]):
    case PICA_REG_INDEX(vs.int_uniforms[1]):
    case PICA_REG_INDEX(vs.int_uniforms[2]):
    case PICA_REG_INDEX(vs.int_uniforms[3]): {
        const u32 index = static_cast<u32>(id - PICA_REG_INDEX(vs.int_uniforms[0]));
        vs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No) {
            gs_setup.WriteUniformIntReg(index, regs.internal.vs.GetIntUniform(index));
        }
        break;
    }

    case PICA_REG_INDEX(vs.uniform_setup.set_value[0]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[1]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[2]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[3]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[4]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[5]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[6]):
    case PICA_REG_INDEX(vs.uniform_setup.set_value[7]): {
        const auto index = vs_setup.WriteUniformFloatReg(regs.internal.vs, value);
        if (!regs.internal.pipeline.gs_unit_exclusive_configuration &&
            regs.internal.pipeline.use_gs == PipelineRegs::UseGS::No && index) {
            gs_setup.uniforms.f[index.value()] = vs_setup.uniforms.f[index.value()];
        }
        break;
    }

    case PICA_REG_INDEX(vs.program.set_word[0]):
    case PICA_REG_INDEX(vs.program.set_word[1]):
    case PICA_REG_INDEX(vs.program.set_word[2]):
    case PICA_REG_INDEX(vs.program.set_word[3]):
    case PICA_REG_INDEX(vs.program.set_word[4]):
    case PICA_REG_INDEX(vs.program.set_word[5]):
    case PICA_REG_INDEX(vs.program.set_word[6]):
    case PICA_REG_INDEX(vs.program.set_word[7]): {
        u32& offset = regs.internal.vs.program.offset;
        if (offset >= 512) {
            LOG_ERROR(HW_GPU, "Invalid VS program offset {}", offset);
        } else {
            vs_setup.program_code[offset] = value;
            vs_setup.MarkProgramCodeDirty();
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration) {
                gs_setup.program_code[offset] = value;
                gs_setup.MarkProgramCodeDirty();
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[0]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[1]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[2]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[3]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[4]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[5]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[6]):
    case PICA_REG_INDEX(vs.swizzle_patterns.set_word[7]): {
        u32& offset = regs.internal.vs.swizzle_patterns.offset;
        if (offset >= vs_setup.swizzle_data.size()) {
            LOG_ERROR(HW_GPU, "Invalid VS swizzle pattern offset {}", offset);
        } else {
            vs_setup.swizzle_data[offset] = value;
            vs_setup.MarkSwizzleDataDirty();
            if (!regs.internal.pipeline.gs_unit_exclusive_configuration) {
                gs_setup.swizzle_data[offset] = value;
                gs_setup.MarkSwizzleDataDirty();
            }
            offset++;
        }
        break;
    }

    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        auto& lut_config = regs.internal.lighting.lut_config;
        ASSERT_MSG(lut_config.index < 256, "lut_config.index exceeded maximum value of 255!");
        lighting.luts[lut_config.type][lut_config.index].raw = value;
        lut_config.index.Assign(lut_config.index + 1);
        break;
    }

    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        fog.lut[regs.internal.texturing.fog_lut_offset % 128].raw = value;
        regs.internal.texturing.fog_lut_offset.Assign(regs.internal.texturing.fog_lut_offset + 1);
        break;

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]): {
        auto& index = regs.internal.texturing.proctex_lut_config.index;

        switch (regs.internal.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            proctex.noise_table[index % proctex.noise_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            proctex.color_map_table[index % proctex.color_map_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            proctex.alpha_map_table[index % proctex.alpha_map_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            proctex.color_table[index % proctex.color_table.size()].raw = value;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            proctex.color_diff_table[index % proctex.color_diff_table.size()].raw = value;
            break;
        }
        index.Assign(index + 1);
        break;
    }

    default:
        break;
    }

    rasterizer->NotifyPicaRegisterChanged(id);
}

void PicaCore::SubmitImmediate(u32 value) {
    if (!immediate.queue.Push(value)) {
        return;
    }

    constexpr std::size_t ImmediateModeIndex = 0xF;

    auto& setup = regs.internal.pipeline.vs_default_attributes_setup;
    if (setup.index > ImmediateModeIndex) {
        LOG_ERROR(HW_GPU, "Invalid VS default attribute index {}", setup.index);
        return;
    }

    const auto attribute = immediate.queue.Get();
    if (setup.index < ImmediateModeIndex) {
        input_default_attributes[setup.index] = attribute;
        setup.index++;
        return;
    }

    immediate.input_vertex[immediate.current_attribute] = attribute;
    if (immediate.current_attribute < regs.internal.pipeline.max_input_attrib_index) {
        immediate.current_attribute++;
        return;
    }

    DrawImmediate();
}

void PicaCore::DrawImmediate() {
    BORKED3DS_PROFILE("PicaCore", "Draw Immediate");

    if (IsPicaDrawTraceEnabled()) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA immediate_draw topology={} current_attr={} max_attr={}",
                 static_cast<u32>(primitive_assembler.GetTopology()), immediate.current_attribute,
                 regs.internal.pipeline.max_input_attrib_index.Value());
    }

    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                               std::addressof(immediate.input_vertex));
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    ShaderUnit shader_unit;
    AttributeBuffer output{};

    shader_unit.LoadInput(regs.internal.vs, immediate.input_vertex);
    shader_engine->Run(vs_setup, shader_unit);
    shader_unit.WriteOutput(regs.internal.vs, output);

    if (immediate.reset_geometry_pipeline) {
        geometry_pipeline.Reconfigure();
        immediate.reset_geometry_pipeline = false;
    }

    ASSERT(!geometry_pipeline.NeedIndexInput());
    geometry_pipeline.Setup(shader_engine.get());
    geometry_pipeline.SubmitVertex(output);

    if (IsPicaDrawTraceEnabled()) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA immediate -> rasterizer->DrawTriangles()");
    }
    rasterizer->DrawTriangles();
    immediate.current_attribute = 0;
}

void PicaCore::DrawArrays(bool is_indexed) {
    BORKED3DS_PROFILE("PicaCore", "Draw Arrays");
    const u64 draw_index = ++g_pica_draw_counter;
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();
    const bool trace_draw = IsPicaDrawTraceEnabled();
    const bool a7z64_ultra_quiet = IsV115DA7Z64PicaDrawArraysUltraQuietBoundaryEnabled();
    const bool a7z65_no_prebackend = IsV115DA7Z65PicaEarlyDirectNoPrebackendLogEnabled();
    const bool a7z72_silent_early_backend = IsV115DA7Z72PicaDrawArraysSilentEarlyBackendEnabled();
    const bool a7z77_single_entry_marker = IsV115DA7Z77PicaDrawArraysSingleEntryMarkerEnabled();

    // A7Z89: ultra-early probe — first thing in DrawArrays, before any gate or decision.
    // If this never appears in HW mode, the crash is before DrawArrays is reached.
    V115DA7Z89EmitDrawArraysUltraEarly(
        draw_index, is_indexed,
        regs.internal.pipeline.num_vertices,
        static_cast<u32>(primitive_assembler.GetTopology()),
        primitive_assembler.IsEmpty(),
        Settings::values.use_hw_shader.GetValue());

    if (a7z77_single_entry_marker) {
        static std::atomic<u64> a7z77_drawarrays_entry_counter{0};
        if (++a7z77_drawarrays_entry_counter == 1) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z77 drawarrays_entry_once");
        }
    }

    if (a7z72_silent_early_backend && IsSafePicaHwDrawAllowed() && IsSafePicaHwEnterAllowed() &&
        IsDirectSafePicaHwHandoffEnabled() && !IsPicaAccelAllowed() && !IsPicaAccelForcedOff()) {
        const bool v115d_a7z72_hw_shader = Settings::values.use_hw_shader.GetValue();
        const bool v115d_a7z72_primitive_empty = primitive_assembler.IsEmpty();
        const u32 v115d_a7z72_max_vertices = GetSafePicaHwMaxVertices();
        const bool v115d_a7z72_textures_disabled = ArePrimaryTexturesDisabled(regs.internal);
        const bool v115d_a7z72_accelerate_shape =
            v115d_a7z72_hw_shader && v115d_a7z72_primitive_empty &&
            regs.internal.pipeline.use_gs != PipelineRegs::UseGS::Yes &&
            regs.internal.pipeline.num_vertices > 0 &&
            regs.internal.pipeline.num_vertices <= v115d_a7z72_max_vertices &&
            (primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::Shader ||
             primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::List
                 ? ((regs.internal.pipeline.num_vertices % 3) == 0)
                 : true);
        const bool v115d_a7z72_safe_candidate =
            v115d_a7z72_accelerate_shape && v115d_a7z72_textures_disabled;
        const u32 v115d_a7z72_budget = GetSafePicaHwDrawBudget();

        if (v115d_a7z72_safe_candidate && v115d_a7z72_budget != 0) {
            const u64 v115d_a7z72_hw_index = ++g_pica_safe_hw_draw_counter;
            if (v115d_a7z72_hw_index <= v115d_a7z72_budget) {
                if (IsSafePicaHwDryRunEnabled()) {
                    return;
                }
                if (IsV115DA7Z76PicaSingleBackendCallMarkerEnabled()) {
                    LOG_WARNING(HW_GPU,
                                "TRACE_DRAW_PICA strict_compat v115d_a7z76 before_accelerate_draw_batch_call");
                }
                V115DA7Z85EmitDrawArraysBackendOnce();
                (void)rasterizer->AccelerateDrawBatch(is_indexed);
                V115DA7Z86EmitDrawArraysAfterBackendOnce();
                return;
            }
        }
    }

    if (a7z64_ultra_quiet) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z64 drawarrays_enter draw_index={} indexed={} num_vertices={} vertex_offset={}",
                    draw_index, static_cast<u32>(is_indexed), regs.internal.pipeline.num_vertices,
                    regs.internal.pipeline.vertex_offset);
    }
    if (a7z65_no_prebackend) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_a7z65 drawarrays_enter draw_index={} indexed={} num_vertices={} vertex_offset={}",
                    draw_index, static_cast<u32>(is_indexed), regs.internal.pipeline.num_vertices,
                    regs.internal.pipeline.vertex_offset);
    }

    V114C6PicaGateFileTraceRaw("v115d_mux drawarrays_enter");
    V114C6PicaGateFileTraceU64("v115d_mux draw_index", draw_index);
    V114C6PicaGateFileTraceU32("v115d_mux draw_indexed", static_cast<u32>(is_indexed));
    V114C6PicaGateFileTraceU32("v115d_mux draw_num_vertices", regs.internal.pipeline.num_vertices);
    V114C6PicaGateFileTraceU32("v115d_mux draw_topology", static_cast<u32>(primitive_assembler.GetTopology()));
    V114C6PicaGateFileTraceU32("v115d_mux draw_use_gs", static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
    V114C6PicaGateFileTraceU32("v115d_mux draw_primitive_empty", static_cast<u32>(primitive_assembler.IsEmpty()));
    V115DA7XPicaTraceRaw("v115d_a7x drawarrays_enter");
    V115DA7XPicaTraceU64("v115d_a7x draw_index", draw_index);
    V115DA7XPicaTraceU32("v115d_a7x draw_indexed", static_cast<u32>(is_indexed));
    V115DA7XPicaTraceU32("v115d_a7x draw_num_vertices", regs.internal.pipeline.num_vertices);
    V115DA7XPicaTraceU32("v115d_a7x draw_topology", static_cast<u32>(primitive_assembler.GetTopology()));
    V115DA7XPicaTraceU32("v115d_a7x draw_primitive_empty", static_cast<u32>(primitive_assembler.IsEmpty()));

    if (IsStrictCompatEnabled() && !a7z64_ultra_quiet && !a7z65_no_prebackend) {
        static std::atomic<u64> drawarrays_console_counter{0};
        const u64 drawarrays_console_index = ++drawarrays_console_counter;
        if (drawarrays_console_index <= 32 || drawarrays_console_index == 64 ||
            drawarrays_console_index == 128 || drawarrays_console_index == 256 ||
            drawarrays_console_index == 512) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_mux drawarrays_enter console_index={} draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} topology={} use_gs={} primitive_empty={}",
                        drawarrays_console_index, draw_index, is_indexed,
                        regs.internal.pipeline.num_vertices, regs.internal.pipeline.vertex_offset,
                        static_cast<u32>(Settings::values.use_hw_shader.GetValue()),
                        static_cast<u32>(primitive_assembler.GetTopology()),
                        static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                        static_cast<u32>(primitive_assembler.IsEmpty()));
        }
    }

    // v115-D-MUX rollback: the v114-C7 log reached trigger_draw_before_drawarrays, drawarrays_enter,
    // and the multiplex marker, then stopped before early_predecision. Move the tiny,
    // untextured predecision path above the generic trace_draw texture-state logging so
    // the next log tells us whether the crash was in LogPicaTextureState() or inside
    // AccelerateDrawBatch stage 1..7.
    if (IsStrictCompatEnabled() && IsSafePicaHwDrawAllowed() && IsSafePicaHwEnterAllowed() &&
        IsDirectSafePicaHwHandoffEnabled() && !IsPicaAccelAllowed() && !IsPicaAccelForcedOff()) {
        if (a7z64_ultra_quiet) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z64 early_direct_gate_enter draw_index={}",
                        draw_index);
        }
        if (a7z65_no_prebackend) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z65 early_direct_gate_enter draw_index={}",
                        draw_index);
        }
        V114C6PicaGateFileTraceRaw("v115d_mux early_direct_accel_begin");
        V114C6PicaGateFileTraceRaw("v115d_mux early_predecision_begin");
        V115DA7XPicaTraceRaw("v115d_a7x early_direct_accel_begin");
        V115DA7XPicaTraceRaw("v115d_a7x early_predecision_begin");
        const bool v115d_mux_early_hw_shader = Settings::values.use_hw_shader.GetValue();
        const bool v115d_mux_early_primitive_empty = primitive_assembler.IsEmpty();
        const u32 v115d_mux_early_topology = static_cast<u32>(primitive_assembler.GetTopology());
        const u32 v115d_mux_early_use_gs = static_cast<u32>(regs.internal.pipeline.use_gs.Value());
        const u32 v115d_mux_early_textures_disabled = ArePrimaryTexturesDisabled(regs.internal) ? 1u : 0u;
        const u32 v115d_mux_early_max_vertices = GetSafePicaHwMaxVertices();
        const bool v115d_mux_early_accelerate_shape =
            v115d_mux_early_hw_shader && v115d_mux_early_primitive_empty &&
            regs.internal.pipeline.use_gs != PipelineRegs::UseGS::Yes &&
            regs.internal.pipeline.num_vertices > 0 &&
            regs.internal.pipeline.num_vertices <= v115d_mux_early_max_vertices &&
            (primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::Shader ||
             primitive_assembler.GetTopology() == PipelineRegs::TriangleTopology::List
                 ? ((regs.internal.pipeline.num_vertices % 3) == 0)
                 : true);
        const bool v115d_mux_early_safe_candidate =
            v115d_mux_early_accelerate_shape && v115d_mux_early_textures_disabled != 0;

        if (a7z64_ultra_quiet && v115d_mux_early_safe_candidate) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z64 early_direct_candidate_true draw_index={} num_vertices={}",
                        draw_index, regs.internal.pipeline.num_vertices);
        }
        if (a7z65_no_prebackend && v115d_mux_early_safe_candidate) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_a7z65 early_direct_candidate_true draw_index={} num_vertices={}",
                        draw_index, regs.internal.pipeline.num_vertices);
        }

        V114C6PicaGateFileTraceU32("v115d_mux early_hw_shader", static_cast<u32>(v115d_mux_early_hw_shader));
        V114C6PicaGateFileTraceU32("v115d_mux early_primitive_empty", static_cast<u32>(v115d_mux_early_primitive_empty));
        V114C6PicaGateFileTraceU32("v115d_mux early_topology", v115d_mux_early_topology);
        V114C6PicaGateFileTraceU32("v115d_mux early_use_gs", v115d_mux_early_use_gs);
        V114C6PicaGateFileTraceU32("v115d_mux early_textures_disabled", v115d_mux_early_textures_disabled);
        V114C6PicaGateFileTraceU32("v115d_mux early_safe_candidate", static_cast<u32>(v115d_mux_early_safe_candidate));
        V115DA7XPicaTraceU32("v115d_a7x early_hw_shader", static_cast<u32>(v115d_mux_early_hw_shader));
        V115DA7XPicaTraceU32("v115d_a7x early_textures_disabled", v115d_mux_early_textures_disabled);
        V115DA7XPicaTraceU32("v115d_a7x early_safe_candidate", static_cast<u32>(v115d_mux_early_safe_candidate));

        static std::atomic<u64> v115d_mux_predecision_console_counter{0};
        const u64 v115d_mux_predecision_console_index = ++v115d_mux_predecision_console_counter;
        if (v115d_mux_predecision_console_index <= 16) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_mux early_direct_accel console_index={} draw_index={} indexed={} num_vertices={} hw_shader={} primitive_empty={} textures_disabled={} topology={} use_gs={} candidate={} budget={} max_vertices={}",
                        v115d_mux_predecision_console_index, draw_index, is_indexed,
                        regs.internal.pipeline.num_vertices,
                        static_cast<u32>(v115d_mux_early_hw_shader),
                        static_cast<u32>(v115d_mux_early_primitive_empty),
                        v115d_mux_early_textures_disabled, v115d_mux_early_topology,
                        v115d_mux_early_use_gs, static_cast<u32>(v115d_mux_early_safe_candidate),
                        GetSafePicaHwDrawBudget(), v115d_mux_early_max_vertices);
        }

        if (v115d_mux_early_safe_candidate) {
            const u64 v115d_mux_early_hw_index = ++g_pica_safe_hw_draw_counter;
            const bool v115d_mux_early_budget_ok =
                GetSafePicaHwDrawBudget() != 0 &&
                v115d_mux_early_hw_index <= GetSafePicaHwDrawBudget();

            // v115-D-MUX rollback: the v114-C9 log reached early_safe_hw_decision but never reached
            // early_before_accelerate_draw_batch or TRACE_ACCEL_STAGE. Avoid all formatted
            // console logging between the budget decision and the backend call; jump straight
            // into AccelerateDrawBatch() so the next log proves whether the backend is reached.
            if (v115d_mux_early_budget_ok) {
                if (a7z65_no_prebackend) {
                    (void)rasterizer->AccelerateDrawBatch(is_indexed);
                    return;
                }
                V114C6PicaGateFileTraceU64("v115d_mux early_direct_safe_hw_index", v115d_mux_early_hw_index);
                V114C6PicaGateFileTraceRaw("v115d_mux early_direct_before_accelerate_draw_batch");
                V115DA7XPicaTraceU64("v115d_a7x early_direct_safe_hw_index", v115d_mux_early_hw_index);
                V115DA7XPicaTraceRaw("v115d_a7x early_direct_before_accelerate_draw_batch");

                if (IsSafePicaHwDryRunEnabled()) {
                    V114C6PicaGateFileTraceRaw("v115d_mux early_direct_dry_run_consumed");
                    return;
                }

                if (IsV115DA7Z19PicaCallBoundaryProbeEnabled()) {
                    V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_probe_begin");
                    V114C6PicaGateFileTraceU64("v115d_a7z19 draw_index", draw_index);
                    V114C6PicaGateFileTraceU32("v115d_a7z19 indexed", static_cast<u32>(is_indexed));
                    V114C6PicaGateFileTraceU32("v115d_a7z19 num_vertices", regs.internal.pipeline.num_vertices);
                    V114C6PicaGateFileTraceU32("v115d_a7z19 vertex_offset", regs.internal.pipeline.vertex_offset);
                    V114C6PicaGateFileTraceU32("v115d_a7z19 skip_backend_call",
                                               static_cast<u32>(IsV115DA7Z19PicaSkipBackendCallEnabled()));

                    if (IsV115DA7Z19PicaSkipBackendCallEnabled()) {
                        V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_skip_backend_return_false");
                        V114C6PicaGateFileTraceU32("v115d_a7z19 accelerate_draw_batch_result", 0);
                        V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_return_controlled_false");
                        return;
                    }

                    V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_before_call");
                    if (IsV115DA7Z21PicaAfterBackendUltraCleanReturnEnabled()) {
                        V114C6PicaGateFileTraceRaw(
                            "v115d_a7z21 pica_after_backend_ultra_clean_begin");
                        (void)rasterizer->AccelerateDrawBatch(is_indexed);
                        return;
                    }
                    if (IsV115DA7Z20PicaAfterBackendControlledReturnEnabled()) {
                        V114C6PicaGateFileTraceRaw(
                            "v115d_a7z20 pica_after_backend_controlled_begin");
                        const bool v115d_a7z20_accelerated =
                            rasterizer->AccelerateDrawBatch(is_indexed);
                        V114C6PicaGateFileTraceRaw(
                            "v115d_a7z20 pica_after_backend_controlled_after_call");
                        V114C6PicaGateFileTraceU32(
                            "v115d_a7z20 pica_after_backend_controlled_result",
                            static_cast<u32>(v115d_a7z20_accelerated));
                        V114C6PicaGateFileTraceRaw(
                            "v115d_a7z20 pica_after_backend_controlled_return");
                        return;
                    }

                    const bool v115d_a7z19_accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
                    V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_after_call");
                    V114C6PicaGateFileTraceU32("v115d_a7z19 accelerate_draw_batch_result",
                                               static_cast<u32>(v115d_a7z19_accelerated));
                    V114C6PicaGateFileTraceRaw("v115d_a7z19 pica_call_boundary_return_controlled");
                    return;
                }

                if (IsEnvEnabled("BORKED3DS_V3DV_PROBE_FIRST_VKCMD_DRAW_ZEROCOUNT_ONLY")) {
                    // v115-D-MUX: call the backend with the original PICA indexed command, but the
                    // Vulkan backend records the final draw with index/vertex count forced to 0.
                    // If DrawArrays returns, trigger_draw_after_drawarrays proves the return path.
                    (void)rasterizer->AccelerateDrawBatch(is_indexed);
                    return;
                }

                V115DA7Z85EmitDrawArraysBackendOnce();
                const bool v115d_mux_early_accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
                V115DA7Z86EmitDrawArraysAfterBackendOnce();
                V114C6PicaGateFileTraceRaw("v115d_mux early_direct_after_accelerate_draw_batch");
                V114C6PicaGateFileTraceU32("v115d_mux early_direct_accelerate_draw_batch_result",
                                           static_cast<u32>(v115d_mux_early_accelerated));
                V115DA7XPicaTraceRaw("v115d_a7x early_direct_after_accelerate_draw_batch");
                V115DA7XPicaTraceU32("v115d_a7x early_direct_accelerate_draw_batch_result",
                                     static_cast<u32>(v115d_mux_early_accelerated));
                LOG_WARNING(HW_GPU,
                            "TRACE_DRAW_PICA strict_compat v115d_mux early_direct_after_accelerate_draw_batch draw_index={} result={}",
                            draw_index, static_cast<u32>(v115d_mux_early_accelerated));

                // vDIRA (Direction A, v119): this early-direct mux used to `return` UNCONDITIONALLY,
                // discarding the AccelerateDrawBatch() result -- a declined draw was simply DROPPED,
                // never reaching the software vertex path. That made the backend's per-draw software
                // fallback (hybrid VSs hit by the V3DV upper-bank miscompile, e.g. the Sonic Lost
                // World glyph VS) a silent no-op in the safe-HW-handoff configuration. When the
                // fallback is enabled and the backend declines, run the software path exactly as the
                // tail of DrawArrays does: LoadVertices() (software VS, carries the v116-B constant
                // attribute fix) followed by rasterizer->DrawTriangles(). Without the flag the
                // historical behaviour (unconditional return) is preserved bit-for-bit.
                if (!v115d_mux_early_accelerated &&
                    IsEnvEnabled("BORKED3DS_V3DV_DIRA_SW_FALLBACK")) {
                    static std::atomic<u64> dira_pica_sw_counter{0};
                    const u64 dira_pica_sw_count = ++dira_pica_sw_counter;
                    static const bool dira_trace =
                        IsEnvEnabled("BORKED3DS_V3DV_TRACE_DIRA");
                    if (dira_trace &&
                        (dira_pica_sw_count <= 4 || (dira_pica_sw_count % 512u) == 0u)) {
                        LOG_INFO(HW_GPU,
                                 "vDIRA pica software_path_taken count={} draw_index={}"
                                 " indexed={} num_vertices={}",
                                 dira_pica_sw_count, draw_index, is_indexed,
                                 regs.internal.pipeline.num_vertices);
                    }
                    LoadVertices(is_indexed);
                    rasterizer->DrawTriangles();
                    return;
                }
                return;
            }

            V114C6PicaGateFileTraceU64("v115d_mux early_safe_hw_index", v115d_mux_early_hw_index);
            V114C6PicaGateFileTraceU32("v115d_mux early_budget_ok", static_cast<u32>(v115d_mux_early_budget_ok));
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_mux early_safe_hw_decision budget_rejected draw_index={} indexed={} num_vertices={} safe_hw_index={} budget_ok={}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        v115d_mux_early_hw_index, static_cast<u32>(v115d_mux_early_budget_ok));
        }
    }


    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::DrawArrays begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={}",
                  draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                  regs.internal.pipeline.vertex_offset, Settings::values.use_hw_shader.GetValue(),
                  Settings::values.skip_slow_draw.GetValue(),
                  static_cast<u32>(primitive_assembler.GetTopology()),
                  static_cast<u32>(regs.internal.pipeline.use_gs.Value()));
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA begin draw_index={} indexed={} num_vertices={} vertex_offset={} use_hw_shader={} skip_slow_draw={} topology={} use_gs={} primitive_empty={}",
                 draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                 regs.internal.pipeline.vertex_offset, Settings::values.use_hw_shader.GetValue(),
                 Settings::values.skip_slow_draw.GetValue(),
                 static_cast<u32>(primitive_assembler.GetTopology()),
                 static_cast<u32>(regs.internal.pipeline.use_gs.Value()), primitive_assembler.IsEmpty());

        if (draw_index == 1) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_mux pica_core_shader_multiplex_quietdisplay_marker stage_stop_after={} force_stage_trace={} entry_only_probe={} enter_safe_hw={} safe_budget={} safe_max_vertices={}",
                        GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE")),
                        static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE")),
                        static_cast<u32>(IsSafePicaHwEnterAllowed()),
                        GetSafePicaHwDrawBudget(), GetSafePicaHwMaxVertices());
        }

        LogPicaTextureState(regs.internal, "drawarrays_begin");
    }

    if (IsEnvEnabled("BORKED3DS_V3DV_BYPASS_FIRST_DRAW") && draw_index == 1) {
        LOG_WARNING(HW_GPU,
                    "V3DV test bypass: skipping first PICA draw because BORKED3DS_V3DV_BYPASS_FIRST_DRAW=1");
        return;
    }

    if (debug_context) {
        debug_context->OnEvent(DebugContext::Event::IncomingPrimitiveBatch, nullptr);
        SCOPE_EXIT(
            { debug_context->OnEvent(DebugContext::Event::FinishedPrimitiveBatch, nullptr); });
    }

    bool accelerate_draw = [this] {
        if (regs.internal.pipeline.use_gs == PipelineRegs::UseGS::Yes) {
            return false;
        }

        bool can_accelerate = Settings::values.use_hw_shader && primitive_assembler.IsEmpty();
        const auto topology = primitive_assembler.GetTopology();
        if (topology == PipelineRegs::TriangleTopology::Shader ||
            topology == PipelineRegs::TriangleTopology::List) {
            can_accelerate = can_accelerate && (regs.internal.pipeline.num_vertices % 3) == 0;
        }
        return can_accelerate;
    }();

    const u32 textures_disabled = ArePrimaryTexturesDisabled(regs.internal) ? 1u : 0u;
    const u32 safe_hw_max_vertices = GetSafePicaHwMaxVertices();
    const bool strict_safe_pica_hw_candidate =
        accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !IsPicaAccelForcedOff() && IsSafePicaHwDrawAllowed() &&
        Settings::values.use_hw_shader.GetValue() && primitive_assembler.IsEmpty() &&
        regs.internal.pipeline.use_gs != PipelineRegs::UseGS::Yes && textures_disabled != 0 &&
        regs.internal.pipeline.num_vertices > 0 &&
        regs.internal.pipeline.num_vertices <= safe_hw_max_vertices;

    u64 strict_safe_pica_hw_index = 0;
    const bool strict_safe_pica_hw_draw = [&] {
        if (!strict_safe_pica_hw_candidate) {
            return false;
        }
        strict_safe_pica_hw_index = ++g_pica_safe_hw_draw_counter;
        const u32 budget = GetSafePicaHwDrawBudget();
        return budget != 0 && strict_safe_pica_hw_index <= budget;
    }();

    const bool strict_safe_pica_hw_enter =
        strict_safe_pica_hw_draw && IsSafePicaHwEnterAllowed();
    const bool strict_safe_pica_hw_dry_run =
        strict_safe_pica_hw_draw && IsSafePicaHwDryRunEnabled();

    // v114: do not copy GVX64's final software-style behavior. Keep broad PICA acceleration
    // blocked. A tiny HW-shader candidate may enter AccelerateDrawBatch only as a grouped
    // stage-6 preflight. The backend may pass geometry-shader gate, topology, vertex analysis,
    // and SetupVertexArray(), but must return before shader setup, pipeline binding,
    // descriptors, or vkCmdDraw/vkCmdDrawIndexed.
    if (accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !strict_safe_pica_hw_draw) {
        if (trace_draw) {
            LOG_INFO(HW_GPU,
                     "TRACE_DRAW_PICA strict_compat v114 forcing software path before broad PICA acceleration draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} allow_pica_accel=0 safe_hw_candidate={} safe_hw_allowed={} safe_hw_enter={} safe_hw_dry_run={} safe_hw_index={} safe_hw_budget={}",
                     draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                     primitive_assembler.IsEmpty(), textures_disabled,
                     static_cast<u32>(primitive_assembler.GetTopology()),
                     static_cast<u32>(strict_safe_pica_hw_candidate),
                     static_cast<u32>(IsSafePicaHwDrawAllowed()),
                     static_cast<u32>(strict_safe_pica_hw_enter),
                     static_cast<u32>(strict_safe_pica_hw_dry_run), strict_safe_pica_hw_index,
                     GetSafePicaHwDrawBudget());
            LogPicaTextureState(regs.internal, "v114_force_software_before_broad_accel");
        }
        if (!g_logged_strict_accel_gate.exchange(true)) {
            LOG_WARNING(HW_GPU,
                        "Pi5/V3DV strict compatibility v114: broad PICA AccelerateDrawBatch remains disabled by default; tiny untextured HW candidates may enter AccelerateDrawBatch only with the v114 raw-enter-simple handoff probe enabled. Keep BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF=1, BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG=1, BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_POSTLOG=0, BORKED3DS_V3DV_ACCEL_SILENT_ENTRY_RETURN=0, BORKED3DS_V3DV_ACCEL_RAW_ENTER_RETURN=0, BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN=0, BORKED3DS_V3DV_ACCEL_ENTRY_ONLY_PROBE=0, BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER=1, BORKED3DS_V3DV_FORCE_ACCEL_STAGE_TRACE=1, BORKED3DS_V3DV_ENTER_SAFE_PICA_HW_DRAWS=1, BORKED3DS_V3DV_USE_TRIVIAL_VERTEX_SHADER_PROBE=0, BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_CONFIG_ONLY=0, BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_ONLY=0, and BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY=1 for normal v114 tests");
        }
        accelerate_draw = false;
    } else if (strict_safe_pica_hw_draw && trace_draw) {
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v114 safe micro PICA HW candidate hw_index={} budget={} draw_index={} indexed={} num_vertices={} primitive_empty={} textures_disabled={} topology={} use_hw_shader={} enter={} dry_run={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                    strict_safe_pica_hw_index, GetSafePicaHwDrawBudget(), draw_index, is_indexed,
                    regs.internal.pipeline.num_vertices, primitive_assembler.IsEmpty(),
                    textures_disabled, static_cast<u32>(primitive_assembler.GetTopology()),
                    static_cast<u32>(Settings::values.use_hw_shader.GetValue()),
                    static_cast<u32>(strict_safe_pica_hw_enter),
                    static_cast<u32>(strict_safe_pica_hw_dry_run),
                    regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                    regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        if (IsVerbosePicaMicroTextureTraceEnabled()) {
            LogPicaTextureState(regs.internal, "v114_safe_micro_hw_candidate");
        }
    }

    V114C6PicaGateFileTraceRaw("v115d_mux safe_hw_decision");
    V114C6PicaGateFileTraceU32("v115d_mux accelerate_draw", static_cast<u32>(accelerate_draw));
    V114C6PicaGateFileTraceU32("v115d_mux textures_disabled", textures_disabled);
    V114C6PicaGateFileTraceU32("v115d_mux safe_hw_candidate", static_cast<u32>(strict_safe_pica_hw_candidate));
    V114C6PicaGateFileTraceU32("v115d_mux safe_hw_draw", static_cast<u32>(strict_safe_pica_hw_draw));
    V114C6PicaGateFileTraceU32("v115d_mux safe_hw_enter", static_cast<u32>(strict_safe_pica_hw_enter));
    V114C6PicaGateFileTraceU32("v115d_mux safe_hw_dry_run", static_cast<u32>(strict_safe_pica_hw_dry_run));
    V114C6PicaGateFileTraceU64("v115d_mux safe_hw_index", strict_safe_pica_hw_index);

    if (IsStrictCompatEnabled()) {
        static std::atomic<u64> safe_decision_console_counter{0};
        const u64 safe_decision_console_index = ++safe_decision_console_counter;
        if (safe_decision_console_index <= 32 || safe_decision_console_index == 64 ||
            safe_decision_console_index == 128 || safe_decision_console_index == 256 ||
            safe_decision_console_index == 512) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v115d_mux safe_hw_decision console_index={} draw_index={} accelerate_draw={} textures_disabled={} safe_hw_candidate={} safe_hw_draw={} safe_hw_enter={} safe_hw_dry_run={} safe_hw_index={} safe_hw_budget={} safe_hw_max_vertices={}",
                        safe_decision_console_index, draw_index, static_cast<u32>(accelerate_draw),
                        textures_disabled, static_cast<u32>(strict_safe_pica_hw_candidate),
                        static_cast<u32>(strict_safe_pica_hw_draw),
                        static_cast<u32>(strict_safe_pica_hw_enter),
                        static_cast<u32>(strict_safe_pica_hw_dry_run), strict_safe_pica_hw_index,
                        GetSafePicaHwDrawBudget(), GetSafePicaHwMaxVertices());
        }
    }

    const bool strict_direct_safe_hw_handoff =
        accelerate_draw && strict_safe_pica_hw_draw && strict_safe_pica_hw_enter &&
        IsDirectSafePicaHwHandoffEnabled();

    if (strict_direct_safe_hw_handoff) {
        V114C6PicaGateFileTraceRaw("v115d_mux before_accelerate_draw_batch");
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_mux before_accelerate_draw_batch draw_index={} indexed={} num_vertices={} safe_hw_index={} stage_stop_after={} generate_guarded_probe={}",
                    draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                    strict_safe_pica_hw_index,
                    GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                    static_cast<u32>(IsEnvEnabled("BORKED3DS_V3DV_PROBE_PROGRAMMABLE_VS_GENERATE_GUARDED_ONLY")));
        // v114:
        // v110 proved raw_enter_noargs survives. Keep pica_core handoff logging suppressed
        // and let the backend emit TRACE_ACCEL_STAGE v114 raw_enter_noargs plus raw_enter_simple
        // before returning with BORKED3DS_V3DV_ACCEL_RAW_ENTER_SIMPLE_RETURN=0.
        if (trace_draw && !IsDirectSafePicaHwHandoffNoPrelogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 pre_call_direct_noargs AccelerateDrawBatch");
        }

        const bool accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
        V114C6PicaGateFileTraceRaw("v115d_mux after_accelerate_draw_batch");
        V114C6PicaGateFileTraceU32("v115d_mux accelerate_draw_batch_result", static_cast<u32>(accelerated));
        LOG_WARNING(HW_GPU,
                    "TRACE_DRAW_PICA strict_compat v115d_mux after_accelerate_draw_batch draw_index={} result={}",
                    draw_index, static_cast<u32>(accelerated));

        if (trace_draw && IsDirectSafePicaHwHandoffPostlogEnabled()) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 post_call_direct AccelerateDrawBatch returned {}",
                        static_cast<u32>(accelerated));
        }

        if (accelerated) {
            return;
        }

        // Do not call the backend twice if the direct diagnostic path returns false.
        accelerate_draw = false;
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays accelerate_draw={}", accelerate_draw);
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA accelerate_draw={} gs_mode={} topology={}",
                 accelerate_draw, static_cast<u32>(regs.internal.pipeline.use_gs.Value()),
                 static_cast<u32>(primitive_assembler.GetTopology()));
    }

    if (strict_safe_pica_hw_dry_run) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114 handoff dry-run consumed safe micro PICA HW candidate before AccelerateDrawBatch draw_index={} indexed={} num_vertices={} hw_index={} budget={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        strict_safe_pica_hw_index, GetSafePicaHwDrawBudget(),
                        regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
            if (IsVerbosePicaMicroTextureTraceEnabled()) {
                LogPicaTextureState(regs.internal, "v114_safe_micro_hw_dry_run_consumed");
            }
        }
        return;
    }

    // v114 last-chance guard: only explicit full diagnosis or explicitly executed tiny HW
    // probes may enter AccelerateDrawBatch in strict mode.
    if (accelerate_draw && IsStrictCompatEnabled() && !IsPicaAccelAllowed() &&
        !strict_safe_pica_hw_enter) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA strict_compat v114 late guard blocked AccelerateDrawBatch draw_index={} indexed={} num_vertices={} textures_disabled={} topology={} safe_hw_candidate={} safe_hw_allowed={} safe_hw_enter={} safe_hw_dry_run={}",
                        draw_index, is_indexed, regs.internal.pipeline.num_vertices,
                        textures_disabled, static_cast<u32>(primitive_assembler.GetTopology()),
                        static_cast<u32>(strict_safe_pica_hw_candidate),
                        static_cast<u32>(IsSafePicaHwDrawAllowed()),
                        static_cast<u32>(strict_safe_pica_hw_enter),
                        static_cast<u32>(strict_safe_pica_hw_dry_run));
            LogPicaTextureState(regs.internal, "v114_late_guard_force_software");
        }
        accelerate_draw = false;
    }

    if (accelerate_draw) {
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                        "TRACE_DRAW_PICA v114 pre_call AccelerateDrawBatch indexed={} draw_index={} num_vertices={} hw_index={} stop_after={} color_addr=0x{:08x} depth_addr=0x{:08x}",
                        is_indexed, draw_index, regs.internal.pipeline.num_vertices,
                        strict_safe_pica_hw_index, GetEnvU32("BORKED3DS_V3DV_ACCEL_STAGE_STOP_AFTER", 0),
                        regs.internal.framebuffer.framebuffer.GetColorBufferPhysicalAddress(),
                        regs.internal.framebuffer.framebuffer.GetDepthBufferPhysicalAddress());
        }
        const bool accelerated = rasterizer->AccelerateDrawBatch(is_indexed);
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays AccelerateDrawBatch returned {}", accelerated);
        }
        if (trace_draw) {
            LOG_WARNING(HW_GPU,
                     "TRACE_DRAW_PICA v114 post_call AccelerateDrawBatch returned {} indexed={} draw_index={} num_vertices={} hw_index={}",
                     accelerated, is_indexed, draw_index, regs.internal.pipeline.num_vertices,
                     strict_safe_pica_hw_index);
        }
        if (accelerated) {
            if (trace_draw) {
                LOG_INFO(HW_GPU, "TRACE_DRAW_PICA returning early after accelerated draw");
            }
            return;
        }
    } else if (Settings::values.skip_slow_draw) {
        if (trace_hotpath) {
            LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays skipping slow draw");
        }
        if (trace_draw) {
            LOG_INFO(HW_GPU, "TRACE_DRAW_PICA skip_slow_draw prevented software fallback");
            LogPicaTextureState(regs.internal, "skip_slow_draw");
        }
        return;
    }

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays falling back to software vertex path");
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA falling back to software vertex path");
    }
    LoadVertices(is_indexed);

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU, "PicaCore::DrawArrays calling rasterizer->DrawTriangles()");
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA software path -> rasterizer->DrawTriangles()");
        LogPicaTextureState(regs.internal, "before_draw_triangles");
    }
    rasterizer->DrawTriangles();
}

void PicaCore::LoadVertices(bool is_indexed) {
    const auto& pipeline = regs.internal.pipeline;
    const bool trace_hotpath = IsPicaHotpathTraceEnabled();
    const bool trace_draw = IsPicaDrawTraceEnabled();

    if (trace_hotpath) {
        LOG_DEBUG(HW_GPU,
                  "PicaCore::LoadVertices begin indexed={} num_vertices={} vertex_offset={} base_address={:#010X}",
                  is_indexed, pipeline.num_vertices, pipeline.vertex_offset,
                  pipeline.vertex_attributes.GetPhysicalBaseAddress());
    }
    if (trace_draw) {
        LOG_INFO(HW_GPU,
                 "TRACE_DRAW_PICA load_vertices begin indexed={} num_vertices={} vertex_offset={} base_address={:#010X}",
                 is_indexed, pipeline.num_vertices, pipeline.vertex_offset,
                 pipeline.vertex_attributes.GetPhysicalBaseAddress());
    }

    const PAddr base_address = pipeline.vertex_attributes.GetPhysicalBaseAddress();
    const auto loader = VertexLoader(memory, pipeline);
    regs.internal.rasterizer.ValidateSemantics();

    const auto& index_info = pipeline.index_array;
    const bool index_u16 = index_info.format != 0;
    const PAddr index_base = base_address + index_info.offset;
    const u32 index_stride = index_u16 ? sizeof(u16) : sizeof(u8);
    const u32 needed_index_bytes = pipeline.num_vertices * index_stride;

    const auto index_ref = memory.GetPhysicalRef(index_base);
    const auto index_ref_size = index_ref.GetSize();
    const u32 available_index_bytes = static_cast<u32>(
        std::min<std::size_t>(static_cast<std::size_t>(index_ref_size),
                              static_cast<std::size_t>(needed_index_bytes)));
    const u32 available_indices = index_stride != 0 ? (available_index_bytes / index_stride) : 0;
    const u8* index_bytes = index_ref.GetPtr();

    if (is_indexed && available_index_bytes < needed_index_bytes) {
        LOG_ERROR(HW_GPU,
                  "PicaCore::LoadVertices index array truncated base={:#010X} needed={} available={} num_vertices={} u16={}",
                  index_base, needed_index_bytes, available_index_bytes, pipeline.num_vertices,
                  index_u16);
    }

    const auto read_index = [&](u32 index) -> u32 {
        if (!is_indexed) {
            return index + pipeline.vertex_offset;
        }
        if (available_indices == 0) {
            return pipeline.vertex_offset;
        }
        const u32 clamped_index = std::min(index, available_indices - 1);
        if (index_u16) {
            u16 value{};
            std::memcpy(&value, index_bytes + clamped_index * sizeof(u16), sizeof(u16));
            return static_cast<u32>(value) + pipeline.vertex_offset;
        }
        return static_cast<u32>(index_bytes[clamped_index]) + pipeline.vertex_offset;
    };

    constexpr std::size_t VertexCacheSize = 64;
    std::array<bool, VertexCacheSize> vertex_cache_valid{};
    std::array<u32, VertexCacheSize> vertex_cache_ids{};
    std::array<AttributeBuffer, VertexCacheSize> vertex_cache{};
    u32 vertex_cache_pos = 0;

    ShaderUnit shader_unit;
    AttributeBuffer vs_output{};

    shader_engine->SetupBatch(vs_setup, regs.internal.vs.main_offset);
    geometry_pipeline.Reconfigure();
    geometry_pipeline.Setup(shader_engine.get());
    ASSERT(!geometry_pipeline.NeedIndexInput() || is_indexed);

    for (u32 index = 0; index < pipeline.num_vertices; ++index) {
        const u32 vertex = read_index(index);
        bool vertex_cache_hit = false;

        if (is_indexed) {
            if (geometry_pipeline.NeedIndexInput()) {
                geometry_pipeline.SubmitIndex(vertex);
                continue;
            }

            for (std::size_t i = 0; i < VertexCacheSize; ++i) {
                if (vertex_cache_valid[i] && vertex_cache_ids[i] == vertex) {
                    vs_output = vertex_cache[i];
                    vertex_cache_hit = true;
                    break;
                }
            }
        }

        if (!vertex_cache_hit) {
            AttributeBuffer input{};
            loader.LoadVertex(base_address, index, vertex, input, input_default_attributes);

            if (debug_context) {
                debug_context->OnEvent(DebugContext::Event::VertexShaderInvocation,
                                       std::addressof(input));
            }

            shader_unit.LoadInput(regs.internal.vs, input);
            shader_engine->Run(vs_setup, shader_unit);
            shader_unit.WriteOutput(regs.internal.vs, vs_output);

            if (is_indexed) {
                vertex_cache_valid[vertex_cache_pos] = true;
                vertex_cache_ids[vertex_cache_pos] = vertex;
                vertex_cache[vertex_cache_pos] = vs_output;
                vertex_cache_pos = (vertex_cache_pos + 1) % VertexCacheSize;
            }
        }

        geometry_pipeline.SubmitVertex(vs_output);
    }

    if (trace_draw) {
        LOG_INFO(HW_GPU, "TRACE_DRAW_PICA load_vertices end indexed={} num_vertices={}", is_indexed,
                 pipeline.num_vertices);
    }
}

PicaCore::RenderPropertiesGuess PicaCore::GuessCmdRenderProperties(PAddr list, u32 size) {
    const u8* head = memory.GetPhysicalPointer(list);
    cmd_list.Reset(list, head, size);

    constexpr size_t max_iterations = 0x100;

    RenderPropertiesGuess find_info{};
    find_info.vp_height = regs.internal.rasterizer.viewport_size_y.Value();
    find_info.paddr = regs.internal.framebuffer.framebuffer.color_buffer_address.Value() * 8;

    auto process_write = [this, &find_info](u32 cmd_id, u32 value) {
        switch (cmd_id) {
        case PICA_REG_INDEX(rasterizer.viewport_size_y):
            find_info.vp_height = value;
            find_info.vp_heigh_found = true;
            break;
        case PICA_REG_INDEX(framebuffer.framebuffer.color_buffer_address):
            find_info.paddr = value * 8;
            find_info.paddr_found = true;
            break;
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[0]):
        [[unlikely]] case PICA_REG_INDEX(pipeline.command_buffer.trigger[1]): {
            const u32 index =
                static_cast<u32>(cmd_id - PICA_REG_INDEX(pipeline.command_buffer.trigger[0]));
            const PAddr addr = regs.internal.pipeline.command_buffer.GetPhysicalAddress(index);
            const u32 size = regs.internal.pipeline.command_buffer.GetSize(index);
            const u8* head = memory.GetPhysicalPointer(addr);
            cmd_list.Reset(addr, head, size);
            break;
        }
        default:
            break;
        }
        return find_info.vp_heigh_found && find_info.paddr_found;
    };

    size_t iterations = 0;
    while (cmd_list.current_index < cmd_list.length && iterations < max_iterations) {
        if (cmd_list.current_index % 2 != 0) {
            cmd_list.current_index++;
        }

        const u32 value = cmd_list.head[cmd_list.current_index++];
        const CommandHeader header{cmd_list.head[cmd_list.current_index++]};

        if (process_write(header.cmd_id, value))
            break;

        for (u32 i = 0; i < header.extra_data_length; ++i) {
            const u32 cmd = header.cmd_id + (header.group_commands ? i + 1 : 0);
            const u32 extra_value = cmd_list.head[cmd_list.current_index++];
            if (process_write(cmd, extra_value))
                break;
        }

        iterations++;
    }

    return find_info;
}

template <class Archive>
void PicaCore::CommandList::serialize(Archive& ar, const u32 file_version) {
    ar & addr;
    ar & length;
    ar & current_index;
    if (Archive::is_loading::value) {
        const u8* ptr = Core::System::GetInstance().Memory().GetPhysicalPointer(addr);
        head = reinterpret_cast<const u32*>(ptr);
    }
}

SERIALIZE_IMPL(PicaCore::CommandList)

} // namespace Pica
