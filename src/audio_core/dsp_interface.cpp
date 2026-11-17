// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include "audio_core/dsp_interface.h"
#include "audio_core/sink.h"
#include "audio_core/sink_details.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"

namespace AudioCore {

namespace {

/// v422: BORKED3DS_V422_AUDIO_OFF=1 turns the latency control off.
bool V422Off() {
    static const bool off = std::getenv("BORKED3DS_V422_AUDIO_OFF") != nullptr;
    return off;
}

/// v422: BORKED3DS_V422_AUDIO_SONDE=1 logs one V422_AUDIO line per second of output.
bool V422Sonde() {
    static const bool on = std::getenv("BORKED3DS_V422_AUDIO_SONDE") != nullptr;
    return on;
}

double V422Ms(u64 frames) {
    return static_cast<double>(frames) * 1000.0 / native_sample_rate;
}

} // namespace

DspInterface::DspInterface(Core::System& system_) : system(system_) {}

DspInterface::~DspInterface() = default;

void DspInterface::SetSink(AudioCore::SinkType sink_type, std::string_view audio_device) {
    // Dispose of the current sink first to avoid contention.
    sink.reset();

    sink = AudioCore::GetSinkDetails(sink_type).create_sink(audio_device);
    sink->SetCallback(
        [this](s16* buffer, std::size_t num_frames) { OutputCallback(buffer, num_frames); });
    time_stretcher.SetOutputSampleRate(sink->GetNativeSampleRate());
}

Sink& DspInterface::GetSink() {
    ASSERT(sink);
    return *sink.get();
}

void DspInterface::EnableStretching(bool enable) {
    enable_time_stretching = enable;
}

void DspInterface::OutputFrame(StereoFrame16 frame) {
    if (!sink) {
        return;
    }

    const std::size_t pushed = fifo.Push(frame.data(), frame.size());
    if (pushed < frame.size()) {
        // FIFO full (250 ms of audio waiting): these samples are lost.
        v422_overflow_frames += frame.size() - pushed;
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioFrame(std::move(frame));
    }
}

void DspInterface::OutputSample(std::array<s16, 2> sample) {
    if (!sink) {
        return;
    }

    if (fifo.Push(&sample, 1) == 0) {
        v422_overflow_frames += 1;
    }

    auto video_dumper = system.GetVideoDumper();
    if (video_dumper && video_dumper->IsDumping()) {
        video_dumper->AddAudioSample(std::move(sample));
    }
}

void DspInterface::OutputCallback(s16* buffer, std::size_t num_frames) {
    // Determine if we should stretch based on the current emulation speed.
    const auto perf_stats = system.GetLastPerfStats();
    const auto should_stretch = enable_time_stretching && perf_stats.emulation_speed <= 95;
    if (performing_time_stretching && !should_stretch) {
        // If we just stopped stretching, flush the stretcher before returning to normal output.
        flushing_time_stretcher = true;
    }
    performing_time_stretching = should_stretch;

    std::size_t frames_written = 0;
    if (performing_time_stretching) {
        const std::vector<s16> in{fifo.Pop()};
        const std::size_t num_in{in.size() / 2};
        frames_written = time_stretcher.Process(in.data(), num_in, buffer, num_frames);
    } else {
        if (flushing_time_stretcher) {
            time_stretcher.Flush();
            frames_written = time_stretcher.Process(nullptr, 0, buffer, num_frames);
            flushing_time_stretcher = false;

            // Make sure any frames that did not fit are cleared from the time stretcher,
            // so that they do not bleed into the next time the stretcher is enabled.
            time_stretcher.Clear();
        }
        frames_written += fifo.Pop(buffer, num_frames - frames_written);
    }

    if (frames_written > 0) {
        std::memcpy(&last_frame[0], buffer + 2 * (frames_written - 1), 2 * sizeof(s16));
    }

    V422Tick(num_frames, frames_written, performing_time_stretching);

    // Hold last emitted frame; this prevents popping.
    for (std::size_t i = frames_written; i < num_frames; i++) {
        std::memcpy(buffer + 2 * i, &last_frame[0], 2 * sizeof(s16));
    }

    // Implementation of the hardware volume slider
    // A cubic curve is used to approximate a linear change in human-perceived loudness
    const float linear_volume = std::clamp(Settings::Volume(), 0.0f, 1.0f);
    if (linear_volume != 1.0) {
        const float volume_scale_factor = linear_volume * linear_volume * linear_volume;
        for (std::size_t i = 0; i < num_frames; i++) {
            buffer[i * 2 + 0] = static_cast<s16>(buffer[i * 2 + 0] * volume_scale_factor);
            buffer[i * 2 + 1] = static_cast<s16>(buffer[i * 2 + 1] * volume_scale_factor);
        }
    }
}

void DspInterface::V422Tick(std::size_t num_frames, std::size_t frames_written,
                            bool stretching) {
    // The lowest FIFO fill seen after each callback, over one second, is audio that never
    // drained: a standing delay. Above 20 ms it is cut back to 10 ms, once per second.
    // While stretching, the stretcher empties the FIFO itself, so those callbacks are not counted.
    constexpr std::size_t kept_frames = native_sample_rate / 100; // 10 ms
    const std::size_t fill = fifo.Size();
    if (!stretching) {
        v422_low_fill = std::min(v422_low_fill, fill);
    } else {
        v422_stretch_frames += num_frames;
    }
    v422_high_fill = std::max(v422_high_fill, fill);
    if (frames_written < num_frames) {
        v422_underrun_frames += num_frames - frames_written;
    }
    v422_window_frames += num_frames;
    if (v422_window_frames < static_cast<std::size_t>(native_sample_rate)) {
        return;
    }

    std::size_t dropped = 0;
    if (!V422Off() && v422_low_fill != ~std::size_t(0) && v422_low_fill > 2 * kept_frames) {
        std::array<std::array<s16, 2>, 512> scratch;
        std::size_t excess = v422_low_fill - kept_frames;
        while (excess > 0) {
            const std::size_t n = fifo.Pop(scratch.data(), std::min(excess, scratch.size()));
            if (n == 0) {
                break;
            }
            excess -= n;
            dropped += n;
        }
        if (!v422_announced) {
            v422_announced = true;
            LOG_WARNING(Audio, "V422_AUDIO latence reduite : {:.1f} ms de son en attente retires "
                               "(BORKED3DS_V422_AUDIO_OFF=1 pour couper)",
                        V422Ms(dropped));
        }
    }

    if (V422Sonde()) {
        LOG_WARNING(Audio,
                    "V422_AUDIO fifo_min_ms={:.1f} fifo_max_ms={:.1f} retire_ms={:.1f} "
                    "debordement_ms={:.1f} manque_ms={:.1f} etirement_ms={:.1f} actif={}",
                    v422_low_fill == ~std::size_t(0) ? -1.0 : V422Ms(v422_low_fill),
                    V422Ms(v422_high_fill), V422Ms(dropped),
                    V422Ms(v422_overflow_frames.exchange(0)), V422Ms(v422_underrun_frames),
                    V422Ms(v422_stretch_frames), V422Off() ? 0 : 1);
    }

    v422_low_fill = ~std::size_t(0);
    v422_high_fill = 0;
    v422_window_frames = 0;
    v422_underrun_frames = 0;
    v422_stretch_frames = 0;
}

} // namespace AudioCore
