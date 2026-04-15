// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <atomic>
#include <string_view>

#include "common/assert.h"
#include "common/color.h"
#include "common/logging/log.h"
#include "common/swap.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"
#include "video_core/texture/etc1.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"

using TextureFormat = Pica::TexturingRegs::TextureFormat;

namespace Pica::Texture {

constexpr std::size_t TILE_SIZE = 8 * 8;
constexpr std::size_t ETC1_SUBTILES = 2 * 2;

namespace {

std::atomic<u32> g_pi5_ui_decode_trace_budget{192};

[[nodiscard]] bool ConsumeTraceBudget(std::atomic<u32>& budget) {
    u32 remaining = budget.load(std::memory_order_relaxed);
    while (remaining != 0) {
        if (budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] constexpr Common::Vec4<u8> MakeBlackAlpha(const u8 alpha) {
    return {0, 0, 0, alpha};
}

[[nodiscard]] constexpr Common::Vec4<u8> MakeWhiteAlpha(const u8 alpha) {
    return {255, 255, 255, alpha};
}

[[nodiscard]] constexpr Common::Vec4<u8> MakeIntensityAlpha(const u8 intensity, const u8 alpha) {
    return {intensity, intensity, intensity, alpha};
}

[[nodiscard]] constexpr Common::Vec4<u8> MakeOpaqueIntensity(const u8 intensity) {
    return {intensity, intensity, intensity, 255};
}

[[nodiscard]] constexpr std::string_view TextureFormatName(TextureFormat format) {
    switch (format) {
    case TextureFormat::RGBA8:
        return "RGBA8";
    case TextureFormat::RGB8:
        return "RGB8";
    case TextureFormat::RGB5A1:
        return "RGB5A1";
    case TextureFormat::RGB565:
        return "RGB565";
    case TextureFormat::RGBA4:
        return "RGBA4";
    case TextureFormat::IA8:
        return "IA8";
    case TextureFormat::RG8:
        return "RG8";
    case TextureFormat::I8:
        return "I8";
    case TextureFormat::A8:
        return "A8";
    case TextureFormat::IA4:
        return "IA4";
    case TextureFormat::I4:
        return "I4";
    case TextureFormat::A4:
        return "A4";
    case TextureFormat::ETC1:
        return "ETC1";
    case TextureFormat::ETC1A4:
        return "ETC1A4";
    default:
        return "Unknown";
    }
}

void TraceDecodedTexel(TextureFormat format, bool disable_alpha, unsigned int x, unsigned int y,
                      const Common::Vec4<u8>& rgba, u32 raw0, u32 raw1 = 0,
                      std::string_view detail = {}) {
    if (!ConsumeTraceBudget(g_pi5_ui_decode_trace_budget)) {
        return;
    }

    LOG_INFO(HW_GPU,
             "TRACE_PI5_UI decode format={} disable_alpha={} x={} y={} raw0=0x{:02X} raw1=0x{:02X} rgba=({}, {}, {}, {}) detail={}",
             TextureFormatName(format), disable_alpha, x, y, raw0 & 0xFFu, raw1 & 0xFFu, rgba[0],
             rgba[1], rgba[2], rgba[3], detail);
}

} // namespace

size_t CalculateTileSize(TextureFormat format) {
    switch (format) {
    case TextureFormat::RGBA8:
        return 4 * TILE_SIZE;

    case TextureFormat::RGB8:
        return 3 * TILE_SIZE;

    case TextureFormat::RGB5A1:
    case TextureFormat::RGB565:
    case TextureFormat::RGBA4:
    case TextureFormat::IA8:
    case TextureFormat::RG8:
        return 2 * TILE_SIZE;

    case TextureFormat::I8:
    case TextureFormat::A8:
    case TextureFormat::IA4:
        return 1 * TILE_SIZE;

    case TextureFormat::I4:
    case TextureFormat::A4:
        return TILE_SIZE / 2;

    case TextureFormat::ETC1:
        return ETC1_SUBTILES * 8;

    case TextureFormat::ETC1A4:
        return ETC1_SUBTILES * 16;

    default: // placeholder for yet unknown formats
        UNIMPLEMENTED();
        return 0;
    }
}

Common::Vec4<u8> LookupTexture(const u8* source, unsigned int x, unsigned int y,
                               const TextureInfo& info, bool disable_alpha) {
    // Coordinate in tiles
    const unsigned int coarse_x = x / 8;
    const unsigned int coarse_y = y / 8;

    // Coordinate inside the tile
    const unsigned int fine_x = x % 8;
    const unsigned int fine_y = y % 8;

    const u8* line = source + coarse_y * info.stride;
    const u8* tile = line + coarse_x * CalculateTileSize(info.format);
    return LookupTexelInTile(tile, fine_x, fine_y, info, disable_alpha);
}

Common::Vec4<u8> LookupTexelInTile(const u8* source, unsigned int x, unsigned int y,
                                   const TextureInfo& info, bool disable_alpha) {
    DEBUG_ASSERT(x < 8);
    DEBUG_ASSERT(y < 8);

    using VideoCore::MortonInterleave;

    switch (info.format) {
    case TextureFormat::RGBA8: {
        auto res = Common::Color::DecodeRGBA8(source + MortonInterleave(x, y) * 4);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB8: {
        auto res = Common::Color::DecodeRGB8(source + MortonInterleave(x, y) * 3);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGB5A1: {
        auto res = Common::Color::DecodeRGB5A1(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::RGB565: {
        auto res = Common::Color::DecodeRGB565(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), 255};
    }

    case TextureFormat::RGBA4: {
        auto res = Common::Color::DecodeRGBA4(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), res.b(), static_cast<u8>(disable_alpha ? 255 : res.a())};
    }

    case TextureFormat::IA8: {
        const u8* source_ptr = source + MortonInterleave(x, y) * 2;
        const u8 alpha = source_ptr[0];
        const u8 intensity = source_ptr[1];

        const Common::Vec4<u8> rgba =
            disable_alpha ? MakeOpaqueIntensity(intensity) : MakeIntensityAlpha(intensity, alpha);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, source_ptr[0], source_ptr[1],
                          disable_alpha ? "ia8_alpha_disabled_as_intensity"
                                        : "ia8_intensity_alpha");
        return rgba;
    }

    case TextureFormat::RG8: {
        auto res = Common::Color::DecodeRG8(source + MortonInterleave(x, y) * 2);
        return {res.r(), res.g(), 0, 255};
    }

    case TextureFormat::I8: {
        const u8* source_ptr = source + MortonInterleave(x, y);
        const Common::Vec4<u8> rgba = MakeIntensityAlpha(*source_ptr, 255);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, *source_ptr, 0,
                          "i8_intensity_opaque");
        return rgba;
    }

    case TextureFormat::A8: {
        const u8* source_ptr = source + MortonInterleave(x, y);
        const u8 alpha = *source_ptr;

        const Common::Vec4<u8> rgba =
            disable_alpha ? MakeOpaqueIntensity(alpha) : MakeWhiteAlpha(alpha);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, alpha, 0,
                          disable_alpha ? "a8_alpha_disabled_as_grayscale"
                                        : "a8_white_alpha");
        return rgba;
    }

    case TextureFormat::IA4: {
        const u8* source_ptr = source + MortonInterleave(x, y);

        const u8 intensity = Common::Color::Convert4To8(((*source_ptr) & 0xF0) >> 4);
        const u8 alpha = Common::Color::Convert4To8((*source_ptr) & 0xF);

        const Common::Vec4<u8> rgba =
            disable_alpha ? MakeOpaqueIntensity(intensity) : MakeIntensityAlpha(intensity, alpha);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, *source_ptr, 0,
                          disable_alpha ? "ia4_alpha_disabled_as_intensity"
                                        : "ia4_hi_intensity_lo_alpha");
        return rgba;
    }

    case TextureFormat::I4: {
        const u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 intensity = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        intensity = Common::Color::Convert4To8(intensity);

        const Common::Vec4<u8> rgba = MakeIntensityAlpha(intensity, 255);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, *source_ptr, 0,
                          (morton_offset % 2) ? "i4_high_nibble" : "i4_low_nibble");
        return rgba;
    }

    case TextureFormat::A4: {
        const u32 morton_offset = MortonInterleave(x, y);
        const u8* source_ptr = source + morton_offset / 2;

        u8 alpha = (morton_offset % 2) ? ((*source_ptr & 0xF0) >> 4) : (*source_ptr & 0xF);
        alpha = Common::Color::Convert4To8(alpha);

        const Common::Vec4<u8> rgba =
            disable_alpha ? MakeOpaqueIntensity(alpha) : MakeWhiteAlpha(alpha);
        TraceDecodedTexel(info.format, disable_alpha, x, y, rgba, *source_ptr, 0,
                          disable_alpha ? ((morton_offset % 2) ? "a4_high_nibble_alpha_disabled"
                                                               : "a4_low_nibble_alpha_disabled")
                                        : ((morton_offset % 2) ? "a4_high_nibble_white_alpha"
                                                               : "a4_low_nibble_white_alpha"));
        return rgba;
    }

    case TextureFormat::ETC1:
    case TextureFormat::ETC1A4: {
        const bool has_alpha = (info.format == TextureFormat::ETC1A4);
        const std::size_t subtile_size = has_alpha ? 16 : 8;

        // ETC1 further subdivides each 8x8 tile into four 4x4 subtiles
        constexpr unsigned int subtile_width = 4;
        constexpr unsigned int subtile_height = 4;

        const unsigned int subtile_index = (x / subtile_width) + 2 * (y / subtile_height);
        x %= subtile_width;
        y %= subtile_height;

        const u8* subtile_ptr = source + subtile_index * subtile_size;

        u8 alpha = 255;
        if (has_alpha) {
            u64_le packed_alpha;
            std::memcpy(&packed_alpha, subtile_ptr, sizeof(u64));
            subtile_ptr += sizeof(u64);

            alpha = Common::Color::Convert4To8(
                (packed_alpha >> (4 * (x * subtile_width + y))) & 0xF);
        }

        u64_le subtile_data;
        std::memcpy(&subtile_data, subtile_ptr, sizeof(u64));

        return Common::MakeVec(SampleETC1Subtile(subtile_data, x, y),
                               disable_alpha ? static_cast<u8>(255) : alpha);
    }

    default:
        LOG_ERROR(HW_GPU, "Unknown texture format: {:x}", static_cast<u32>(info.format));
        DEBUG_ASSERT(false);
        return {};
    }
}

TextureInfo TextureInfo::FromPicaRegister(const TexturingRegs::TextureConfig& config,
                                          const TexturingRegs::TextureFormat& format) {
    TextureInfo info;
    info.physical_address = config.GetPhysicalAddress();
    info.width = config.width;
    info.height = config.height;
    info.format = format;
    info.SetDefaultStride();
    return info;
}

} // namespace Pica::Texture
