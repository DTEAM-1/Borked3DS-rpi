// Copyright Citra Emulator Project / Azahar Emulator Project
// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

// v396 (Azahar cf87efa3) : minimum et maximum vectorises d'un tableau d'indices de sommets,
// utilises par RasterizerAccelerated::AnalyzeVertexArray (chemin chaud de chaque draw indexe).
// NEON sur ARM64 (Raspberry Pi 5), SSE4.2 sur x86 si disponible, boucle scalaire sinon.

#include <algorithm>
#include "common/math_util.h"

#if defined(__aarch64__) || defined(__ARM_NEON)
#define BORKED3DS_MINMAX_NEON
#include <arm_neon.h>
#elif defined(__SSE4_2__) || defined(__SSE4_1__)
#define BORKED3DS_MINMAX_SSE
#include <smmintrin.h>
#endif

namespace Common {

std::pair<u8, u8> FindMinMax(const std::span<const u8>& data) {
    const std::size_t count = data.size();
    const u8* ptr = data.data();
    u8 lo = 0xFF;
    u8 hi = 0;
    std::size_t i = 0;
    constexpr std::size_t lanes = 16;
    if (count >= lanes * 2) {
#if defined(BORKED3DS_MINMAX_NEON)
        uint8x16_t vmin = vdupq_n_u8(0xFF);
        uint8x16_t vmax = vdupq_n_u8(0);
        for (; i + lanes <= count; i += lanes) {
            const uint8x16_t v = vld1q_u8(ptr + i);
            vmin = vminq_u8(vmin, v);
            vmax = vmaxq_u8(vmax, v);
        }
        lo = vminvq_u8(vmin);
        hi = vmaxvq_u8(vmax);
#elif defined(BORKED3DS_MINMAX_SSE)
        __m128i vmin = _mm_set1_epi8(static_cast<char>(0xFF));
        __m128i vmax = _mm_setzero_si128();
        for (; i + lanes <= count; i += lanes) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            vmin = _mm_min_epu8(vmin, v);
            vmax = _mm_max_epu8(vmax, v);
        }
        alignas(16) u8 tmp[lanes];
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), vmin);
        lo = *std::min_element(tmp, tmp + lanes);
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), vmax);
        hi = *std::max_element(tmp, tmp + lanes);
#endif
    }
    for (; i < count; ++i) {
        lo = std::min(lo, ptr[i]);
        hi = std::max(hi, ptr[i]);
    }
    return {lo, hi};
}

std::pair<u16, u16> FindMinMax(const std::span<const u16>& data) {
    const std::size_t count = data.size();
    const u16* ptr = data.data();
    u16 lo = 0xFFFF;
    u16 hi = 0;
    std::size_t i = 0;
    constexpr std::size_t lanes = 8;
    if (count >= lanes * 2) {
#if defined(BORKED3DS_MINMAX_NEON)
        uint16x8_t vmin = vdupq_n_u16(0xFFFF);
        uint16x8_t vmax = vdupq_n_u16(0);
        for (; i + lanes <= count; i += lanes) {
            const uint16x8_t v = vld1q_u16(ptr + i);
            vmin = vminq_u16(vmin, v);
            vmax = vmaxq_u16(vmax, v);
        }
        lo = vminvq_u16(vmin);
        hi = vmaxvq_u16(vmax);
#elif defined(BORKED3DS_MINMAX_SSE)
        __m128i vmin = _mm_set1_epi16(static_cast<short>(0xFFFF));
        __m128i vmax = _mm_setzero_si128();
        for (; i + lanes <= count; i += lanes) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            vmin = _mm_min_epu16(vmin, v);
            vmax = _mm_max_epu16(vmax, v);
        }
        alignas(16) u16 tmp[lanes];
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), vmin);
        lo = *std::min_element(tmp, tmp + lanes);
        _mm_store_si128(reinterpret_cast<__m128i*>(tmp), vmax);
        hi = *std::max_element(tmp, tmp + lanes);
#endif
    }
    for (; i < count; ++i) {
        lo = std::min(lo, ptr[i]);
        hi = std::max(hi, ptr[i]);
    }
    return {lo, hi};
}

} // namespace Common
