// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <cstdlib>
#include "common/common_types.h"

// V433 -- LOT D D'AZAHAR (27 nov. 2026). Correctifs amont ACTIFS PAR DEFAUT ; chacun se coupe par
// sa variable (valeur non vide et differente de "0", meme convention que BORKED3DS_V3DV_V408_OFF) :
//   D1  BORKED3DS_V3DV_V433_NO_STRIDE     validation de l'aliasing de stride (Azahar #2451)
//                                         -- rasterizer_cache/rasterizer_cache.h
//   D2  BORKED3DS_V3DV_V433_NO_FAST_PUMP  lecture de la manette toutes les 1 ms (Azahar #2700)
//                                         -- input_common/sdl/sdl_impl.cpp
//   D3  BORKED3DS_V3DV_V433_NO_MEMFILL    une seule interruption PSC0 quand le GSP demande les
//                                         deux remplissages ensemble (Azahar #1218) -- gpu.cpp
// Le vidage du retard audio d'Azahar #2700 existe deja depuis v422 (V422Tick) : non repris.
//
// Journal (niveau Warning) : marqueur V433_AZAHAR_LOT_D au demarrage, puis une ligne
// V433_COMPTEURS par minute emulee (gpu.cpp, VBlankCallback).
// En-tete seul (variables inline) : aucun changement de CMakeLists.

namespace VideoCore::V433 {

[[nodiscard]] inline bool EnvOff(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] inline bool StrideFixEnabled() {
    static const bool enabled = !EnvOff("BORKED3DS_V3DV_V433_NO_STRIDE");
    return enabled;
}

[[nodiscard]] inline bool FastPumpEnabled() {
    static const bool enabled = !EnvOff("BORKED3DS_V3DV_V433_NO_FAST_PUMP");
    return enabled;
}

[[nodiscard]] inline bool MemFillFixEnabled() {
    static const bool enabled = !EnvOff("BORKED3DS_V3DV_V433_NO_MEMFILL");
    return enabled;
}

/// D1 : validations que l'ancien raccourci de stride aurait sautees.
inline std::atomic<u64> g_stride_forcees{0};

/// D3 : remplissages des deux tampons demandes ensemble par le GSP (une seule interruption PSC0
/// chacun quand D3 est actif, deux sinon). 0 = le jeu n'utilise pas ce cas.
inline std::atomic<u64> g_memfill_paires{0};

} // namespace VideoCore::V433
