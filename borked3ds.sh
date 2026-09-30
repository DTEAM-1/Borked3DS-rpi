#!/usr/bin/env bash

rp_module_id="borked3ds"
rp_module_desc="Borked3DS – Nintendo 3DS Emulator (Pi5 Vulkan V3DV + Qt)"
rp_module_help="ROM Extensions: .3ds .cia .cxi"
rp_module_licence="GPL3 https://github.com/DTEAM-1/Borked3DS-rpi"
rp_module_repo=""
rp_module_section="exp"
rp_module_flags="!all rpi5"

function depends_borked3ds() {
    getDepends \
        cmake ninja-build build-essential git pkg-config \
        clang \
        libx11-dev libxrandr-dev libxi-dev \
        libgl1-mesa-dev libglu1-mesa-dev \
        libsdl2-dev libevdev-dev \
        libpulse-dev libasound2-dev \
        qt6-base-dev qt6-base-dev-tools qt6-tools-dev \
        libboost-all-dev libcrypto++-dev
}

function sources_borked3ds() {

    ################################
    # SOURCE SELECTION
    #
    # Default: clone DTEAM-1/Borked3DS-rpi from GitHub. The repository is the source of
    # truth; a local tree is never picked up automatically, so a build can never silently
    # compile a stale copy.
    #
    # To build a local tree (quick test without pushing), ask for it explicitly:
    #     BORKED3DS_LOCAL_SRC=/home/pi/Borked3DS-rpi-master sudo -E ./retropie_setup.sh
    #
    # The selected source is printed in both cases.
    ################################

    local local_src="${BORKED3DS_LOCAL_SRC:-}"

    ################################
    # BUILD DIRECTORY CLEANUP
    #
    # Empty $md_build without deleting it:
    #   - "rm -rf $md_build/*" skips hidden files, so the previous .git survived and the
    #     next "git clone" failed ("destination path already exists").
    #   - "rm -rf $md_build" is wrong too: RetroPie-Setup has already pushd'ed into
    #     $md_build, so deleting it removes the shell's working directory (git then fails
    #     and "git -C" reports the RetroPie-Setup commit instead of ours).
    # find -mindepth 1 also removes hidden files and keeps the directory itself.
    ################################

    mkdir -p "$md_build"
    find "$md_build" -mindepth 1 -delete 2>/dev/null
    cd "$md_build" || exit 1

    if [ -n "$local_src" ]; then
        if [ ! -f "$local_src/CMakeLists.txt" ] || [ ! -d "$local_src/src" ]; then
            echo "BORKED3DS_LOCAL_SRC=$local_src ne contient pas un arbre valide -- abandon."
            exit 1
        fi
        echo "=========================================================="
        echo "SOURCES: arbre LOCAL (demande explicitement) -> $local_src"
        echo "=========================================================="
        cp -a "$local_src"/. "$md_build"/
        rm -rf "$md_build/build"
        echo "LOCAL-$(date +%Y%m%d-%H%M%S)" > "$md_build/.borked3ds_commit"
        cd "$md_build" || exit 1
        if [ -d "$md_build/.git" ]; then
            git submodule update --init --recursive
        fi
    else
        echo "=========================================================="
        echo "SOURCES: clone GitHub DTEAM-1/Borked3DS-rpi (defaut)"
        echo "=========================================================="
        ################################
        # CLONE WITH RETRIES, STOP ON FAILURE
        #
        # A network error used to let the script continue: the patches below ran on an
        # empty tree and "git -C" reported the parent RetroPie-Setup commit. The clone is
        # now retried 3 times (GitHub outages are often transient) and the build stops if
        # it still fails. The directory is emptied between attempts.
        ################################

        local clone_ok=0
        local attempt
        for attempt in 1 2 3; do
            echo "Clone du depot -- tentative $attempt/3..."
            if git clone --recursive https://github.com/DTEAM-1/Borked3DS-rpi "$md_build"; then
                clone_ok=1
                break
            fi
            echo "Tentative $attempt echouee."
            if [ "$attempt" -lt 3 ]; then
                echo "Nettoyage du repertoire et nouvelle tentative dans 5 s..."
                find "$md_build" -mindepth 1 -delete 2>/dev/null
                sleep 5
            fi
        done

        if [ "$clone_ok" -ne 1 ]; then
            echo ""
            echo "!! CLONE IMPOSSIBLE apres 3 tentatives."
            echo "!! Cause typique : coupure reseau ou GitHub temporairement injoignable."
            echo "!! Verifier la connexion puis relancer :"
            echo "!!     git ls-remote https://github.com/DTEAM-1/Borked3DS-rpi HEAD"
            echo "!! Ne pas poursuivre : le build compilerait autre chose."
            exit 1
        fi

        cd "$md_build" || exit 1

        # Guard: without CMakeLists.txt the clone is incomplete even if git succeeded.
        if [ ! -f "$md_build/CMakeLists.txt" ]; then
            echo "!! Clone incomplet : CMakeLists.txt absent. Abandon."
            exit 1
        fi

        git submodule update --init --recursive

        # Read the commit only if $md_build is the root of a git repository; otherwise
        # "git -C" would climb to the parent repository and report a foreign commit.
        if [ ! -d "$md_build/.git" ]; then
            echo "!! $md_build n'est pas la racine d'un depot git -- commit non fiable. Abandon."
            exit 1
        fi
        git -C "$md_build" rev-parse --short HEAD > "$md_build/.borked3ds_commit"
        echo "Commit compile : $(cat "$md_build/.borked3ds_commit") $(git -C "$md_build" log -1 --format=%s)"
    fi

    ################################
    # FIX CMAKE PKGCONFIG BUG
    ################################

    if [ -f "$md_build/externals/cmake-modules/Findcryptopp.cmake" ]; then
        sed -i '1ifind_package(PkgConfig)' \
        "$md_build/externals/cmake-modules/Findcryptopp.cmake"
    fi

    ################################
    # FIX CLANG -Werror INCOMPATIBILITY
    #
    # The fork sets -Werror in its cmake files, so Clang rejects code that GCC accepts.
    # CMAKE_CXX_FLAGS cannot override a -Werror set by target_compile_options().
    #   Fix 1: strip bare -Werror from every cmake file (-Werror=... forms are kept).
    #   Fix 2: patch the two offending source files directly.
    #
    # TODO (packaging): the two source fixes should be committed to the repository and
    # removed from here; if the upstream pattern changes, the sed silently does nothing.
    ################################

    find "$md_build" \( -name "CMakeLists.txt" -o -name "*.cmake" \) | \
        xargs grep -l "\-Werror" 2>/dev/null | while read -r f; do
        echo "Patching -Werror out of: $f"
        sed -i 's/[[:space:]]-Werror[[:space:]]/ /g' "$f"
        sed -i 's/[[:space:]]-Werror"/ "/g' "$f"
        sed -i 's/-Werror\([^=]\)/\1/g; s/-Werror$//' "$f"
    done

    # Check: no bare -Werror may remain (-Werror=... is legitimate).
    local werror_left
    werror_left="$(grep -rn -- "-Werror" "$md_build" --include="CMakeLists.txt" --include="*.cmake" 2>/dev/null | grep -v -- "-Werror=" | wc -l)"
    if [ "$werror_left" -ne 0 ]; then
        echo "ATTENTION : $werror_left occurrence(s) de -Werror subsistent dans les fichiers cmake."
    else
        echo "-Werror nu : aucune occurrence restante."
    fi

    # Source fix 1: glsl_fs_shader_gen.cpp
    # Logical '||' with a constant operand (GL_SHADER_IMAGE_ATOMIC is an int constant).
    # Clang rejects it; bitwise '|' is equivalent here.
    local fs_gen="$md_build/src/video_core/shader/generator/glsl_fs_shader_gen.cpp"
    if [ -f "$fs_gen" ]; then
        if grep -q "GLAD_GL_ARB_shader_image_load_store || GL_SHADER_IMAGE_ATOMIC" "$fs_gen"; then
            sed -i 's/GLAD_GL_ARB_shader_image_load_store || GL_SHADER_IMAGE_ATOMIC/GLAD_GL_ARB_shader_image_load_store | GL_SHADER_IMAGE_ATOMIC/' "$fs_gen"
            echo "Patched glsl_fs_shader_gen.cpp: || -> | pour GL_SHADER_IMAGE_ATOMIC"
        else
            echo "NOTE: motif GL_SHADER_IMAGE_ATOMIC absent de glsl_fs_shader_gen.cpp (deja corrige en amont ?)"
        fi
    fi

    # Source fix 2: texture_decode.cpp
    # Unused function 'MakeBlackAlpha': [[nodiscard]] -> [[maybe_unused]]
    local tex_decode="$md_build/src/video_core/texture/texture_decode.cpp"
    if [ -f "$tex_decode" ]; then
        if grep -q "\[\[nodiscard\]\] constexpr Common::Vec4<u8> MakeBlackAlpha" "$tex_decode"; then
            sed -i 's/\[\[nodiscard\]\] constexpr Common::Vec4<u8> MakeBlackAlpha/[[maybe_unused]] constexpr Common::Vec4<u8> MakeBlackAlpha/' "$tex_decode"
            echo "Patched texture_decode.cpp: [[nodiscard]] -> [[maybe_unused]] pour MakeBlackAlpha"
        else
            echo "NOTE: motif MakeBlackAlpha absent de texture_decode.cpp (deja corrige en amont ?)"
        fi
    fi

    ################################
    # EXIT GUARD
    #
    # Make sure the source tree is in place before build_borked3ds() runs; otherwise a
    # failed clone only shows up later as an obscure CMake error.
    ################################

    if [ ! -f "$md_build/CMakeLists.txt" ] || [ ! -d "$md_build/src" ]; then
        echo ""
        echo "!! ARBRE SOURCE INCOMPLET dans $md_build"
        echo "!! CMakeLists.txt ou src/ manquant -- le clone ou la copie a echoue."
        echo "!! Ne pas poursuivre : le build compilerait autre chose ou echouerait plus loin."
        exit 1
    fi
    echo "Arbre source verifie : CMakeLists.txt et src/ presents."
}

function build_borked3ds() {

    cd "$md_build" || exit 1

    mkdir -p build
    cd build || exit 1

    # Build with Clang: upstream notes that "Vulkan may crash if the executable was
    # compiled with GCC". -Werror is stripped in sources_borked3ds() above.
    cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_FLAGS="-march=armv8.2-a+crc+crypto -O3" \
        -DCMAKE_CXX_FLAGS="-march=armv8.2-a+crc+crypto -O3" \
        -DENABLE_QT=ON \
        -DENABLE_SDL2=ON \
        -DENABLE_TESTS=OFF \
        -DUSE_SYSTEM_LIBS=ON

    if [ $? -ne 0 ]; then
        echo "CMake configuration failed"
        exit 1
    fi

    ninja -j$(nproc)

    if [ $? -ne 0 ]; then
        echo "Build failed"
        exit 1
    fi
}

function install_borked3ds() {

    mkdir -p "$md_inst"

    # The Qt executable is named borked3ds (not borked3ds-qt).
    if [ -f "$md_build/build/bin/Release/borked3ds" ]; then
        cp "$md_build/build/bin/Release/borked3ds" "$md_inst/borked3ds"
        chmod +x "$md_inst/borked3ds"
    else
        echo "Binary borked3ds missing — build may have failed"
        exit 1
    fi

    ################################
    # USER HOME
    #
    # The scriptmodule runs as root ($HOME is /root). Files created below must go to the
    # user's home; RetroPie-Setup provides $home and $__user for that.
    ################################
    local user_home="${home:-/home/${__user:-pi}}"

    ################################
    # MINIMAL SAVEDATA
    #
    # archive_source_sd_savedata.cpp mounts SaveData from
    # sdmc/.../title/{high}/{low}/data/00000001/. Sonic Lost World reads network_id.dat
    # at startup; if it is missing, the unhandled FILE_NOT_FOUND crashes the ARM thread.
    #
    # TODO (packaging): this is specific to the project's test games and does not belong
    # in a distributable package.
    ################################

    local sdmc_base="$user_home/.local/share/borked3ds-emu/sdmc/Nintendo 3DS"
    local sdmc_id0="00000000000000000000000000000000"
    local sdmc_id1="00000000000000000000000000000000"
    local sdmc="$sdmc_base/$sdmc_id0/$sdmc_id1"

    # Sonic Lost World US (00040000000C8C00) — network_id.dat
    # 16 bytes: LocalFriendCodeSeed (8 non-zero bytes) + NetworkID (8 bytes)
    local sonic_us_data="$sdmc/title/00040000/000c8c00/data/00000001"
    if [ ! -f "$sonic_us_data/network_id.dat" ]; then
        mkdir -p "$sonic_us_data"
        python3 -c "
import struct
data = struct.pack('<Q', 0x0123456789ABCDEF) + bytes(8)
open('$sonic_us_data/network_id.dat', 'wb').write(data)
" && echo "Created network_id.dat for Sonic Lost World US" \
          || echo "WARNING: failed to create network_id.dat"
    fi

    # Sonic Lost World EU (00040000000C8D00) — same layout
    local sonic_eu_data="$sdmc/title/00040000/000c8d00/data/00000001"
    if [ ! -f "$sonic_eu_data/network_id.dat" ]; then
        mkdir -p "$sonic_eu_data"
        python3 -c "
import struct
data = struct.pack('<Q', 0x0123456789ABCDEF) + bytes(8)
open('$sonic_eu_data/network_id.dat', 'wb').write(data)
" && echo "Created network_id.dat for Sonic Lost World EU" \
          || echo "WARNING: failed to create network_id.dat EU"
    fi

    ################################
    # POST-INSTALL VERIFICATION
    #
    # Every expected marker string is searched for in the installed binary, so that no
    # test is ever run on a stale binary. A single ABSENT marker invalidates the test
    # cycle: do not launch a game, re-upload the files and rebuild.
    #
    # Each marker proves that a given fix or probe is present, e.g.:
    #   A7Z12_FRAME_CENSUS / swhist_le8= / rp_switch= / cpu_pct= / sub_lag= / f_fb= /
    #     seq_count= / A7Z12_FB_IDENT / c_addr= / A7Z12_RP_END_SITE : frame census fields
    #   BORKED3DS_V3DV_DISABLE_LAZY_COPY_VIEW : escape hatch of the TB33 fix (no image copy
    #     on every draw, on by default); if it disappears, the fix was lost
    #   BORKED3DS_V3DV_TRACE_BLEND / TRACE_DISPLAY_TRANSFER : heavy traces are opt-in (TB34)
    #   V385_SPECIALISATION : vertex shader specialization (v385)
    #   V387_FS_NO_ROBUST / BORKED3DS_V3DV_V387_FS_ROBUST : non-robust fragment shaders (v388)
    #   V389_DEFAUTS_VULKAN : Vulkan defaults set by the program (v389)
    #   V390_PRIORITE_COMPILATION : compile threads run at background priority (v390)
    #   V391_ECLAIRAGE_ALLEGE : reduced lighting for Luigi's Mansion 2 only (v391)
    #   V393_OMBRES : PICA shadow maps in Vulkan, opt-in with BORKED3DS_V3DV_V393_SHADOWS=1 (v393/v394)
    #   V394_LOGICOP : logic op NoOp masks color writes in strict-compat (Mario 3D Land shadows) (v394)
    #   V395_AZAHAR_LOT_A : small Azahar fixes (KeepAll2 cull mode, zero-area draws, invalid vertex arrays, null cube units, malformed GS, FillScreen, present sampler) (v395)
    #   V396_LOT_C : identical PICA shader words no longer mark the program dirty; SIMD index min/max (v396)
    #   V397_CHEMIN_CHAUD : env lookups cached without std::string; descriptor writes batched in strict-compat (v397)
    # To add a marker, append it to borked3ds_markers.
    ################################

    echo ""
    echo "=========================================================="
    echo "VERIFICATION DU BINAIRE INSTALLE"
    echo "=========================================================="
    if [ -f "$md_build/.borked3ds_commit" ]; then
        echo "Commit compile : $(cat "$md_build/.borked3ds_commit")"
        cp "$md_build/.borked3ds_commit" "$md_inst/.borked3ds_commit"
    else
        echo "Commit compile : INCONNU"
    fi

    local borked3ds_markers=(
        "TRACE_DISPLAY_TRANSFER src="
        "shifts the bottom screen"
        "BORKED3DS_V3DV_TRACE_SCREEN_RECT"
        "BORKED3DS_V3DV_DIRA_SW_FALLBACK"
        "BORKED3DS_V3DV_TRACE_SYNC"
        "TRACE_SYNC finish="
        "BORKED3DS_V3DV_STRICT_SERIALIZE_SW_DRAWS"
        "BORKED3DS_V3DV_STRICT_FLUSH_SW_DRAWS"
        "v3dv_zband"
        "streambuf_wait="
        "TRACE_PIPELINE_BUILD compile="
        "TRACE_PIPELINE_POISON hash="
        "BORKED3DS_V3DV_DISABLE_EDS"
        "BORKED3DS_V3DV_DIRA_WIDE"
        "BORKED3DS_V3DV_DIRA_ALL"
        "TRACE_VSDECIDE main_offset="
        "BORKED3DS_V3DV_A7Z12_FRAME_CENSUS"
        "swhist_le8="
        "rp_switch="
        "BORKED3DS_V3DV_MIN_DRAWS_TO_FLUSH"
        "BORKED3DS_V3DV_DISABLE_RENDERPASS_FLUSH"
        "cpu_pct="
        "sub_lag="
        "f_fb="
        "seq_count="
        "A7Z12_FB_IDENT"
        "c_addr="
        "A7Z12_RP_END_SITE"
        "BORKED3DS_V3DV_DISABLE_LAZY_COPY_VIEW"
        "BORKED3DS_V3DV_TRACE_BLEND"
        "BORKED3DS_V3DV_TRACE_DISPLAY_TRANSFER"
        "V385_SPECIALISATION programme="
        "V387_FS_NO_ROBUST actif"
        "BORKED3DS_V3DV_V387_FS_ROBUST"
        "V389_DEFAUTS_VULKAN"
        "V390_PRIORITE_COMPILATION"
        "V391_ECLAIRAGE_ALLEGE"
        "V393_OMBRES actif="
        "V394_LOGICOP actif="
        "V395_AZAHAR_LOT_A actif="
        "V396_LOT_C actif="
        "V397_CHEMIN_CHAUD actif="
    )

    local borked3ds_missing=0
    local m
    for m in "${borked3ds_markers[@]}"; do
        if strings -a "$md_inst/borked3ds" | grep -aqF "$m"; then
            printf "  OK      %s\n" "$m"
        else
            printf "  ABSENT  %s\n" "$m"
            borked3ds_missing=1
        fi
    done

    # Removed code: finding any of these strings means the binary predates v389.
    local borked3ds_removed=(
        "BORKED3DS_V3DV_V382_TEXSYNC"
        "BORKED3DS_V3DV_V386_EZ"
        "BORKED3DS_V3DV_V386_SKIP_DARK"
        "V386_LUMIERES"
        "BORKED3DS_V3DV_DIRA_Z_BIAS"
        "BORKED3DS_V3DV_DIRA_FULLSCREEN_TRI"
        "BORKED3DS_V3DV_DIRA_FORCE_DYNSTATE"
    )
    for m in "${borked3ds_removed[@]}"; do
        if strings -a "$md_inst/borked3ds" | grep -aqF "$m"; then
            printf "  PERIME  %s (devrait avoir disparu)\n" "$m"
            borked3ds_missing=1
        fi
    done

    if [ "$borked3ds_missing" -ne 0 ]; then
        echo ""
        echo "!! BINAIRE NON CONFORME -- tout releve fait avec celui-ci serait invalide."
        echo "!! Verifier que les fichiers patches ont bien ete pousses avant le build."
    else
        echo ""
        echo "Binaire conforme."
    fi

    ################################
    # VULKAN SHADER CACHE (v392): KEPT ACROSS REBUILDS
    #
    # The caches survive a rebuild, so areas already visited in a game never recompile
    # (Luigi's Mansion 2 needs 1-2 s per new pipeline):
    #   - <cache>/*.bin      : VkPipelineCache, validated by the driver itself (vendor, device,
    #                          pipelineCacheUUID); a Mesa update simply invalidates it.
    #   - <cache>/spirv/*.spv: GLSL -> SPIR-V results, keyed by a hash of the GLSL text, so a
    #                          change in the shader generators produces new keys, never stale hits.
    # Only the GLSL -> SPIR-V conversion itself can make old .spv files wrong (glslang version or
    # vk_shader_util.cpp options). The spirv/ directory is therefore purged only when that key
    # changes. BORKED3DS_PURGE_SHADER_CACHE=1 forces a full purge (cold measurements).
    ################################
    local vk_cache="$user_home/.local/share/borked3ds-emu/shaders/vulkan"
    local spirv_key
    spirv_key="$(git -C "$md_build" rev-parse HEAD:externals/glslang 2>/dev/null)-$(md5sum "$md_build/src/video_core/renderer_vulkan/vk_shader_util.cpp" 2>/dev/null | cut -c1-32)"
    if [ -n "${BORKED3DS_PURGE_SHADER_CACHE:-}" ]; then
        rm -rf "$vk_cache"
        echo "Vulkan shader cache: full purge (BORKED3DS_PURGE_SHADER_CACHE)."
    elif [ -d "$vk_cache" ]; then
        if [ "$(cat "$vk_cache/.spirv_key" 2>/dev/null)" != "$spirv_key" ]; then
            rm -rf "$vk_cache/spirv"
            echo "Vulkan shader cache: kept, SPIR-V part purged (GLSL -> SPIR-V conversion changed)."
        else
            echo "Vulkan shader cache: kept ($(ls "$vk_cache"/spirv 2>/dev/null | wc -l) SPIR-V files)."
        fi
    fi
    mkdir -p "$vk_cache"
    echo "$spirv_key" > "$vk_cache/.spirv_key"
    chown -R "${__user:-pi}": "$user_home/.local/share/borked3ds-emu/shaders" 2>/dev/null
    echo "=========================================================="
    echo ""
}

function configure_borked3ds() {

    mkRomDir "3ds"

    ################################
    # LAUNCH LINES (v389)
    #
    # The tuned settings no longer live in emulators.cfg:
    #   - BORKED3DS_V3DV_* variables are set by the program when a game starts, only when
    #     the selected API is Vulkan (under OpenGL they make the emulator exit); see
    #     V389ApplyV3dvDefaults() in src/borked3ds_qt/main.cpp, log line V389_DEFAUTS_VULKAN;
    #   - launcher environment (xcb, SDL, GL_OES_texture_buffer, Vulkan layers) is set at
    #     the start of main();
    #   - graphics_api=Vulkan and use_disk_shader_cache=true are code defaults (settings.h).
    # A variable set by hand on a line still takes priority (tests).
    # Escape hatch: BORKED3DS_V3DV_NO_DEFAULTS=1.
    #
    # Never add V3D_DEBUG=opt_compile_time back: with non-robust fragment shaders (v388)
    # it makes V3D spill registers (Luigi's Mansion 2: 38 -> 126 ms per frame).
    #
    # Measurement probes (no longer set by default), to add by hand on a test line:
    #   BORKED3DS_V3DV_A7Z12_FRAME_CENSUS=1 BORKED3DS_V3DV_A7Z12_CENSUS_PERIOD=61
    #   BORKED3DS_V3DV_TRACE_PIPELINE_BUILD=1
    # The census is logged at Info level: log_filter must include Render.Vulkan:Info.
    #
    # Three lines:
    #   borked3ds          : game, OpenGL or Vulkan depending on the setting (default)
    #   borked3ds-ui       : Qt interface only
    #   borked3ds-ui-qt06  : Qt interface only, 0.6 scale (small screens)
    # Old test lines (borked3ds_*) are removed.
    ################################

    addEmulator 1 "$md_id" "3ds" "XINIT-WM:$md_inst/borked3ds -f %ROM%"
    addEmulator 0 "${md_id}-ui" "3ds" "XINIT-WMC:$md_inst/borked3ds"
    addEmulator 0 "${md_id}-ui-qt06" "3ds" "XINIT-WMC:QT_SCALE_FACTOR=0.6 $md_inst/borked3ds"

    local _emucfg="$configdir/3ds/emulators.cfg"
    if [[ -f "$_emucfg" ]]; then
        sed -i '/^borked3ds_/d' "$_emucfg"
    fi

    addSystem "3ds"

    echo ""
    echo "Ligne de lancement installee :"
    grep -a "^borked3ds" "$configdir/3ds/emulators.cfg" 2>/dev/null || \
        echo "  (introuvable -- verifier $configdir/3ds/emulators.cfg)"
    echo ""
}
