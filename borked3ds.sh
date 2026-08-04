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
    # SOURCE DES SOURCES
    #
    # DEFAUT = clone GitHub. Le depot DTEAM-1/Borked3DS-rpi est la source de verite du
    # projet : le flux normal est editer -> pousser -> compiler. Ne PAS privilegier un
    # arbre local automatiquement, sinon un build compilerait silencieusement une copie
    # en retard sur le depot -- exactement le meme genre de piege, en sens inverse.
    #
    # Pour compiler un arbre local (test rapide sans pousser), le demander EXPLICITEMENT :
    #     BORKED3DS_LOCAL_SRC=/home/pi/Borked3DS-rpi-master sudo -E ./retropie_setup.sh
    #
    # Dans les deux cas la source retenue est affichee en clair. Ne jamais lancer un build
    # sans avoir lu cette ligne.
    ################################

    local local_src="${BORKED3DS_LOCAL_SRC:-}"

    ################################
    # NETTOYAGE DU REPERTOIRE DE BUILD
    #
    # Probleme d'origine : "rm -rf $md_build/*" utilise un glob, qui n'inclut PAS les
    # fichiers caches. Le .git du clone precedent survivait au nettoyage, et le
    # "git clone" suivant echouait avec :
    #     fatal: destination path '...' already exists and is not an empty directory
    #
    # PIEGE (rencontre v157) : ne PAS corriger avec "rm -rf $md_build". RetroPie-Setup
    # fait un pushd DANS $md_build avant d'appeler cette fonction : supprimer le
    # repertoire detruit le repertoire courant du shell. Symptomes observes :
    #     fatal: Unable to read current working directory: No such file or directory
    #     fatal: remote helper 'https' aborted session
    #   puis "git -C" remonte au depot RetroPie-Setup (faux numero de commit affiche),
    #   puis CMake echoue : "does not appear to contain CMakeLists.txt".
    #
    # Solution : VIDER le repertoire sans le supprimer. find -mindepth 1 traite aussi
    # les fichiers caches (.git inclus) et laisse le repertoire -- donc le CWD -- intact.
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
        # CLONE AVEC REESSAIS ET ARRET IMMEDIAT EN CAS D'ECHEC
        #
        # PIEGE AVERE (03/08/2026) : un echec reseau ("Recv failure: Connection reset
        # by peer") laissait le script CONTINUER. Consequences observees dans le log :
        #   - les sed -Werror et les fixes source s'appliquaient dans le vide ;
        #   - "git -C $md_build rev-parse" remontait au depot PARENT (RetroPie-Setup)
        #     et affichait "Commit compile : dd6475c2", un commit qui n'a rien a voir
        #     avec le fork -- donc un faux numero de commit dans les traces de test.
        # Seule la garde de sortie en fin de fonction arretait finalement le build.
        #
        # Correctif : verifier le code de retour du clone, reessayer (les coupures
        # GitHub sont souvent transitoires), et SORTIR si les tentatives echouent.
        # Le repertoire est vide entre deux tentatives, sinon git refuse de cloner.
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

        # Garde : sans CMakeLists.txt, le clone est incomplet meme s'il a "reussi".
        if [ ! -f "$md_build/CMakeLists.txt" ]; then
            echo "!! Clone incomplet : CMakeLists.txt absent. Abandon."
            exit 1
        fi

        git submodule update --init --recursive

        # Le commit est lu DANS le depot clone. Si -C echouait, on remonterait au
        # depot parent et on afficherait un commit etranger : on verifie donc que le
        # repertoire est bien la racine d'un depot git avant de s'y fier.
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
    # Le fork pose -Werror dans CMakeLists.txt / fichiers cmake, ce qui fait rejeter par
    # Clang du code que GCC tolere. CMAKE_CXX_FLAGS ne peut pas annuler un -Werror pose
    # par target_compile_options() : les flags de cible priment sur la ligne de commande.
    #
    # Fix 1 : retirer -Werror de TOUS les fichiers cmake apres le clone.
    # Fix 2 : patcher directement les deux fichiers source problematiques.
    #
    # CORRECTIF (etape 2) : le troisieme sed etait "s/-Werror//g", trop agressif -- il
    # transformait "-Werror=return-type" en "=return-type", c'est-a-dire un flag corrompu
    # passe au compilateur. Le motif epargne desormais les formes "-Werror=...".
    #
    # DETTE (etape 6 / packaging) : les deux fixes source ci-dessous sont appliques par
    # sed APRES clone, donc absents du depot -- le depot ne compile pas seul. Pire, si le
    # motif change en amont, le sed devient un no-op SILENCIEUX. A pousser dans le depot,
    # puis retirer d'ici.
    ################################

    find "$md_build" \( -name "CMakeLists.txt" -o -name "*.cmake" \) | \
        xargs grep -l "\-Werror" 2>/dev/null | while read -r f; do
        echo "Patching -Werror out of: $f"
        sed -i 's/[[:space:]]-Werror[[:space:]]/ /g' "$f"
        sed -i 's/[[:space:]]-Werror"/ "/g' "$f"
        sed -i 's/-Werror\([^=]\)/\1/g; s/-Werror$//' "$f"
    done

    # Controle : plus aucun -Werror "nu" ne doit subsister (les -Werror=... sont legitimes).
    local werror_left
    werror_left="$(grep -rn -- "-Werror" "$md_build" --include="CMakeLists.txt" --include="*.cmake" 2>/dev/null | grep -v -- "-Werror=" | wc -l)"
    if [ "$werror_left" -ne 0 ]; then
        echo "ATTENTION : $werror_left occurrence(s) de -Werror subsistent dans les fichiers cmake."
    else
        echo "-Werror nu : aucune occurrence restante."
    fi

    # Source fix 1: glsl_fs_shader_gen.cpp
    # logical '||' with constant operand — GL_SHADER_IMAGE_ATOMIC est une constante int.
    # Clang le rejette ; '|' bitwise est semantiquement identique ici.
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
    # fonction 'MakeBlackAlpha' inutilisee — [[nodiscard]] -> [[maybe_unused]]
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
    # GARDE DE SORTIE
    #
    # Verifie que l'arbre source est reellement en place AVANT de rendre la main a
    # build_borked3ds(). Sans cela, un clone avorte se manifeste plus tard par une erreur
    # CMake obscure ("does not appear to contain CMakeLists.txt"), plusieurs etapes apres
    # la vraie cause.
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

    # BUG FIX: Borked3DS Vulkan plante s'il est compile avec GCC sur Pi5.
    # Note amont : "Vulkan may crash if the executable was compiled with GCC."
    # On utilise Clang. Installation : sudo apt install clang
    # Note : -Werror est retire des fichiers cmake dans sources_borked3ds() ci-dessus.
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

    # BUG FIX: le binaire CLI s'appelle borked3ds, pas borked3ds-qt
    if [ -f "$md_build/build/bin/Release/borked3ds" ]; then
        cp "$md_build/build/bin/Release/borked3ds" "$md_inst/borked3ds"
        chmod +x "$md_inst/borked3ds"
    else
        echo "Binary borked3ds missing — build may have failed"
        exit 1
    fi

    ################################
    # HOME UTILISATEUR
    #
    # Le scriptmodule s'execute sous root : $HOME vaut /root, pas /home/pi. Les fichiers
    # crees plus bas doivent aller dans le home de l'utilisateur, sinon l'emulateur ne
    # les trouve jamais. RetroPie-Setup expose $home et $__user pour ca.
    ################################
    local user_home="${home:-/home/${__user:-pi}}"

    ################################
    # NAND/SaveData minimal requis
    # archive_source_sd_savedata.cpp monte le SaveData dans :
    # sdmc/.../title/{high}/{low}/data/00000001/
    # Certains jeux (ex: Sonic Lost World) lisent network_id.dat au demarrage.
    # Si absent -> FILE_NOT_FOUND non gere -> crash silencieux dans le thread ARM.
    #
    # DETTE (etape 6 / packaging) : ce bloc est specifique aux jeux de test du projet.
    # Il n'a rien a faire dans un package RetroPie distribuable -- a sortir avant
    # publication (ou a conditionner a une variable de developpement).
    ################################

    local sdmc_base="$user_home/.local/share/borked3ds-emu/sdmc/Nintendo 3DS"
    local sdmc_id0="00000000000000000000000000000000"
    local sdmc_id1="00000000000000000000000000000000"
    local sdmc="$sdmc_base/$sdmc_id0/$sdmc_id1"

    # Sonic Lost World US (00040000000C8C00) — network_id.dat
    # 16 octets : LocalFriendCodeSeed (8 octets non nuls) + NetworkID (8 octets)
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

    # Sonic Lost World EU (00040000000C8D00) — meme structure
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
    # VERIFICATION POST-INSTALLATION
    #
    # But : rendre impossible un releve fait sur un binaire perime. Chaque marqueur
    # attendu est cherche dans le binaire installe. Un seul ABSENT invalide le cycle
    # de test -- ne pas lancer le jeu, reprendre au push.
    #
    # Pour ajouter un marqueur au fil des sessions, l'ecrire dans borked3ds_markers.
    #
    # v158 : A7Z12_FRAME_CENSUS avait ete ajoute au code SANS entrer ici -- la
    # verification post-installation ne le couvrait donc pas, et un binaire perime
    # serait passe "conforme" en ne produisant simplement aucun recensement.
    # Deux marqueurs sont poses plutot qu'un :
    #   BORKED3DS_V3DV_A7Z12_FRAME_CENSUS -> la sonde existe (depuis 9f7b67e) ;
    #   swhist_le8=                       -> c'est bien la version v158 du census
    #                                        (sommets + histogramme + frame_us).
    # Le second est le seul discriminant : sans lui, un binaire 342556a repondrait
    # OK au premier et le releve serait fait avec une sonde a 9 champs.
    #
    # TB14/TB15 : "rp_switch=" atteste du census enrichi des compteurs de render pass
    # (rp_begin/rp_switch/rp_area/rp_end/rp_flush). Les deux variables MIN_DRAWS_TO_FLUSH
    # et DISABLE_RENDERPASS_FLUSH attestent du seuil de flush rendu reglable dans
    # vk_render_manager.cpp. Sans "rp_switch=", le binaire est anterieur a TB14 et
    # le releve ne produirait aucun compteur -- silencieusement.
    #
    # TB16 : "cpu_pct=" atteste de la sonde d'occupation CPU de l'EmuThread
    # (cpu_us/wall_us/cpu_pct/pframes/tid). Elle tranche CPU-bound vs attente.
    #
    # TB24 : "sub_lag=" atteste des compteurs de soumission GPU
    # (sub_n/sub_us/sub_max_us/sub_lag), poses dans vk_master_semaphore.cpp.
    #
    # TB26 : "f_fb=" atteste de la decomposition des causes de bascule de render pass
    # (d_fb/d_rp/d_ar/d_cl, f_fb/f_rp/f_ar/f_cl, fbn), posee dans vk_render_manager.cpp.
    #
    # TB27 : "seq_count=" atteste de la mesure de faisabilite du regroupement -- longueur
    # des runs consecutifs par cible (seq_count/seq_draws) et histogramme par framebuffer
    # (fbh0..fbh5), pose dans vk_render_manager.cpp. Sans lui, le binaire est anterieur a
    # TB27 et le releve ne produirait aucun de ces champs -- silencieusement.
    #
    # TB28a/b : "A7Z12_FB_IDENT" atteste de l'identification des cibles de rendu
    # (color_id/depth_id/shadow/dimensions/formats par framebuffer), posee dans
    # vk_render_manager.cpp et emise par le census. C'est elle qui dit si les cibles
    # jumelles de TB27 sont la meme surface ou deux cibles distinctes. "c_addr=" atteste
    # de TB28b : adresses physiques 3DS par cible (discriminant stereo gauche/droite).
    #
    # TB32 : "A7Z12_RP_END_SITE" atteste du comptage des fermetures de render pass PAR
    # SITE D'APPEL. Cette sonde a montre que les 168 bascules "f_rp" ne sont pas des
    # changements de cible mais des fermetures forcees par du code hors chemin de draw.
    # Toujours actif avec le census, cout negligeable (quelques lignes par periode).
    #
    # TB33 : "BORKED3DS_V3DV_DISABLE_LAZY_COPY_VIEW" atteste du correctif MAJEUR de
    # l'axe B, desormais actif PAR DEFAUT (la chaine n'est plus qu'une echappatoire).
    # Surface::CopyImageView() ferme le render pass et blitte l'image entiere ; elle
    # etait appelee pour CHAQUE texture de CHAQUE draw alors que son resultat n'est
    # utilise qu'en cas de feedback direct. TB32 : 303 fermetures/frame sur 315.
    # Mesure : Metroid 49,3 % -> 88,7 %, garde-fou non-regression passe sur les trois
    # temoins. Si ce marqueur disparait, le correctif a ete perdu par un re-upload.
    #
    # TB34 : "BORKED3DS_V3DV_TRACE_BLEND" et "BORKED3DS_V3DV_TRACE_DISPLAY_TRANSFER"
    # attestent que les deux traces lourdes sont desormais OPT-IN. Elles etaient
    # inconditionnelles (~266 lignes/s, 11 Mo/session). Depuis TB33 le CPU est le mur
    # (cpu_pct=99), donc toute charge CPU retiree se lit directement en vitesse.
    # Gater l'emission ne retire pas les chaines du binaire : le marqueur
    # "TRACE_DISPLAY_TRANSFER src=" ci-dessus reste valide.
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

    ################################
    # Sondes retirees : leur presence signale un binaire perime.
    #
    # ETAPE 3 (nettoyage) -- a deplacer ICI une fois le code nettoye :
    #   BORKED3DS_V3DV_A7Z8_DUMP_ALL_SPIRV      (piste precision fp16 : close)
    #   BORKED3DS_V3DV_A7Z9_DUMP_COMPILESPV     (piste precision fp16 : close)
    #   BORKED3DS_V3DV_A7Z10_CLEAR_NEW_SURFACES (flash : parque, sonde sans effet)
    #   BORKED3DS_V3DV_A7Z11_TRACE_COLOR_ALLOC  (flash : parque, sonde sans effet)
    #   BORKED3DS_V3DV_DIRA_WIDE / DIRA_ALL     (a retirer aussi des marqueurs ci-dessus)
    # Tant qu'elles sont dans le code, elles doivent RESTER hors de cette liste, sinon
    # chaque build sera declare non conforme.
    ################################
    local borked3ds_removed=(
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
    # PURGE DU CACHE SHADER VULKAN
    # Systematique : un cache issu du binaire precedent fausse le premier lancement.
    ################################
    rm -rf "$user_home/.local/share/borked3ds-emu/shaders/vulkan"
    echo "Cache shader Vulkan purge."
    echo "=========================================================="
    echo ""
}

function configure_borked3ds() {

    mkRomDir "3ds"

    ################################
    # LIGNE DE LANCEMENT
    #
    # Nommee "borked3ds_test4" : c'est ce que le runcommand appelle.
    #
    # ATTENTION -- PIEGE AVERE : cette fonction REECRIT emulators.cfg a chaque
    # installation. Une variable ajoutee a la main dans emulators.cfg est donc
    # perdue au build suivant. Toute nouvelle sonde doit etre ajoutee ICI, dans
    # cette ligne, sinon elle sera compilee dans le binaire mais jamais activee.
    # Cas reels : TRACE_SYNC (un cycle de test perdu), puis A7Z8/A7Z10 (sondes
    # compilees mais silencieusement inactives au run suivant le rebuild).
    #
    # ------------------------------------------------------------------
    # EXCEPTION ASSUMEE (v160) : BORKED3DS_V3DV_A7Z12_FRAME_CENSUS=1 est desormais
    # DANS cette ligne. Motif : le census est l'instrument de mesure de reference du
    # projet, et son absence apres un rebuild a coute deux cycles de test dans la
    # meme session (le jeu tourne, le log s'ecrit, mais zero ligne census -- panne
    # silencieuse). Son cout est negligeable (~7 lignes/s, periode 60 frames).
    # A RETIRER a l'etape de livraison, quand les mesures seront closes.
    #
    # Ligne v157 -- BASELINE PROPRE. Ne contient AUCUNE autre sonde diagnostique.
    #
    # Regle : les sondes se posent a la main pour un test, JAMAIS ici. En
    # particulier, ne jamais laisser dans cette ligne :
    #   BORKED3DS_V3DV_TRACE_DRAW=1  -- ~19 000 lignes de log/seconde, ralentit
    #     fortement le jeu. Cause de la fausse "regression de vitesse" v156.
    #   BORKED3DS_V3DV_SOFTWARE_CLEAR_TILE_BUDGET / SAFE_UNTEXTURED_DRAW_BUDGET
    #     portes a 1e9 -- font executer tous les clears et draws software
    #     auparavant consommes en no-op : cout reel.
    #
    # Etat des artefacts visuels a la v157 :
    #   - Banding (boule Sonic) : PARQUE. VS et FS prouves fp32 (dump SPIR-V :
    #     RelaxedPrecision=0, OpTypeFloat 16=0), relaxed_precision off, present
    #     en 8 bits, Mesa 26.1.2 deja au plus recent. Cause interne a V3D,
    #     non corrigeable dans nos shaders. Present a l'identique sur gvx64.
    #   - Flash au demarrage 3D (one-shot, plein ecran, jaune sous Metroid) :
    #     PARQUE. Cosmetique, une seule occurrence par session. Hypotheses
    #     eliminees : async_shader_compilation, surface couleur non initialisee
    #     (A7Z10), budgets cumulatifs, clears de debug present (opt-in, inactifs).
    #
    # Ligne v152 :
    #   DIRA_MAX_VERTICES 128 -> 65536 : le plafond de 128 sommets coupait le draw
    #   d'un calque clair de Sonic Lost World (rendu "trop fonce"). Mesure : essai
    #   a 65536 = calque revenu, aucun impact vitesse. L'autre essai (retrait
    #   complet de DIRA_SW_FALLBACK) a echoue : perte de texte + lenteur.
    #
    # Ligne v151 (figement RESOLU) :
    #   L'extended dynamic state est ACTIF PAR DEFAUT sous strict-compat (voir
    #   vk_graphics_pipeline.cpp). Mesure Kid Icarus : pire compilation de pipeline
    #   12993 ms -> ~51 ms, 43 pipelines "poison" -> 0, plus aucun gel.
    #   Echappatoire sans rebuild : ajouter BORKED3DS_V3DV_DISABLE_EDS=1 a la main.
    #   Le bug de coherence hash/build qui bloquait ce chantier est RESOLU :
    #   Hash() et Build() partagent desormais la meme expression de gating.
    #
    # v148 avait corrige SAFE_PICA_HW_DRAW_BUDGET=1 / MAX_VERTICES=6, valeurs de
    # bisection oubliees qui bridaient le rendu, portees a 9999 / 65536.
    ################################

    addEmulator 1 "${md_id}_test4" "3ds" "XINIT-WM:QT_QPA_PLATFORM=xcb QT_SCALE_FACTOR=0.6 SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0 MESA_EXTENSION_OVERRIDE=GL_OES_texture_buffer VK_INSTANCE_LAYERS= VK_LAYER_PATH= BORKED3DS_V3DV_STRICT_COMPAT=1 BORKED3DS_V3DV_HIGH_SWITCH=1 BORKED3DS_V3DV_ALLOW_PICA_ACCEL=1 BORKED3DS_V3DV_ALLOW_SOFTWARE_TEXTURES=1 BORKED3DS_V3DV_DISABLE_SOFTWARE_QUARANTINE=1 BORKED3DS_V3DV_DIRA_SW_FALLBACK=1 BORKED3DS_V3DV_A7Z71_PICA_TRIGGER_SILENT_DRAWARRAYS=1 BORKED3DS_V3DV_A7Z72_PICA_DRAWARRAYS_SILENT_EARLY_BACKEND=1 BORKED3DS_V3DV_A7Z73_SUPPRESS_RAW_ENTER_SIMPLE_LOG=1 BORKED3DS_V3DV_A7Z74_SILENT_OUTER_ENTRY_TO_STAGE=1 BORKED3DS_V3DV_DISABLE_ACCEL_INTERNAL_DRY_RUN=1 BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF=1 BORKED3DS_V3DV_DIRECT_SAFE_HW_HANDOFF_NO_PRELOG=1 BORKED3DS_V3DV_ALLOW_SAFE_PICA_HW_DRAWS=1 BORKED3DS_V3DV_ENTER_SAFE_PICA_HW_DRAWS=1 BORKED3DS_V3DV_SAFE_PICA_HW_DRAW_BUDGET=9999 BORKED3DS_V3DV_SAFE_PICA_HW_MAX_VERTICES=65536 BORKED3DS_V3DV_A7Z12_FRAME_CENSUS=1 $md_inst/borked3ds %ROM%"

    addSystem "3ds"

    echo ""
    echo "Ligne de lancement installee :"
    grep -a "^borked3ds" "$configdir/3ds/emulators.cfg" 2>/dev/null || \
        echo "  (introuvable -- verifier $configdir/3ds/emulators.cfg)"
    echo ""
}
