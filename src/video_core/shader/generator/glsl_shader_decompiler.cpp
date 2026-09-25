// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/common_types.h"
#include "video_core/shader/generator/glsl_shader_decompiler.h"

namespace Pica::Shader::Generator::GLSL {

using nihstro::DestRegister;
using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

constexpr u32 PROGRAM_END = MAX_PROGRAM_CODE_LENGTH;

class DecompileFail : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// v364 -- plafonds des garde-fous, lus une seule fois. 0 = illimite (comportement d'origine).
[[nodiscard]] u32 MaxSubroutines() {
    static const u32 cap = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_MAX_SUBROUTINES");
        if (v == nullptr || v[0] == '\0') {
            return 1024u;
        }
        return static_cast<u32>(std::strtoul(v, nullptr, 10));
    }();
    return cap;
}

[[nodiscard]] std::size_t MaxShaderBytes() {
    static const std::size_t cap = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_MAX_SHADER_BYTES");
        if (v == nullptr || v[0] == '\0') {
            return std::size_t{16} * 1024 * 1024;
        }
        return static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
    }();
    return cap;
}

/// v377 -- forme des vertex shaders PICA traduits.
///
/// Mesure au banc v3dbench (Pi 5, Mesa 26.1.2, shader poison de Sonic, 26 900 mots SPIR-V) :
///   - forme d'origine (boucle `while (true) { switch (jmp_to) }` + lecture d'uniforme gardee par
///     un `if`) : 17 a 77 s en jeu, et plantage du pilote dans le banc ;
///   - chaine acyclique `if (jmp_to == ...)` + lecture sans branche : 2,0 s, calcul identique.
/// La boucle d'aiguillage oblige V3D a garder tous les registres PICA vivants a chaque tour ;
/// les 84 `if` de controle de bornes coupent le code en blocs et empechent l'ordonnancement
/// des lectures TMU. Les deux causes font echouer l'allocation de registres a repetition.
///
/// Par defaut : nouvelle forme. BORKED3DS_V3DV_V377_LEGACY_VS=1 rend la forme d'origine
/// (comparaison A/B ; ne jamais poser =0, l'absence de la variable suffit).
[[nodiscard]] bool V377LegacyVertexShaderForm() {
    static const bool legacy = std::getenv("BORKED3DS_V3DV_V377_LEGACY_VS") != nullptr;
    return legacy;
}

/// Describes the behaviour of code path of a given entry point and a return point.
enum class ExitMethod {
    Undetermined, ///< Internal value. Only occur when analyzing JMP loop.
    AlwaysReturn, ///< All code paths reach the return point.
    Conditional,  ///< Code path reaches the return point or an END instruction conditionally.
    AlwaysEnd,    ///< All code paths reach a END instruction.
};

/// A subroutine is a range of code refereced by a CALL, IF or LOOP instruction.
struct Subroutine {
    /// Generates a name suitable for GLSL source code.
    std::string GetName() const {
        return "sub_" + std::to_string(begin) + "_" + std::to_string(end);
    }

    u32 begin;              ///< Entry point of the subroutine.
    u32 end;                ///< Return point of the subroutine.
    ExitMethod exit_method; ///< Exit method of the subroutine.
    std::set<u32> labels;   ///< Addresses refereced by JMP instructions.

    bool operator<(const Subroutine& rhs) const {
        return std::tie(begin, end) < std::tie(rhs.begin, rhs.end);
    }
};

/// Analyzes shader code and produces a set of subroutines.
class ControlFlowAnalyzer {
public:
    ControlFlowAnalyzer(const ProgramCode& program_code, u32 main_offset)
        : program_code(program_code) {

        // Recursively finds all subroutines.
        const Subroutine& program_main = AddSubroutine(main_offset, PROGRAM_END);
        if (program_main.exit_method != ExitMethod::AlwaysEnd)
            throw DecompileFail("Program does not always end");
    }

    std::set<Subroutine> MoveSubroutines() {
        return std::move(subroutines);
    }

private:
    const ProgramCode& program_code;
    std::set<Subroutine> subroutines;
    std::map<std::pair<u32, u32>, ExitMethod> exit_method_map;

    /// Adds and analyzes a new subroutine if it is not added yet.
    const Subroutine& AddSubroutine(u32 begin, u32 end) {
        auto iter = subroutines.find(Subroutine{begin, end});
        if (iter != subroutines.end())
            return *iter;

        // v364 -- GARDE-FOU 1 : plafond du nombre de sous-routines.
        //
        // Les sous-routines sont indexees par le couple (begin, end) sur un espace de
        // MAX_PROGRAM_CODE_LENGTH^2 = 4096^2 = 16,7 MILLIONS de couples possibles, et chacune est
        // ensuite emise comme une fonction GLSL couvrant sa plage d'instructions. Sans plafond, un
        // programme PICA au flot de controle tres imbrique fait exploser l'enumeration : le source
        // GLSL genere passe de ~40 Ko a plusieurs gigaoctets. Mesure sur Pi5 (runs O a U, Kid
        // Icarus) : EmuThread a 4,4 Go residents et 10,9 Go virtuels, puis quatre demandes
        // refusees de 12 Gio + quelques pages (identiques a l'octet entre les runs), famine
        // memoire de toute la machine et mort de l'emulateur.
        //
        // Au-dela du plafond on leve DecompileFail : DecompileProgram le rattrape deja et renvoie
        // une chaine vide, et UseProgrammableVertexShader retombe alors sur le chemin logiciel
        // pour CETTE configuration uniquement. Tous les autres shaders restent inchanges.
        // Plafond ajustable par BORKED3DS_V3DV_MAX_SUBROUTINES (0 = illimite, comportement
        // d'origine). Les shaders sains en utilisent une poignee.
        if (const u32 cap = MaxSubroutines(); cap != 0 && subroutines.size() >= cap) {
            throw DecompileFail(
                fmt::format("Subroutine cap reached: {} subroutines, {} exit-method entries, "
                            "while adding range [{}, {}) -- runaway control-flow analysis",
                            subroutines.size(), exit_method_map.size(), begin, end));
        }

        Subroutine subroutine{begin, end};
        subroutine.exit_method = Scan(begin, end, subroutine.labels);
        if (subroutine.exit_method == ExitMethod::Undetermined)
            throw DecompileFail("Recursive function detected");
        return *subroutines.insert(std::move(subroutine)).first;
    }

    /// Merges exit method of two parallel branches.
    static ExitMethod ParallelExit(ExitMethod a, ExitMethod b) {
        if (a == ExitMethod::Undetermined) {
            return b;
        }
        if (b == ExitMethod::Undetermined) {
            return a;
        }
        if (a == b) {
            return a;
        }
        return ExitMethod::Conditional;
    }

    /// Cascades exit method of two blocks of code.
    static ExitMethod SeriesExit(ExitMethod a, ExitMethod b) {
        // This should be handled before evaluating b.
        DEBUG_ASSERT(a != ExitMethod::AlwaysEnd);

        if (a == ExitMethod::Undetermined) {
            return ExitMethod::Undetermined;
        }

        if (a == ExitMethod::AlwaysReturn) {
            return b;
        }

        if (b == ExitMethod::Undetermined || b == ExitMethod::AlwaysEnd) {
            return ExitMethod::AlwaysEnd;
        }

        return ExitMethod::Conditional;
    }

    /// Scans a range of code for labels and determines the exit method.
    ExitMethod Scan(u32 begin, u32 end, std::set<u32>& labels) {
        auto [iter, inserted] =
            exit_method_map.emplace(std::make_pair(begin, end), ExitMethod::Undetermined);
        ExitMethod& exit_method = iter->second;
        if (!inserted)
            return exit_method;

        for (u32 offset = begin; offset != end && offset != PROGRAM_END; ++offset) {
            const Instruction instr(program_code[offset]);
            switch (instr.opcode.Value()) {
            case OpCode::Id::END: {
                return exit_method = ExitMethod::AlwaysEnd;
            }
            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                labels.insert(instr.flow_control.dest_offset);
                ExitMethod no_jmp = Scan(offset + 1, end, labels);
                ExitMethod jmp = Scan(instr.flow_control.dest_offset, end, labels);
                return exit_method = ParallelExit(no_jmp, jmp);
            }
            case OpCode::Id::CALL: {
                auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                           instr.flow_control.dest_offset +
                                               instr.flow_control.num_instructions);
                if (call.exit_method == ExitMethod::AlwaysEnd)
                    return exit_method = ExitMethod::AlwaysEnd;
                ExitMethod after_call = Scan(offset + 1, end, labels);
                return exit_method = SeriesExit(call.exit_method, after_call);
            }
            case OpCode::Id::LOOP: {
                auto& loop = AddSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
                if (loop.exit_method == ExitMethod::AlwaysEnd)
                    return exit_method = ExitMethod::AlwaysEnd;
                ExitMethod after_loop = Scan(instr.flow_control.dest_offset + 1, end, labels);
                return exit_method = SeriesExit(loop.exit_method, after_loop);
            }
            case OpCode::Id::CALLC:
            case OpCode::Id::CALLU: {
                auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                           instr.flow_control.dest_offset +
                                               instr.flow_control.num_instructions);
                ExitMethod after_call = Scan(offset + 1, end, labels);
                return exit_method = SeriesExit(
                           ParallelExit(call.exit_method, ExitMethod::AlwaysReturn), after_call);
            }
            case OpCode::Id::IFU:
            case OpCode::Id::IFC: {
                auto& if_sub = AddSubroutine(offset + 1, instr.flow_control.dest_offset);
                ExitMethod else_method;
                if (instr.flow_control.num_instructions != 0) {
                    auto& else_sub = AddSubroutine(instr.flow_control.dest_offset,
                                                   instr.flow_control.dest_offset +
                                                       instr.flow_control.num_instructions);
                    else_method = else_sub.exit_method;
                } else {
                    else_method = ExitMethod::AlwaysReturn;
                }

                ExitMethod both = ParallelExit(if_sub.exit_method, else_method);
                if (both == ExitMethod::AlwaysEnd)
                    return exit_method = ExitMethod::AlwaysEnd;
                ExitMethod after_call =
                    Scan(instr.flow_control.dest_offset + instr.flow_control.num_instructions, end,
                         labels);
                return exit_method = SeriesExit(both, after_call);
            }
            default:
                break;
            }
        }
        return exit_method = ExitMethod::AlwaysReturn;
    }
};

// v117c-MIRROR (Plan A gating, refined): the V3DV low-bank mirror overwrites ONLY f[0..31], so a
// draw is corrupted iff it reads any uniform f[<32] (e.g. 3D transform matrices in the low slots).
// We therefore mirror a draw iff it (a) reads the upper bank f[64..95] via an address-register
// (dynamic) index -- the pattern routed to get_offset_register_sw -- AND (b) never reads any uniform
// below f[32]. The dialogue-glyph VS reads only f[32 + aL.x] (position) and f[64 + aL.y] (texcoord),
// so it qualifies; matrix-reading 3D shaders are excluded and left intact. Reachability is computed
// with the same ControlFlowAnalyzer used for generation, bounded by END, so unreachable tail words
// are never misdecoded.
// vDIRA (Direction A, v119) refactor: the per-bank uniform-read scan below used to live inside
// VertexShaderLowMirrorPlan. It is now shared between the mirror gate (v117c/v118) and the new
// software-VS-fallback gate (VertexShaderNeedsSoftwareVSFallback), so both make their decision from
// the SAME reachable-code analysis and can never disagree on what a VS reads. Behaviour of the
// mirror path is bit-for-bit identical to v118.
namespace {

struct UniformReadScan {
    bool analyzed;      ///< control-flow analysis succeeded; when false the sets are empty
    std::set<u32> low;  ///< f[<32] reads (static or dynamic): must be preserved by a mirror window
    std::set<u32> mid;  ///< f[32..63] (e.g. address-indexed position) and STATIC upper-bank reads
    std::set<u32> high; ///< f[64..95] reads via an address register: the pattern V3D miscompiles
    /// v131-MIRROR : bases INFERIEURES A 64 lues a travers un registre d'adresse. Une telle
    /// lecture balaie f[base + aL] : tous les slots AU-DESSUS de la base sont atteignables et ne
    /// peuvent donc pas servir de destination au miroir. v130 ne retenait que les bases, ce qui
    /// faisait passer pour libres des plages que le shader allait lire -- le miroir les ecrasait
    /// et un sommet partait a une position absurde (tige traversant le personnage, mesuree sur
    /// Sonic Lost World ET Metroid Samus Returns en v300).
    std::set<u32> dyn;
};

UniformReadScan ScanVertexShaderUniformReads(const ProgramCode& program_code, u32 main_offset) {
    UniformReadScan scan{};

    std::set<Subroutine> subroutines;
    try {
        subroutines = ControlFlowAnalyzer(program_code, main_offset).MoveSubroutines();
    } catch (const std::exception&) {
        // If control-flow analysis fails the generator falls back too; report "not analyzed" so
        // both gates (mirror and software fallback) stay conservative.
        scan.analyzed = false;
        return scan;
    }
    scan.analyzed = true;

    // Collect the exact uniform indices read, split by bank.
    const auto collect = [&](const SourceRegister& reg, u32 addr_index) {
        if (reg.GetRegisterType() != RegisterType::FloatUniform) {
            return;
        }
        const u32 base = static_cast<u32>(reg.GetIndex());
        if (addr_index != 0 && base < 64u) {
            // v131-MIRROR : lecture indexee sous 64 -- portee inconnue vers le haut.
            scan.dyn.insert(base);
        }
        if (base < 32) {
            scan.low.insert(base);
        } else if (addr_index != 0 && base >= 64) {
            scan.high.insert(base);
        } else {
            scan.mid.insert(base);
        }
    };
    for (const auto& sub : subroutines) {
        for (u32 offset = sub.begin; offset < sub.end && offset < PROGRAM_END; ++offset) {
            const Instruction instr = {program_code[offset]};
            if (instr.opcode.Value() == OpCode::Id::END) {
                break; // bounds the loose main subroutine range at its terminator
            }
            switch (instr.opcode.Value().GetInfo().type) {
            case OpCode::Type::Arithmetic: {
                const bool inv =
                    (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));
                collect(instr.common.GetSrc1(inv), !inv * instr.common.address_register_index);
                collect(instr.common.GetSrc2(inv), inv * instr.common.address_register_index);
                break;
            }
            case OpCode::Type::MultiplyAdd: {
                const bool inv = (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI);
                collect(instr.mad.GetSrc1(inv), 0);
                collect(instr.mad.GetSrc2(inv), !inv * instr.mad.address_register_index);
                collect(instr.mad.GetSrc3(inv), inv * instr.mad.address_register_index);
                break;
            }
            default:
                break;
            }
        }
    }

    // v118-DIAG: BORKED3DS_V3DV_TRACE_MIRROR_MAP=1 logs each distinct VS's read map once, so a hybrid
    // VS (low AND high) can be characterized and the chosen window verified. Pure diagnostic. The
    // seen-set dedups by main_offset, so a VS scanned by both gates is still logged only once.
    static const bool trace_mirror_map = std::getenv("BORKED3DS_V3DV_TRACE_MIRROR_MAP") != nullptr;
    if (trace_mirror_map) {
        static std::set<u32> seen_map_offsets;
        if (seen_map_offsets.insert(main_offset).second) {
            const auto join_set = [](const std::set<u32>& s) {
                std::string out;
                for (const u32 v : s) {
                    out += std::to_string(v);
                    out += ' ';
                }
                return out;
            };
            LOG_INFO(HW_GPU,
                     "TRACE_MIRROR_MAP main_offset={} low=[ {}] mid=[ {}] high_indexed=[ {}] "
                     "dyn_below64=[ {}]",
                     main_offset, join_set(scan.low), join_set(scan.mid), join_set(scan.high),
                     join_set(scan.dyn));
        }
    }

    return scan;
}

} // Anonymous namespace

LowMirrorPlan VertexShaderLowMirrorPlan(const ProgramCode& program_code, u32 main_offset) {
    // v118-MIRROR (Plan A, per-VS base): V3D miscompiles DYNAMIC indexed reads of the upper float
    // uniform bank f[64..95] (used by the dialogue-glyph texcoord), while it handles dynamic reads
    // of the lower banks correctly (the position already uses f[32 + aL.x]). The fix mirrors the
    // needed upper-bank slots into a CONTIGUOUS window the VS does not otherwise read, then
    // re-fetches them through a dynamic index into that window.
    //
    // v130-MIRROR : deux corrections, toutes deux dictees par la mesure v298.
    //
    //   (1) FENETRE SOURCE. v128 recopiait f[64 + i] et le shader relisait
    //       f[base + clamp(index - 64, 0, count-1)]. La base source etait codee en dur a 64.
    //       Or TRACE_MIRROR_MAP sur Sonic Lost World donne, pour les VS main_offset 457, 463 et
    //       468, high_indexed = { 79 80 81 82 83 84 }. Avec la source a 64, index - 64 vaut 15 a
    //       20, que le clamp ecrase a count-1 : les six emplacements de glyphes lisaient tous le
    //       MEME slot, et un slot faux. La base source devient donc min(scan.high).
    //
    //   (2) FENETRE DESTINATION. v128 la placait juste au-dessus de la plus haute lecture basse
    //       (base = highest_low + 1), ce qui ne laissait que 4 slots pour les VS 457/463/468
    //       (highest_low = 27) alors qu'il en faut 6. Mais la contrainte materielle porte sur
    //       f[64..95] SEULEMENT : la lecture dynamique de la banque mediane f[32..63] est le
    //       chemin que la position emprunte deja (f[32 + aL.x]) et que V3DV compile correctement.
    //       La destination est donc cherchee dans f[0..63] entier, hors lectures basses et hors
    //       lectures statiques inferieures a 64 : pour ces trois VS, mid = { 86..95 } uniquement,
    //       donc f[32..63] est entierement libre -- 32 slots contigus au lieu de 4.
    //
    // Comportement inchange quand BORKED3DS_V3DV_LOW_MIRROR n'est pas arme, et identique a v118
    // pour les VS purs texte dont la fenetre dynamique commence a 64.
    const LowMirrorPlan kNoMirror{false, 0, 0, 64};

    const UniformReadScan scan = ScanVertexShaderUniformReads(program_code, main_offset);
    if (!scan.analyzed) {
        // If control-flow analysis fails the generator falls back too; never mirror in that case.
        return kNoMirror;
    }

    if (scan.high.empty()) {
        return kNoMirror; // no dynamic upper-bank read -> nothing to mirror
    }

    // Fenetre source reellement necessaire, bornee a la banque haute.
    const u32 src_base = *scan.high.begin();
    const u32 src_span = *scan.high.rbegin() - src_base + 1u;
    const u32 src_room = 96u - src_base;

    // v118 a l'identique : VS dedie a la banque haute dont la fenetre commence a 64. Aucun autre
    // jeu ne change de comportement par ce patch.
    // v131-MIRROR : le cas v118 historique (VS dedie banque haute, fenetre f[0..31]) n'est
    // conserve que s'il est SUR, c'est-a-dire si aucune lecture indexee sous 64 ne peut balayer
    // f[0..31]. Sinon on retombe sur la recherche generale ci-dessous, qui refusera au besoin.
    if (scan.low.empty() && src_base == 64u && (scan.dyn.empty() || *scan.dyn.begin() >= 32u)) {
        return LowMirrorPlan{true, 0u, 32u, 64u};
    }

    // Un VS hybride (lectures basses ET hautes) reste derriere BORKED3DS_V3DV_HYBRID_MIRROR.
    // Justification v128 : v127 a mesure samples=0 pour chaque draw software-A8, donc le repli
    // logiciel ne sert pas ces draws ; le miroir est la seule voie qui reste pour le chemin
    // accelere. Le risque nomme par v128 -- ecraser des slots bas utilises par la 3D du VS
    // partage -- est fortement reduit ici puisque la destination evite desormais toute lecture
    // connue, basse comme mediane.
    if (!scan.low.empty()) {
        static const bool hybrid_mirror =
            std::getenv("BORKED3DS_V3DV_HYBRID_MIRROR") != nullptr;
        if (!hybrid_mirror) {
            return kNoMirror;
        }
    }

    // Plus grande plage contigue libre dans f[0..63].
    // v131-MIRROR : plafond de recherche. Une lecture indexee de base B balaie f[B + aL] vers le
    // HAUT, sans borne connue a la compilation. Tout slot >= min(dyn) est donc potentiellement lu
    // et ne peut pas servir de destination. v130 ignorait ce point : il annoncait free_len=36 sur
    // Sonic et 42 sur Metroid alors que la zone etait balayee, et le miroir ecrasait des donnees
    // de transformation -- d'ou la tige. Mieux vaut refuser un miroir que corrompre la geometrie.
    const u32 dyn_ceiling = scan.dyn.empty() ? 64u : *scan.dyn.begin();

    std::array<bool, 64> used{};
    for (const u32 v : scan.low) {
        if (v < 64u) {
            used[v] = true;
        }
    }
    for (const u32 v : scan.mid) {
        if (v < 64u) {
            used[v] = true;
        }
    }
    for (u32 i = dyn_ceiling; i < 64u; ++i) {
        used[i] = true;
    }
    u32 best_base = 0u;
    u32 best_len = 0u;
    u32 cur_base = 0u;
    u32 cur_len = 0u;
    for (u32 i = 0; i < 64u; ++i) {
        if (!used[i]) {
            if (cur_len == 0u) {
                cur_base = i;
            }
            ++cur_len;
            if (cur_len > best_len) {
                best_len = cur_len;
                best_base = cur_base;
            }
        } else {
            cur_len = 0u;
        }
    }

    // v132-MIRROR : une fenetre plus COURTE que la portee demandee n'est pas fausse, elle est
    // PARTIELLE -- et la distinction est decisive.
    //
    // Le shader lit uniforms.f[dst + clamp(index - src_base, 0, count-1)] et l'upload recopie
    // f[dst + i] = f[src_base + i]. Pour tout index couvert par la fenetre, la valeur rendue est
    // donc EXACTEMENT celle que le jeu demandait ; seuls les index au-dela sont ecrases par le
    // clamp. Cinq glyphes justes sur six, et non six faux.
    //
    // C'est ce qui separe ce cas de v128 : v128 cumulait une fenetre courte ET une base source
    // figee a 64, si bien que les six emplacements lisaient tous la meme valeur erronee. La base
    // source etant desormais correcte, tronquer degrade proprement au lieu de corrompre.
    //
    // Mesure v302, apres application du plafond dyn_ceiling : Metroid VS 75 obtient 6 slots sur 7
    // et Sonic VS 463 en obtient 5 sur 6 -- il manquait a chacun EXACTEMENT un slot. Refuser
    // transformait "presque tout le texte" en "rien du tout".
    //
    // On ne refuse donc plus que sous deux slots : a un seul slot le miroir renvoie une constante,
    // c'est-a-dire exactement le comportement gele qu'il est cense corriger, sans aucun gain.
    // Le plafond dyn_ceiling reste entier : la fenetre ne deborde jamais sur une plage balayee par
    // un registre d'adresse, donc aucune corruption de geometrie n'est reintroduite.
    if (best_len < 2u) {
        static const bool trace_mirror_map =
            std::getenv("BORKED3DS_V3DV_TRACE_MIRROR_MAP") != nullptr;
        if (trace_mirror_map) {
            static std::set<u32> seen_refused;
            if (seen_refused.insert(main_offset).second) {
                LOG_INFO(HW_GPU,
                         "v132 mirror REFUSED main_offset={} src_base={} src_span={} "
                         "best_free_base={} best_free_len={} dyn_ceiling={}",
                         main_offset, src_base, src_span, best_base, best_len, dyn_ceiling);
            }
        }
        return kNoMirror;
    }

    const u32 count = std::min(best_len, src_room);

    // One-shot log per distinct VS so the applied plan is visible in the field.
    static const bool trace_mirror_map =
        std::getenv("BORKED3DS_V3DV_TRACE_MIRROR_MAP") != nullptr;
    if (trace_mirror_map) {
        static std::set<u32> seen_plan;
        if (seen_plan.insert(main_offset).second) {
            LOG_INFO(HW_GPU,
                     "v132 mirror plan main_offset={} src_base={} src_span={} dst_base={} "
                     "count={} free_len={} dyn_ceiling={} covered={} missing={} partial={} "
                     "hybrid={}",
                     main_offset, src_base, src_span, best_base, count, best_len, dyn_ceiling,
                     std::min(count, src_span), src_span > count ? src_span - count : 0u,
                     static_cast<u32>(count < src_span), static_cast<u32>(!scan.low.empty()));
        }
    }
    return LowMirrorPlan{true, best_base, count, src_base};
}

bool VertexShaderWantsLowMirror(const ProgramCode& program_code, u32 main_offset) {
    return VertexShaderLowMirrorPlan(program_code, main_offset).ok;
}

bool VertexShaderNeedsSoftwareVSFallback(const ProgramCode& program_code, u32 main_offset) {
    // vDIRA (Direction A, v119): a HYBRID VS -- one that reads BOTH the low bank f[<32] (3D
    // matrices / constants) AND the upper bank f[64..95] through a dynamic index (glyph texcoords)
    // -- can be fixed neither by the hardware GLSL path (V3D 7.1 freezes the dynamic upper-bank
    // index into a constant) nor by the v118 low-bank mirror (any fixed low window clobbers or
    // starves its low reads). The only correct execution for such a draw is the SOFTWARE vertex
    // shader (mainline Citra's use_hw_shader=0 path), applied per draw: the Vulkan rasterizer
    // returns false from AccelerateDrawBatch for exactly these draws, and PicaCore::DrawArrays
    // falls through to LoadVertices. Everything else (pure-text VSs -> mirror, pure-3D VSs ->
    // hardware) is unaffected, keeping the software cost limited to the miscompiled draws.
    const UniformReadScan scan = ScanVertexShaderUniformReads(program_code, main_offset);
    const bool is_hybrid = scan.analyzed && !scan.high.empty() && !scan.low.empty();

    // ---------------------------------------------------------------------
    // BORKED3DS_V3DV_TRACE_VSDECIDE -- sonde de DECISION de routage.
    //
    // But : repondre a la seule question restee ouverte sur la boule de Sonic,
    // dont le rendu change a CHAQUE lancement alors que l'async, la memoire non
    // initialisee et les budgets cumulatifs ont ete elimines.
    //
    // Emet UNE ligne par couple (main_offset, empreinte du programme VS) distinct :
    //   prog_hash  : empreinte du code du VS. S'il DIFFERE entre deux lancements
    //                pour un meme main_offset, l'alea est EN AMONT de nous (etat
    //                emule / upload), pas dans le routage.
    //   analyzed / n_high / n_low : les entrees exactes de la decision.
    //   decision   : le routage retenu (1 = software, 0 = materiel).
    //
    // Methode : capturer deux lancements, rediriger vers deux fichiers, puis
    // comparer avec diff. Toute ligne qui differe designe la cause.
    // Purement numerique (compatible daltonisme), inerte hors variable.
    static const bool trace_vsdecide = std::getenv("BORKED3DS_V3DV_TRACE_VSDECIDE") != nullptr;
    if (trace_vsdecide) {
        u64 prog_hash = 1469598103934665603ull; // FNV-1a 64
        for (const u32 word : program_code) {
            prog_hash ^= static_cast<u64>(word);
            prog_hash *= 1099511628211ull;
        }
        static std::mutex seen_mutex;
        static std::set<std::pair<u32, u64>> seen;
        bool first = false;
        {
            std::scoped_lock lock{seen_mutex};
            first = seen.insert({main_offset, prog_hash}).second;
        }
        if (first) {
            LOG_INFO(Render_Vulkan,
                     "TRACE_VSDECIDE main_offset={} prog_hash={:#018x} analyzed={} "
                     "n_high={} n_low={} hybrid={}",
                     main_offset, prog_hash, static_cast<u32>(scan.analyzed),
                     static_cast<u32>(scan.high.size()), static_cast<u32>(scan.low.size()),
                     static_cast<u32>(is_hybrid));
        }
    }

    // vDIRA v152 (BORKED3DS_V3DV_DIRA_ALL) : sonde "marteau".
    //
    // Route TOUS les draws vers le VS software, sans aucune analyse. Lent par
    // nature -- sonde de diagnostic, pas un reglage.
    static const bool dira_all = std::getenv("BORKED3DS_V3DV_DIRA_ALL") != nullptr;
    if (dira_all) {
        return true;
    }

    // vDIRA v151 (BORKED3DS_V3DV_DIRA_WIDE) : elargissement mesure sur Sonic.
    //
    // Le critere "hybride" exige de lire la banque HAUTE *et* la banque BASSE.
    // Les VS qui ne lisent QUE la banque haute retombent donc sur le chemin
    // materiel, ou V3D 7.1 fige l'index dynamique -> rendu fonce. D'ou deux
    // rendus differents dans un meme objet.
    //
    // v152 : le flag couvre desormais AUSSI les VS dont l'analyse de flot de
    // controle echoue (!scan.analyzed). Le comportement historique les laissait
    // au materiel "par prudence", mais c'est precisement le cas ou l'on ne peut
    // PAS prouver l'absence de lecture dynamique -- donc le cas a router.
    static const bool dira_wide = std::getenv("BORKED3DS_V3DV_DIRA_WIDE") != nullptr;
    const bool needs_software =
        dira_wide ? (!scan.analyzed || !scan.high.empty()) : is_hybrid;

    if (!needs_software) {
        return false;
    }
    // vDIRA v128: v127 proved samples=0 for every software-A8 draw -- the software Vulkan path is
    // dead at the V3DV driver level. When BORKED3DS_V3DV_HYBRID_MIRROR=1 grants this hybrid VS a
    // usable mirror plan, PREFER the accelerated path (its low-bank read is V3DV-safe) over the
    // dead software path. Reevaluated per call: LowMirrorPlan is a pure function of program_code
    // + main_offset + env, so this stays consistent with the upload and shader code paths that
    // already use LowMirrorPlan directly. If HYBRID_MIRROR is off, LowMirrorPlan still returns
    // kNoMirror for hybrids -> the historical software route is preserved for A/B comparison.
    if (VertexShaderLowMirrorPlan(program_code, main_offset).ok) {
        return false;
    }
    return true;
}

class ShaderWriter {
public:
    // Forwards all arguments directly to libfmt.
    // Note that all formatting requirements for fmt must be
    // obeyed when using this function. (e.g. {{ must be used
    // printing the character '{' is desirable. Ditto for }} and '}',
    // etc).
    template <typename... Args>
    void AddLine(fmt::format_string<Args...> text, Args&&... args) {
        AddExpression(fmt::format(text, std::forward<Args>(args)...));
        AddNewLine();
    }

    void AddNewLine() {
        DEBUG_ASSERT(scope >= 0);
        shader_source += '\n';
        // v364 -- GARDE-FOU 2 : plafond de taille du source GLSL emis. Independant du garde-fou 1 :
        // il attrape tout emballement de l'etage d'emission, meme avec peu de sous-routines. Un
        // vertex shader PICA sain fait ~40 Ko ; le plafond par defaut est a 16 Mo, soit 400 fois
        // plus. Ajustable par BORKED3DS_V3DV_MAX_SHADER_BYTES (0 = illimite).
        if (const std::size_t cap = MaxShaderBytes();
            cap != 0 && shader_source.size() > cap) [[unlikely]] {
            throw DecompileFail(fmt::format(
                "Emitted GLSL exceeded {} bytes ({} emitted) -- runaway shader emission", cap,
                shader_source.size()));
        }
    }

    std::string MoveResult() {
        return std::move(shader_source);
    }

    int scope = 0;

private:
    void AddExpression(std::string_view text) {
        if (!text.empty()) {
            shader_source.append(static_cast<std::size_t>(scope) * 4, ' ');
        }
        shader_source += text;
    }

    std::string shader_source;
};

/// An adaptor for getting swizzle pattern string from nihstro interfaces.
template <SwizzlePattern::Selector (SwizzlePattern::*getter)(int) const>
std::string GetSelectorSrc(const SwizzlePattern& pattern) {
    std::string out;
    for (int i = 0; i < 4; ++i) {
        switch ((pattern.*getter)(i)) {
        case SwizzlePattern::Selector::x:
            out += 'x';
            break;
        case SwizzlePattern::Selector::y:
            out += 'y';
            break;
        case SwizzlePattern::Selector::z:
            out += 'z';
            break;
        case SwizzlePattern::Selector::w:
            out += 'w';
            break;
        default:
            UNREACHABLE();
            return "";
        }
    }
    return out;
}

constexpr auto GetSelectorSrc1 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc1>;
constexpr auto GetSelectorSrc2 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc2>;
constexpr auto GetSelectorSrc3 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc3>;

class GLSLGenerator {
public:
    GLSLGenerator(const std::set<Subroutine>& subroutines, const ProgramCode& program_code,
                  const SwizzleData& swizzle_data, u32 main_offset,
                  const RegGetter& inputreg_getter, const RegGetter& outputreg_getter,
                  bool sanitize_mul, u16 jmpu_spec_mask = 0, u16 jmpu_spec_values = 0,
                  u16* cyclic_jmpu_mask_out = nullptr, u8 loop_spec_mask = 0,
                  const std::array<u32, 4>& loop_spec_values = {},
                  u16* used_bools_out = nullptr, u8* used_loops_out = nullptr)
        : subroutines(subroutines), program_code(program_code), swizzle_data(swizzle_data),
          main_offset(main_offset), inputreg_getter(inputreg_getter),
          outputreg_getter(outputreg_getter), sanitize_mul(sanitize_mul),
          v380_spec_mask(jmpu_spec_mask), v380_spec_values(jmpu_spec_values),
          v380_cyclic_mask_out(cyclic_jmpu_mask_out), v385_loop_mask(loop_spec_mask),
          v385_loop_values(loop_spec_values), v385_used_bools_out(used_bools_out),
          v385_used_loops_out(used_loops_out) {

        Generate();
    }

    std::string MoveShaderCode() {
        return shader.MoveResult();
    }

private:
    /// Gets the Subroutine object corresponding to the specified address.
    const Subroutine& GetSubroutine(u32 begin, u32 end) const {
        auto iter = subroutines.find(Subroutine{begin, end});
        ASSERT(iter != subroutines.end());
        return *iter;
    }

    /// Generates condition evaluation code for the flow control instruction.
    static std::string EvaluateCondition(Instruction::FlowControlType flow_control) {
        using Op = Instruction::FlowControlType::Op;

        const std::string_view result_x =
            flow_control.refx.Value() ? "conditional_code.x" : "!conditional_code.x";
        const std::string_view result_y =
            flow_control.refy.Value() ? "conditional_code.y" : "!conditional_code.y";

        switch (flow_control.op) {
        case Op::JustX:
            return std::string(result_x);
        case Op::JustY:
            return std::string(result_y);
        case Op::Or:
        case Op::And: {
            const std::string_view and_or = flow_control.op == Op::Or ? "any" : "all";
            std::string bvec;
            if (flow_control.refx.Value() && flow_control.refy.Value()) {
                bvec = "conditional_code";
            } else if (!flow_control.refx.Value() && !flow_control.refy.Value()) {
                bvec = "not(conditional_code)";
            } else {
                bvec = fmt::format("bvec2({}, {})", result_x, result_y);
            }
            return fmt::format("{}({})", and_or, bvec);
        }
        default:
            UNREACHABLE();
            return "";
        }
    }

    /// Generates code representing a source register.
    std::string GetSourceRegister(const SourceRegister& source_reg,
                                  u32 address_register_index) const {
        const u32 index = static_cast<u32>(source_reg.GetIndex());

        switch (source_reg.GetRegisterType()) {
        case RegisterType::Input:
            return inputreg_getter(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        case RegisterType::FloatUniform:
            if (address_register_index != 0) {
                // v116-SW: route the upper-bank (texcoord) reads through the constant-index
                // switch variant on V3D; keep the dynamic read for the lower bank (position/color),
                // which works. `index` here is the compile-time base uniform index.
                const char* off_fn =
                    (std::getenv("BORKED3DS_V3DV_HIGH_SWITCH") != nullptr && index >= 64)
                        ? "get_offset_register_sw"
                        : "get_offset_register";
                return fmt::format("{}({}, address_registers.{})", off_fn, index,
                                   "xyz"[address_register_index - 1]);
            }
            return fmt::format("uniforms.f[{}]", index);
        default:
            UNREACHABLE();
            return "";
        }
    }

    /// Generates code representing a destination register.
    std::string GetDestRegister(const DestRegister& dest_reg) const {
        const u32 index = static_cast<u32>(dest_reg.GetIndex());

        switch (dest_reg.GetRegisterType()) {
        case RegisterType::Output:
            return outputreg_getter(index);
        case RegisterType::Temporary:
            return fmt::format("reg_tmp{}", index);
        default:
            UNREACHABLE();
            return "";
        }
    }

    /// Generates code representing a bool uniform
    /// v385 : un booleen dont la valeur est connue pour ce draw (bit de v380_spec_mask) est emis
    /// comme constante : IFU / CALLU / JMPU deviennent des tests sur `true` / `false`, que le
    /// compilateur du pilote elimine. Sinon : lecture de l'uniforme, comme avant.
    std::string GetUniformBool(u32 index) const {
        if (index < 16) {
            if (v385_used_bools_out != nullptr) {
                *v385_used_bools_out = static_cast<u16>(*v385_used_bools_out | (1u << index));
            }
            if (((v380_spec_mask >> index) & 1u) != 0) {
                return ((v380_spec_values >> index) & 1u) != 0 ? "true" : "false";
            }
        }
        return fmt::format("uniforms.b[{}]", index);
    }

    /**
     * Adds code that calls a subroutine.
     * @param subroutine the subroutine to call.
     */
    void CallSubroutine(const Subroutine& subroutine) {
        if (subroutine.exit_method == ExitMethod::AlwaysEnd) {
            shader.AddLine("{}();", subroutine.GetName());
            shader.AddLine("return true;");
        } else if (subroutine.exit_method == ExitMethod::Conditional) {
            shader.AddLine("if ({}()) {{ return true; }}", subroutine.GetName());
        } else {
            shader.AddLine("{}();", subroutine.GetName());
        }
    }

    /**
     * Writes code that does an assignment operation.
     * @param swizzle the swizzle data of the current instruction.
     * @param reg the destination register code.
     * @param value the code representing the value to assign.
     * @param dest_num_components number of components of the destination register.
     * @param value_num_components number of components of the value to assign.
     */
    void SetDest(const SwizzlePattern& swizzle, std::string_view reg, std::string_view value,
                 u32 dest_num_components, u32 value_num_components) {
        u32 dest_mask_num_components = 0;
        std::string dest_mask_swizzle = ".";

        for (u32 i = 0; i < dest_num_components; ++i) {
            if (swizzle.DestComponentEnabled(static_cast<int>(i))) {
                dest_mask_swizzle += "xyzw"[i];
                ++dest_mask_num_components;
            }
        }

        if (reg.empty() || dest_mask_num_components == 0) {
            return;
        }
        DEBUG_ASSERT(value_num_components >= dest_num_components || value_num_components == 1);

        const std::string dest =
            fmt::format("{}{}", reg, dest_num_components != 1 ? dest_mask_swizzle : "");

        std::string src{value};
        if (value_num_components == 1) {
            if (dest_mask_num_components != 1) {
                src = fmt::format("vec{}({})", dest_mask_num_components, value);
            }
        } else if (value_num_components != dest_mask_num_components) {
            src = fmt::format("({}){}", value, dest_mask_swizzle);
        }

        shader.AddLine("{} = {};", dest, src);
    }

    /**
     * Compiles a single instruction from PICA to GLSL.
     * @param offset the offset of the PICA shader instruction.
     * @return the offset of the next instruction to execute. Usually it is the current offset + 1.
     * If the current instruction is IF or LOOP, the next instruction is after the IF or LOOP block.
     * If the current instruction always terminates the program, returns PROGRAM_END.
     */
    u32 CompileInstr(u32 offset) {
        const Instruction instr = {program_code[offset]};

        std::size_t swizzle_offset =
            instr.opcode.Value().GetInfo().type == OpCode::Type::MultiplyAdd
                ? instr.mad.operand_desc_id
                : instr.common.operand_desc_id;
        const SwizzlePattern swizzle(swizzle_data[swizzle_offset]);

        shader.AddLine("// {}: {}", offset, instr.opcode.Value().GetInfo().name);

        switch (instr.opcode.Value().GetInfo().type) {
        case OpCode::Type::Arithmetic: {
            const bool is_inverted =
                (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

            std::string src1 = swizzle.negate_src1 ? "-" : "";
            src1 += GetSourceRegister(instr.common.GetSrc1(is_inverted),
                                      !is_inverted * instr.common.address_register_index);
            src1 += "." + GetSelectorSrc1(swizzle);

            std::string src2 = swizzle.negate_src2 ? "-" : "";
            src2 += GetSourceRegister(instr.common.GetSrc2(is_inverted),
                                      is_inverted * instr.common.address_register_index);
            src2 += "." + GetSelectorSrc2(swizzle);

            std::string dest_reg = GetDestRegister(instr.common.dest.Value());

            switch (instr.opcode.Value().EffectiveOpCode()) {
            case OpCode::Id::ADD: {
                SetDest(swizzle, dest_reg, fmt::format("{} + {}", src1, src2), 4, 4);
                break;
            }

            case OpCode::Id::MUL: {
                if (sanitize_mul) {
                    SetDest(swizzle, dest_reg, fmt::format("sanitize_mul({}, {})", src1, src2), 4,
                            4);
                } else {
                    SetDest(swizzle, dest_reg, fmt::format("{} * {}", src1, src2), 4, 4);
                }
                break;
            }

            case OpCode::Id::FLR: {
                SetDest(swizzle, dest_reg, fmt::format("floor({})", src1), 4, 4);
                break;
            }

            case OpCode::Id::MAX: {
                if (sanitize_mul) {
                    SetDest(swizzle, dest_reg,
                            fmt::format("mix({1}, {0}, greaterThan({0}, {1}))", src1, src2), 4, 4);
                } else {
                    SetDest(swizzle, dest_reg, fmt::format("max({}, {})", src1, src2), 4, 4);
                }
                break;
            }

            case OpCode::Id::MIN: {
                if (sanitize_mul) {
                    SetDest(swizzle, dest_reg,
                            fmt::format("mix({1}, {0}, lessThan({0}, {1}))", src1, src2), 4, 4);
                } else {
                    SetDest(swizzle, dest_reg, fmt::format("min({}, {})", src1, src2), 4, 4);
                }
                break;
            }

            case OpCode::Id::DP3:
            case OpCode::Id::DP4:
            case OpCode::Id::DPH:
            case OpCode::Id::DPHI: {
                OpCode::Id opcode = instr.opcode.Value().EffectiveOpCode();
                std::string dot;
                if (opcode == OpCode::Id::DP3) {
                    if (sanitize_mul) {
                        dot = fmt::format("dot(vec3(sanitize_mul({}, {})), vec3(1.0))", src1, src2);
                    } else {
                        dot = fmt::format("dot(vec3({}), vec3({}))", src1, src2);
                    }
                } else {
                    if (sanitize_mul) {
                        const std::string src1_ =
                            (opcode == OpCode::Id::DPH || opcode == OpCode::Id::DPHI)
                                ? fmt::format("vec4({}.xyz, 1.0)", src1)
                                : std::move(src1);

                        dot = fmt::format("dot(sanitize_mul({}, {}), vec4(1.0))", src1_, src2);
                    } else {
                        dot = fmt::format("dot({}, {})", src1, src2);
                    }
                }

                SetDest(swizzle, dest_reg, dot, 4, 1);
                break;
            }

            case OpCode::Id::RCP: {
                if (!sanitize_mul) {
                    // When accurate multiplication is OFF, NaN are not really handled. This is a
                    // workaround to cheaply avoid NaN. Fixes graphical issues in Ocarina of Time.
                    shader.AddLine("if ({}.x != 0.0)", src1);
                }
                SetDest(swizzle, dest_reg, fmt::format("(1.0 / {}.x)", src1), 4, 1);
                break;
            }

            case OpCode::Id::RSQ: {
                if (!sanitize_mul) {
                    // When accurate multiplication is OFF, NaN are not really handled. This is a
                    // workaround to cheaply avoid NaN. Fixes graphical issues in Ocarina of Time.
                    shader.AddLine("if ({}.x > 0.0)", src1);
                }
                SetDest(swizzle, dest_reg, fmt::format("inversesqrt({}.x)", src1), 4, 1);
                break;
            }

            case OpCode::Id::MOVA: {
                SetDest(swizzle, "address_registers", fmt::format("ivec2({})", src1), 2, 2);
                break;
            }

            case OpCode::Id::MOV: {
                SetDest(swizzle, dest_reg, src1, 4, 4);
                break;
            }

            case OpCode::Id::SGE:
            case OpCode::Id::SGEI: {
                SetDest(swizzle, dest_reg,
                        fmt::format("vec4(greaterThanEqual({}, {}))", src1, src2), 4, 4);
                break;
            }

            case OpCode::Id::SLT:
            case OpCode::Id::SLTI: {
                SetDest(swizzle, dest_reg, fmt::format("vec4(lessThan({}, {}))", src1, src2), 4, 4);
                break;
            }

            case OpCode::Id::CMP: {
                using CompareOp = Instruction::Common::CompareOpType::Op;
                const std::map<CompareOp, std::pair<std::string_view, std::string_view>> cmp_ops{
                    {CompareOp::Equal, {"==", "equal"}},
                    {CompareOp::NotEqual, {"!=", "notEqual"}},
                    {CompareOp::LessThan, {"<", "lessThan"}},
                    {CompareOp::LessEqual, {"<=", "lessThanEqual"}},
                    {CompareOp::GreaterThan, {">", "greaterThan"}},
                    {CompareOp::GreaterEqual, {">=", "greaterThanEqual"}},
                };

                const CompareOp op_x = instr.common.compare_op.x.Value();
                const CompareOp op_y = instr.common.compare_op.y.Value();

                // v116-EQ (restaure v155) : sur Pi5/V3DV, l'ecriture texcoord des glyphes de
                // dialogue (sub_67_86, reg_tmp5) est gardee par une egalite flottante EXACTE
                // `reg_tmp8 == f[5].y`. reg_tmp8 derive de quelques ULP sur l'ALU V3D, donc le ==
                // exact est faux la ou il est vrai sous GL desktop -> comparaison faussee ->
                // rendu pale / banding. Emet une comparaison tolerante (|a-b| <= eps) pour
                // Equal / NotEqual seulement ; les comparaisons ordonnees (<,<=,>,>=) intactes.
                // Off -> comportement exact d'origine. Fix valide (recap v116 SS4), perdu au
                // commit 542a238af (upload du 24 juin), restaure ici.
                const bool tolerant_eq =
                    std::getenv("BORKED3DS_V3DV_TOLERANT_EQ") != nullptr;
                const bool force_eq_true =
                    std::getenv("BORKED3DS_V3DV_FORCE_EQ_TRUE") != nullptr;
                constexpr std::string_view kEqEps = "1e-2";

                const auto emit_component = [&](const char* comp, CompareOp op) {
                    if (force_eq_true && op == CompareOp::Equal) {
                        shader.AddLine("conditional_code.{} = true;", comp);
                        return true;
                    }
                    if (tolerant_eq && op == CompareOp::Equal) {
                        shader.AddLine("conditional_code.{} = abs({}.{} - {}.{}) <= {};", comp, src1,
                                       comp, src2, comp, kEqEps);
                        return true;
                    }
                    if (tolerant_eq && op == CompareOp::NotEqual) {
                        shader.AddLine("conditional_code.{} = abs({}.{} - {}.{}) > {};", comp, src1,
                                       comp, src2, comp, kEqEps);
                        return true;
                    }
                    return false;
                };

                if (cmp_ops.find(op_x) == cmp_ops.end()) {
                    LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", op_x);
                } else if (cmp_ops.find(op_y) == cmp_ops.end()) {
                    LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", op_y);
                } else if (force_eq_true && op_x == op_y && op_x == CompareOp::Equal) {
                    shader.AddLine("conditional_code = bvec2(true);");
                } else if (tolerant_eq && op_x == op_y &&
                           (op_x == CompareOp::Equal || op_x == CompareOp::NotEqual)) {
                    // Same tolerant op on both components: keep the vec2 form.
                    const std::string_view fn = op_x == CompareOp::Equal ? "lessThanEqual"
                                                                         : "greaterThan";
                    shader.AddLine("conditional_code = {}(abs(vec2({}) - vec2({})), vec2({}));", fn,
                                   src1, src2, kEqEps);
                } else if (op_x != op_y) {
                    if (!emit_component("x", op_x)) {
                        shader.AddLine("conditional_code.x = {}.x {} {}.x;", src1,
                                       cmp_ops.find(op_x)->second.first, src2);
                    }
                    if (!emit_component("y", op_y)) {
                        shader.AddLine("conditional_code.y = {}.y {} {}.y;", src1,
                                       cmp_ops.find(op_y)->second.first, src2);
                    }
                } else {
                    shader.AddLine("conditional_code = {}(vec2({}), vec2({}));",
                                   cmp_ops.find(op_x)->second.second, src1, src2);
                }
                break;
            }

            case OpCode::Id::EX2: {
                SetDest(swizzle, dest_reg, fmt::format("exp2({}.x)", src1), 4, 1);
                break;
            }

            case OpCode::Id::LG2: {
                SetDest(swizzle, dest_reg, fmt::format("log2({}.x)", src1), 4, 1);
                break;
            }

            default: {
                LOG_ERROR(HW_GPU, "Unhandled arithmetic instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
                break;
            }
            }

            break;
        }

        case OpCode::Type::MultiplyAdd: {
            if ((instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD) ||
                (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI)) {
                bool is_inverted = (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI);

                std::string src1 = swizzle.negate_src1 ? "-" : "";
                src1 += GetSourceRegister(instr.mad.GetSrc1(is_inverted), 0);
                src1 += "." + GetSelectorSrc1(swizzle);

                std::string src2 = swizzle.negate_src2 ? "-" : "";
                src2 += GetSourceRegister(instr.mad.GetSrc2(is_inverted),
                                          !is_inverted * instr.mad.address_register_index);
                src2 += "." + GetSelectorSrc2(swizzle);

                std::string src3 = swizzle.negate_src3 ? "-" : "";
                src3 += GetSourceRegister(instr.mad.GetSrc3(is_inverted),
                                          is_inverted * instr.mad.address_register_index);
                src3 += "." + GetSelectorSrc3(swizzle);

                std::string dest_reg =
                    (instr.mad.dest.Value() < 0x10)
                        ? outputreg_getter(static_cast<u32>(instr.mad.dest.Value().GetIndex()))
                    : (instr.mad.dest.Value() < 0x20)
                        ? "reg_tmp" + std::to_string(instr.mad.dest.Value().GetIndex())
                        : "";

                if (sanitize_mul) {
                    SetDest(swizzle, dest_reg,
                            fmt::format("sanitize_mul({}, {}) + {}", src1, src2, src3), 4, 4);
                } else {
                    SetDest(swizzle, dest_reg, fmt::format("{} * {} + {}", src1, src2, src3), 4, 4);
                }
            } else {
                LOG_ERROR(HW_GPU, "Unhandled multiply-add instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
            }
            break;
        }

        default: {
            switch (instr.opcode.Value()) {
            case OpCode::Id::END: {
                shader.AddLine("return true;");
                offset = PROGRAM_END - 1;
                break;
            }

            case OpCode::Id::JMPC:
            case OpCode::Id::JMPU: {
                std::string condition;
                // v380 : JMPU dont le booleen est specialise (valeur connue pour ce draw, voir
                // DecompileProgram) : le saut est soit toujours pris, soit jamais.
                bool v380_resolved = false;
                bool v380_taken = false;
                int v380_bool_id = -1;
                if (instr.opcode.Value() == OpCode::Id::JMPC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else {
                    bool invert_test = instr.flow_control.num_instructions & 1;
                    const u32 bool_id = instr.flow_control.bool_uniform_id.Value();
                    v380_bool_id = static_cast<int>(bool_id);
                    condition = (invert_test ? "!" : "") + GetUniformBool(bool_id);
                    if (bool_id < 16 && ((v380_spec_mask >> bool_id) & 1u) != 0) {
                        v380_resolved = true;
                        v380_taken = (((v380_spec_values >> bool_id) & 1u) != 0) != invert_test;
                    }
                }

                const u32 jmp_dest = instr.flow_control.dest_offset.Value();
                if (v380_resolved && !v380_taken) {
                    // Jamais pris pour cette valeur du booleen : aucune arete, aucun code.
                    shader.AddLine("// v380 : saut specialise, jamais pris");
                    break;
                }
                if (v380_resolved) {
                    // Toujours pris : la suite du bloc est morte. CompileRange s'arrete apres
                    // cette instruction (pas de poursuite en fin de bloc).
                    condition = "true";
                    v380_stop = true;
                }
                if (v377_acyclic) {
                    // v379 : saut dans la chaine ordonnee par le flot (voir V379TryStructured).
                    // Passe de releve : on note l'arete bloc -> cible, le texte est jete.
                    // Passe d'emission : une cible placee PLUS LOIN dans l'ordre d'emission se
                    // traite comme en v377 (on note la cible, la suite du bloc est gardee) ; une
                    // cible placee AVANT ou sur le bloc courant appartient forcement a la meme
                    // composante cyclique, emise dans un `while (true)` : `continue` repart en
                    // tete de cette boucle et les gardes sautent jusqu'au bloc vise.
                    if (v379_collect) {
                        v379_edges.emplace_back(v377_block_label, jmp_dest);
                        if (v380_bool_id >= 0 && !v380_resolved) {
                            v380_block_bools[v377_block_label] |=
                                static_cast<u16>(1u << (v380_bool_id & 15));
                        }
                    }
                    bool backward = false;
                    if (v379_pos != nullptr) {
                        const auto dest_it = v379_pos->find(jmp_dest);
                        if (dest_it == v379_pos->end()) {
                            // v379b : cible sans etiquette = sortie (`default: return false`
                            // de la forme d'origine) ; etiquette inatteignable = incoherence.
                            if (v379_labels == nullptr || v379_labels->count(jmp_dest)) {
                                v379_inconsistent = true;
                            }
                        } else {
                            backward = dest_it->second <= v379_block_pos;
                        }
                    }
                    shader.AddLine("if ({}) {{", condition);
                    ++shader.scope;
                    if (backward) {
                        shader.AddLine("jmp_to = {}u; continue;", jmp_dest);
                    } else {
                        shader.AddLine("jmp_to = {}u;", jmp_dest);
                    }
                    --shader.scope;
                    shader.AddLine("}}");
                    if (!backward) {
                        shader.AddLine("if (jmp_to == {}u) {{", v377_block_label);
                        ++shader.scope;
                        ++v377_open_guards;
                    }
                    break;
                }

                shader.AddLine("if ({}) {{", condition);
                ++shader.scope;
                shader.AddLine("{{ jmp_to = {}u; break; }}", jmp_dest);

                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::CALL:
            case OpCode::Id::CALLC:
            case OpCode::Id::CALLU: {
                std::string condition;
                if (instr.opcode.Value() == OpCode::Id::CALLC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else if (instr.opcode.Value() == OpCode::Id::CALLU) {
                    condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                }

                if (condition.empty()) {
                    shader.AddLine("{{");
                } else {
                    shader.AddLine("if ({}) {{", condition);
                }
                ++shader.scope;

                auto& call_sub = GetSubroutine(instr.flow_control.dest_offset,
                                               instr.flow_control.dest_offset +
                                                   instr.flow_control.num_instructions);

                CallSubroutine(call_sub);
                if (instr.opcode.Value() == OpCode::Id::CALL &&
                    call_sub.exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }

                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::NOP: {
                break;
            }

            case OpCode::Id::IFC:
            case OpCode::Id::IFU: {
                std::string condition;
                if (instr.opcode.Value() == OpCode::Id::IFC) {
                    condition = EvaluateCondition(instr.flow_control);
                } else {
                    condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                }

                const u32 if_offset = offset + 1;
                const u32 else_offset = instr.flow_control.dest_offset;
                const u32 endif_offset =
                    instr.flow_control.dest_offset + instr.flow_control.num_instructions;

                shader.AddLine("if ({}) {{", condition);
                ++shader.scope;

                auto& if_sub = GetSubroutine(if_offset, else_offset);
                CallSubroutine(if_sub);
                offset = else_offset - 1;

                if (instr.flow_control.num_instructions != 0) {
                    --shader.scope;
                    shader.AddLine("}} else {{");
                    ++shader.scope;

                    auto& else_sub = GetSubroutine(else_offset, endif_offset);
                    CallSubroutine(else_sub);
                    offset = endif_offset - 1;

                    if (if_sub.exit_method == ExitMethod::AlwaysEnd &&
                        else_sub.exit_method == ExitMethod::AlwaysEnd) {
                        offset = PROGRAM_END - 1;
                    }
                }

                --shader.scope;
                shader.AddLine("}}");
                break;
            }

            case OpCode::Id::LOOP: {
                // v385 : compteur de boucle connu pour ce draw -> litteral (bornes constantes,
                // le pilote peut derouler). Sinon : lecture de l'uniforme, comme avant.
                const u32 int_id = instr.flow_control.int_uniform_id.Value();
                if (v385_used_loops_out != nullptr && int_id < 4) {
                    *v385_used_loops_out = static_cast<u8>(*v385_used_loops_out | (1u << int_id));
                }
                std::string int_uniform;
                if (int_id < 4 && ((v385_loop_mask >> int_id) & 1u) != 0) {
                    const u32 v = v385_loop_values[int_id];
                    int_uniform = fmt::format("uvec3({}u, {}u, {}u)", v & 0xFFu, (v >> 8) & 0xFFu,
                                              (v >> 16) & 0xFFu);
                } else {
                    int_uniform = fmt::format("uniforms.i[{}]", int_id);
                }

                shader.AddLine("address_registers.z = int({}.y);", int_uniform);

                const std::string loop_var = fmt::format("loop{}", offset);
                shader.AddLine(
                    "for (uint {} = 0u; {} <= {}.x; address_registers.z += int({}.z), ++{}) {{",
                    loop_var, loop_var, int_uniform, int_uniform, loop_var);
                ++shader.scope;

                auto& loop_sub = GetSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
                CallSubroutine(loop_sub);
                offset = instr.flow_control.dest_offset;

                --shader.scope;
                shader.AddLine("}}");

                if (loop_sub.exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }

                break;
            }

            case OpCode::Id::EMIT:
            case OpCode::Id::SETEMIT:
                LOG_ERROR(HW_GPU, "Geometry shader operation detected in vertex shader");
                break;

            default: {
                LOG_ERROR(HW_GPU, "Unhandled instruction: 0x{:02x} ({}): 0x{:08x}",
                          (int)instr.opcode.Value().EffectiveOpCode(),
                          instr.opcode.Value().GetInfo().name, instr.hex);
                throw DecompileFail("Unhandled instruction");
                break;
            }
            }

            break;
        }
        }
        return offset + 1;
    }

    /**
     * Compiles a range of instructions from PICA to GLSL.
     * @param begin the offset of the starting instruction.
     * @param end the offset where the compilation should stop (exclusive).
     * @return the offset of the next instruction to compile. PROGRAM_END if the program terminates.
     */
    u32 CompileRange(u32 begin, u32 end) {
        u32 program_counter;
        for (program_counter = begin; program_counter < (begin > end ? PROGRAM_END : end);) {
            program_counter = CompileInstr(program_counter);
            if (v380_stop) {
                // v380 : saut specialise toujours pris. Le reste du bloc est mort et il n'y a
                // pas de poursuite : meme valeur de retour qu'un END.
                v380_stop = false;
                return PROGRAM_END;
            }
        }
        return program_counter;
    }

    void Generate() {
        if (sanitize_mul) {
            shader.AddLine("vec4 sanitize_mul(vec4 lhs, vec4 rhs) {{");
            ++shader.scope;
            shader.AddLine("vec4 product = lhs * rhs;");
            shader.AddLine("return mix(product, mix(mix(vec4(0.0), product, isnan(rhs)), product, "
                           "isnan(lhs)), isnan(product));");
            --shader.scope;
            shader.AddLine("}}\n");
        }

        shader.AddLine("vec4 get_offset_register(int base_index, int offset) {{");
        ++shader.scope;
        shader.AddLine("int fixed_offset = offset >= -128 && offset <= 127 ? offset : 0;");
        shader.AddLine("uint index = uint((base_index + fixed_offset) & 0x7F);");
        // v115-I Pi5/V3DV test: position (constant index 32) renders fine, but the texcoord
        // uses a runtime index (64 + address_registers.y). If that index lands >= 96 on V3DV,
        // the original code returns the constant vec4(1.0) -> flat UV -> invisible dialogue
        // text. When BORKED3DS_V3DV_CLAMP_OFFSET_INDEX is set we read the nearest in-range
        // uniform instead of the constant fallback: if the text reappears (even with imperfect
        // UVs) the out-of-range fallback was the culprit. Off by default -> original behaviour.
        if (std::getenv("BORKED3DS_V3DV_CLAMP_OFFSET_INDEX") != nullptr) {
            shader.AddLine("return uniforms.f[min(index, 95u)];");
        } else if (V377LegacyVertexShaderForm()) {
            shader.AddLine("return index < 96u ? uniforms.f[index] : vec4(1.0);");
        } else {
            // v377 : meme resultat que la forme d'origine (vec4(1.0) hors plage), mais sans
            // branche : la lecture se fait toujours a un index borne, puis une selection garde
            // la valeur de repli. Inlinee a chaque lecture indexee (84 fois dans le poison de
            // Sonic), la forme `?:` produisait autant de `if` autour de lectures TMU.
            shader.AddLine(
                "return mix(uniforms.f[min(index, 95u)], vec4(1.0), bvec4(index >= 96u));");
        }
        --shader.scope;
        shader.AddLine("}}\n");

        // v116c-TBO: V3D miscompiles DYNAMIC indexed reads of the upper uniform bank
        // (f[64..95], used by the dialogue-glyph texcoord f[64 + address_registers.y]) while it
        // handles the lower bank (position) and CONSTANT indices correctly. The constant-index
        // switch (v116b) was still folded back to a dynamic load by V3D (text stayed flat) and,
        // inlined at each base>=64 call site, also blew the VS up (~34-46k words) and hung the GPU.
        // get_offset_register_sw now reads f[] through a UNIFORM TEXEL BUFFER (the texture unit) --
        // a different hardware path that V3D compiles correctly here. This is the exact mechanism
        // the FS fog/lighting LUTs already use: texelFetch(lut, per_draw_base + i) over a
        // whole-buffer view. f_texel_base is the per-draw texel index of f[0], written by the
        // Vulkan uniform upload and read at a CONSTANT index (V3D-safe). One change kills BOTH the
        // invisibility and the giant-switch hang. Emitted only when enabled; used ONLY for
        // base_index >= 64 (see call site). vs_pica_f_tbo is bound at set=0 binding=6, eVertex.
        if (std::getenv("BORKED3DS_V3DV_HIGH_SWITCH") != nullptr) {
            shader.AddLine("#ifdef VULKAN");
            shader.AddLine("layout(set = 0, binding = 6) uniform samplerBuffer vs_pica_f_tbo;");
            shader.AddLine("#endif");
            shader.AddLine("vec4 get_offset_register_sw(int base_index, int offset) {{");
            ++shader.scope;
            shader.AddLine("int fixed_offset = offset >= -128 && offset <= 127 ? offset : 0;");
            shader.AddLine("int index = min((base_index + fixed_offset) & 0x7F, 95);");
            // v116c-DIAG / v129 recalibrated: BORKED3DS_V3DV_TBO_INDEX_TEST=1 bypasses texelFetch
            // entirely and returns a HIGH-CONTRAST parity pattern derived from `index`. Run with
            // FS_SHOW_UV=1. The v116c shallow ramp ((index-64)/31) was unusable for VSs whose
            // dynamic upper-bank read spans only a few slots (Sonic Lost World: index 64..69 ->
            // ramp 0.00..0.16, six near-black grays indistinguishable to the eye). Parity maps
            // adjacent glyph indices to full black vs full white, so a per-glyph checkerboard is
            // unmistakable in pure luminance regardless of index span or how FS_SHOW_UV maps the
            // texcoord channels:
            //   - alternating dark/bright per glyph cell -> `index` (hence aL.y) VARIES and reaches
            //     the output -> the fault is the texelFetch READ (V3D texel-buffer path) -> the
            //     2D-image fix (GL LUT analog) applies.
            //   - uniform (all one level) -> `index` is FROZEN upstream (address register / MOVA) ->
            //     no storage change helps; the address-register computation is the target.
            // parity mapped to 0.25 vs 0.75 (NOT 0/1): FS_SHOW_UV displays fract(abs(texcoord0.x)),
            // and fract(1.0)==0.0 would collapse a 0/1 parity to all-black. 0.25 and 0.75 both
            // survive fract intact -> a clean dark/bright alternation per glyph. All three
            // components carry the same value, so FS_SHOW_UV_AXIS (x or y) is irrelevant here.
            if (std::getenv("BORKED3DS_V3DV_TBO_INDEX_TEST") != nullptr) {
                shader.AddLine("return vec4(vec3(0.25 + 0.5 * float((index - 64) & 1)), 1.0);");
            } else if (std::getenv("BORKED3DS_V3DV_LOW_MIRROR") != nullptr) {
                // v118-MIRROR (Plan A, per-VS base): the needed upper-bank slots f[64..64+count) are
                // mirrored by the Vulkan uniform upload (RasterizerVulkan::UploadUniforms) into a
                // conflict-free low window f[base..base+count) chosen by VertexShaderLowMirrorPlan
                // (base = highest f[<32] slot this VS reads, + 1). DYNAMIC indexed reads of the LOW
                // bank are the only uniform path V3D compiles correctly here (the position already
                // uses f[32 + aL.x]); both the upper-bank dynamic read AND the VS texel-buffer
                // (texelFetch) path miscompile to a constant on V3DV. Read the mirror with a dynamic
                // LOW index at the per-VS base. `index` is in [64,95] here -> index-64 in [0,31];
                // clamp into the mirrored window. For a pure upper-bank VS, base=0, count=32 -> the
                // previous f[clamp(index-64,0,31)] behaviour exactly, so other games are unchanged.
                const LowMirrorPlan plan =
                    VertexShaderLowMirrorPlan(program_code, main_offset);
                const u32 base = plan.ok ? plan.base : 0u;
                const u32 count = plan.ok ? plan.count : 32u;
                // v130-MIRROR : la relecture part de la base source reelle du plan, plus de 64.
                const u32 src_base = plan.ok ? plan.src_base : 64u;
                shader.AddLine("return uniforms.f[{} + clamp(index - {}, 0, {})];", base, src_base,
                               count - 1u);
            } else {
                shader.AddLine("return texelFetch(vs_pica_f_tbo, int(f_texel_base) + index);");
            }
            --shader.scope;
            shader.AddLine("}}\n");
        }

        shader.AddLine("bvec2 conditional_code = bvec2(false);");
        shader.AddLine("ivec3 address_registers = ivec3(0);");
        for (int i = 0; i < 16; ++i) {
            shader.AddLine("vec4 reg_tmp{} = vec4(0.0, 0.0, 0.0, 1.0);", i);
        }
        shader.AddNewLine();

        // Add declarations for all subroutines
        for (const auto& subroutine : subroutines) {
            shader.AddLine("bool {}();", subroutine.GetName());
        }
        shader.AddNewLine();

        // Add the main entry point
        shader.AddLine("bool exec_shader() {{");
        ++shader.scope;
        CallSubroutine(GetSubroutine(main_offset, PROGRAM_END));
        --shader.scope;
        shader.AddLine("}}\n");

        // Add definitions for all subroutines
        for (const auto& subroutine : subroutines) {
            std::set<u32> labels = subroutine.labels;

            shader.AddLine("bool {}() {{", subroutine.GetName());
            ++shader.scope;

            if (labels.empty()) {
                if (CompileRange(subroutine.begin, subroutine.end) != PROGRAM_END) {
                    shader.AddLine("return false;");
                }
            } else if (!V377LegacyVertexShaderForm() && V379TryStructured(subroutine)) {
                // v379 : chaine ordonnee par le flot, boucles limitees aux cycles reels.
            } else {
                labels.insert(subroutine.begin);
                shader.AddLine("uint jmp_to = {}u;", subroutine.begin);
                shader.AddLine("while (true) {{");
                ++shader.scope;

                shader.AddLine("switch (jmp_to) {{");

                for (auto label : labels) {
                    shader.AddLine("case {}u: {{", label);
                    ++shader.scope;

                    auto next_it = labels.lower_bound(label + 1);
                    u32 next_label = next_it == labels.end() ? subroutine.end : *next_it;

                    u32 compile_end = CompileRange(label, next_label);
                    if (compile_end > next_label && compile_end != PROGRAM_END) {
                        // This happens only when there is a label inside a IF/LOOP block
                        shader.AddLine("{{ jmp_to = {}u; break; }}", compile_end);
                        labels.emplace(compile_end);
                    }

                    --shader.scope;
                    shader.AddLine("}}");
                }

                shader.AddLine("default: return false;");
                shader.AddLine("}}");

                --shader.scope;
                shader.AddLine("}}");

                shader.AddLine("return false;");
            }

            --shader.scope;
            shader.AddLine("}}\n");

            DEBUG_ASSERT(shader.scope == 0);
        }
    }

    /**
     * v379 -- emet une sous-routine a etiquettes sous forme de chaine ORDONNEE PAR LE FLOT :
     *
     *     uint jmp_to = BEGIN;
     *     if (jmp_to == A) { ...bloc A... ; if (jmp_to == A) { jmp_to = B; } }
     *     while (true) {                                   // composante cyclique seulement
     *         if (jmp_to == C) { ... if (x) { jmp_to = C; continue; } ... }
     *         if (jmp_to == D) { ... }
     *         break;
     *     }
     *     return false;
     *
     * v377 placait les blocs dans l'ordre des ADRESSES et abandonnait (forme `while/switch`
     * d'origine) des qu'un saut visait une adresse plus basse. Luigi's Mansion 2 le montre : son
     * programme principal commence a 159 et saute vers du code partage place en 2 et 118. Ce ne
     * sont pas des boucles, seulement du code range plus haut ; v377 retombait pourtant sur la
     * forme d'origine, et ses deux vertex shaders (56 000 mots) compilaient en 84 et 98 s.
     *
     * Deux passes :
     *   1. releve : chaque bloc [L, etiquette suivante) est compile a blanc ; on note ses aretes
     *      (sauts et poursuite en fin de bloc). Le texte est jete.
     *   2. emission : blocs atteignables depuis BEGIN, regroupes en composantes fortement
     *      connexes (Tarjan), composantes dans l'ordre topologique, blocs d'une composante dans
     *      l'ordre des adresses. Toute arete qui va plus loin dans cet ordre est traitee comme en
     *      v377 (gardes). Toute arete qui revient en arriere reste dans sa composante, qui est
     *      alors emise dans un `while (true)` ; l'arete devient `continue` et les gardes sautent
     *      jusqu'au bloc vise. Une entree par le milieu d'une composante marche de la meme facon.
     *
     * Sans cycle, le resultat est la chaine acyclique de v377 (dans un autre ordre si du code
     * est range plus haut). Avec cycles, seule la boucle reelle est emise en boucle, sans
     * `switch`. Les etendues de blocs, la poursuite apres un bloc IF/LOOP (etiquette ajoutee en
     * cours de route) et le retour `false` final sont ceux de la forme d'origine.
     *
     * Retourne false (texte intact) si la structure est incoherente (cible inconnue) :
     * l'appelant emet alors la forme d'origine.
     */
    bool V379TryStructured(const Subroutine& subroutine) {
        const ShaderWriter saved = shader;
        std::set<u32> labels = subroutine.labels;
        labels.insert(subroutine.begin);

        const auto next_label_of = [&](std::set<u32>::const_iterator it) {
            const auto next_it = std::next(it);
            return next_it == labels.end() ? subroutine.end : *next_it;
        };

        // --- Passe 1 : releve des aretes -------------------------------------------------------
        std::map<u32, std::vector<u32>> succ;
        v377_acyclic = true;
        v379_collect = true;
        v379_pos = nullptr;
        v379_inconsistent = false;
        v379_edges.clear();
        v380_block_bools.clear();
        for (auto it = labels.begin(); it != labels.end(); ++it) {
            const u32 label = *it;
            const u32 next_label = next_label_of(it);
            v377_block_label = label;
            v377_open_guards = 0;
            const u32 compile_end = CompileRange(label, next_label);
            auto& out = succ[label];
            // (les aretes de saut sont ajoutees apres la boucle, etiquettes definitives)
            if (compile_end != PROGRAM_END) {
                if (compile_end > next_label) {
                    // Etiquette situee dans un bloc IF/LOOP deja execute : meme traitement que la
                    // forme d'origine, on poursuit directement apres ce bloc.
                    labels.emplace(compile_end);
                }
                // v379b : poursuite vers une adresse qui n'est pas une etiquette (fin de la
                // sous-routine) ou bloc vide : c'est une sortie (`return false` de la forme
                // d'origine, via `default`), pas une arete. Sans cette regle, toute
                // sous-routine appelee par CALL retombait sur la forme d'origine (sub_105_134
                // de Luigi's Mansion 2, inlinee 4 fois par vertex shader).
                if (compile_end != label && labels.count(compile_end)) {
                    out.push_back(compile_end);
                }
            }
        }
        v379_collect = false;
        shader = saved;
        // Sauts : seules les cibles qui sont des etiquettes forment une arete. Un saut vers une
        // adresse sans etiquette (jamais parcourue par l'analyse) tombe, dans la forme d'origine,
        // sur `default: return false` : c'est une sortie, traitee comme telle a l'emission.
        for (const auto& [from, dest] : v379_edges) {
            if (labels.count(dest)) {
                succ[from].push_back(dest);
            }
        }

        // --- Blocs atteignables depuis BEGIN --------------------------------------------------
        std::set<u32> reach;
        std::vector<u32> todo{subroutine.begin};
        while (!todo.empty()) {
            const u32 n = todo.back();
            todo.pop_back();
            if (!labels.count(n)) {
                v377_acyclic = false;
                V380Count(V379Form::Fallback, 0, 0, subroutine);
                return false; // cible hors des etiquettes connues : forme d'origine
            }
            if (!reach.insert(n).second) {
                continue;
            }
            for (const u32 s : succ[n]) {
                todo.push_back(s);
            }
        }

        // --- Composantes fortement connexes (Tarjan) -------------------------------------------
        std::map<u32, int> index_of, low_of, comp_of;
        std::vector<u32> stack;
        std::set<u32> on_stack;
        std::vector<std::vector<u32>> comps;
        int next_index = 0;
        std::function<void(u32)> strong = [&](u32 v) {
            index_of[v] = low_of[v] = next_index++;
            stack.push_back(v);
            on_stack.insert(v);
            for (const u32 w : succ[v]) {
                if (!index_of.count(w)) {
                    strong(w);
                    low_of[v] = std::min(low_of[v], low_of[w]);
                } else if (on_stack.count(w)) {
                    low_of[v] = std::min(low_of[v], index_of[w]);
                }
            }
            if (low_of[v] == index_of[v]) {
                std::vector<u32> comp;
                u32 w;
                do {
                    w = stack.back();
                    stack.pop_back();
                    on_stack.erase(w);
                    comp_of[w] = static_cast<int>(comps.size());
                    comp.push_back(w);
                } while (w != v);
                std::sort(comp.begin(), comp.end());
                comps.push_back(std::move(comp));
            }
        };
        for (const u32 n : reach) {
            if (!index_of.count(n)) {
                strong(n);
            }
        }

        // --- Ordre topologique des composantes (la plus basse adresse d'abord a egalite) --------
        const std::size_t ncomp = comps.size();
        std::vector<int> indegree(ncomp, 0);
        std::vector<std::set<int>> comp_succ(ncomp);
        std::vector<bool> cyclic(ncomp, false);
        for (const u32 n : reach) {
            const int cn = comp_of[n];
            for (const u32 s : succ[n]) {
                const int cs = comp_of[s];
                if (cs == cn) {
                    cyclic[cn] = true;
                } else if (comp_succ[cn].insert(cs).second) {
                    ++indegree[cs];
                }
            }
        }
        std::set<std::pair<u32, int>> ready;
        for (std::size_t c = 0; c < ncomp; ++c) {
            if (indegree[c] == 0) {
                ready.emplace(comps[c].front(), static_cast<int>(c));
            }
        }
        std::vector<int> order;
        while (!ready.empty()) {
            const int c = ready.begin()->second;
            ready.erase(ready.begin());
            order.push_back(c);
            for (const int s : comp_succ[c]) {
                if (--indegree[s] == 0) {
                    ready.emplace(comps[s].front(), s);
                }
            }
        }
        if (order.size() != ncomp || comp_of[subroutine.begin] != order.front()) {
            v377_acyclic = false;
            V380Count(V379Form::Fallback, 0, 0, subroutine);
            return false;
        }

        // v380 : booleens des JMPU situes dans une composante cyclique. Rapportes a l'appelant
        // (CyclicJumpBoolMask) pour specialiser ces sauts : dans Luigi's Mansion 2, chaque
        // valeur des booleens donne un graphe sans cycle.
        if (v380_cyclic_mask_out != nullptr) {
            for (const u32 n : reach) {
                if (cyclic[comp_of[n]]) {
                    const auto b = v380_block_bools.find(n);
                    if (b != v380_block_bools.end()) {
                        *v380_cyclic_mask_out |= b->second;
                    }
                }
            }
        }

        std::map<u32, u32> pos;
        u32 loop_blocks = 0;
        for (const int c : order) {
            for (const u32 n : comps[c]) {
                pos.emplace(n, static_cast<u32>(pos.size()));
            }
            if (cyclic[c]) {
                loop_blocks += static_cast<u32>(comps[c].size());
            }
        }

        // --- Passe 2 : emission ----------------------------------------------------------------
        v379_pos = &pos;
        v379_labels = &labels;
        v379_inconsistent = false;
        shader.AddLine("uint jmp_to = {}u;", subroutine.begin);
        for (const int c : order) {
            if (cyclic[c]) {
                shader.AddLine("while (true) {{");
                ++shader.scope;
            }
            for (const u32 label : comps[c]) {
                const u32 next_label = next_label_of(labels.find(label));
                shader.AddLine("if (jmp_to == {}u) {{", label);
                ++shader.scope;
                v377_block_label = label;
                v379_block_pos = pos.at(label);
                v377_open_guards = 0;
                const u32 compile_end = CompileRange(label, next_label);
                for (; v377_open_guards > 0; --v377_open_guards) {
                    --shader.scope;
                    shader.AddLine("}}");
                }
                if (compile_end != PROGRAM_END) {
                    const auto end_it = pos.find(compile_end);
                    if (compile_end == label || !labels.count(compile_end)) {
                        // v379b : sortie de la sous-routine (voir la passe 1), jamais `continue`.
                        shader.AddLine("if (jmp_to == {}u) {{ jmp_to = {}u; }}", label,
                                       compile_end);
                    } else if (end_it == pos.end()) {
                        v379_inconsistent = true;
                    } else if (end_it->second <= v379_block_pos) {
                        shader.AddLine("if (jmp_to == {}u) {{ jmp_to = {}u; continue; }}", label,
                                       compile_end);
                    } else {
                        shader.AddLine("if (jmp_to == {}u) {{ jmp_to = {}u; }}", label,
                                       compile_end);
                    }
                }
                --shader.scope;
                shader.AddLine("}}");
            }
            if (cyclic[c]) {
                shader.AddLine("break;");
                --shader.scope;
                shader.AddLine("}}");
            }
        }
        shader.AddLine("return false;");
        v377_acyclic = false;
        v379_pos = nullptr;
        v379_labels = nullptr;

        if (v379_inconsistent) {
            shader = saved;
            V380Count(V379Form::Fallback, 0, 0, subroutine);
            return false;
        }
        V380Count(loop_blocks == 0 ? V379Form::Acyclic : V379Form::Loops, loop_blocks,
                            static_cast<u32>(reach.size()), subroutine);
        return true;
    }

    enum class V379Form { Acyclic, Loops, Fallback };

    /// v380 -- pas de journal pendant l'analyse de CyclicJumpBoolMask (compilation a blanc).
    void V380Count(V379Form form, u32 loop_blocks, u32 blocks, const Subroutine& subroutine) {
        if (v380_cyclic_mask_out == nullptr) {
            V379CountSubroutine(form, loop_blocks, blocks, subroutine);
        }
    }

    /// v379 -- journal : quelques lignes au debut, puis toute forme avec boucle ou repli.
    /// Le nom V377_FORME et le compteur boucle_conservee (= replis sur la forme d'origine) sont
    /// gardes pour que les commandes de resume existantes restent valables.
    static void V379CountSubroutine(V379Form form, u32 loop_blocks, u32 blocks,
                                    const Subroutine& subroutine) {
        static std::mutex mutex;
        static u64 acyclic_count = 0;
        static u64 structured_count = 0;
        static u64 fallback_count = 0;
        std::scoped_lock lock{mutex};
        switch (form) {
        case V379Form::Acyclic:
            ++acyclic_count;
            break;
        case V379Form::Loops:
            ++structured_count;
            break;
        case V379Form::Fallback:
            ++fallback_count;
            break;
        }
        const u64 total = acyclic_count + structured_count + fallback_count;
        if (form != V379Form::Acyclic || total <= 8 || (total % 256) == 0) {
            const std::string what =
                form == V379Form::Acyclic
                    ? std::string{"acyclique"}
                    : form == V379Form::Loops
                          ? fmt::format("boucle_reelle({}/{}_blocs)", loop_blocks, blocks)
                          : std::string{"boucle_conservee(repli)"};
            LOG_WARNING(HW_GPU,
                        "V377_FORME {} {} | cumul: acyclique={} boucle_reelle={} "
                        "boucle_conservee={}",
                        what, subroutine.GetName(), acyclic_count, structured_count,
                        fallback_count);
        }
    }

private:
    const std::set<Subroutine>& subroutines;
    const ProgramCode& program_code;
    const SwizzleData& swizzle_data;
    const u32 main_offset;
    const RegGetter& inputreg_getter;
    const RegGetter& outputreg_getter;
    const bool sanitize_mul;

    ShaderWriter shader;

    // v377 -- etat de l'emission de la sous-routine en cours.
    bool v377_acyclic = false;
    u32 v377_block_label = 0;
    int v377_open_guards = 0;

    // v379 -- releve des aretes (passe 1) et position des blocs dans l'ordre d'emission (passe 2).
    bool v379_collect = false;
    std::vector<std::pair<u32, u32>> v379_edges;
    const std::map<u32, u32>* v379_pos = nullptr;
    const std::set<u32>* v379_labels = nullptr;

    // v380 -- specialisation des JMPU (booleens connus pour ce draw) et releve des booleens
    // de sauts situes dans des composantes cycliques.
    const u16 v380_spec_mask;
    const u16 v380_spec_values;
    u16* const v380_cyclic_mask_out;
    bool v380_stop = false;
    std::map<u32, u16> v380_block_bools;
    u32 v379_block_pos = 0;
    bool v379_inconsistent = false;

    // v385 -- compteurs de boucle specialises (bits 0-3 = i[0..3], valeur x | y << 8 | z << 16)
    // et releve des booleens / compteurs lus par le programme (compilation a blanc).
    const u8 v385_loop_mask;
    const std::array<u32, 4> v385_loop_values;
    u16* const v385_used_bools_out;
    u8* const v385_used_loops_out;
};

std::string DecompileProgram(const ProgramCode& program_code, const SwizzleData& swizzle_data,
                             u32 main_offset, const RegGetter& inputreg_getter,
                             const RegGetter& outputreg_getter, bool sanitize_mul,
                             u16 jmpu_spec_mask, u16 jmpu_spec_values, u8 loop_spec_mask,
                             const std::array<u32, 4>& loop_spec_values) {

    try {
        auto subroutines = ControlFlowAnalyzer(program_code, main_offset).MoveSubroutines();
        GLSLGenerator generator(subroutines, program_code, swizzle_data, main_offset,
                                inputreg_getter, outputreg_getter, sanitize_mul, jmpu_spec_mask,
                                jmpu_spec_values, nullptr, loop_spec_mask, loop_spec_values);
        return generator.MoveShaderCode();
    } catch (const DecompileFail& exception) {
        LOG_INFO(HW_GPU, "Shader decompilation failed: {}", exception.what());
        return "";
    }
}

u16 CyclicJumpBoolMask(const ProgramCode& program_code, const SwizzleData& swizzle_data,
                       u32 main_offset) {
    u16 mask = 0;
    try {
        auto subroutines = ControlFlowAnalyzer(program_code, main_offset).MoveSubroutines();
        const RegGetter any_reg = [](u32) { return std::string{"reg_tmp0"}; };
        GLSLGenerator generator(subroutines, program_code, swizzle_data, main_offset, any_reg,
                                any_reg, false, 0, 0, &mask);
    } catch (const std::exception&) {
        return 0;
    }
    return mask;
}

u16 CyclicJumpBoolMaskCached(const ProgramCode& program_code, const SwizzleData& swizzle_data,
                             u64 program_hash, u64 swizzle_hash, u32 main_offset) {
    static const bool disabled = std::getenv("BORKED3DS_V3DV_V380_NO_JMPU_SPEC") != nullptr;
    if (disabled) {
        return 0;
    }
    static std::mutex mutex;
    static std::map<std::tuple<u64, u64, u32>, u16> cache;
    const auto key = std::make_tuple(program_hash, swizzle_hash, main_offset);
    {
        std::scoped_lock lock{mutex};
        if (const auto it = cache.find(key); it != cache.end()) {
            return it->second;
        }
    }
    const u16 mask = CyclicJumpBoolMask(program_code, swizzle_data, main_offset);
    {
        std::scoped_lock lock{mutex};
        cache.emplace(key, mask);
    }
    if (mask != 0) {
        LOG_WARNING(HW_GPU, "V380_SPECIALISATION programme={:016x} entree={} booleens=0x{:04x}",
                    program_hash, main_offset, mask);
    }
    return mask;
}

// ---------------------------------------------------------------------------------------------
// v385 -- SPECIALISATION DE TOUS LES BOOLEENS ET COMPTEURS DE BOUCLE LUS PAR LE PROGRAMME.
//
// Mesure TB46-TB51 (six jeux, caches vides) : les pipelines lourds (>= 0,5 s) reviennent a
// quelques programmes PICA (Kid Icarus : 82 pipelines, 2 programmes) dont le VS est recompile par
// V3D pour chaque pipeline. Banc TB52 (VS poison de Sonic, opt_compile_time) : booleens b[] et
// compteur i0 figes en constantes -> 1 086 -> 560 ms (-48 %), avec GS 878 -> 420 ms (-52 %).
//
// Masque par programme (compilation a blanc, en cache) = booleens lus par IFU / CALLU / JMPU et
// compteurs lus par LOOP. Le masque ne change jamais le resultat : les valeurs figees sont celles du
// draw courant et font partie de la cle du VS. Il ne change que le nombre de variantes, borne par
// un plafond par programme (BORKED3DS_V3DV_V385_MAX_VARIANTS, defaut 4) : au-dela, les nouvelles
// combinaisons prennent la forme generique (seuls restent figes les JMPU cycliques de v380).
// Echappatoire (A/B) : BORKED3DS_V3DV_V385_NO_SPEC=1.
// ---------------------------------------------------------------------------------------------
namespace {
struct V385ProgramInfo {
    u16 cyclic_bools = 0;
    u16 used_bools = 0;
    u8 used_loops = 0;
    std::set<std::pair<u16, std::array<u32, 4>>> variants;
    bool capped = false;
};
} // Anonymous namespace

V385Spec V385ComputeSpec(const ProgramCode& program_code, const SwizzleData& swizzle_data,
                         u64 program_hash, u64 swizzle_hash, u32 main_offset,
                         const std::array<bool, 16>& bools,
                         const std::array<u32, 4>& loop_values) {
    static const bool v380_disabled =
        std::getenv("BORKED3DS_V3DV_V380_NO_JMPU_SPEC") != nullptr;
    static const bool v385_disabled = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_V385_NO_SPEC");
        return v != nullptr && v[0] != '\0';
    }();
    static const std::size_t max_variants = [] {
        const char* v = std::getenv("BORKED3DS_V3DV_V385_MAX_VARIANTS");
        const unsigned long n = (v != nullptr && v[0] != '\0') ? std::strtoul(v, nullptr, 10) : 4;
        return static_cast<std::size_t>(n == 0 ? 1 : n);
    }();
    static std::mutex mutex;
    static std::map<std::tuple<u64, u64, u32>, V385ProgramInfo> programs;

    const auto key = std::make_tuple(program_hash, swizzle_hash, main_offset);
    std::unique_lock lock{mutex};
    auto it = programs.find(key);
    if (it == programs.end()) {
        lock.unlock();
        V385ProgramInfo info;
        try {
            auto subroutines = ControlFlowAnalyzer(program_code, main_offset).MoveSubroutines();
            const RegGetter any_reg = [](u32) { return std::string{"reg_tmp0"}; };
            u16 cyclic = 0;
            GLSLGenerator generator(subroutines, program_code, swizzle_data, main_offset, any_reg,
                                    any_reg, false, 0, 0, &cyclic, 0, {}, &info.used_bools,
                                    &info.used_loops);
            info.cyclic_bools = v380_disabled ? 0 : cyclic;
        } catch (const std::exception&) {
            info = V385ProgramInfo{};
        }
        if (v385_disabled) {
            info.used_bools = 0;
            info.used_loops = 0;
        }
        if (info.cyclic_bools != 0) {
            LOG_WARNING(HW_GPU,
                        "V380_SPECIALISATION programme={:016x} entree={} booleens=0x{:04x}",
                        program_hash, main_offset, info.cyclic_bools);
        }
        if (info.used_bools != 0 || info.used_loops != 0) {
            LOG_WARNING(HW_GPU,
                        "V385_SPECIALISATION programme={:016x} entree={} booleens=0x{:04x} "
                        "boucles=0x{:x} plafond={}",
                        program_hash, main_offset, info.used_bools, info.used_loops,
                        max_variants);
        }
        lock.lock();
        it = programs.emplace(key, std::move(info)).first;
    }
    V385ProgramInfo& info = it->second;

    const auto bool_values_of = [&](u16 mask) {
        u16 values = 0;
        for (u32 i = 0; i < 16; ++i) {
            if (((mask >> i) & 1u) != 0 && bools[i]) {
                values = static_cast<u16>(values | (1u << i));
            }
        }
        return values;
    };

    V385Spec spec{};
    const u16 full_mask = static_cast<u16>(info.cyclic_bools | info.used_bools);
    if (info.used_bools != 0 || info.used_loops != 0) {
        const u16 full_values = bool_values_of(full_mask);
        std::array<u32, 4> loops{};
        for (u32 i = 0; i < 4; ++i) {
            if (((info.used_loops >> i) & 1u) != 0) {
                loops[i] = loop_values[i] & 0xFFFFFFu;
            }
        }
        const auto variant = std::make_pair(full_values, loops);
        bool admitted = info.variants.count(variant) != 0;
        if (!admitted && info.variants.size() < max_variants) {
            info.variants.insert(variant);
            admitted = true;
            LOG_WARNING(HW_GPU,
                        "V385_VARIANTE programme={:016x} entree={} n={} booleens=0x{:04x} "
                        "i0={:06x} i1={:06x} i2={:06x} i3={:06x}",
                        program_hash, main_offset, info.variants.size(), full_values, loops[0],
                        loops[1], loops[2], loops[3]);
        }
        if (admitted) {
            spec.bool_mask = full_mask;
            spec.bool_values = full_values;
            spec.loop_mask = info.used_loops;
            spec.loop_values = loops;
            return spec;
        }
        if (!info.capped) {
            info.capped = true;
            LOG_WARNING(HW_GPU,
                        "V385_PLAFOND programme={:016x} entree={} plafond={} : combinaisons "
                        "suivantes en forme generique",
                        program_hash, main_offset, max_variants);
        }
    }
    // Forme generique : seuls les JMPU cycliques de v380 restent figes (indispensable pour Luigi).
    spec.bool_mask = info.cyclic_bools;
    spec.bool_values = bool_values_of(info.cyclic_bools);
    return spec;
}

} // namespace Pica::Shader::Generator::GLSL
