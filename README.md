# Borked3DS-rpi

**Nintendo 3DS emulation for the Raspberry Pi 4, the Raspberry Pi 5 and x86-64 PCs, installed in one step from RetroPie.**

> **This is not a new emulator.** Borked3DS-rpi is an adaptation of [Borked3DS](https://github.com/Borked3DS/Borked3DS), one of the continuations of **Citra**, the open-source Nintendo 3DS emulator. Borked3DS itself builds on PabloMK7's Citra fork, Lime3DS and Mandarine. The main active continuation of Citra today is **[Azahar](https://github.com/azahar-emu/azahar)**, and several of its fixes were ported here.
>
> Citra and its forks have run 3DS games well on PC (Windows, Linux, macOS) and Android for years; they just did not run properly on the Raspberry Pi. The emulator itself is the work of the Citra, Borked3DS and Azahar developers. The work done here — by [gvx64](https://github.com/gvx64/Borked3DS-rpi) for the Pi 4, and by DTEAM-1 with Claude for the Pi 5 — was to make it build, run and perform well on the Pi: GPU driver issues, the OpenGL ES and Vulkan (V3DV) paths, performance tuning, a few fixes found along the way, and the RetroPie install script.

> **Project status: no longer in active development.** The Pi 5 work is complete and this repository is kept as is: the install script, the build and the documentation stay available, but no new features or optimizations are planned. Fixes useful to other emulators are being offered to Azahar.

Support thread: [3DS emulator for Pi users based on Borked 3DS](https://retropie.org.uk/forum/topic/37122/3ds-emulator-for-pi-users-based-on-borked-3ds) (RetroPie forum).

---

## What's new (November 2026, Pi 5 / Vulkan)

The Pi 5 build is now heavily optimized and debugged. The biggest find: **the emulated 3DS GPU ran in zero emulated time**, so some games computed frames that were never shown. An automatic per-frame pacing now fixes this for every game, with no per-game setting. Full details and measurements are in the [optimization journal](OPTIMIZATIONS_SINCE_LETTERS_BUG_v421.pdf).

| Game (Pi 5, Vulkan) | Before | Now |
|---|---|---|
| Castlevania: Lords of Shadow – Mirror of Fate | 33 % speed, sliding character (a known bug on Citra/Azahar, even on PC) | **100 % speed, 60 fps, character fixed** |
| Luigi's Mansion 2 | ~55 % speed, lighting reduced to 6 lights | **100 % speed, original 8 lights, no glitch** |
| Every game | — | **PGO + LTO build**: 8 to 17 % less CPU on the emulation thread |

Also new: the invisible right eye is no longer drawn when 3D is off (detected automatically, only when it is safe), and the sound of LEGO Ninjago: Shadow of Ronin cut-scenes is fixed (streamed audio buffers are now read as they play, as on the real DSP).

**RetroArch-style hotkeys (Pi 5 and x86-64).** Hold **Select** and press a second button (Select + Start to quit, Select + R1 to quick save, etc.), with a short on-screen message confirming the action. It works out of the box on a new install; see [Controller and hotkeys](#controller-and-hotkeys). Already installed? `borked3ds.sh` is unchanged, just rebuild borked3ds from source.

---

## Supported systems

One install script, three targets. It detects the machine and builds the right emulator for it.

| | Raspberry Pi 5 (Pi 500, CM5) | Raspberry Pi 4 (Pi 400, CM4) | x86-64 PC |
|---|---|---|---|
| OS | 64-bit. **Trixie recommended**, with an up-to-date Mesa ([trixie-backports](https://tracker.debian.org/pkg/mesa), tested with 26.1.2). Bookworm works; its Mesa 24.x is not tested with Vulkan | 64-bit, Bookworm or Trixie | 64-bit Debian or Ubuntu (GCC 13+ or Clang 16+) |
| Renderer | **Vulkan** (default, recommended) | **OpenGL with GLES** (set by the script) | Vulkan (default) |
| Emulator source | [DTEAM-1/Borked3DS-rpi](https://github.com/DTEAM-1/Borked3DS-rpi) (this repository) | [gvx64/Borked3DS-rpi](https://github.com/gvx64/Borked3DS-rpi) | [DTEAM-1/Borked3DS-rpi](https://github.com/DTEAM-1/Borked3DS-rpi) (this repository) |
| Build | Clang, `-mcpu=cortex-a76+crypto -O3`, ThinLTO, PGO | System compiler, RetroPie flags (as RetroPie-Extra builds it) | System compiler, `-O3`, no `-march` (runs on any x86-64 CPU) |

32-bit systems are not supported.

### You also need

- Decrypted 3DS ROMs. No BIOS needed.
- A mouse for navigating the user interface when needed. A touchscreen works too.
- On the Pi 5, a good power supply (the official 27 W one is what the measurements were made with).

---

## Installation (RetroPie)

1. Use [RetroPie-Extra](https://github.com/Exarkuniv/RetroPie-Extra) and install [`borked3ds.sh`](borked3ds.sh) **from source**. The script detects your board, installs the dependencies, builds the right emulator and writes the optimal settings for your board. There is nothing else to configure for performance.
2. **Pi 5:** the build is automatically optimized with PGO and LTO (about 45 min on a Pi 5 8 GB). At the end of the install you should see `PGO : actif`. To check later:
   ```
   cat /opt/retropie/emulators/borked3ds/.borked3ds_pgo
   ```
   A Pi 5 with less than 6 GB of RAM builds without LTO (PGO still applies). If anything prevents PGO, the script says why and builds without it: PGO can never break a build.
3. **Already a user?** Back up, then delete, your **`qt-config.ini`** (`~/.config/borked3ds-emu/`) and your **`emulators.cfg`** (`/opt/retropie/configs/3ds/`) before updating, so you start from the built-in optimal settings and the new default hotkeys. The Vulkan shader cache is kept across updates.
4. Add decrypted 3DS ROMs to your ROM folder (`~/RetroPie/roms/3ds`).
5. **OpenGL/GLES (Pi 4):** modify the `<command>` line of the 3ds system entry to declare the following environment variable, in `/etc/emulationstation/es_systems.cfg` or `/opt/retropie/configs/all/emulationstation/es_systems.cfg` (it is harmless with Vulkan):

```xml
<system>
  <name>3ds</name>
  <fullname>3ds</fullname>
  <path>/home/pi/RetroPie/roms/3ds</path>
  <extension>.3ds .cci .3DS .CCI</extension>
  <command>env MESA_EXTENSION_OVERRIDE="GL_OES_texture_buffer" /opt/retropie/supplementary/runcommand/runcommand.sh 0 _SYS_ 3ds %ROM%</command>
  <platform>3ds</platform>
  <theme>3ds</theme>
</system>
```

`MESA_EXTENSION_OVERRIDE="GL_OES_texture_buffer"` enables the GLES 3.2 extension `GL_OES_texture_buffer`, which the Pi 4/Pi 5 driver supports incompletely but well enough for this emulator.

---

## Launch options (runcommand menu)

| Option | Use |
|---|---|
| **borked3ds** | Play (default) |
| **borked3ds-ui** | Opens the emulator's main window without starting the game, to set controls, hotkeys and per-game options |
| **borked3ds-ui-qt06** | Same, with smaller menus that fit an 800×480 screen better (recommended on small screens) |

---

## Pi 5 tips

- **No graphics settings to change.** Vulkan, the shader cache and every Pi 5 setting are built into the emulator.
- **Leave "Disable Right Eye Rendering" unchecked.** The emulator now skips the invisible right eye by itself when 3D is off, and only when it is safe (some games, such as Metroid, reuse the right eye for effects and would flicker).
- **First run of a game or area:** shaders are compiled in the background, so you may see a brief flash, an object appearing late or a short stutter. Play on: the next time is smooth, and the cache is kept even after an emulator update.
- **Prefer in-game saves to savestates.** Loading a savestate rebuilds the whole system and can leave a screen undrawn until it is redrawn.
- **Troubleshooting only:** the two automatic features can be turned off by adding `BORKED3DS_V3DV_V415_OFF=1` (frame pacing) or `BORKED3DS_V3DV_V408_OFF=1` (right-eye skip) in front of the launch line in `emulators.cfg`. If you need one of them, please report the game.

---

## Controller and hotkeys

Set up your controller first (**Emulation → Configure… → Controls**). The hotkeys follow this mapping.

### Default hotkeys (Pi 5 and x86-64)

Like RetroArch / RetroPie: hold **Select** (the hotkey enable button), then press the second button.

| Combo | Action |
|---|---|
| Select + Start | Exit Borked3DS (back to EmulationStation) |
| Select + R1 | Quick Save |
| Select + L1 | Quick Load |
| Select + X | Menu: pauses and leaves fullscreen so the menus can be used with the mouse; press again to go back to fullscreen and resume |
| Select + B | Reset (restarts the game) |
| Select + L3 | Screenshot |
| Select + R3 | Start / stop video recording |

- A small RetroArch-style message (white text on dark blue, bottom left, 2 seconds) confirms each action: *Saving state...*, *Loading state...* (or *No quick save to load*), *Screenshot taken*, *Video recording started*, *Video recording stopped*. It is not included in screenshots or videos.
- "Select", "Start", "R1"… mean the buttons mapped to the 3DS Select, Start, R… in Controls, so the defaults work with any controller; nothing is hard-coded for a specific gamepad. L3 / R3 (absent on a 3DS) are found through SDL's controller database, or else through EmulationStation's input configuration (*left thumb* / *right thumb*). If Select + L3 / R3 does nothing, assign them manually (step 2 below).
- As in RetroArch, the game also receives the combo buttons (for example Select and R1).
- Go to **Emulation → Configure… → General → General** and **uncheck** "Confirm Exit While Emulation is Running", so Select + Start quits without a confirmation window.
- **New install, or no `qt-config.ini` yet:** everything is automatic.
- **Existing `qt-config.ini`:** hotkeys you set yourself in the past are kept and take priority over the new defaults (a rebuild never changes your settings). Either delete `qt-config.ini` as explained in [Installation](#installation-retropie), or switch to the new defaults from the UI (steps below).

The Pi 4 builds [gvx64's fork](https://github.com/gvx64/Borked3DS-rpi), which keeps its own hotkey behaviour (single buttons) unless it adopts this change.

### Changing the hotkeys from the UI launch line

**1. Open Borked3DS in UI mode (before playing)**

- In EmulationStation, open the 3DS system and highlight any game.
- Press A to launch it. When the RetroPie runcommand screen appears ("Press a key to configure launch options"), immediately press A again (or any key).
- Choose **Select emulator for ROM**.
- Choose **borked3ds-ui** for normal-size menus, or **borked3ds-ui-qt06** for smaller menus that fit an 800×480 screen better.
- Choose **Launch**. Borked3DS opens its main window without starting the game.

**2. Map the hotkeys**

- In the menu bar, open **Emulation → Configure…**
- In the left panel, select **Hotkeys** (in the Controls group).
- The line at the top, **Controller hotkey enable button**, sets the button to hold: **Default (Select)**, **None** (old behaviour, each hotkey is a single button), or click the button field and press another gamepad button.
- Each action has two columns: **Hotkey** for the keyboard, **Controller** for the gamepad. To assign a gamepad button, double-click the action's cell in the Controller column and press **only the second button** of the combo (not the enable button). To reset or clear a mapping, right-click the cell and choose **Restore Default** or **Clear**.
- To switch an existing configuration to the new defaults: click **Default (Select)**, then **Restore Defaults** (this also resets the keyboard shortcuts), or right-click each Controller cell → **Restore Default**.
- If Select + L3 / R3 does nothing: double-click the Controller cell of **Capture Screenshot** and press L3; same with **Toggle Video Dumping** and R3.

**3. Other settings**

- You can set a **gamepad touch cursor** (per game only): in the UI, right-click a game, open **Properties**, last tab (**Controls**), and check **Enable Gamepad Touch Cursor**. You can then set the sensitivity and more.
- Finally, open **View** (top menu) and select **Fullscreen**.

Once everything is saved, exit the UI and run a game again with the **borked3ds** option (the default) in the runcommand menu. You are now all set for the Nintendo 3DS.

---

## Pi 4 settings

The Pi 4 builds [gvx64's fork](https://github.com/gvx64/Borked3DS-rpi), maintained for the Pi 4's GPU (V3D 4.2) and its OpenGL ES path.

- **Graphics API: OpenGL** (essential; Vulkan is not usable on the Pi 4) — set by the script
- **Use GLES extensions: enabled** — set by the script
- **Hardware Shader Acceleration: enabled** (essential on the Pi 4, but causes long pauses and occasional fatal hangs, due to the Pi 4's GPU, until the shader cache is sufficiently built up; without it most games are too slow to be playable)
- **Accurate Multiplication: disabled** (reduces the length and frequency of pauses and fatal hangs during shader compilation). For now, set it in the **per-game** configuration menu: the checkbox only becomes accessible after toggling Hardware Shader Acceleration off and on, and it tends to be re-enabled when set in the common configuration menu.
- **Shader JIT Engine: enabled**
- **Async Presentation: enabled**
- **Async Shader Compilation: enabled**
- **Use Disk Shader Cache: enabled** (Preload Shader Count: 1000–1500)

Optional: moving the Borked3DS shader cache to a USB drive is described in [post 221](https://retropie.org.uk/forum/topic/37122/3ds-emulator-for-pi-users-based-on-borked-3ds/221) of the support thread.

### Running outside RetroPie (Pi 4, OpenGL)

From the PIXEL desktop:

```
env MESA_EXTENSION_OVERRIDE="GL_OES_texture_buffer" /path/to/borked3ds
```

Without a desktop (for example Raspberry Pi OS Lite):

```
env MESA_EXTENSION_OVERRIDE="GL_OES_texture_buffer" xinit /path/to/borked3ds
```

---

## Building

The supported way to build is the RetroPie script [`borked3ds.sh`](borked3ds.sh): it is also the reference for the exact dependencies, compiler, CMake options and post-install checks of each target. In short:

- **Board detection:** `/proc/device-tree/model`, then RetroPie's platform flags; `uname -m` for x86-64. It can be forced with `BORKED3DS_TARGET=pi4`, `pi5` or `x86_64`, and the repository with `BORKED3DS_REPO=<url>` (pass them through `sudo -E`).
- **Libraries:** the versions pinned by this repository's submodules, except Qt and SDL2, which come from the system.
- **Bookworm:** the tree needs CMake 3.26+ and Bookworm ships 3.25, so CMake 3.31 is downloaded into RetroPie-Setup's `tmp` folder (nothing installed system-wide). The Pi 5 build uses `clang-16` from Bookworm's repository.
- **PGO profile:** `dist/pgo/borked3ds.proftext.xz`, with its checksum and [`dist/pgo/PROFIL.txt`](dist/pgo/PROFIL.txt) (how it was recorded and how to regenerate it). `BORKED3DS_NO_PGO=1` forces a build without it.
- **Pi 4:** see the build instructions in [gvx64/Borked3DS-rpi](https://github.com/gvx64/Borked3DS-rpi).

---

## Paths

| What | Where |
|---|---|
| Emulator | `/opt/retropie/emulators/borked3ds/borked3ds` |
| Settings | `~/.config/borked3ds-emu/qt-config.ini` |
| Log | `~/.local/share/borked3ds-emu/log/borked3ds_log.txt` |
| Vulkan shader cache | `~/.local/share/borked3ds-emu/shaders/vulkan` |
| Launch lines | `/opt/retropie/configs/3ds/emulators.cfg` |
| Savestates | `~/.local/share/borked3ds-emu/states/` (`<TitleID>.00.cst` is the quick save) |
| In-game saves | `~/.local/share/borked3ds-emu/sdmc/Nintendo 3DS/` |
| Screenshots | `~/.local/share/borked3ds-emu/screenshots/` (unless another folder is set in the settings) |
| Videos | `~/.local/share/borked3ds-emu/videos/` (or the last folder used in **Emulation → Dump Video**), named `<game>_<date_time>.<ext>` |

The default video format (webm / VP9) is heavy for a Pi 5; a lighter format can be chosen once in **Emulation → Dump Video** and is remembered.

---

## Feedback

Please tell us whether everything works for you, or not: **board, OS, renderer and game**. On a Pi 5, the result of `cat /opt/retropie/emulators/borked3ds/.borked3ds_pgo` helps too, and for the hotkeys, your controller model. Post in the [support thread](https://retropie.org.uk/forum/topic/37122/3ds-emulator-for-pi-users-based-on-borked-3ds) or open an issue in this repository. As the project is no longer in active development, reports are welcome but may not lead to a fix.

---

## Where this comes from

**Citra** was the open-source Nintendo 3DS emulator. After it shut down, several forks continued the work: [PabloMK7's fork](https://github.com/PabloMK7/citra) (now archived), [Lime3DS](https://github.com/Lime3DS/Lime3DS) (now defunct), [Mandarine](https://github.com/mandarine3ds/mandarine), and **[Azahar](https://github.com/azahar-emu/azahar)**, which unifies the Lime3DS and PabloMK7 work and is today the main active continuation.

**[Borked3DS](https://github.com/Borked3DS/Borked3DS)**, by rtiangha, is one of those Citra-derived projects. Its lineage comes directly from PabloMK7's fork, with elements of Lime3DS and Mandarine, and its own additions: more recent third-party dependencies built from source, configurable SPIR-V shader optimization in Vulkan, better logging and tooltips, and more. Its author describes it as an experimental, hobby, "canary" project.

**Borked3DS-rpi** started from Borked3DS's OpenGL ES development branch, adapted by gvx64 for the Raspberry Pi 4 and 5 ([gvx64/Borked3DS-rpi](https://github.com/gvx64/Borked3DS-rpi), OpenGL/GLES). This repository continues from there for the Pi 5's Vulkan driver (V3DV) and for x86-64, and integrates gvx64's improvements rather than the reverse. The Pi 4 keeps gvx64's fork, chosen automatically by the install script.

This repository is no longer in active development. Changes made here that could be useful upstream are being offered to Azahar.

---

## Credits

- **Citra** project developers
- **rtiangha** — [Borked3DS](https://github.com/Borked3DS/Borked3DS), the original project
- **Azahar** project developers (several fixes ported)
- **gvx64** and **Claude** — adapting the emulator for Pi and x86-64 users
- **folly** and **dteam** — the RetroPie installation script and testing
- **miken32** — SDL2 issue
- **roslof** and **RapidEdwin08** — build reports
- **ExarKunIv** — [RetroPie-Extra](https://github.com/Exarkuniv/RetroPie-Extra)

Thanks to everyone else who tested the emulator.

## License

GNU General Public License v2, see [`license.txt`](license.txt), as inherited from Citra.
