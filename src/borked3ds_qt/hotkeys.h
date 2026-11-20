// Copyright 2014 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <string>
#include <QKeySequence>
#include <QString>

class QDialog;
class QSettings;
class QShortcut;
class QWidget;

class HotkeyRegistry final {
public:
    friend class ConfigureHotkeys;
    friend class GMainWindow; //gvx64

    explicit HotkeyRegistry();
    ~HotkeyRegistry();

    /**
     * Loads hotkeys from the settings file.
     *
     * @note Yet unregistered hotkeys which are present in the settings will automatically be
     *       registered.
     */
    void LoadHotkeys();

    /**
     * Saves all registered hotkeys to the settings file.
     *
     * @note Each hotkey group will be stored a settings group; For each hotkey inside that group, a
     *       settings group will be created to store the key sequence and the hotkey context.
     */
    void SaveHotkeys();

    /**
     * Returns a QShortcut object whose activated() signal can be connected to other QObjects'
     * slots.
     *
     * @param group  General group this hotkey belongs to (e.g. "Main Window", "Debugger").
     * @param action Name of the action (e.g. "Start Emulation", "Load Image").
     * @param widget Parent widget of the returned QShortcut.
     * @warning If multiple QWidgets' call this function for the same action, the returned QShortcut
     *          will be the same. Thus, you shouldn't rely on the caller really being the
     *          QShortcut's parent.
     */
    QShortcut* GetHotkey(const QString& group, const QString& action, QObject* widget);

    /**
     * Returns a QKeySequence object whose signal can be connected to QAction::setShortcut.
     *
     * @param group  General group this hotkey belongs to (e.g. "Main Window", "Debugger").
     * @param action Name of the action (e.g. "Start Emulation", "Load Image").
     */
    QKeySequence GetKeySequence(const QString& group, const QString& action);

    /**
     * Returns a Qt::ShortcutContext object who can be connected to other
     * QAction::setShortcutContext.
     *
     * @param group  General group this shortcut context belongs to (e.g. "Main Window",
     * "Debugger").
     * @param action Name of the action (e.g. "Start Emulation", "Load Image").
     */
    Qt::ShortcutContext GetShortcutContext(const QString& group, const QString& action);

private:
    struct Hotkey {
        QKeySequence keyseq;
        QString controller_keyseq;
        std::map<QString, QShortcut*> shortcuts;
        Qt::ShortcutContext context = Qt::WindowShortcut;
    };

    using HotkeyMap = std::map<QString, Hotkey>;
    using HotkeyGroupMap = std::map<QString, HotkeyMap>;

    HotkeyGroupMap hotkey_groups;
};

/**
 * v426 - Controller hotkeys "a la RetroPie" (hotkey enable button + button).
 *
 * A controller hotkey binding is either a raw input ParamPackage recorded by ControllerDialog
 * (engine:sdl,guid:...,port:...,button:N) or one of two symbolic forms that never hard-code a
 * controller, so the defaults work with whatever gamepad the player configured:
 *   engine:hotkey_3ds,button:<name>  -> the input bound to that 3DS button in the current input
 *                                       profile (name = select, start, l, r, x, b, ...)
 *   engine:hotkey_pad,button:<name>  -> a named gamepad button the 3DS does not have
 *                                       (leftstick = L3, rightstick = R3, guide), looked up on the
 *                                       controller of the current input profile through its
 *                                       SDL_GameController mapping, then EmulationStation's
 *                                       es_input.cfg (leftthumb/rightthumb) as a fallback.
 */
namespace ControllerHotkey {

/// Default hotkey enable button (modifier): the button mapped to 3DS Select.
QString DefaultModifier();

/// Resolves a binding (raw or symbolic) to a concrete input ParamPackage string; empty on failure.
std::string Resolve(const QString& binding);

/// Converts a raw binding to its symbolic form when it matches a 3DS button of the current input
/// profile (or L3/R3), so it keeps working with another controller. Returns the input otherwise.
QString Symbolize(const QString& raw_binding);

/// True when both resolved ParamPackage strings designate the same physical input.
bool SameInput(const std::string& a, const std::string& b);

/// Human readable name of a binding for the configuration UI.
QString DisplayName(const QString& binding);

} // namespace ControllerHotkey
