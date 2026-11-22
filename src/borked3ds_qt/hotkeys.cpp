// Copyright 2014 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QShortcut>
#include <QXmlStreamReader>
#include <QtGlobal>
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/uisettings.h"
#include "common/param_package.h"
#include "common/settings.h"
#include "input_common/main.h"

HotkeyRegistry::HotkeyRegistry() = default;

HotkeyRegistry::~HotkeyRegistry() = default;

void HotkeyRegistry::SaveHotkeys() { //gvx64 - replace
    UISettings::values.shortcuts.clear();
    for (const auto& group : hotkey_groups) {
        for (const auto& hotkey : group.second) {
            UISettings::Shortcut shortcut;
            shortcut.name = hotkey.first;
            shortcut.group = group.first;
            shortcut.shortcut = UISettings::ContextualShortcut{
                hotkey.second.keyseq.toString(),
                hotkey.second.context
            };
            shortcut.controller_shortcut = hotkey.second.controller_keyseq;

            UISettings::values.shortcuts.push_back(shortcut);
        }
    }
}

void HotkeyRegistry::LoadHotkeys() { //gvx64
    for (auto shortcut : UISettings::values.shortcuts) {
        Hotkey& hk = hotkey_groups[shortcut.group][shortcut.name];

        // Load keyboard shortcut
        if (!shortcut.shortcut.keyseq.isEmpty()) {
            hk.keyseq = QKeySequence::fromString(shortcut.shortcut.keyseq, QKeySequence::NativeText);
            hk.context = static_cast<Qt::ShortcutContext>(shortcut.shortcut.context);
        }

        // Load controller shortcut - gvx64
        if (!shortcut.controller_shortcut.isEmpty()) {
            hk.controller_keyseq = shortcut.controller_shortcut;
        }

        // Update QShortcut objects
        for (auto const& [_, hotkey_shortcut] : hk.shortcuts) {
            if (hotkey_shortcut) {
                hotkey_shortcut->disconnect();
                hotkey_shortcut->setKey(hk.keyseq);
            }
        }
    }
}

QShortcut* HotkeyRegistry::GetHotkey(const QString& group, const QString& action, QObject* widget) {
    Hotkey& hk = hotkey_groups[group][action];
    const auto widget_name = widget->objectName();

    if (!hk.shortcuts[widget_name]) {
        hk.shortcuts[widget_name] = new QShortcut(hk.keyseq, widget, nullptr, nullptr, hk.context);
    }

    return hk.shortcuts[widget_name];
}

QKeySequence HotkeyRegistry::GetKeySequence(const QString& group, const QString& action) {
    Hotkey& hk = hotkey_groups[group][action];
    return hk.keyseq;
}

Qt::ShortcutContext HotkeyRegistry::GetShortcutContext(const QString& group,
                                                       const QString& action) {
    Hotkey& hk = hotkey_groups[group][action];
    return hk.context;
}

// ---------------------------------------------------------------------------------------------
// v426 - Controller hotkeys (hotkey enable button + button), symbolic bindings
// ---------------------------------------------------------------------------------------------
namespace ControllerHotkey {

namespace {

constexpr char ENGINE_3DS[] = "hotkey_3ds";
constexpr char ENGINE_PAD[] = "hotkey_pad";

// Named gamepad buttons that can be stored symbolically (SDL_GameController names).
constexpr std::array<const char*, 3> PAD_BUTTONS{{"leftstick", "rightstick", "guide"}};

QString Translate(const char* text) {
    return QCoreApplication::translate("ControllerHotkey", text);
}

int NativeButtonIndex(const std::string& name) {
    const std::string key = "button_" + name;
    for (int i = 0; i < Settings::NativeButton::NumButtons; ++i) {
        if (key == Settings::NativeButton::mapping[i]) {
            return i;
        }
    }
    return -1;
}

// Controller (engine/guid/port) of the current input profile, taken from the 3DS Select binding
// first, then Start, then any SDL binding.
bool FindProfileController(Common::ParamPackage& out) {
    const auto& buttons = Settings::values.current_input_profile.buttons;
    std::array<int, Settings::NativeButton::NumButtons + 2> order{};
    std::size_t n = 0;
    order[n++] = Settings::NativeButton::Select;
    order[n++] = Settings::NativeButton::Start;
    for (int i = 0; i < Settings::NativeButton::NumButtons; ++i) {
        order[n++] = i;
    }
    for (std::size_t k = 0; k < n; ++k) {
        const Common::ParamPackage params(buttons[order[k]]);
        if (params.Get("engine", "") == "sdl" && params.Has("guid")) {
            out = Common::ParamPackage{};
            out.Set("engine", "sdl");
            out.Set("guid", params.Get("guid", "0"));
            out.Set("port", params.Get("port", 0));
            return true;
        }
    }
    return false;
}

// EmulationStation (RetroPie) fallback: es_input.cfg uses SDL joystick indices and the same GUID
// string format as the SDL backend, so its entries map directly. Finds input `es_name` of the
// joystick `guid` and returns its type ("button", "hat", "axis"), id and value.
bool FindEmulationStationInput(const std::string& guid, const QString& es_name, QString& type,
                               int& id, int& value) {
    const std::array<QString, 2> candidates{{
        QDir::homePath() + QStringLiteral("/.emulationstation/es_input.cfg"),
        QStringLiteral("/opt/retropie/configs/all/emulationstation/es_input.cfg"),
    }};

    for (const QString& path : candidates) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            continue;
        }

        QXmlStreamReader xml(&file);
        bool in_target = false;
        while (!xml.atEnd()) {
            xml.readNext();
            if (xml.isStartElement()) {
                const auto attrs = xml.attributes();
                if (xml.name() == QStringLiteral("inputConfig")) {
                    in_target =
                        attrs.value(QStringLiteral("type")) == QStringLiteral("joystick") &&
                        attrs.value(QStringLiteral("deviceGUID")).toString().toStdString() == guid;
                } else if (in_target && xml.name() == QStringLiteral("input") &&
                           attrs.value(QStringLiteral("name")) == es_name) {
                    type = attrs.value(QStringLiteral("type")).toString();
                    id = attrs.value(QStringLiteral("id")).toInt();
                    value = attrs.value(QStringLiteral("value")).toInt();
                    return true;
                }
            } else if (xml.isEndElement() && xml.name() == QStringLiteral("inputConfig")) {
                in_target = false;
            }
        }
    }
    return false;
}

std::string FromEmulationStation(const std::string& guid, int port, const QString& es_name) {
    QString type;
    int id = 0;
    int value = 0;
    if (!FindEmulationStationInput(guid, es_name, type, id, value)) {
        return {};
    }

    Common::ParamPackage params;
    params.Set("engine", "sdl");
    params.Set("guid", guid);
    params.Set("port", port);
    if (type == QStringLiteral("button")) {
        params.Set("button", id);
    } else if (type == QStringLiteral("hat")) {
        params.Set("hat", id);
        switch (value) {
        case 1:
            params.Set("direction", "up");
            break;
        case 2:
            params.Set("direction", "right");
            break;
        case 4:
            params.Set("direction", "down");
            break;
        case 8:
            params.Set("direction", "left");
            break;
        default:
            return {};
        }
    } else if (type == QStringLiteral("axis")) {
        params.Set("axis", id);
        params.Set("direction", value < 0 ? "-" : "+");
        params.Set("threshold", value < 0 ? -0.5f : 0.5f);
    } else {
        return {};
    }
    return params.Serialize();
}

std::string ResolvePadButton(const std::string& name) {
    Common::ParamPackage controller;
    if (!FindProfileController(controller)) {
        return {};
    }

    const Common::ParamPackage bind = InputCommon::GetControllerNamedButtonBind(controller, name);
    if (bind.Has("engine")) {
        return bind.Serialize();
    }

    QString es_name;
    if (name == "leftstick") {
        es_name = QStringLiteral("leftthumb");
    } else if (name == "rightstick") {
        es_name = QStringLiteral("rightthumb");
    } else {
        return {};
    }
    return FromEmulationStation(controller.Get("guid", "0"), controller.Get("port", 0), es_name);
}

QString Display3DSButton(const std::string& name) {
    static const std::array<std::pair<const char*, const char*>, 18> names{{
        {"a", "A"},         {"b", "B"},           {"x", "X"},
        {"y", "Y"},         {"up", "D-Pad Up"},   {"down", "D-Pad Down"},
        {"left", "D-Pad Left"}, {"right", "D-Pad Right"}, {"l", "L"},
        {"r", "R"},         {"start", "Start"},   {"select", "Select"},
        {"debug", "Debug"}, {"gpio14", "Gpio14"}, {"zl", "ZL"},
        {"zr", "ZR"},       {"home", "Home"},     {"power", "Power"},
    }};
    for (const auto& [key, label] : names) {
        if (name == key) {
            return QString::fromLatin1(label);
        }
    }
    return QString::fromStdString(name);
}

} // Anonymous namespace

QString DefaultModifier() {
    return QStringLiteral("engine:hotkey_3ds,button:select");
}

std::string Resolve(const QString& binding) {
    if (binding.isEmpty()) {
        return {};
    }
    const Common::ParamPackage params(binding.toStdString());
    const std::string engine = params.Get("engine", "");
    if (engine.empty()) {
        return {};
    }
    if (engine == ENGINE_3DS) {
        const int index = NativeButtonIndex(params.Get("button", ""));
        if (index < 0) {
            return {};
        }
        return Settings::values.current_input_profile.buttons[index];
    }
    if (engine == ENGINE_PAD) {
        return ResolvePadButton(params.Get("button", ""));
    }
    return binding.toStdString();
}

std::string ResolveStick(bool right, bool& invert_x, bool& invert_y) {
    invert_x = false;
    invert_y = false;

    // 1. The stick bound in Controls: right = 3DS C-Stick, left = 3DS Circle Pad
    const auto& analogs = Settings::values.current_input_profile.analogs;
    const std::string& bound =
        analogs[right ? Settings::NativeAnalog::CStick : Settings::NativeAnalog::CirclePad];
    if (!Common::ParamPackage(bound).Get("engine", "").empty()) {
        return bound;
    }

    Common::ParamPackage controller;
    if (!FindProfileController(controller)) {
        return {};
    }

    // 2. SDL's controller database
    const Common::ParamPackage axes = InputCommon::GetControllerNamedAxesBind(
        controller, right ? "rightx" : "leftx", right ? "righty" : "lefty");
    if (axes.Has("engine")) {
        return axes.Serialize();
    }

    // 3. EmulationStation: <side>analogleft / <side>analogup, value -1 when not inverted
    const QString side = right ? QStringLiteral("right") : QStringLiteral("left");
    const std::string guid = controller.Get("guid", "0");
    QString type_x, type_y;
    int id_x = 0, id_y = 0, value_x = 0, value_y = 0;
    if (!FindEmulationStationInput(guid, side + QStringLiteral("analogleft"), type_x, id_x,
                                   value_x) ||
        !FindEmulationStationInput(guid, side + QStringLiteral("analogup"), type_y, id_y,
                                   value_y) ||
        type_x != QStringLiteral("axis") || type_y != QStringLiteral("axis")) {
        return {};
    }
    invert_x = value_x > 0;
    invert_y = value_y > 0;

    Common::ParamPackage params;
    params.Set("engine", "sdl");
    params.Set("guid", guid);
    params.Set("port", controller.Get("port", 0));
    params.Set("axis_x", id_x);
    params.Set("axis_y", id_y);
    params.Set("deadzone", 0.1f);
    return params.Serialize();
}

bool SameInput(const std::string& a, const std::string& b) {
    if (a.empty() || b.empty()) {
        return false;
    }
    if (a == b) {
        return true;
    }
    const Common::ParamPackage pa(a);
    const Common::ParamPackage pb(b);
    const std::string engine = pa.Get("engine", "");
    if (engine != pb.Get("engine", "")) {
        return false;
    }
    if (engine == "sdl") {
        if (pa.Get("guid", "") != pb.Get("guid", "") || pa.Get("port", 0) != pb.Get("port", 0)) {
            return false;
        }
        if (pa.Has("button") || pb.Has("button")) {
            return pa.Has("button") && pb.Has("button") &&
                   pa.Get("button", -1) == pb.Get("button", -1);
        }
        if (pa.Has("hat") || pb.Has("hat")) {
            return pa.Has("hat") && pb.Has("hat") && pa.Get("hat", -1) == pb.Get("hat", -1) &&
                   pa.Get("direction", "") == pb.Get("direction", "");
        }
        if (pa.Has("axis") || pb.Has("axis")) {
            return pa.Has("axis") && pb.Has("axis") && pa.Get("axis", -1) == pb.Get("axis", -1) &&
                   pa.Get("direction", "") == pb.Get("direction", "");
        }
        return false;
    }
    if (engine == "keyboard") {
        return pa.Get("code", -1) == pb.Get("code", -2);
    }
    return false;
}

QString Symbolize(const QString& raw_binding) {
    const Common::ParamPackage params(raw_binding.toStdString());
    const std::string engine = params.Get("engine", "");
    if (engine.empty() || engine == ENGINE_3DS || engine == ENGINE_PAD) {
        return raw_binding;
    }
    const std::string raw = raw_binding.toStdString();

    for (int i = 0; i < Settings::NativeButton::NumButtons; ++i) {
        if (SameInput(Settings::values.current_input_profile.buttons[i], raw)) {
            const std::string name = std::string(Settings::NativeButton::mapping[i]).substr(7);
            return QStringLiteral("engine:hotkey_3ds,button:%1").arg(QString::fromStdString(name));
        }
    }
    for (const char* name : PAD_BUTTONS) {
        if (SameInput(ResolvePadButton(name), raw)) {
            return QStringLiteral("engine:hotkey_pad,button:%1").arg(QString::fromLatin1(name));
        }
    }
    return raw_binding;
}

QString DisplayName(const QString& binding) {
    if (binding.isEmpty()) {
        return {};
    }
    const Common::ParamPackage params(binding.toStdString());
    const std::string engine = params.Get("engine", "");
    if (engine == ENGINE_3DS) {
        return Translate("%1 (3DS)").arg(Display3DSButton(params.Get("button", "")));
    }
    if (engine == ENGINE_PAD) {
        const std::string name = params.Get("button", "");
        if (name == "leftstick") {
            return QStringLiteral("L3");
        }
        if (name == "rightstick") {
            return QStringLiteral("R3");
        }
        if (name == "guide") {
            return QStringLiteral("Guide");
        }
        return QString::fromStdString(name);
    }
    if (engine == "sdl") {
        if (params.Has("button")) {
            return Translate("Button %1").arg(params.Get("button", 0));
        }
        if (params.Has("hat")) {
            return Translate("Hat %1 %2")
                .arg(params.Get("hat", 0))
                .arg(QString::fromStdString(params.Get("direction", "")));
        }
        if (params.Has("axis")) {
            return Translate("Axis %1%2")
                .arg(params.Get("axis", 0))
                .arg(QString::fromStdString(params.Get("direction", "")));
        }
    }
    if (engine == "keyboard") {
        return QKeySequence(params.Get("code", 0)).toString(QKeySequence::NativeText);
    }
    return binding;
}

} // namespace ControllerHotkey
