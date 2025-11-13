// Copyright 2014 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QShortcut>
#include <QtGlobal>
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/uisettings.h"

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
