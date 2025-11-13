// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <QWidget>

namespace Ui {
class ConfigureHotkeys;
}

class HotkeyRegistry;
class QStandardItemModel;

class ConfigureHotkeys : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureHotkeys(QWidget* parent = nullptr);
    ~ConfigureHotkeys() override;

    void ApplyConfiguration(HotkeyRegistry& registry);
    void RetranslateUI();

    void EmitHotkeysChanged();

    /**
     * Populates the hotkey list widget using data from the provided registry.
     * Called everytime the Configure dialog is opened.
     * @param registry The HotkeyRegistry whose data is used to populate the list.
     */
    void Populate(const HotkeyRegistry& registry);

public slots:
    void OnInputKeysChanged(QList<QKeySequence> new_key_list);

signals:
    void HotkeysChanged(QList<QKeySequence> new_key_list);

private:
    void Configure(QModelIndex index);
    void ConfigureKeyboardHotkey(QModelIndex index);  // gvx64
    void ConfigureControllerHotkey(QModelIndex index); // gvx64
    std::pair<bool, QString> IsUsedKey(QKeySequence key_sequence) const;
    std::pair<bool, QString> IsUsedControllerButton(const QString& button_string) const; // gvx64
    QList<QKeySequence> GetUsedKeyList() const;

    void RestoreDefaults();
    void ClearAll();
    void PopupContextMenu(const QPoint& menu_location);
    void ShowKeyboardContextMenu(QModelIndex index, const QPoint& menu_location); // gvx64
    void ShowControllerContextMenu(QModelIndex index, const QPoint& menu_location); // gvx64
    void RestoreHotkey(QModelIndex index);

    /**
     * List of keyboard keys currently registered to any of the 3DS inputs.
     * These can't be bound to any hotkey.
     * Synchronised with ConfigureInput via signal-slot.
     */
    QList<QKeySequence> input_keys_list;

    std::unique_ptr<Ui::ConfigureHotkeys> ui;

    QStandardItemModel* model;
};
