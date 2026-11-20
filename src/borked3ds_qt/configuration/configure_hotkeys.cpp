// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QStandardItemModel>
#include "borked3ds_qt/configuration/config.h"
#include "borked3ds_qt/configuration/configure_hotkeys.h"
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/uisettings.h"
#include "borked3ds_qt/util/sequence_dialog/sequence_dialog.h"
#include "ui_configure_hotkeys.h"
#include "borked3ds_qt/util/controller_dialog/controller_dialog.h" // gvx64

constexpr int name_column = 0;
constexpr int hotkey_column = 1;
constexpr int controller_column = 2;            // gvx64
constexpr int controller_binding_role = Qt::UserRole + 1; // v426 : raw/symbolic binding

ConfigureHotkeys::ConfigureHotkeys(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureHotkeys>()) {
    ui->setupUi(this);
    setFocusPolicy(Qt::ClickFocus);

    // v426 : controller hotkey enable button (modifier), RetroPie style ("Select + Start", ...)
    auto* modifier_layout = new QHBoxLayout();
    auto* modifier_label = new QLabel(tr("Controller hotkey enable button:"), this);
    modifier_button = new QPushButton(this);
    auto* modifier_clear = new QPushButton(tr("None"), this);
    auto* modifier_default = new QPushButton(tr("Default (Select)"), this);
    const QString modifier_tooltip =
        tr("Hold this button, then press the button of a controller hotkey (e.g. Select + Start "
           "to quit).\n\"(3DS)\" bindings follow the buttons configured in Controls, so they work "
           "with any controller.\nNone: controller hotkeys trigger on a single button.");
    modifier_label->setToolTip(modifier_tooltip);
    modifier_button->setToolTip(modifier_tooltip);
    modifier_layout->addWidget(modifier_label);
    modifier_layout->addWidget(modifier_button, 1);
    modifier_layout->addWidget(modifier_clear);
    modifier_layout->addWidget(modifier_default);
    ui->verticalLayout->insertLayout(1, modifier_layout);

    connect(modifier_button, &QPushButton::clicked, this, &ConfigureHotkeys::ConfigureModifier);
    connect(modifier_clear, &QPushButton::clicked, this, [this] { SetModifier(QString{}); });
    connect(modifier_default, &QPushButton::clicked, this,
            [this] { SetModifier(ControllerHotkey::DefaultModifier()); });

    model = new QStandardItemModel(this);
    model->setColumnCount(3); // Changed from 2 - gvx64
    model->setHorizontalHeaderLabels({tr("Action"), tr("Hotkey"), tr("Controller")}); // gvx64

    connect(ui->hotkey_list, &QTreeView::doubleClicked, this, &ConfigureHotkeys::Configure);
    connect(ui->hotkey_list, &QTreeView::customContextMenuRequested, this,
            &ConfigureHotkeys::PopupContextMenu);
    ui->hotkey_list->setContextMenuPolicy(Qt::CustomContextMenu);
    ui->hotkey_list->setModel(model);

    ui->hotkey_list->setColumnWidth(0, 200); // Reduced from 250 - gvx64
    ui->hotkey_list->setColumnWidth(1, 150); // gvx64
    ui->hotkey_list->resizeColumnToContents(2); // gvx64 - controller column

    connect(ui->button_restore_defaults, &QPushButton::clicked, this,
            &ConfigureHotkeys::RestoreDefaults);
    connect(ui->button_clear_all, &QPushButton::clicked, this, &ConfigureHotkeys::ClearAll);

    SetModifier(UISettings::values.controller_hotkey_modifier);
}

ConfigureHotkeys::~ConfigureHotkeys() = default;

void ConfigureHotkeys::EmitHotkeysChanged() {
    emit HotkeysChanged(GetUsedKeyList());
}

QList<QKeySequence> ConfigureHotkeys::GetUsedKeyList() const {
    QList<QKeySequence> list;
    for (int r = 0; r < model->rowCount(); r++) {
        QStandardItem* parent = model->item(r, 0);
        for (int r2 = 0; r2 < parent->rowCount(); r2++) {
            QStandardItem* keyseq = parent->child(r2, 1);
            list << QKeySequence::fromString(keyseq->text(), QKeySequence::NativeText);
        }
    }
    return list;
}

void ConfigureHotkeys::SetControllerBinding(QStandardItem* item, const QString& binding) {
    item->setData(binding, controller_binding_role);
    item->setText(ControllerHotkey::DisplayName(binding));
    item->setToolTip(binding);
}

QString ConfigureHotkeys::GetControllerBinding(const QStandardItem* item) const {
    return item->data(controller_binding_role).toString();
}

void ConfigureHotkeys::SetModifier(const QString& binding) {
    modifier_binding = binding;
    if (binding.isEmpty()) {
        modifier_button->setText(tr("None (single-button controller hotkeys)"));
    } else {
        modifier_button->setText(ControllerHotkey::DisplayName(binding));
    }
}

void ConfigureHotkeys::ConfigureModifier() {
    ControllerDialog controller_dialog{this};

    const int return_code = controller_dialog.exec();
    const QString raw = controller_dialog.GetButtonString();
    if (return_code == QDialog::Rejected || raw.isEmpty()) {
        return;
    }
    const QString binding = ControllerHotkey::Symbolize(raw);

    // The enable button cannot also be the button of a hotkey
    const std::string resolved = ControllerHotkey::Resolve(binding);
    for (int r = 0; r < model->rowCount(); ++r) {
        const QStandardItem* const parent = model->item(r, 0);
        for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
            const QString other = GetControllerBinding(parent->child(r2, controller_column));
            if (other == binding ||
                ControllerHotkey::SameInput(ControllerHotkey::Resolve(other), resolved)) {
                QMessageBox::warning(
                    this, tr("Conflicting Button"),
                    tr("The entered button is already assigned to: %1")
                        .arg(parent->child(r2, name_column)->text()));
                return;
            }
        }
    }
    SetModifier(binding);
}

void ConfigureHotkeys::Populate(const HotkeyRegistry& registry) {
    model->clear(); // gvx64 - Clear before repopulating
    model->setColumnCount(3); // gvx64
    model->setHorizontalHeaderLabels({tr("Action"), tr("Hotkey"), tr("Controller")}); // gvx64

    for (const auto& group : registry.hotkey_groups) {
        QStandardItem* parent_item = new QStandardItem(group.first);
        parent_item->setEditable(false);

        for (const auto& hotkey : group.second) {
            QStandardItem* action = new QStandardItem(hotkey.first);
            QStandardItem* keyseq =
                new QStandardItem(hotkey.second.keyseq.toString(QKeySequence::NativeText));
            QStandardItem* controller_keyseq = new QStandardItem(); // gvx64
            SetControllerBinding(controller_keyseq, hotkey.second.controller_keyseq); // v426

            action->setEditable(false);
            keyseq->setEditable(false);
            controller_keyseq->setEditable(false); // gvx64

            parent_item->appendRow({action, keyseq, controller_keyseq}); // gvx64 - 3 items
        }
        model->appendRow(parent_item);
    }

    ui->hotkey_list->expandAll();
    ui->hotkey_list->resizeColumnToContents(controller_column);

    SetModifier(UISettings::values.controller_hotkey_modifier); // v426
}

void ConfigureHotkeys::OnInputKeysChanged(QList<QKeySequence> new_key_list) {
    input_keys_list = new_key_list;
}

void ConfigureHotkeys::Configure(QModelIndex index) {
    if (!index.parent().isValid()) {
        return;
    }

    // Determine which column was clicked - gvx64
    const int column = index.column();

    if (column == 1) {
        // Keyboard hotkey column (existing functionality)
        ConfigureKeyboardHotkey(index);
    } else if (column == 2) {
        // Controller button column - gvx64
        ConfigureControllerHotkey(index);
    }
}

void ConfigureHotkeys::ConfigureKeyboardHotkey(QModelIndex index) {
    // Swap to the hotkey column
    index = index.sibling(index.row(), 1); // Column 1 = keyboard hotkey

    const auto previous_key = model->data(index);

    SequenceDialog hotkey_dialog{this};

    const int return_code = hotkey_dialog.exec();
    const auto key_sequence = hotkey_dialog.GetSequence();
    if (return_code == QDialog::Rejected || key_sequence.isEmpty()) {
        return;
    }
    const auto [key_sequence_used, used_action] = IsUsedKey(key_sequence);

    // Check for turbo/per-game speed conflict. Needed to prevent the user from binding both hotkeys
    // to the same action. Which causes problems resetting the frame limit to the initial value.
    const QString current_action =
        model->data(model->index(index.row(), 0, index.parent())).toString();
    const bool is_turbo = current_action == tr("Toggle Turbo Mode");
    const bool is_per_game = current_action == tr("Toggle Per-Game Speed");

    if (is_turbo || is_per_game) {
        QString other_action = is_turbo ? tr("Toggle Per-Game Speed") : tr("Toggle Turbo Mode");
        QKeySequence other_sequence;

        for (int r = 0; r < model->rowCount(); ++r) {
            const QStandardItem* const parent = model->item(r, 0);
            for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
                if (parent->child(r2, 0)->text() == other_action) {
                    other_sequence = QKeySequence::fromString(
                        parent->child(r2, 1)->text(), QKeySequence::NativeText);
                    break;
                }
            }
        }

        // Show warning if either hotkey is already set
        if (!key_sequence.isEmpty() && !other_sequence.isEmpty()) {
            QMessageBox::warning(
                this, tr("Conflicting Key Sequence"),
                tr("The per-game speed and turbo speed hotkeys cannot be bound at the same time."));
            return;
        }
    }

    if (key_sequence_used && key_sequence != QKeySequence(previous_key.toString())) {
        QMessageBox::warning(
            this, tr("Conflicting Key Sequence"),
            tr("The entered key sequence is already assigned to: %1").arg(used_action));
    } else {
        model->setData(index, key_sequence.toString(QKeySequence::NativeText));
        EmitHotkeysChanged();
    }
}

void ConfigureHotkeys::ConfigureControllerHotkey(QModelIndex index) { //gvx64
    // Swap to the controller column
    index = index.sibling(index.row(), controller_column); // Column 2 = controller - gvx64
    QStandardItem* const item = model->itemFromIndex(index);
    if (!item) {
        return;
    }

    const QString previous_button = GetControllerBinding(item);

    ControllerDialog controller_dialog{this};

    const int return_code = controller_dialog.exec();
    const QString raw_button = controller_dialog.GetButtonString();

    if (return_code == QDialog::Rejected || raw_button.isEmpty()) {
        return;
    }

    // v426 : store "(3DS)" / L3 / R3 bindings symbolically so they follow the configured controls
    const QString button_string = ControllerHotkey::Symbolize(raw_button);

    // v426 : the hotkey enable button cannot also be a hotkey button
    if (!modifier_binding.isEmpty() &&
        (button_string == modifier_binding ||
         ControllerHotkey::SameInput(ControllerHotkey::Resolve(button_string),
                                     ControllerHotkey::Resolve(modifier_binding)))) {
        QMessageBox::warning(this, tr("Conflicting Button"),
                             tr("This button is the controller hotkey enable button."));
        return;
    }

    // Check if this button is already used
    const auto [button_used, used_action] = IsUsedControllerButton(button_string);

    if (button_used && button_string != previous_button) {
        QMessageBox::warning(
            this, tr("Conflicting Button"),
            tr("The entered button is already assigned to: %1").arg(used_action));
    } else {
        SetControllerBinding(item, button_string);
    }
}

std::pair<bool, QString> ConfigureHotkeys::IsUsedControllerButton(const QString& button_string) const { //gvx64
    if (button_string.isEmpty()) {
        return std::make_pair(false, QString());
    }
    const std::string resolved = ControllerHotkey::Resolve(button_string); // v426

    for (int r = 0; r < model->rowCount(); ++r) {
        const QStandardItem* const parent = model->item(r, 0);

        for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
            const QStandardItem* const controller_item = parent->child(r2, controller_column);
            const auto controller_str = GetControllerBinding(controller_item);

            if (button_string == controller_str ||
                ControllerHotkey::SameInput(ControllerHotkey::Resolve(controller_str),
                                            resolved)) {
                return std::make_pair(true, parent->child(r2, 0)->text());
            }
        }
    }

    return std::make_pair(false, QString());
}

std::pair<bool, QString> ConfigureHotkeys::IsUsedKey(QKeySequence key_sequence) const {
    if (key_sequence == QKeySequence::fromString(QStringLiteral(""), QKeySequence::NativeText)) {
        return std::make_pair(false, QString());
    }

    if (input_keys_list.contains(key_sequence)) {
        return std::make_pair(true, tr("A 3ds button"));
    }

    for (int r = 0; r < model->rowCount(); ++r) {
        const QStandardItem* const parent = model->item(r, 0);

        for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
            const QStandardItem* const key_seq_item = parent->child(r2, hotkey_column);
            const auto key_seq_str = key_seq_item->text();
            const auto key_seq = QKeySequence::fromString(key_seq_str, QKeySequence::NativeText);

            if (key_sequence == key_seq) {
                return std::make_pair(true, parent->child(r2, 0)->text());
            }
        }
    }

    return std::make_pair(false, QString());
}

void ConfigureHotkeys::ApplyConfiguration(HotkeyRegistry& registry) {
    for (int key_id = 0; key_id < model->rowCount(); key_id++) {
        QStandardItem* parent = model->item(key_id, 0);
        for (int key_column_id = 0; key_column_id < parent->rowCount(); key_column_id++) {
            const QStandardItem* action = parent->child(key_column_id, 0);
            const QStandardItem* keyseq = parent->child(key_column_id, 1);
            const QStandardItem* controller_keyseq = parent->child(key_column_id, 2); // gvx64

            for (auto& [group, sub_actions] : registry.hotkey_groups) {
                if (group != parent->text())
                    continue;
                for (auto& [action_name, hotkey] : sub_actions) {
                    if (action_name != action->text())
                        continue;
                    hotkey.keyseq = QKeySequence(keyseq->text());
                    hotkey.controller_keyseq = GetControllerBinding(controller_keyseq); // v426
                }
            }
        }
    }

    UISettings::values.controller_hotkey_modifier = modifier_binding; // v426
    registry.SaveHotkeys();
}

void ConfigureHotkeys::RestoreDefaults() {
    for (int r = 0; r < model->rowCount(); ++r) {
        const QStandardItem* parent = model->item(r, 0);

        for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
            // v426 : defaults looked up by name (the array order used to be relied upon)
            const UISettings::Shortcut* default_hotkey =
                Config::FindDefaultHotkey(parent->text(), parent->child(r2, name_column)->text());
            model->item(r, 0)->child(r2, 1)->setText(
                default_hotkey ? default_hotkey->shortcut.keyseq : QString{});
            SetControllerBinding(model->item(r, 0)->child(r2, controller_column),
                                 default_hotkey ? default_hotkey->controller_shortcut : QString{});
        }
    }
    SetModifier(ControllerHotkey::DefaultModifier()); // v426
}

void ConfigureHotkeys::ClearAll() {
    for (int r = 0; r < model->rowCount(); ++r) {
        const QStandardItem* parent = model->item(r, 0);

        for (int r2 = 0; r2 < parent->rowCount(); ++r2) {
            model->item(r, 0)->child(r2, 1)->setText(QString{}); // Keyboard
            SetControllerBinding(model->item(r, 0)->child(r2, controller_column),
                                 QString{}); // Controller - gvx64
        }
    }
}

void ConfigureHotkeys::PopupContextMenu(const QPoint& menu_location) {
    const auto index = ui->hotkey_list->indexAt(menu_location);
    if (!index.parent().isValid()) {
        return;
    }

    // Determine which column was clicked
    const int column = index.column();

    if (column == 1) {
        // Keyboard hotkey column
        ShowKeyboardContextMenu(index, menu_location);
    } else if (column == 2) {
        // Controller button column - gvx64
        ShowControllerContextMenu(index, menu_location);
    }
}

void ConfigureHotkeys::ShowKeyboardContextMenu(QModelIndex index, const QPoint& menu_location) {
    QMenu context_menu;
    QAction* restore_default = context_menu.addAction(tr("Restore Default"));
    QAction* clear = context_menu.addAction(tr("Clear"));

    const auto hotkey_index = index.sibling(index.row(), 1); // Keyboard column
    connect(restore_default, &QAction::triggered, this,
            [this, hotkey_index] { RestoreHotkey(hotkey_index); });
    connect(clear, &QAction::triggered, this,
            [this, hotkey_index] { model->setData(hotkey_index, QString{}); });

    context_menu.exec(ui->hotkey_list->viewport()->mapToGlobal(menu_location));
}

void ConfigureHotkeys::ShowControllerContextMenu(QModelIndex index, const QPoint& menu_location) {
    QMenu context_menu;
    QAction* restore_default = context_menu.addAction(tr("Restore Default")); // v426
    QAction* clear = context_menu.addAction(tr("Clear"));

    const auto controller_index = index.sibling(index.row(), controller_column); // Controller column
    connect(restore_default, &QAction::triggered, this,
            [this, controller_index] { RestoreControllerHotkey(controller_index); });
    connect(clear, &QAction::triggered, this, [this, controller_index] {
        if (QStandardItem* item = model->itemFromIndex(controller_index)) {
            SetControllerBinding(item, QString{});
        }
    });

    context_menu.exec(ui->hotkey_list->viewport()->mapToGlobal(menu_location));
}

void ConfigureHotkeys::RestoreHotkey(QModelIndex index) {
    // v426 : default looked up by group/action name instead of the row index
    const QString group = index.parent().data().toString();
    const QString action = index.sibling(index.row(), name_column).data().toString();
    const UISettings::Shortcut* default_hotkey = Config::FindDefaultHotkey(group, action);
    if (!default_hotkey) {
        return;
    }
    const QKeySequence& default_key_sequence =
        QKeySequence::fromString(default_hotkey->shortcut.keyseq, QKeySequence::NativeText);
    const auto [key_sequence_used, used_action] = IsUsedKey(default_key_sequence);

    if (key_sequence_used && default_key_sequence != QKeySequence(model->data(index).toString())) {
        QMessageBox::warning(
            this, tr("Conflicting Key Sequence"),
            tr("The default key sequence is already assigned to: %1").arg(used_action));
    } else {
        model->setData(index, default_key_sequence.toString(QKeySequence::NativeText));
    }
}

void ConfigureHotkeys::RestoreControllerHotkey(QModelIndex index) { // v426
    QStandardItem* const item = model->itemFromIndex(index);
    if (!item) {
        return;
    }
    const QString group = index.parent().data().toString();
    const QString action = index.sibling(index.row(), name_column).data().toString();
    const UISettings::Shortcut* default_hotkey = Config::FindDefaultHotkey(group, action);
    const QString binding = default_hotkey ? default_hotkey->controller_shortcut : QString{};

    const auto [button_used, used_action] = IsUsedControllerButton(binding);
    if (button_used && binding != GetControllerBinding(item)) {
        QMessageBox::warning(
            this, tr("Conflicting Button"),
            tr("The default button is already assigned to: %1").arg(used_action));
    } else {
        SetControllerBinding(item, binding);
    }
}

void ConfigureHotkeys::RetranslateUI() {
    ui->retranslateUi(this);
}
