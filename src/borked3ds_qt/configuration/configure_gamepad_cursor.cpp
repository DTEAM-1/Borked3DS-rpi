// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include "borked3ds_qt/configuration/configure_gamepad_cursor.h"
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/util/controller_dialog/controller_dialog.h"
#include "common/settings.h"

ConfigureGamepadCursor::ConfigureGamepadCursor(QWidget* parent) : QWidget(parent) {
    auto* const layout = new QVBoxLayout(this);

    enable_checkbox = new QCheckBox(this);
    layout->addWidget(enable_checkbox);

    description_label = new QLabel(this);
    description_label->setWordWrap(true);
    layout->addWidget(description_label);

    options_group = new QGroupBox(this);
    auto* const form = new QFormLayout(options_group);

    in_games_checkbox = new QCheckBox(options_group);
    form->addRow(in_games_checkbox);

    stick_label = new QLabel(options_group);
    stick_combobox = new QComboBox(options_group);
    stick_combobox->addItem(QString{}, 0u);
    stick_combobox->addItem(QString{}, 1u);
    form->addRow(stick_label, stick_combobox);

    // v429 : sensitivity 1 - 100 %, default 50 % (interface and games)
    speed_label = new QLabel(options_group);
    speed_slider = new QSlider(Qt::Horizontal, options_group);
    speed_slider->setRange(1, 100);
    speed_slider->setSingleStep(1);
    speed_slider->setPageStep(10);
    speed_slider->setTickInterval(10);
    speed_slider->setTickPosition(QSlider::TicksBelow);
    speed_value_label = new QLabel(options_group);
    speed_value_label->setMinimumWidth(40);
    auto* const speed_layout = new QHBoxLayout();
    speed_layout->addWidget(speed_slider, 1);
    speed_layout->addWidget(speed_value_label);
    form->addRow(speed_label, speed_layout);

    for (std::size_t i = 0; i < click_buttons.size(); ++i) {
        click_labels[i] = new QLabel(options_group);
        click_buttons[i] = new QPushButton(options_group);
        clear_buttons[i] = new QPushButton(options_group);
        auto* const click_layout = new QHBoxLayout();
        click_layout->addWidget(click_buttons[i], 1);
        click_layout->addWidget(clear_buttons[i]);
        form->addRow(click_labels[i], click_layout);
        connect(click_buttons[i], &QPushButton::clicked, this, [this, i] { ConfigureClick(i); });
        connect(clear_buttons[i], &QPushButton::clicked, this, [this, i] { SetClick(i, {}); });
    }

    defaults_button = new QPushButton(options_group);
    form->addRow(QString{}, defaults_button);
    layout->addWidget(options_group);

    status_label = new QLabel(this);
    status_label->setWordWrap(true);
    layout->addWidget(status_label);
    layout->addStretch(1);

    connect(enable_checkbox, &QCheckBox::toggled, this, [this] { UpdateState(); });
    connect(stick_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this] { UpdateStatus(); });
    connect(speed_slider, &QSlider::valueChanged, this, [this] { UpdateSpeedLabel(); });
    connect(defaults_button, &QPushButton::clicked, this, [this] {
        stick_combobox->setCurrentIndex(stick_combobox->findData(1u));
        in_games_checkbox->setChecked(true);
        speed_slider->setValue(
            static_cast<int>(Settings::values.gamepad_cursor_sensitivity.GetDefault()));
        SetClick(0, QString::fromStdString(Settings::values.gamepad_cursor_click1.GetDefault()));
        SetClick(1, QString::fromStdString(Settings::values.gamepad_cursor_click2.GetDefault()));
    });

    ApplyTexts();
    SetConfiguration();
}

ConfigureGamepadCursor::~ConfigureGamepadCursor() = default;

void ConfigureGamepadCursor::ApplyTexts() {
    enable_checkbox->setText(tr("Enable the gamepad cursor (interface and games)"));
    description_label->setText(
        tr("Moves the mouse pointer with a stick, everywhere: menus, game list, settings and the "
           "3DS touch screen. L3 or R3 clicks (hold to drag, click twice to open a game). The "
           "arrow hides after 1 second without movement, comes back as soon as the stick moves "
           "and never shows for the touch screen. The mouse and the touch screen keep "
           "working.\nWhen enabled, the per-game touch cursor (game Properties > Controls) is "
           "not used."));
    in_games_checkbox->setText(tr("Also in games (off: only in the interface and when the game "
                                  "is paused, e.g. with the Select + X menu)"));
    options_group->setTitle(tr("Cursor"));
    stick_label->setText(tr("Stick:"));
    stick_combobox->setItemText(0, tr("Left stick"));
    stick_combobox->setItemText(1, tr("Right stick"));
    speed_label->setText(tr("Sensitivity:"));
    speed_slider->setToolTip(tr("Pointer speed, in the interface and in games (default 50 %)"));
    click_labels[0]->setText(tr("Click button 1:"));
    click_labels[1]->setText(tr("Click button 2:"));
    for (QPushButton* const button : click_buttons) {
        button->setToolTip(tr("Click, then press the gamepad button that clicks. Choosing one "
                              "button replaces the automatic L3 / R3 pair: add a second one "
                              "here if you want two."));
    }
    for (QPushButton* const button : clear_buttons) {
        button->setText(tr("Clear"));
    }
    defaults_button->setText(tr("Restore Defaults (right stick, also in games, 50 %, automatic "
                                "L3 and R3)"));
    for (std::size_t i = 0; i < click_buttons.size(); ++i) {
        SetClick(i, click_bindings[i]);
    }
    UpdateSpeedLabel();
    UpdateStatus();
}

void ConfigureGamepadCursor::SetConfiguration() {
    enable_checkbox->setChecked(Settings::values.gamepad_cursor_enabled.GetValue());
    in_games_checkbox->setChecked(Settings::values.gamepad_cursor_in_games.GetValue());
    const int stick_index =
        stick_combobox->findData(Settings::values.gamepad_cursor_stick.GetValue() != 0 ? 1u : 0u);
    stick_combobox->setCurrentIndex(stick_index < 0 ? 1 : stick_index);
    speed_slider->setValue(static_cast<int>(Settings::values.gamepad_cursor_sensitivity.GetValue()));
    SetClick(0, QString::fromStdString(Settings::values.gamepad_cursor_click1.GetValue()));
    SetClick(1, QString::fromStdString(Settings::values.gamepad_cursor_click2.GetValue()));
    UpdateState();
}

void ConfigureGamepadCursor::ApplyConfiguration() {
    Settings::values.gamepad_cursor_enabled = enable_checkbox->isChecked();
    Settings::values.gamepad_cursor_in_games = in_games_checkbox->isChecked();
    Settings::values.gamepad_cursor_stick = stick_combobox->currentData().toUInt();
    Settings::values.gamepad_cursor_sensitivity = static_cast<u32>(speed_slider->value());
    Settings::values.gamepad_cursor_click1 = click_bindings[0].toStdString();
    Settings::values.gamepad_cursor_click2 = click_bindings[1].toStdString();
}

void ConfigureGamepadCursor::RetranslateUI() {
    ApplyTexts();
}

void ConfigureGamepadCursor::UpdateState() {
    options_group->setEnabled(enable_checkbox->isChecked());
    UpdateStatus();
}

void ConfigureGamepadCursor::UpdateSpeedLabel() {
    speed_value_label->setText(QStringLiteral("%1 %").arg(speed_slider->value()));
}

void ConfigureGamepadCursor::SetClick(std::size_t index, const QString& binding) {
    click_bindings[index] = binding;
    const QString automatic =
        index == 0 ? QString::fromStdString(Settings::values.gamepad_cursor_click1.GetDefault())
                   : QString::fromStdString(Settings::values.gamepad_cursor_click2.GetDefault());
    QString text = binding.isEmpty() ? tr("[none]") : ControllerHotkey::DisplayName(binding);
    if (!binding.isEmpty() && binding == automatic) {
        text = tr("%1 (automatic)").arg(text);
    }
    click_buttons[index]->setText(text);
    if (clear_buttons[index]) {
        clear_buttons[index]->setEnabled(!binding.isEmpty());
    }
    UpdateStatus();
}

void ConfigureGamepadCursor::ConfigureClick(std::size_t index) {
    ControllerDialog dialog{this};
    const int result = dialog.exec();
    const QString raw = dialog.GetButtonString();
    if (result == QDialog::Rejected || raw.isEmpty()) {
        return;
    }
    // Stored as "L3" / "R3" / "(3DS)" when it matches, so it follows the configured controls
    SetClick(index, ControllerHotkey::Symbolize(raw));

    // A button chosen by the player replaces the automatic pair: the other slot is cleared if it
    // still holds its automatic value (a second button can then be chosen there).
    const std::size_t other = index == 0 ? 1 : 0;
    const QString other_automatic =
        other == 0 ? QString::fromStdString(Settings::values.gamepad_cursor_click1.GetDefault())
                   : QString::fromStdString(Settings::values.gamepad_cursor_click2.GetDefault());
    if (click_bindings[other] == other_automatic && click_bindings[other] != click_bindings[index]) {
        SetClick(other, {});
    }
}

void ConfigureGamepadCursor::UpdateStatus() {
    if (!status_label || !stick_combobox) {
        return;
    }
    if (!enable_checkbox->isChecked()) {
        status_label->setText(tr("Gamepad cursor disabled. The per-game touch cursor (game "
                                 "Properties > Controls) can be used instead."));
        return;
    }

    bool invert_x = false;
    bool invert_y = false;
    const bool right = stick_combobox->currentData().toUInt() != 0;
    const bool stick_found = !ControllerHotkey::ResolveStick(right, invert_x, invert_y).empty();

    QStringList lines;
    lines << (stick_found
                  ? tr("Stick: found.")
                  : tr("Stick: not found. Bind the %1 in Input, or configure the %2 analog stick "
                       "in EmulationStation.")
                        .arg(right ? tr("C-Stick") : tr("Circle Pad"),
                             right ? tr("right") : tr("left")));
    for (std::size_t i = 0; i < click_bindings.size(); ++i) {
        if (!click_bindings[i].isEmpty() &&
            ControllerHotkey::Resolve(click_bindings[i]).empty()) {
            lines << tr("Click button %1 (%2): not found on this controller, click the button "
                        "above and press it.")
                         .arg(i + 1)
                         .arg(ControllerHotkey::DisplayName(click_bindings[i]));
        }
    }
    if (click_bindings[0].isEmpty() && click_bindings[1].isEmpty()) {
        lines << tr("No click button: the stick moves the arrow only.");
    }
    lines << tr("No click while the controller hotkey enable button is held (Select + L3 / R3 "
                "are hotkeys).");
    status_label->setText(lines.join(QLatin1Char('\n')));
}
