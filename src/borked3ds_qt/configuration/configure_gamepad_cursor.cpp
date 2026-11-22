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

    stick_label = new QLabel(options_group);
    stick_combobox = new QComboBox(options_group);
    stick_combobox->addItem(QString{}, 0u);
    stick_combobox->addItem(QString{}, 1u);
    form->addRow(stick_label, stick_combobox);

    // Sensitivity 0.1x - 5.0x mapped to 1 - 50, as gvx64's per-game cursor
    speed_label = new QLabel(options_group);
    speed_slider = new QSlider(Qt::Horizontal, options_group);
    speed_slider->setRange(1, 50);
    speed_slider->setTickInterval(5);
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
        form->addRow(click_labels[i], click_buttons[i]);
        connect(click_buttons[i], &QPushButton::clicked, this, [this, i] { ConfigureClick(i); });
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
        speed_slider->setValue(10);
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
           "pointer hides after 3 seconds and comes back as soon as the stick moves. The mouse "
           "and the touch screen keep working.\nWhen enabled, the per-game touch cursor (game "
           "Properties > Controls) is not used."));
    options_group->setTitle(tr("Cursor"));
    stick_label->setText(tr("Stick:"));
    stick_combobox->setItemText(0, tr("Left stick"));
    stick_combobox->setItemText(1, tr("Right stick"));
    speed_label->setText(tr("Speed:"));
    speed_slider->setToolTip(tr("Pointer speed at full tilt (0.1x - 5.0x)"));
    click_labels[0]->setText(tr("Click button 1:"));
    click_labels[1]->setText(tr("Click button 2:"));
    for (QPushButton* const button : click_buttons) {
        button->setToolTip(tr("Click, then press the gamepad button that clicks"));
    }
    defaults_button->setText(tr("Restore Defaults (right stick, L3 and R3)"));
    for (std::size_t i = 0; i < click_buttons.size(); ++i) {
        SetClick(i, click_bindings[i]);
    }
    UpdateSpeedLabel();
    UpdateStatus();
}

void ConfigureGamepadCursor::SetConfiguration() {
    enable_checkbox->setChecked(Settings::values.gamepad_cursor_enabled.GetValue());
    const int stick_index =
        stick_combobox->findData(Settings::values.gamepad_cursor_stick.GetValue() != 0 ? 1u : 0u);
    stick_combobox->setCurrentIndex(stick_index < 0 ? 1 : stick_index);
    speed_slider->setValue(
        static_cast<int>(Settings::values.gamepad_cursor_speed.GetValue() * 10.0f + 0.5f));
    SetClick(0, QString::fromStdString(Settings::values.gamepad_cursor_click1.GetValue()));
    SetClick(1, QString::fromStdString(Settings::values.gamepad_cursor_click2.GetValue()));
    UpdateState();
}

void ConfigureGamepadCursor::ApplyConfiguration() {
    Settings::values.gamepad_cursor_enabled = enable_checkbox->isChecked();
    Settings::values.gamepad_cursor_stick = stick_combobox->currentData().toUInt();
    Settings::values.gamepad_cursor_speed = static_cast<float>(speed_slider->value()) / 10.0f;
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
    speed_value_label->setText(
        QStringLiteral("%1x").arg(static_cast<double>(speed_slider->value()) / 10.0, 0, 'f', 1));
}

void ConfigureGamepadCursor::SetClick(std::size_t index, const QString& binding) {
    click_bindings[index] = binding;
    click_buttons[index]->setText(binding.isEmpty() ? tr("[not set]")
                                                    : ControllerHotkey::DisplayName(binding));
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
    lines << tr("No click while the controller hotkey enable button is held (Select + L3 / R3 "
                "are hotkeys).");
    status_label->setText(lines.join(QLatin1Char('\n')));
}
