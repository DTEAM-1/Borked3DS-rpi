// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "borked3ds_qt/configuration/configure_touch_cursor.h"
#include "common/settings.h"
#include "ui_configure_touch_cursor.h"

ConfigureTouchCursor::ConfigureTouchCursor(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureTouchCursor>()) {
    ui->setupUi(this);

    // Populate analog stick combo box
    ui->analog_stick_combobox->addItem(tr("Circle Pad"), 0);
    ui->analog_stick_combobox->addItem(tr("C-Stick"), 1);

    // Populate touch button combo box
    // Using same button names as in configure_input.cpp
    ui->touch_button_combobox->addItem(tr("A"), Settings::NativeButton::A);
    ui->touch_button_combobox->addItem(tr("B"), Settings::NativeButton::B);
    ui->touch_button_combobox->addItem(tr("X"), Settings::NativeButton::X);
    ui->touch_button_combobox->addItem(tr("Y"), Settings::NativeButton::Y);
    ui->touch_button_combobox->addItem(tr("L"), Settings::NativeButton::L);
    ui->touch_button_combobox->addItem(tr("R"), Settings::NativeButton::R);
    ui->touch_button_combobox->addItem(tr("Start"), Settings::NativeButton::Start);
    ui->touch_button_combobox->addItem(tr("Select"), Settings::NativeButton::Select);
    ui->touch_button_combobox->addItem(tr("Debug"), Settings::NativeButton::Debug);
    ui->touch_button_combobox->addItem(tr("Gpio14"), Settings::NativeButton::Gpio14); // gvx64 - rarely used button for touch cursor

    // Set sensitivity slider range (0.1 to 5.0, mapped to 1-50)
    ui->sensitivity_slider->setMinimum(1);
    ui->sensitivity_slider->setMaximum(50);
    ui->sensitivity_slider->setTickInterval(5);
    ui->sensitivity_slider->setTickPosition(QSlider::TicksBelow);

    SetConfiguration();

    // Connect signals
    connect(ui->enable_touch_cursor, &QCheckBox::toggled, this,
            &ConfigureTouchCursor::OnEnabledToggled);
    connect(ui->analog_stick_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureTouchCursor::OnAnalogStickChanged);
    connect(ui->touch_button_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureTouchCursor::OnTouchButtonChanged);
    connect(ui->sensitivity_slider, &QSlider::valueChanged, this,
            &ConfigureTouchCursor::OnSensitivityChanged);
}

ConfigureTouchCursor::~ConfigureTouchCursor() = default;

void ConfigureTouchCursor::SetConfiguration() {
    ui->enable_touch_cursor->setChecked(Settings::values.touch_cursor_enabled.GetValue());

    // Set analog stick selection
    int analog_index = ui->analog_stick_combobox->findData(
        Settings::values.touch_cursor_analog_stick.GetValue());
    if (analog_index != -1) {
        ui->analog_stick_combobox->setCurrentIndex(analog_index);
    }

    // Set touch button selection
    int button_index = ui->touch_button_combobox->findData(
        Settings::values.touch_cursor_button.GetValue());
    if (button_index != -1) {
        ui->touch_button_combobox->setCurrentIndex(button_index);
    }

    // Set sensitivity (convert 0.1-5.0 to 1-50)
    float sensitivity = Settings::values.touch_cursor_sensitivity.GetValue();
    int slider_value = static_cast<int>(sensitivity * 10.0f);
    ui->sensitivity_slider->setValue(slider_value);
//gvx64    ui->sensitivity_value_label->setText(QString::number(sensitivity, 'f', 1) + "x");
    ui->sensitivity_value_label->setText(QStringLiteral("%1x").arg(sensitivity, 0, 'f', 1));
    UpdateUIState();
}

void ConfigureTouchCursor::ApplyConfiguration() {
    Settings::values.touch_cursor_enabled = ui->enable_touch_cursor->isChecked();
    Settings::values.touch_cursor_analog_stick =
        ui->analog_stick_combobox->currentData().toUInt();
    Settings::values.touch_cursor_button = ui->touch_button_combobox->currentData().toUInt();

    // Convert slider value (1-50) back to sensitivity (0.1-5.0)
    float sensitivity = ui->sensitivity_slider->value() / 10.0f;
    Settings::values.touch_cursor_sensitivity = sensitivity;
}

void ConfigureTouchCursor::UpdateUIState() {
    bool enabled = ui->enable_touch_cursor->isChecked();
    ui->analog_stick_label->setEnabled(enabled);
    ui->analog_stick_combobox->setEnabled(enabled);
    ui->touch_button_label->setEnabled(enabled);
    ui->touch_button_combobox->setEnabled(enabled);
    ui->sensitivity_label->setEnabled(enabled);
    ui->sensitivity_slider->setEnabled(enabled);
    ui->sensitivity_value_label->setEnabled(enabled);
}

void ConfigureTouchCursor::OnEnabledToggled(bool enabled) {
    UpdateUIState();
}

void ConfigureTouchCursor::OnAnalogStickChanged([[maybe_unused]] int index) {
    // No immediate action needed, saved on Apply
}

void ConfigureTouchCursor::OnTouchButtonChanged([[maybe_unused]] int index) {
    // No immediate action needed, saved on Apply
}

void ConfigureTouchCursor::OnSensitivityChanged(int value) {
    float sensitivity = value / 10.0f;
//gvx64    ui->sensitivity_value_label->setText(QString::number(sensitivity, 'f', 1) + "x");
    ui->sensitivity_value_label->setText(QStringLiteral("%1x").arg(sensitivity, 0, 'f', 1));
}

void ConfigureTouchCursor::RetranslateUI() {
    ui->retranslateUi(this);
}
