// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "borked3ds_qt/configuration/configure_controls.h"
#include "common/settings.h"
#include "ui_configure_controls.h"

ConfigureControls::ConfigureControls(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureControls>()) {
    ui->setupUi(this);

    // Populate profile combo box - gvx64
    for (size_t i = 0; i < Settings::values.input_profiles.size(); ++i) {
        ui->profile_combobox->addItem(
            QString::fromStdString(Settings::values.input_profiles[i].name),
            static_cast<int>(i));
    }

    // Populate analog stick combo box
    ui->analog_stick_combobox->addItem(tr("Circle Pad"), 0);
    ui->analog_stick_combobox->addItem(tr("C-Stick"), 1);

    // Populate touch button combo box
    ui->touch_button_combobox->addItem(tr("A"), Settings::NativeButton::A);
    ui->touch_button_combobox->addItem(tr("B"), Settings::NativeButton::B);
    ui->touch_button_combobox->addItem(tr("X"), Settings::NativeButton::X);
    ui->touch_button_combobox->addItem(tr("Y"), Settings::NativeButton::Y);
    ui->touch_button_combobox->addItem(tr("L"), Settings::NativeButton::L);
    ui->touch_button_combobox->addItem(tr("R"), Settings::NativeButton::R);
    ui->touch_button_combobox->addItem(tr("Start"), Settings::NativeButton::Start);
    ui->touch_button_combobox->addItem(tr("Select"), Settings::NativeButton::Select);
    ui->touch_button_combobox->addItem(tr("Debug"), Settings::NativeButton::Debug);
    ui->touch_button_combobox->addItem(tr("Gpio14"), Settings::NativeButton::Gpio14);

    // Set sensitivity slider range (0.1 to 5.0, mapped to 1-50)
    ui->sensitivity_slider->setMinimum(1);
    ui->sensitivity_slider->setMaximum(50);
    ui->sensitivity_slider->setTickInterval(5);
    ui->sensitivity_slider->setTickPosition(QSlider::TicksBelow);

    SetConfiguration();

    // Connect signals
    connect(ui->profile_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureControls::OnProfileChanged);
    connect(ui->enable_touch_cursor, &QCheckBox::toggled, this,
            &ConfigureControls::OnTouchCursorEnabledToggled);
    connect(ui->analog_stick_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureControls::OnAnalogStickChanged);
    connect(ui->touch_button_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureControls::OnTouchButtonChanged);
    connect(ui->sensitivity_slider, &QSlider::valueChanged, this,
            &ConfigureControls::OnSensitivityChanged);
}

ConfigureControls::~ConfigureControls() = default;

void ConfigureControls::SetConfiguration() {
    // Set profile selection - gvx64
    int profile_index = Settings::values.input_profile_index.GetValue();

    // If no profile is set (-1), default to the first profile (usually "default")
    if (profile_index < 0 || profile_index >= static_cast<int>(Settings::values.input_profiles.size())) {
        profile_index = 0;
    }

    int combo_index = ui->profile_combobox->findData(profile_index);
    if (combo_index != -1) {
        ui->profile_combobox->setCurrentIndex(combo_index);
    } else {
        ui->profile_combobox->setCurrentIndex(0); // Fallback to first profile
    }
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

    // Set sensitivity
    float sensitivity = Settings::values.touch_cursor_sensitivity.GetValue();
    int slider_value = static_cast<int>(sensitivity * 10.0f);
    ui->sensitivity_slider->setValue(slider_value);
    ui->sensitivity_value_label->setText(QStringLiteral("%1x").arg(sensitivity, 0, 'f', 1));

    UpdateUIState();
}

void ConfigureControls::ApplyConfiguration() {
    // Apply profile selection - gvx64
    Settings::values.input_profile_index = ui->profile_combobox->currentData().toInt();

    Settings::values.touch_cursor_enabled = ui->enable_touch_cursor->isChecked();
    Settings::values.touch_cursor_analog_stick =
        ui->analog_stick_combobox->currentData().toUInt();
    Settings::values.touch_cursor_button = ui->touch_button_combobox->currentData().toUInt();

    // Convert slider value (1-50) back to sensitivity (0.1-5.0)
    float sensitivity = ui->sensitivity_slider->value() / 10.0f;
    Settings::values.touch_cursor_sensitivity = sensitivity;
}

void ConfigureControls::UpdateUIState() {
    bool enabled = ui->enable_touch_cursor->isChecked();
    ui->analog_stick_label->setEnabled(enabled);
    ui->analog_stick_combobox->setEnabled(enabled);
    ui->touch_button_label->setEnabled(enabled);
    ui->touch_button_combobox->setEnabled(enabled);
    ui->sensitivity_label->setEnabled(enabled);
    ui->sensitivity_slider->setEnabled(enabled);
    ui->sensitivity_value_label->setEnabled(enabled);
}

void ConfigureControls::OnProfileChanged([[maybe_unused]] int index) {
    // No immediate action needed, saved on Apply
}

void ConfigureControls::OnTouchCursorEnabledToggled(bool enabled) {
    UpdateUIState();
}

void ConfigureControls::OnAnalogStickChanged([[maybe_unused]] int index) {
    // No immediate action needed, saved on Apply
}

void ConfigureControls::OnTouchButtonChanged([[maybe_unused]] int index) {
    // No immediate action needed, saved on Apply
}

void ConfigureControls::OnSensitivityChanged(int value) {
    float sensitivity = value / 10.0f;
    ui->sensitivity_value_label->setText(QStringLiteral("%1x").arg(sensitivity, 0, 'f', 1));
}

void ConfigureControls::RetranslateUI() {
    ui->retranslateUi(this);
}
