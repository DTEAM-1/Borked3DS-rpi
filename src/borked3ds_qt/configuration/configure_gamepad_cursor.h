// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <QString>
#include <QWidget>

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QPushButton;
class QSlider;

/// v428 : Configure > Controls > Gamepad Cursor (global, interface and games)
class ConfigureGamepadCursor : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureGamepadCursor(QWidget* parent = nullptr);
    ~ConfigureGamepadCursor() override;

    void SetConfiguration();
    void ApplyConfiguration();
    void RetranslateUI();

private:
    void ApplyTexts();
    void UpdateState();
    void UpdateStatus();
    void UpdateSpeedLabel();
    void ConfigureClick(std::size_t index);
    void SetClick(std::size_t index, const QString& binding);

    QCheckBox* enable_checkbox = nullptr;
    QLabel* description_label = nullptr;
    QGroupBox* options_group = nullptr;
    QLabel* stick_label = nullptr;
    QComboBox* stick_combobox = nullptr;
    QLabel* speed_label = nullptr;
    QSlider* speed_slider = nullptr;
    QLabel* speed_value_label = nullptr;
    std::array<QLabel*, 2> click_labels{};
    std::array<QPushButton*, 2> click_buttons{};
    std::array<QString, 2> click_bindings;
    QPushButton* defaults_button = nullptr;
    QLabel* status_label = nullptr;
};
