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
/// v431 : separate stick and click buttons for the interface and for games
class ConfigureGamepadCursor : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureGamepadCursor(QWidget* parent = nullptr);
    ~ConfigureGamepadCursor() override;

    void SetConfiguration();
    void ApplyConfiguration();
    void RetranslateUI();

private:
    enum Set : std::size_t { Interface = 0, Game = 1 };

    /// Stick + two click buttons of one set (interface or game)
    struct ControlSet {
        QGroupBox* group = nullptr;
        QLabel* stick_label = nullptr;
        QComboBox* stick_combobox = nullptr;
        std::array<QLabel*, 2> click_labels{};
        std::array<QPushButton*, 2> click_buttons{};
        std::array<QPushButton*, 2> clear_buttons{};
        std::array<QString, 2> click_bindings;
    };

    void CreateSet(Set set);
    void ApplyTexts();
    void UpdateState();
    void UpdateStatus();
    void UpdateSpeedLabel();
    void ConfigureClick(Set set, std::size_t index);
    void SetClick(Set set, std::size_t index, const QString& binding);
    QString AutomaticClick(Set set, std::size_t index) const;
    unsigned DefaultStick(Set set) const;

    QCheckBox* enable_checkbox = nullptr;
    QLabel* description_label = nullptr;
    QLabel* speed_label = nullptr;
    QSlider* speed_slider = nullptr;
    QLabel* speed_value_label = nullptr;
    QCheckBox* in_games_checkbox = nullptr;
    std::array<ControlSet, 2> sets;
    QPushButton* defaults_button = nullptr;
    QLabel* status_label = nullptr;
};
