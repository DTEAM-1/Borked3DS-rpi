// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <QWidget>

namespace Ui {
class ConfigureControls;
}

class ConfigureControls : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureControls(QWidget* parent = nullptr);
    ~ConfigureControls() override;

    void ApplyConfiguration();
    void RetranslateUI();
    void SetConfiguration();

private:
    void UpdateUIState();
    void OnProfileChanged(int index);
    void OnAnalogStickChanged(int index);
    void OnTouchButtonChanged(int index);
    void OnSensitivityChanged(int value);
    void OnTouchCursorEnabledToggled(bool enabled);

    std::unique_ptr<Ui::ConfigureControls> ui;
};
