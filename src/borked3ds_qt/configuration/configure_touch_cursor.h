// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <QWidget>

namespace Ui {
class ConfigureTouchCursor;
}

class ConfigureTouchCursor : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureTouchCursor(QWidget* parent = nullptr);
    ~ConfigureTouchCursor() override;

    void ApplyConfiguration();
    void RetranslateUI();
    void SetConfiguration();

private:
    void UpdateUIState();
    void OnAnalogStickChanged(int index);
    void OnTouchButtonChanged(int index);
    void OnSensitivityChanged(int value);
    void OnEnabledToggled(bool enabled);

    std::unique_ptr<Ui::ConfigureTouchCursor> ui;
};
