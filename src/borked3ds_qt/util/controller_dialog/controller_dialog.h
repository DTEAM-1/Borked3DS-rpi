// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <QDialog>
#include <QTimer>
#include "input_common/main.h"

class QLabel;

class ControllerDialog : public QDialog {
    Q_OBJECT

public:
    explicit ControllerDialog(QWidget* parent = nullptr);
    ~ControllerDialog() override;

    QString GetButtonString() const;
    void closeEvent(QCloseEvent*) override;

private:
    void StartPolling();
    void StopPolling();
    void OnButtonDetected(const Common::ParamPackage& params);

    QLabel* status_label;
    QTimer* poll_timer;
    QTimer* timeout_timer;

    std::vector<std::unique_ptr<InputCommon::Polling::DevicePoller>> device_pollers;
    QString detected_button;

    static constexpr int POLL_INTERVAL_MS = 50;
    static constexpr int TIMEOUT_MS = 5000;
};
