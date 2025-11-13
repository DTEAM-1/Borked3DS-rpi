// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>
#include "borked3ds_qt/util/controller_dialog/controller_dialog.h"
#include "common/param_package.h"

ControllerDialog::ControllerDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Press a controller button"));
    setModal(true);

    status_label = new QLabel(tr("Press any button on your controller...\n(5 second timeout)"));
    status_label->setAlignment(Qt::AlignCenter);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel);
    buttons->setCenterButtons(true);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(status_label);
    layout->addWidget(buttons);

    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // Set up timers
    poll_timer = new QTimer(this);
    timeout_timer = new QTimer(this);
    timeout_timer->setSingleShot(true);

    connect(poll_timer, &QTimer::timeout, this, [this]() {
        for (auto& poller : device_pollers) {
            Common::ParamPackage params = poller->GetNextInput();
            if (params.Has("engine")) {
                OnButtonDetected(params);
                return;
            }
        }
    });

    connect(timeout_timer, &QTimer::timeout, this, [this]() {
        status_label->setText(tr("Timed out! No button detected."));
        StopPolling();
        QTimer::singleShot(1000, this, &QDialog::reject);
    });

    // Start polling when dialog opens
    StartPolling();
}

ControllerDialog::~ControllerDialog() {
    StopPolling();
}

void ControllerDialog::StartPolling() {
    device_pollers = InputCommon::Polling::GetPollers(InputCommon::Polling::DeviceType::Button);

    for (auto& poller : device_pollers) {
        poller->Start();
    }

    poll_timer->start(POLL_INTERVAL_MS);
    timeout_timer->start(TIMEOUT_MS);
}

void ControllerDialog::StopPolling() {
    poll_timer->stop();
    timeout_timer->stop();

    for (auto& poller : device_pollers) {
        poller->Stop();
    }

    device_pollers.clear();
}

void ControllerDialog::OnButtonDetected(const Common::ParamPackage& params) {
    detected_button = QString::fromStdString(params.Serialize());

    // Show what was detected
    QString display_text = tr("Detected: ");
    if (params.Get("engine", "") == "sdl") {
        if (params.Has("button")) {
            display_text += tr("Button %1").arg(QString::fromStdString(params.Get("button", "")));
        } else if (params.Has("hat")) {
            display_text += tr("Hat %1 %2").arg(
                QString::fromStdString(params.Get("hat", "")),
                QString::fromStdString(params.Get("direction", "")));
        } else if (params.Has("axis")) {
            display_text += tr("Axis %1%2").arg(
                QString::fromStdString(params.Get("axis", "")),
                QString::fromStdString(params.Get("direction", "")));
        }
    }

    status_label->setText(display_text);

    StopPolling();

    // Auto-accept after showing detection
    QTimer::singleShot(500, this, &QDialog::accept);
}

QString ControllerDialog::GetButtonString() const {
    return detected_button;
}

void ControllerDialog::closeEvent(QCloseEvent*) {
    StopPolling();
    reject();
}
