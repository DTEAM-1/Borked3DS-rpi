// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cmath>
#include <QApplication>
#include <QCursor>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QScreen>
#include <QWidget>
#include "borked3ds_qt/gamepad_cursor.h"
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/uisettings.h"
#include "common/logging/log.h"
#include "common/settings.h"

namespace {

constexpr int POLL_INTERVAL_MS = 16;  // ~60 Hz, like the controller hotkeys
constexpr qint64 HIDE_DELAY_MS = 3000; // pointer hidden after 3 s without stick activity
constexpr float DEAD_ZONE = 0.20f;     // radial, on top of the device's own dead zone
// Full tilt at speed 1.0: 1.6 screen heights per second (about 770 px/s on a 480 px screen)
constexpr float FULL_TILT_SCREENS_PER_SECOND = 1.6f;

// Dialogs that read the gamepad themselves: never click behind them
bool IsInputCaptureDialog(const QWidget* widget) {
    return widget &&
           (widget->inherits("ControllerDialog") || widget->inherits("SequenceDialog"));
}

} // Anonymous namespace

GamepadCursor::GamepadCursor(QObject* parent) : QObject(parent) {
    poll_timer.setInterval(POLL_INTERVAL_MS);
    connect(&poll_timer, &QTimer::timeout, this, [this] { Poll(); });
    qApp->installEventFilter(this);
    Reload();
}

GamepadCursor::~GamepadCursor() {
    qApp->removeEventFilter(this);
    ShowPointer();
}

void GamepadCursor::Reload() {
    built = false;
    if (Settings::values.gamepad_cursor_enabled.GetValue()) {
        if (!poll_timer.isActive()) {
            frame_clock.start();
            poll_timer.start();
        }
    } else {
        poll_timer.stop();
        if (pressed) {
            Release(QCursor::pos());
        }
        stick.reset();
        for (auto& click : clicks) {
            click.reset();
        }
        modifier.reset();
        ShowPointer();
    }
}

void GamepadCursor::Build() {
    built = true;
    enabled = Settings::values.gamepad_cursor_enabled.GetValue();
    stick.reset();
    for (auto& click : clicks) {
        click.reset();
    }
    modifier.reset();
    click_was_pressed = true;
    accumulated_x = 0.0f;
    accumulated_y = 0.0f;
    if (!enabled) {
        return;
    }

    const bool right = Settings::values.gamepad_cursor_stick.GetValue() != 0;
    const std::string stick_params = ControllerHotkey::ResolveStick(right, invert_x, invert_y);
    if (!stick_params.empty()) {
        stick = Input::CreateDevice<Input::AnalogDevice>(stick_params);
    }

    const std::array<std::string, 2> click_bindings{
        Settings::values.gamepad_cursor_click1.GetValue(),
        Settings::values.gamepad_cursor_click2.GetValue(),
    };
    std::array<bool, 2> click_found{};
    for (std::size_t i = 0; i < clicks.size(); ++i) {
        const std::string params =
            ControllerHotkey::Resolve(QString::fromStdString(click_bindings[i]));
        if (!params.empty()) {
            clicks[i] = Input::CreateDevice<Input::ButtonDevice>(params);
            click_found[i] = true;
        }
    }

    // The hotkey enable button (Select) + L3/R3 are hotkeys: no click while it is held
    const QString& modifier_binding = UISettings::values.controller_hotkey_modifier;
    if (!modifier_binding.isEmpty()) {
        const std::string params = ControllerHotkey::Resolve(modifier_binding);
        if (!params.empty()) {
            modifier = Input::CreateDevice<Input::ButtonDevice>(params);
        }
    }

    LOG_WARNING(Frontend,
                "V428_CURSOR ready: stick={} ({}{}) click1={} click2={} speed={:.1f}",
                right ? "right" : "left", stick ? "found" : "NOT FOUND",
                (invert_x || invert_y) ? ", inverted from EmulationStation" : "",
                click_found[0] ? "found" : "NOT FOUND", click_found[1] ? "found" : "NOT FOUND",
                Settings::values.gamepad_cursor_speed.GetValue());
}

void GamepadCursor::Poll() {
    if (!built) {
        Build();
    }
    // Seconds since the previous poll, bounded so a stalled event loop cannot fling the pointer
    const float dt = std::clamp(static_cast<float>(frame_clock.restart()) / 1000.0f, 0.0f, 0.05f);
    if (!enabled) {
        return;
    }

    QWidget* const modal = QApplication::activeModalWidget();
    const bool capturing = IsInputCaptureDialog(modal);

    QPoint pos = QCursor::pos();

    // --- Movement -------------------------------------------------------------------------
    if (stick && !capturing) {
        auto [x, y] = stick->GetStatus();
        if (invert_x) {
            x = -x;
        }
        if (invert_y) {
            y = -y;
        }
        const float r = std::sqrt(x * x + y * y);
        if (r > DEAD_ZONE) {
            const float t = std::min(1.0f, (r - DEAD_ZONE) / (1.0f - DEAD_ZONE));
            QScreen* screen = QGuiApplication::screenAt(pos);
            if (!screen) {
                screen = QGuiApplication::primaryScreen();
            }
            const QRect area = screen ? screen->geometry() : QRect(0, 0, 800, 480);
            const float pixels_per_second = static_cast<float>(area.height()) *
                                            FULL_TILT_SCREENS_PER_SECOND *
                                            Settings::values.gamepad_cursor_speed.GetValue();
            const float step = t * t * pixels_per_second * dt;
            // 3DS convention: y > 0 is up, the screen goes down
            accumulated_x += x / r * step;
            accumulated_y += -y / r * step;

            const int move_x = static_cast<int>(accumulated_x);
            const int move_y = static_cast<int>(accumulated_y);
            if (move_x != 0 || move_y != 0) {
                accumulated_x -= static_cast<float>(move_x);
                accumulated_y -= static_cast<float>(move_y);
                pos = QPoint(std::clamp(pos.x() + move_x, area.left(), area.right()),
                             std::clamp(pos.y() + move_y, area.top(), area.bottom()));
                last_warp_pos = pos;
                QCursor::setPos(pos);
                NoteActivity();
                if (pressed && press_target) {
                    SendMouse(QEvent::MouseMove, pos, Qt::NoButton, Qt::LeftButton,
                              press_target.data());
                }
            }
        } else {
            accumulated_x = 0.0f;
            accumulated_y = 0.0f;
        }
    }

    // --- Clicks (L3 or R3) ----------------------------------------------------------------
    bool click_now = false;
    for (const auto& click : clicks) {
        if (click && click->GetStatus()) {
            click_now = true;
        }
    }
    const bool modifier_held = modifier && modifier->GetStatus();
    if (click_now && !click_was_pressed && !modifier_held && !capturing) {
        last_warp_pos = pos;
        NoteActivity();
        Press(pos);
    } else if (!click_now && click_was_pressed && pressed) {
        Release(pos);
    }
    click_was_pressed = click_now;

    // --- Auto-hide ------------------------------------------------------------------------
    if (gamepad_moved_pointer && !hidden && !pressed &&
        activity_clock.isValid() && activity_clock.elapsed() >= HIDE_DELAY_MS) {
        HidePointer();
    }
}

QWidget* GamepadCursor::TargetAt(const QPoint& global) const {
    QWidget* const under = QApplication::widgetAt(global);
    if (QWidget* const popup = QApplication::activePopupWidget()) {
        // Outside an open menu: the menu gets the click and closes, as with a real mouse
        if (!under || under->window() != popup) {
            return popup;
        }
    }
    return under;
}

void GamepadCursor::SendMouse(QEvent::Type type, const QPoint& global, Qt::MouseButton button,
                              Qt::MouseButtons buttons, QWidget* target) {
    if (!target) {
        return;
    }
    const QPointF global_f(global);
    QMouseEvent event(type, target->mapFromGlobal(global_f), global_f, button, buttons,
                      Qt::NoModifier);
    QApplication::sendEvent(target, &event);
}

void GamepadCursor::Press(const QPoint& global) {
    QWidget* const target = TargetAt(global);
    if (!target) {
        return;
    }
    if (!QApplication::activePopupWidget()) {
        if (QWidget* const window = target->window()) {
            window->activateWindow();
        }
    }
    const bool double_click = last_press_clock.isValid() &&
                              last_press_clock.elapsed() < QApplication::doubleClickInterval() &&
                              (global - last_press_pos).manhattanLength() <=
                                  QApplication::startDragDistance();
    last_press_clock.start();
    last_press_pos = global;

    pressed = true;
    press_target = target;
    SendMouse(QEvent::MouseButtonPress, global, Qt::LeftButton, Qt::LeftButton, target);
    if (double_click && press_target) {
        // Qt 6 order: press, release, press, double click, release
        SendMouse(QEvent::MouseButtonDblClick, global, Qt::LeftButton, Qt::LeftButton,
                  press_target.data());
        last_press_clock.invalidate();
    }
}

void GamepadCursor::Release(const QPoint& global) {
    pressed = false;
    QWidget* const target = press_target ? press_target.data() : TargetAt(global);
    press_target.clear();
    SendMouse(QEvent::MouseButtonRelease, global, Qt::LeftButton, Qt::NoButton, target);
}

void GamepadCursor::NoteActivity() {
    gamepad_moved_pointer = true;
    activity_clock.start();
    ShowPointer();
}

void GamepadCursor::ShowPointer() {
    if (hidden) {
        hidden = false;
        QGuiApplication::restoreOverrideCursor();
    }
}

void GamepadCursor::HidePointer() {
    if (!hidden) {
        hidden = true;
        QGuiApplication::setOverrideCursor(QCursor(Qt::BlankCursor));
    }
}

bool GamepadCursor::eventFilter(QObject* watched, QEvent* event) {
    // A real mouse movement (not one of our warps): the mouse is in use, never hide its pointer
    // The X server delivers the motion of our own warps asynchronously, possibly one poll late,
    // so moves within 200 ms of stick activity are ours.
    if (event->type() == QEvent::MouseMove && enabled) {
        const auto* const mouse = static_cast<QMouseEvent*>(event);
        const QPoint global = mouse->globalPosition().toPoint();
        const bool recent_stick_activity = gamepad_moved_pointer && activity_clock.isValid() &&
                                           activity_clock.elapsed() < 200;
        if (!recent_stick_activity && (global - last_warp_pos).manhattanLength() > 2) {
            gamepad_moved_pointer = false;
            ShowPointer();
        }
    }
    return QObject::eventFilter(watched, event);
}
