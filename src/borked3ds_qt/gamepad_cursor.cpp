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
#if defined(__linux__)
#include <dlfcn.h>
#include <QtGui/qguiapplication_platform.h>
#endif
#include "borked3ds_qt/gamepad_cursor.h"
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/uisettings.h"
#include "common/logging/log.h"
#include "common/settings.h"

namespace {

constexpr int POLL_INTERVAL_MS = 16;  // ~60 Hz, like the controller hotkeys
constexpr qint64 HIDE_DELAY_MS = 1000; // pointer hidden after 1 s without stick activity
constexpr float DEAD_ZONE = 0.20f;     // radial, on top of the device's own dead zone
// Full tilt at 80 % sensitivity (the v428 speed): 1.6 screen heights per second, about
// 770 px/s on a 480 px screen. The default 50 % gives 1.0 screen height per second.
constexpr float FULL_TILT_SCREENS_PER_SECOND = 1.6f;
constexpr float REFERENCE_SENSITIVITY = 80.0f;

float SensitivityFactor() {
    return static_cast<float>(Settings::values.gamepad_cursor_sensitivity.GetValue()) /
           REFERENCE_SENSITIVITY;
}

// Dialogs that read the gamepad themselves: never click behind them
bool IsInputCaptureDialog(const QWidget* widget) {
    return widget &&
           (widget->inherits("ControllerDialog") || widget->inherits("SequenceDialog"));
}

// v432 : real mouse button through the X server (XTest, as xdotool does), loaded at run time so
// there is no build dependency. A real button works everywhere a mouse does: window manager
// buttons (the window's close button), native drags (sliders, scroll bars, title bars), Qt's own
// double click and popup handling, and the game window. When libXtst or X11 is not available
// (Wayland, Windows, macOS), the Qt events fallback below is used.
class XTestButton {
public:
    static XTestButton& Get() {
        static XTestButton instance;
        return instance;
    }

    bool Available() const {
        return fake_button != nullptr;
    }

    void Set(bool press) const {
        if (!fake_button) {
            return;
        }
        fake_button(display, 1, press ? 1 : 0, 0);
        flush(display);
    }

private:
    using FakeButtonFn = int (*)(void*, unsigned int, int, unsigned long);
    using FlushFn = int (*)(void*);

    XTestButton() {
#if defined(__linux__) && QT_CONFIG(xcb)
        auto* const x11 = qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
        if (!x11 || !x11->display()) {
            return;
        }
        void* const xtst = dlopen("libXtst.so.6", RTLD_NOW | RTLD_LOCAL);
        void* const xlib = dlopen("libX11.so.6", RTLD_NOW | RTLD_LOCAL);
        if (!xtst || !xlib) {
            return;
        }
        auto* const button = reinterpret_cast<FakeButtonFn>(dlsym(xtst, "XTestFakeButtonEvent"));
        auto* const flush_fn = reinterpret_cast<FlushFn>(dlsym(xlib, "XFlush"));
        if (!button || !flush_fn) {
            return;
        }
        display = x11->display();
        flush = flush_fn;
        fake_button = button;
#endif
    }

    void* display = nullptr;
    FakeButtonFn fake_button = nullptr;
    FlushFn flush = nullptr;
};

// Mouse events that the touch screen produces (Qt or the X server emulating a pointer)
bool IsFromTouchScreen(const QMouseEvent* mouse) {
    const QPointingDevice* const device = mouse->pointingDevice();
    return (device && device->type() == QInputDevice::DeviceType::TouchScreen) ||
           mouse->source() != Qt::MouseEventNotSynthesized;
}

} // Anonymous namespace

GamepadCursor::GamepadCursor(QObject* parent) : QObject(parent) {
    poll_timer.setInterval(POLL_INTERVAL_MS);
    connect(&poll_timer, &QTimer::timeout, this, [this] { Poll(); });
    qApp->installEventFilter(this);
    // The arrow is hidden after 1 s only while a game runs (from the first poll in a game, like
    // after a stick movement); in the interface it stays visible.
    gamepad_moved_pointer = true;
    last_warp_pos = QCursor::pos(); // events at the starting position are not "real mouse"
    activity_clock.start();
    Reload();
}

void GamepadCursor::SetGameRunningCallback(std::function<bool()> callback) {
    game_running = std::move(callback);
}

bool GamepadCursor::InGame() const {
    return game_running && game_running();
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
        enabled = false;
        if (pressed) {
            Release(QCursor::pos());
        }
        interface_controls.Reset();
        game_controls.Reset();
        modifier.reset();
        ShowPointer();
    }
}

void GamepadCursor::Controls::Reset() {
    stick.reset();
    invert_x = false;
    invert_y = false;
    for (auto& click : clicks) {
        click.reset();
    }
}

void GamepadCursor::BuildControls(Controls& controls, bool right_stick, const std::string& click1,
                                  const std::string& click2, const char* name) {
    controls.Reset();
    const std::string stick_params =
        ControllerHotkey::ResolveStick(right_stick, controls.invert_x, controls.invert_y);
    if (!stick_params.empty()) {
        controls.stick = Input::CreateDevice<Input::AnalogDevice>(stick_params);
    }

    const std::array<std::string, 2> bindings{click1, click2};
    std::array<const char*, 2> state{"none", "none"};
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        if (bindings[i].empty()) {
            continue;
        }
        const std::string params = ControllerHotkey::Resolve(QString::fromStdString(bindings[i]));
        if (params.empty()) {
            state[i] = "NOT FOUND";
            continue;
        }
        controls.clicks[i] = Input::CreateDevice<Input::ButtonDevice>(params);
        state[i] = "found";
    }

    LOG_WARNING(Frontend, "V431_CURSOR {}: stick={} ({}{}) click1={} click2={}", name,
                right_stick ? "right" : "left", controls.stick ? "found" : "NOT FOUND",
                (controls.invert_x || controls.invert_y) ? ", inverted from EmulationStation"
                                                         : "",
                state[0], state[1]);
}

void GamepadCursor::Build() {
    built = true;
    enabled = Settings::values.gamepad_cursor_enabled.GetValue();
    in_games = Settings::values.gamepad_cursor_in_games.GetValue();
    interface_controls.Reset();
    game_controls.Reset();
    modifier.reset();
    click_was_pressed = true;
    accumulated_x = 0.0f;
    accumulated_y = 0.0f;
    if (!enabled) {
        return;
    }

    // Interface: left stick and A by default. Game: right stick, L3 and R3 by default.
    BuildControls(interface_controls, Settings::values.gamepad_cursor_ui_stick.GetValue() != 0,
                  Settings::values.gamepad_cursor_ui_click1.GetValue(),
                  Settings::values.gamepad_cursor_ui_click2.GetValue(), "interface");
    BuildControls(game_controls, Settings::values.gamepad_cursor_stick.GetValue() != 0,
                  Settings::values.gamepad_cursor_click1.GetValue(),
                  Settings::values.gamepad_cursor_click2.GetValue(), "game");

    // The hotkey enable button (Select) + L3/R3 are hotkeys: no click while it is held
    const QString& modifier_binding = UISettings::values.controller_hotkey_modifier;
    if (!modifier_binding.isEmpty()) {
        const std::string params = ControllerHotkey::Resolve(modifier_binding);
        if (!params.empty()) {
            modifier = Input::CreateDevice<Input::ButtonDevice>(params);
        }
    }

    LOG_WARNING(Frontend, "V428_CURSOR ready: sensitivity={}% in_games={} clicks={}",
                Settings::values.gamepad_cursor_sensitivity.GetValue(), in_games ? "yes" : "no",
                XTestButton::Get().Available() ? "V432 X11 XTest (real mouse button)"
                                               : "V432 Qt events (XTest unavailable)");
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

    QPoint pos = QCursor::pos();

    // v431 : interface set or game set. Switching (game starts, pauses, resumes) ends a click in
    // progress, and the new set's buttons must be released before they can click.
    const bool in_game = InGame();
    if (in_game != using_game_controls) {
        using_game_controls = in_game;
        if (pressed) {
            Release(pos);
        }
        click_was_pressed = true;
        accumulated_x = 0.0f;
        accumulated_y = 0.0f;
    }
    const Controls& controls = in_game ? game_controls : interface_controls;

    QWidget* const modal = QApplication::activeModalWidget();
    // "Also in games" off: nothing while a game runs (the stick belongs to the game)
    const bool game_blocked = !in_games && in_game;
    const bool capturing = IsInputCaptureDialog(modal) || game_blocked;
    if (game_blocked && pressed) {
        Release(pos);
    }

    // --- Movement -------------------------------------------------------------------------
    if (controls.stick && !capturing) {
        auto [x, y] = controls.stick->GetStatus();
        if (controls.invert_x) {
            x = -x;
        }
        if (controls.invert_y) {
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
                                            FULL_TILT_SCREENS_PER_SECOND * SensitivityFactor();
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
                    // Button held: drag (scroll bar, slider, list selection, touch slide)
                    SendMouse(QEvent::MouseMove, pos, Qt::NoButton, Qt::LeftButton,
                              press_target.data());
                }
            }
        } else {
            accumulated_x = 0.0f;
            accumulated_y = 0.0f;
        }
    }

    // --- Click buttons: press on push, held while held, release on release -------------------
    bool click_now = false;
    for (const auto& click : controls.clicks) {
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

    // --- Auto-hide, only while a game runs (v430: always visible in the interface) --------
    if (in_game && !was_in_game) {
        // A game starts or resumes: the arrow hides 1 s later, as after a stick movement
        gamepad_moved_pointer = true;
        activity_clock.start();
    }
    was_in_game = in_game;
    if (!in_game) {
        ShowPointer();
    } else if (gamepad_moved_pointer && !hidden && !pressed && activity_clock.isValid() &&
               activity_clock.elapsed() >= HIDE_DELAY_MS) {
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
    // v430 : posted, never sent from inside the poll timer. A click that opens a dialog or a
    // menu with exec() runs a nested event loop in its handler; Qt never re-fires a timer whose
    // slot has not returned, so a click delivered from Poll() froze the stick and the clicks
    // until that dialog closed. Posted events are delivered by the event loop after Poll().
    const QPointF global_f(global);
    QCoreApplication::postEvent(target, new QMouseEvent(type, target->mapFromGlobal(global_f),
                                                        global_f, button, buttons,
                                                        Qt::NoModifier));
}

void GamepadCursor::Press(const QPoint& global) {
    if (XTestButton::Get().Available()) {
        // Real button: the X server and Qt do the rest (grab, drag, double click, menus)
        pressed = true;
        press_target.clear();
        XTestButton::Get().Set(true);
        return;
    }

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
    if (XTestButton::Get().Available()) {
        pressed = false;
        press_target.clear();
        XTestButton::Get().Set(false);
        return;
    }

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
    if (!enabled) {
        return QObject::eventFilter(watched, event);
    }
    // Touch screen in a game: the arrow never shows (the X server moves the pointer to the touch
    // point). In the interface the arrow stays visible.
    const QEvent::Type type = event->type();
    const bool touch_event =
        type == QEvent::TouchBegin || type == QEvent::TouchUpdate || type == QEvent::TouchEnd ||
        ((type == QEvent::MouseMove || type == QEvent::MouseButtonPress ||
          type == QEvent::MouseButtonRelease) &&
         IsFromTouchScreen(static_cast<QMouseEvent*>(event)));
    if (touch_event) {
        if (InGame()) {
            gamepad_moved_pointer = false;
            HidePointer();
        }
        return QObject::eventFilter(watched, event);
    }

    // A real mouse movement (not one of our warps): the mouse is in use, never hide its pointer
    // The X server delivers the motion of our own warps asynchronously, possibly one poll late,
    // so moves within 200 ms of stick activity are ours.
    if (type == QEvent::MouseMove) {
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
