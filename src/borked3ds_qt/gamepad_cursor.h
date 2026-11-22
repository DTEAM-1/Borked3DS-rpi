// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>
#include <QElapsedTimer>
#include <QObject>
#include <QPoint>
#include <QPointer>
#include <QTimer>
#include "core/frontend/input.h"

class QWidget;

/**
 * v428 - Global gamepad cursor (Configure > Controls > Gamepad Cursor).
 *
 * Moves the real mouse pointer with a stick, everywhere: menus, game list, dialogs and the game
 * window (where a click is a touch of the 3DS bottom screen, through the normal mouse path, so it
 * works with every renderer). L3 or R3 clicks, held = drag; double click opens a game in the list.
 * The pointer hides after 3 s without stick activity and comes back as soon as the stick moves.
 * The real mouse and the touch screen keep working: a real mouse movement shows the pointer again
 * and the gamepad never hides a pointer it did not move.
 *
 * Built on gvx64's per-game touch cursor (sub-pixel accumulation, sensitivity), improved with a
 * radial dead zone and a quadratic response curve (precise near the centre, fast at full tilt,
 * like Steam Input / Dolphin's default curves), time-based speed, and the interface support.
 */
class GamepadCursor : public QObject {
public:
    explicit GamepadCursor(QObject* parent = nullptr);
    ~GamepadCursor() override;

    /// Settings or input profile changed: devices are rebuilt on the next poll.
    void Reload();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void Build();
    void Poll();
    void Press(const QPoint& global);
    void Release(const QPoint& global);
    void SendMouse(QEvent::Type type, const QPoint& global, Qt::MouseButton button,
                   Qt::MouseButtons buttons, QWidget* target);
    QWidget* TargetAt(const QPoint& global) const;
    void NoteActivity();
    void ShowPointer();
    void HidePointer();

    QTimer poll_timer;
    QElapsedTimer frame_clock;
    QElapsedTimer activity_clock;
    QElapsedTimer last_press_clock;

    bool built = false;
    bool enabled = false;

    std::unique_ptr<Input::AnalogDevice> stick;
    bool invert_x = false;
    bool invert_y = false;
    std::array<std::unique_ptr<Input::ButtonDevice>, 2> clicks;
    std::unique_ptr<Input::ButtonDevice> modifier;
    bool click_was_pressed = true; // must be seen released once before it can click

    float accumulated_x = 0.0f;
    float accumulated_y = 0.0f;

    bool pressed = false;
    QPointer<QWidget> press_target;
    QPoint last_press_pos;

    bool gamepad_moved_pointer = false; // the pointer was last moved by the stick
    bool hidden = false;                // override cursor (blank) installed by us
    QPoint last_warp_pos;
};
