#pragma once

#include <cstdint>
#include <memory>
#include "common/common_types.h"

namespace Core {
class System;
}

namespace Service::HID {

struct TouchCursorConfig {
    bool enabled = false;
    u32 analog_stick_button = 0;  // Which analog to use (0=CirclePad, 1=CStick)
    u32 touch_button = 0;          // Which button triggers touch
    float sensitivity = 1.0f;      // Cursor movement sensitivity (0.1 - 5.0)
    int cursor_size = 8;           // Cursor indicator size in pixels
};

class TouchCursorController {
public:
    explicit TouchCursorController(Core::System& system);
    ~TouchCursorController();

    void Update(float circle_pad_x, float circle_pad_y, bool touch_pressed);

    void SetConfig(const TouchCursorConfig& config);
    const TouchCursorConfig& GetConfig() const { return config; }

    u16 GetCursorX() const { return cursor_x; }
    u16 GetCursorY() const { return cursor_y; }
    bool IsTouching() const { return is_touching; }

private:
    Core::System& system;
    TouchCursorConfig config;

    u16 cursor_x = 160;  // Center of bottom screen (320 width)
    u16 cursor_y = 120;  // Center of bottom screen (240 height)
    bool is_touching = false;

    // Direct velocity (no acceleration)
    float velocity_x = 0.0f;
    float velocity_y = 0.0f;

    void UpdateCursorPosition();
    // Remove ClampCursorToScreen() declaration if it's still there
};

} // namespace Service::HID
