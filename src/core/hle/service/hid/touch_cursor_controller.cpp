#include "core/hle/service/hid/touch_cursor_controller.h"
#include "core/3ds.h"
#include "core/core.h"
#include <algorithm>
#include <cmath>

namespace Service::HID {

TouchCursorController::TouchCursorController(Core::System& system) : system(system) {
    // Start at center of bottom screen
    cursor_x = Core::kScreenBottomWidth / 2;
    cursor_y = Core::kScreenBottomHeight / 2;
}

TouchCursorController::~TouchCursorController() = default;

void TouchCursorController::Update(float circle_pad_x, float circle_pad_y, bool touch_pressed) {
    if (!config.enabled) {
        return;
    }

    // Invert Y axis for touchscreen coordinates
    circle_pad_y = -circle_pad_y;

    // Use independent axis deadzones (square deadzone) for equal response in all directions
    constexpr float DEADZONE = 0.05f;

    // Apply deadzone per axis
    float processed_x = std::abs(circle_pad_x) > DEADZONE ? circle_pad_x : 0.0f;
    float processed_y = std::abs(circle_pad_y) > DEADZONE ? circle_pad_y : 0.0f;

    if (processed_x == 0.0f && processed_y == 0.0f) {
        // Stop all movement when stick is centered
        velocity_x = 0.0f;
        velocity_y = 0.0f;
    } else {
        // Calculate magnitude from processed values
        float magnitude = std::sqrt(processed_x * processed_x + processed_y * processed_y);

        // Normalize to get direction
        float normalized_x = processed_x / magnitude;
        float normalized_y = processed_y / magnitude;

        // Linear speed based on stick push amount
        float push_amount = std::min(1.0f, magnitude);

        // Apply sensitivity with linear scaling
        float base_speed = 1.0f;
        float speed = base_speed * config.sensitivity * push_amount;

        velocity_x = normalized_x * speed;
        velocity_y = normalized_y * speed;
    }

    // Update cursor position
    UpdateCursorPosition();

    // Update touch state
    is_touching = touch_pressed;
}

void TouchCursorController::UpdateCursorPosition() {
    // Use float accumulation to avoid rounding issues at low speeds
    static float accumulated_x = 0.0f;
    static float accumulated_y = 0.0f;

    // Accumulate sub-pixel movement
    accumulated_x += velocity_x;
    accumulated_y += velocity_y;

    // Only update cursor position when we've accumulated at least 1 pixel of movement
    int delta_x = static_cast<int>(accumulated_x);
    int delta_y = static_cast<int>(accumulated_y);

    if (delta_x != 0 || delta_y != 0) {
        // Apply the movement
        int new_x = static_cast<int>(cursor_x) + delta_x;
        int new_y = static_cast<int>(cursor_y) + delta_y;

        // Clamp to screen boundaries
        cursor_x = static_cast<u16>(std::clamp(new_x, 0, static_cast<int>(Core::kScreenBottomWidth - 1)));
        cursor_y = static_cast<u16>(std::clamp(new_y, 0, static_cast<int>(Core::kScreenBottomHeight - 1)));

        // Subtract the integer part we just applied
        accumulated_x -= delta_x;
        accumulated_y -= delta_y;
    }

    // Reset accumulator if velocity is zero (stick centered)
    if (velocity_x == 0.0f && velocity_y == 0.0f) {
        accumulated_x = 0.0f;
        accumulated_y = 0.0f;
    }
}

void TouchCursorController::SetConfig(const TouchCursorConfig& new_config) {
    config = new_config;
}

} // namespace Service::HID
