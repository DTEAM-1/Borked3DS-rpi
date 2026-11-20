// Copyright 2017 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace Common {
class ParamPackage;
}

namespace InputCommon {

/// Initializes and registers all built-in input device factories.
void Init();

/// Deregisters all built-in input device factories and shuts them down.
void Shutdown();

class Keyboard;

/// Gets the keyboard button device factory.
Keyboard* GetKeyboard();

class MotionEmu;

/// Gets the motion emulation factory.
MotionEmu* GetMotionEmu();

/// Generates a serialized param package for creating a keyboard button device
std::string GenerateKeyboardParam(int key_code);

/// Generates a serialized param package for creating an analog device taking input from keyboard
std::string GenerateAnalogParamFromKeys(int key_up, int key_down, int key_left, int key_right,
                                        int key_modifier, float modifier_scale);

Common::ParamPackage GetControllerButtonBinds(const Common::ParamPackage& params, int button);
Common::ParamPackage GetControllerAnalogBinds(const Common::ParamPackage& params, int analog);

/**
 * v426 : resolves a named SDL_GameController button ("leftstick", "rightstick", "guide", ...) on
 * the controller identified by params (engine/guid/port). Used by the controller hotkeys for
 * buttons the 3DS does not have (L3/R3). Returns an empty package if it cannot be resolved.
 */
Common::ParamPackage GetControllerNamedButtonBind(const Common::ParamPackage& params,
                                                  const std::string& button_name);

/// Reloads the input devices
void ReloadInputDevices();

namespace Polling {

enum class DeviceType { Button, Analog };

/**
 * A class that can be used to get inputs from an input device like controllers without having to
 * poll the device's status yourself
 */
class DevicePoller {
public:
    virtual ~DevicePoller() = default;
    /// Setup and start polling for inputs, should be called before GetNextInput
    virtual void Start() = 0;
    /// Stop polling
    virtual void Stop() = 0;
    /**
     * Every call to this function returns the next input recorded since calling Start
     * @return A ParamPackage of the recorded input, which can be used to create an InputDevice.
     *         If there has been no input, the package is empty
     */
    virtual Common::ParamPackage GetNextInput() = 0;
};

// Get all DevicePoller from all backends for a specific device type
std::vector<std::unique_ptr<DevicePoller>> GetPollers(DeviceType type);
} // namespace Polling
} // namespace InputCommon
