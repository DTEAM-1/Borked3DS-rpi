// Copyright 2026 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include "borked3ds_qt/configuration/configure_gamepad_cursor.h"
#include "borked3ds_qt/hotkeys.h"
#include "borked3ds_qt/util/controller_dialog/controller_dialog.h"
#include "common/settings.h"

ConfigureGamepadCursor::ConfigureGamepadCursor(QWidget* parent) : QWidget(parent) {
    auto* const layout = new QVBoxLayout(this);

    enable_checkbox = new QCheckBox(this);
    layout->addWidget(enable_checkbox);

    description_label = new QLabel(this);
    description_label->setWordWrap(true);
    layout->addWidget(description_label);

    // Sensitivity 1 - 100 %, default 50 %, shared by the interface and games (v429)
    auto* const speed_form = new QFormLayout();
    speed_label = new QLabel(this);
    speed_slider = new QSlider(Qt::Horizontal, this);
    speed_slider->setRange(1, 100);
    speed_slider->setSingleStep(1);
    speed_slider->setPageStep(10);
    speed_slider->setTickInterval(10);
    speed_slider->setTickPosition(QSlider::TicksBelow);
    speed_value_label = new QLabel(this);
    speed_value_label->setMinimumWidth(40);
    auto* const speed_layout = new QHBoxLayout();
    speed_layout->addWidget(speed_slider, 1);
    speed_layout->addWidget(speed_value_label);
    speed_form->addRow(speed_label, speed_layout);
    layout->addLayout(speed_form);

    CreateSet(Interface);
    layout->addWidget(sets[Interface].group);

    // The game set starts with the "Also in games" switch
    in_games_checkbox = new QCheckBox(this);
    CreateSet(Game);
    if (auto* const game_form = qobject_cast<QFormLayout*>(sets[Game].group->layout())) {
        game_form->insertRow(0, in_games_checkbox);
    }
    layout->addWidget(sets[Game].group);

    defaults_button = new QPushButton(this);
    layout->addWidget(defaults_button, 0, Qt::AlignLeft);

    status_label = new QLabel(this);
    status_label->setWordWrap(true);
    layout->addWidget(status_label);
    layout->addStretch(1);

    connect(enable_checkbox, &QCheckBox::toggled, this, [this] { UpdateState(); });
    connect(in_games_checkbox, &QCheckBox::toggled, this, [this] { UpdateState(); });
    connect(speed_slider, &QSlider::valueChanged, this, [this] { UpdateSpeedLabel(); });
    connect(defaults_button, &QPushButton::clicked, this, [this] {
        speed_slider->setValue(
            static_cast<int>(Settings::values.gamepad_cursor_sensitivity.GetDefault()));
        in_games_checkbox->setChecked(Settings::values.gamepad_cursor_in_games.GetDefault());
        for (const Set set : {Interface, Game}) {
            ControlSet& controls = sets[set];
            controls.stick_combobox->setCurrentIndex(
                controls.stick_combobox->findData(DefaultStick(set)));
            for (std::size_t i = 0; i < controls.click_bindings.size(); ++i) {
                SetClick(set, i, AutomaticClick(set, i));
            }
        }
    });

    ApplyTexts();
    SetConfiguration();
}

ConfigureGamepadCursor::~ConfigureGamepadCursor() = default;

void ConfigureGamepadCursor::CreateSet(Set set) {
    ControlSet& controls = sets[set];
    controls.group = new QGroupBox(this);
    auto* const form = new QFormLayout(controls.group);

    controls.stick_label = new QLabel(controls.group);
    controls.stick_combobox = new QComboBox(controls.group);
    controls.stick_combobox->addItem(QString{}, 0u);
    controls.stick_combobox->addItem(QString{}, 1u);
    form->addRow(controls.stick_label, controls.stick_combobox);
    connect(controls.stick_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this] { UpdateStatus(); });

    for (std::size_t i = 0; i < controls.click_buttons.size(); ++i) {
        controls.click_labels[i] = new QLabel(controls.group);
        controls.click_buttons[i] = new QPushButton(controls.group);
        controls.clear_buttons[i] = new QPushButton(controls.group);
        auto* const click_layout = new QHBoxLayout();
        click_layout->addWidget(controls.click_buttons[i], 1);
        click_layout->addWidget(controls.clear_buttons[i]);
        form->addRow(controls.click_labels[i], click_layout);
        connect(controls.click_buttons[i], &QPushButton::clicked, this,
                [this, set, i] { ConfigureClick(set, i); });
        connect(controls.clear_buttons[i], &QPushButton::clicked, this,
                [this, set, i] { SetClick(set, i, {}); });
    }
}

QString ConfigureGamepadCursor::AutomaticClick(Set set, std::size_t index) const {
    if (set == Interface) {
        return QString::fromStdString(index == 0
                                          ? Settings::values.gamepad_cursor_ui_click1.GetDefault()
                                          : Settings::values.gamepad_cursor_ui_click2.GetDefault());
    }
    return QString::fromStdString(index == 0 ? Settings::values.gamepad_cursor_click1.GetDefault()
                                             : Settings::values.gamepad_cursor_click2.GetDefault());
}

unsigned ConfigureGamepadCursor::DefaultStick(Set set) const {
    return set == Interface ? Settings::values.gamepad_cursor_ui_stick.GetDefault()
                            : Settings::values.gamepad_cursor_stick.GetDefault();
}

void ConfigureGamepadCursor::ApplyTexts() {
    enable_checkbox->setText(tr("Enable the gamepad cursor (interface and games)"));
    description_label->setText(
        tr("Moves the mouse pointer with a stick: menus, game list, settings and the 3DS touch "
           "screen. Hold a click button to keep it pressed (drag a scroll bar or a slider, slide "
           "on the touch screen); click twice to open a game. The interface and games have their "
           "own stick and click buttons. In a game the arrow hides after 1 second and never "
           "shows for the touch screen; in the interface it stays visible. The mouse and the "
           "touch screen keep working.\nWhen enabled, the per-game touch cursor (game "
           "Properties > Controls) is not used."));
    speed_label->setText(tr("Sensitivity:"));
    speed_slider->setToolTip(tr("Pointer speed, in the interface and in games (default 50 %)"));

    sets[Interface].group->setTitle(
        tr("Interface (menus, game list, settings, and a paused game)"));
    sets[Game].group->setTitle(tr("Games"));
    in_games_checkbox->setText(tr("Also in games (off: the cursor is not used while a game runs)"));

    for (const Set set : {Interface, Game}) {
        ControlSet& controls = sets[set];
        controls.stick_label->setText(tr("Stick:"));
        controls.stick_combobox->setItemText(0, tr("Left stick"));
        controls.stick_combobox->setItemText(1, tr("Right stick"));
        controls.click_labels[0]->setText(tr("Click button 1:"));
        controls.click_labels[1]->setText(tr("Click button 2:"));
        for (QPushButton* const button : controls.click_buttons) {
            button->setToolTip(tr("Click, then press the gamepad button that clicks. Choosing a "
                                  "button replaces the automatic one(s); add a second one in "
                                  "the other field if you want two."));
        }
        for (QPushButton* const button : controls.clear_buttons) {
            button->setText(tr("Clear"));
        }
        for (std::size_t i = 0; i < controls.click_bindings.size(); ++i) {
            SetClick(set, i, controls.click_bindings[i]);
        }
    }
    defaults_button->setText(tr("Restore Defaults (interface: left stick + A; games: right stick "
                                "+ L3 / R3; 50 %)"));
    UpdateSpeedLabel();
    UpdateStatus();
}

void ConfigureGamepadCursor::SetConfiguration() {
    enable_checkbox->setChecked(Settings::values.gamepad_cursor_enabled.GetValue());
    in_games_checkbox->setChecked(Settings::values.gamepad_cursor_in_games.GetValue());
    speed_slider->setValue(static_cast<int>(Settings::values.gamepad_cursor_sensitivity.GetValue()));

    const std::array<unsigned, 2> sticks{
        Settings::values.gamepad_cursor_ui_stick.GetValue() != 0 ? 1u : 0u,
        Settings::values.gamepad_cursor_stick.GetValue() != 0 ? 1u : 0u,
    };
    const std::array<std::array<std::string, 2>, 2> clicks{{
        {Settings::values.gamepad_cursor_ui_click1.GetValue(),
         Settings::values.gamepad_cursor_ui_click2.GetValue()},
        {Settings::values.gamepad_cursor_click1.GetValue(),
         Settings::values.gamepad_cursor_click2.GetValue()},
    }};
    for (const Set set : {Interface, Game}) {
        ControlSet& controls = sets[set];
        const int index = controls.stick_combobox->findData(sticks[set]);
        controls.stick_combobox->setCurrentIndex(index < 0 ? 0 : index);
        for (std::size_t i = 0; i < controls.click_bindings.size(); ++i) {
            SetClick(set, i, QString::fromStdString(clicks[set][i]));
        }
    }
    UpdateState();
}

void ConfigureGamepadCursor::ApplyConfiguration() {
    Settings::values.gamepad_cursor_enabled = enable_checkbox->isChecked();
    Settings::values.gamepad_cursor_in_games = in_games_checkbox->isChecked();
    Settings::values.gamepad_cursor_sensitivity = static_cast<u32>(speed_slider->value());

    const ControlSet& ui = sets[Interface];
    Settings::values.gamepad_cursor_ui_stick = ui.stick_combobox->currentData().toUInt();
    Settings::values.gamepad_cursor_ui_click1 = ui.click_bindings[0].toStdString();
    Settings::values.gamepad_cursor_ui_click2 = ui.click_bindings[1].toStdString();

    const ControlSet& game = sets[Game];
    Settings::values.gamepad_cursor_stick = game.stick_combobox->currentData().toUInt();
    Settings::values.gamepad_cursor_click1 = game.click_bindings[0].toStdString();
    Settings::values.gamepad_cursor_click2 = game.click_bindings[1].toStdString();
}

void ConfigureGamepadCursor::RetranslateUI() {
    ApplyTexts();
}

void ConfigureGamepadCursor::UpdateState() {
    const bool enabled = enable_checkbox->isChecked();
    speed_slider->setEnabled(enabled);
    speed_label->setEnabled(enabled);
    sets[Interface].group->setEnabled(enabled);
    sets[Game].group->setEnabled(enabled);

    // "Also in games" off: the game stick and buttons are unused
    const bool game_enabled = enabled && in_games_checkbox->isChecked();
    ControlSet& game = sets[Game];
    game.stick_label->setEnabled(game_enabled);
    game.stick_combobox->setEnabled(game_enabled);
    for (std::size_t i = 0; i < game.click_buttons.size(); ++i) {
        game.click_labels[i]->setEnabled(game_enabled);
        game.click_buttons[i]->setEnabled(game_enabled);
        game.clear_buttons[i]->setEnabled(game_enabled && !game.click_bindings[i].isEmpty());
    }
    UpdateStatus();
}

void ConfigureGamepadCursor::UpdateSpeedLabel() {
    speed_value_label->setText(QStringLiteral("%1 %").arg(speed_slider->value()));
}

void ConfigureGamepadCursor::SetClick(Set set, std::size_t index, const QString& binding) {
    ControlSet& controls = sets[set];
    controls.click_bindings[index] = binding;
    QString text = binding.isEmpty() ? tr("[none]") : ControllerHotkey::DisplayName(binding);
    if (!binding.isEmpty() && binding == AutomaticClick(set, index)) {
        text = tr("%1 (automatic)").arg(text);
    }
    controls.click_buttons[index]->setText(text);
    if (controls.clear_buttons[index]) {
        controls.clear_buttons[index]->setEnabled(!binding.isEmpty() &&
                                                  controls.click_buttons[index]->isEnabled());
    }
    UpdateStatus();
}

void ConfigureGamepadCursor::ConfigureClick(Set set, std::size_t index) {
    ControllerDialog dialog{this};
    const int result = dialog.exec();
    const QString raw = dialog.GetButtonString();
    if (result == QDialog::Rejected || raw.isEmpty()) {
        return;
    }
    // Stored as "A (3DS)" / "L3" / "R3" when it matches, so it follows the configured controls
    SetClick(set, index, ControllerHotkey::Symbolize(raw));

    // A button chosen by the player replaces the automatic one(s): the other field is cleared if
    // it still holds its automatic value (a second button can then be chosen there).
    ControlSet& controls = sets[set];
    const std::size_t other = index == 0 ? 1 : 0;
    const QString other_automatic = AutomaticClick(set, other);
    if (!other_automatic.isEmpty() && controls.click_bindings[other] == other_automatic &&
        controls.click_bindings[other] != controls.click_bindings[index]) {
        SetClick(set, other, {});
    }
}

void ConfigureGamepadCursor::UpdateStatus() {
    if (!status_label || !sets[Game].stick_combobox) {
        return;
    }
    if (!enable_checkbox->isChecked()) {
        status_label->setText(tr("Gamepad cursor disabled. The per-game touch cursor (game "
                                 "Properties > Controls) can be used instead."));
        return;
    }

    QStringList lines;
    for (const Set set : {Interface, Game}) {
        if (set == Game && !in_games_checkbox->isChecked()) {
            lines << tr("Games: cursor not used while a game runs.");
            continue;
        }
        const ControlSet& controls = sets[set];
        const QString name = set == Interface ? tr("Interface") : tr("Games");
        bool invert_x = false;
        bool invert_y = false;
        const bool right = controls.stick_combobox->currentData().toUInt() != 0;
        if (ControllerHotkey::ResolveStick(right, invert_x, invert_y).empty()) {
            lines << tr("%1: stick not found. Bind the %2 in Input, or configure the %3 analog "
                        "stick in EmulationStation.")
                         .arg(name, right ? tr("C-Stick") : tr("Circle Pad"),
                              right ? tr("right") : tr("left"));
        } else {
            lines << tr("%1: stick found.").arg(name);
        }
        for (std::size_t i = 0; i < controls.click_bindings.size(); ++i) {
            const QString& binding = controls.click_bindings[i];
            if (!binding.isEmpty() && ControllerHotkey::Resolve(binding).empty()) {
                lines << tr("%1: click button %2 (%3) not found on this controller, click the "
                            "field and press the button.")
                             .arg(name)
                             .arg(i + 1)
                             .arg(ControllerHotkey::DisplayName(binding));
            }
        }
        if (controls.click_bindings[0].isEmpty() && controls.click_bindings[1].isEmpty()) {
            lines << tr("%1: no click button, the stick moves the arrow only.").arg(name);
        }
    }
    lines << tr("No click while the controller hotkey enable button is held (Select + L3 / R3 "
                "are hotkeys).");
    status_label->setText(lines.join(QLatin1Char('\n')));
}
