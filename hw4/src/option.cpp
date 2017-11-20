#include "option.hpp"

using namespace px;

const float Option::CELL_SIZE = 5.0f;
const float Option::CELL_HEIGHT = 5.0f;
const float Option::WALL_THICKNESS = 2.5f;

const bool Option::INVERT_Y = true;
const float Option::MOUSE_SEN = 0.05f;


Option::Option()
    : cell_size(CELL_SIZE), cell_height(CELL_HEIGHT), wall_thickness(WALL_THICKNESS),

      mouse_sensitivity(MOUSE_SEN), invert_y(INVERT_Y)
{}

void Option::setCellSize(float s)
{
    cell_size = s;
}

void Option::setCellHeight(float h)
{
    cell_height = h;
}

void Option::setWallThickness(float w)
{
    wall_thickness = w;
}

void Option::setMouseSensitivity(float s)
{
    mouse_sensitivity = s;
}

void Option::setInvertY(bool enable)
{
    invert_y = enable;
}

void Option::resetShortcuts()
{
    shortcuts.reset();
}

void Option::resetGameParams()
{
    cell_size = CELL_SIZE;
    cell_height = CELL_HEIGHT;
    wall_thickness = WALL_THICKNESS;
}

void Option::resetOpts()
{
    invert_y = INVERT_Y;
    mouse_sensitivity = MOUSE_SEN;
}

Option::Shortcuts::Shortcuts()
    : shortcuts(KEYBOARD_SHORTCUTS)
{}

void Option::Shortcuts::reset()
{
    shortcuts = KEYBOARD_SHORTCUTS;
}

void Option::Shortcuts::set(Action a, decltype(GLFW_KEY_0) key)
{
    auto index = static_cast<unsigned int>(a);
    auto used = std::find(shortcuts.begin(), shortcuts.end(), key);
    if (used != shortcuts.end())
        shortcuts[used - shortcuts.begin()] = GLFW_KEY_UNKNOWN;
    shortcuts[index] = key;
}