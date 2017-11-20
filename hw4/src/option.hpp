#ifndef PX_CG_OPTION_HPP
#define PX_CG_OPTION_HPP

#include <array>
#include <algorithm>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace px {

#define N_ACTIONS 9
enum class Action : unsigned int
{
    Pause = 0,

    MoveForward = 1,
    MoveBackward = 2,
    MoveLeft = 3,
    MoveRight = 4,
    TurnLeft = 5,
    TurnRight = 6,
    Jump = 7,
    Run = 8
};

static constexpr
std::array<decltype(GLFW_KEY_0), N_ACTIONS> KEYBOARD_SHORTCUTS = {
        // system related
        GLFW_KEY_ESCAPE,    // Pause = 0,
        // game control related
        GLFW_KEY_W,         // MoveForward = 1,
        GLFW_KEY_S,         // MoveBackward = 2,
        GLFW_KEY_A,         // MoveLeft = 3,
        GLFW_KEY_D,         // MoveRight = 4,
        GLFW_KEY_Q,   // TurnLeft = 5,
        GLFW_KEY_E,   // TurnRight = 6
        GLFW_KEY_SPACE,     // Jump = 7
        GLFW_KEY_LEFT_SHIFT // Run = 8, Modifier
};

class Option;
}

class px::Option
{
public:
    // game fixed parameters
    static const float CELL_SIZE;
    static const float CELL_HEIGHT;
    static const float WALL_THICKNESS;

    // game options
    static const float MOUSE_SEN;
    static const bool INVERT_Y;

    class Shortcuts
    {
    protected:
        std::array<decltype(GLFW_KEY_0), N_ACTIONS> shortcuts;
    public:
        Shortcuts();
        ~Shortcuts() = default;
        Shortcuts &operator=(Shortcuts const &) = default;
        Shortcuts &operator=(Shortcuts &&) = default;

        decltype(GLFW_KEY_0) operator[](Action a)
        {
            return shortcuts[static_cast<unsigned int>(a)];
        }
        void set(Action a, decltype(GLFW_KEY_0) key);
        void reset();
    } shortcuts;

public:
    Option();

    inline const float &cellSize() const noexcept {return cell_size; }
    inline const float &cellHeight() const noexcept {return cell_height; }
    inline const float &wallThickness() const noexcept {return wall_thickness; }

    inline const bool &invertY() const noexcept {return invert_y;}
    inline const float &mouseSensitivity() const noexcept {return mouse_sensitivity;}


    void setCellSize(float s);
    void setCellHeight(float h);
    void setWallThickness(float w);

    void setInvertY(bool enable);
    void setMouseSensitivity(float s);

    void resetShortcuts();
    void resetGameParams();
    void resetOpts();

    ~Option() = default;
    Option &operator=(Option const &) = default;
    Option &operator=(Option &&) = default;

protected:
    float cell_size;
    float cell_height;
    float wall_thickness;

    float mouse_sensitivity;
    bool  invert_y;
};

#endif
