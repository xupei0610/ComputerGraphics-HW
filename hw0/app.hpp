#ifndef APP_HPP
#define APP_HPP

#include "glad/glad.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <array>
#include <string>
#include "global.hpp"
#include "shape.hpp"
#include "shader.hpp"

#define SDL_ERROR_CHECK                                 \
    {                                                   \
        const char *err_msg = SDL_GetError();           \
        if (err_msg[0] != '\0')                         \
            err("Failed to initialize OpenGL context"); \
    }

class App
{
public:
    static App* getAppInstance();

public:
    using Precision = float;

    constexpr static auto WIN_TITLE = "OpenGL Square by Pei Xu";
    constexpr static std::size_t SCREEN_DEF_WIDTH  = 800;
    constexpr static std::size_t SCREEN_DEF_HEIGHT = 800;
    constexpr static auto WIN_INIT_POS_X = SDL_WINDOWPOS_CENTERED;
    constexpr static auto WIN_INIT_POS_Y = SDL_WINDOWPOS_CENTERED;

    constexpr static Precision DEPTH_OFFSET = 0.01;
    constexpr static Precision MOTION_SPEED = 0.015;
    constexpr static Precision MIN_SHAPE_SIZE = Point<Precision>::CLOSE_THRESHOLD*5;
    constexpr static Precision MAX_SHAPE_SIZE = 0.8;

    enum class Action : int
    {
        None = -1,

        AutoDrift = 0, // Click on the exterior region and
                       // then shapes go to the clicked point automatically

        // the following is mouse action
        Trans = 1,  // Drag inside
        Scale = 2,  // Drag on border
        Rotate = 3, // Drag at corner

        // the following is keyboard action
        MoveUp = 4,     // w or Up
        MoveDown = 5,   // s or Down
        MoveLeft = 6,   // q
        MoveRight = 7,  // e
        RotateLeft = 8, // a or Left
        RotateRight = 9,// d or Right
        ZoomIn = 10,    // z or + or =
        ZoomOut = 11,   // x or - or _
        Reset = 12      // r
    };

public:
    std::array<BasicShader<float>*, 4> shader_programs;

    Action action;
    std::array<
        BasicShape<Precision>*,
        std::tuple_size<std::remove_const<decltype(shader_programs)>::type>::value
    > shapes;
    std::array<
        Precision, std::tuple_size<decltype(shapes)>::value
    > depth;
    int              selected;
    Point<Precision> cursor_pos;
    Point<Precision> clicked_pos; // position where the left button of mouse is pressed
    bool             mouse_down;

protected:
    std::size_t   screen_width;
    std::size_t   screen_height;
    Precision     aspect; // sreen_width / screen_height
    SDL_Window   *window;
    SDL_GLContext context;
    SDL_Cursor   *cursor;
    SDL_Event     event;
    decltype(SDL_GetTicks()) last_time;
    float delta_time;

public:
    ~App();

    [[ noreturn ]]
    void err(std::string const &err_msg);
    void init();
    bool run();
    void clean();

    void over();
    void fullscreen();
    void setWindowSize(std::size_t const &w, std::size_t const &h);

    std::string getActionName(Action const &action) const;

protected:
    App();

    void loadShaders();

    bool processInput();
    void processAction();
    void refreshCursor(Point<Precision> const &cursor_pos);
    void renderShapes();

    void mouseClicked(Point<Precision> const &cursor_pos);
    void mouseDragged(Point<Precision> const &cursor_pos,
                      Point<Precision> const &clicked_pos);
    void keyPressed();
    void updateScreenSize(std::size_t const &w, std::size_t const &h);

    void updateDepth(const int &new_selected);
    void updateVertices();
    Action getAvailableAction(Point<Precision> const &cursor_pos,
                              int &tar_shape) const;
    Action getAvailableAction(Point<Precision> const &cursor_pos) const;

public:
    DISABLE_DEFAULT_CONSTRUCTOR(App)

};

#endif // APP_HPP
