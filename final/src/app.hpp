#ifndef PX_CG_WINDOW_HPP
#define PX_CG_WINDOW_HPP

#include <string>
#include <exception>

#include "timer.hpp"
#include "glfw.hpp"
#include "option.hpp"
#include "scene.hpp"
#include "menu.hpp"

namespace px {

class WindowError;
class App;

}

class px::WindowError : public std::exception
{

public:
    WindowError(const std::string &msg, const int code=0)
            : msg(msg), err_code(code)
    {}
    const char *what() const noexcept override
    {
        return msg.data();
    }
    inline int code() const
    {
        return err_code;
    }

protected:
    std::string msg;
    int err_code;
};

class px::App
{
public:
    static const int WIN_HEIGHT;
    static const int WIN_WIDTH;
    static const char *WIN_TITLE;

    static const float MOUSE_SENSITIVITY;

    static const float MOVE_SPEED;

public:
    Option opt;
    Scene scene;
    Menu menu;

    enum class State
    {
        Pausing,
        Running,
        Win,
        Loose
    } state;

public:
    static App * getInstance();

    [[noreturn]]
    void err(std::string const &msg);

    void init(bool fullscreen = true);
    bool run();
    void restart();
    void close();
    void togglePause();
    void toggleFullscreen();

    inline const int &height() const noexcept {return _height;}
    inline const int &width() const noexcept {return _width;}
    inline const std::string &title() const noexcept {return _title;}
    inline const float &timeGap() const noexcept {return time_gap;}
    inline const int &fps() const noexcept {return _fps;}

    void gameGUI();

    void setSize(int width, int height);
    void setTitle(std::string const &title);

    void processEvents();
    void scroll(float x_offset, float y_offset);
    void cursor(float x_pos, float y_pos);
    void click(int button, int action);

protected:
    App();
    ~App();

    void updateWindowSize();
    void updateFrameBufferSize();
    void updateTimeGap();

    static void keyCallback(GLFWwindow *, int key, int scancode, int action, int mods);
    static void mouseCallback(GLFWwindow *, int button, int action, int mods);
    static void scrollCallback(GLFWwindow *, double x_offset, double y_offset);
    static void cursorPosCallback(GLFWwindow *, double x_pos, double y_pos);
    static void windowSizeCallback(GLFWwindow *window, int width, int height);
    static void frameBufferSizeCallback(GLFWwindow *window, int width, int height);

protected:
    static App * instance;

    Timer timer;
    GLFWwindow * window;
    float time_gap;
    bool mouse_detected;

    bool action[N_ACTIONS];

private:
    int _height;
    int _width;
    std::string _title;
    float _center_y;
    float _center_x;

    int _fps;

    // gui related
    bool _full_screen;

    App &operator=(App const &) = delete;
    App &operator=(App &&) = delete;

};

#endif
