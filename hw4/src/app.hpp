#ifndef PX_CG_WINDOW_HPP
#define PX_CG_WINDOW_HPP

#include <thread>
#include <atomic>
#include <string>
#include <stdexcept>
#include <atomic>

#include "timer.hpp"
#include "glfw.hpp"
#include "option.hpp"
#include "scene.hpp"
#include "shader/text.hpp"
#include "shader/rectangle.hpp"
#include "maze.hpp"

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

private:
    int _height;
    int _width;
    float _center_y;
    float _center_x;
    float _frame_center_y;
    float _frame_center_x;

public:
    Option opt;
    Scene scene;

public:
    static App * getInstance();

    [[noreturn]]
    void err(std::string const &msg);

    void init(bool window_mode = false);
    bool run();
    void restart();
    void togglePause();
    void toggleFullscreen();

    inline const int &height() const noexcept {return _height;}
    inline const int &width() const noexcept {return _width;}
    inline const float &timeGap() const noexcept {return time_gap;}
    inline const int &fps() const noexcept {return _fps;}
    inline const int &lvl() const noexcept {return _lvl; }

    inline const std::atomic<bool> & willStop() {return _game_stop_request; }

    inline const std::string &title() const noexcept {return _title;}

    void setSize(int width, int height);
    void setTitle(std::string const &title);
    void setLvl(int lvl);

    void processEvents();
    void scroll(float x_offset, float y_offset);
    void cursor(float x_pos, float y_pos);
    void click(int button, int action);

    void renderScene();
    void gameGUI();
    void pauseScene();
    void loadingScreen();
    void launchScreen();
    void genNextLvl();
    void genGameScene();

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

    void initShaders();

protected:
    Timer timer;
    GLFWwindow * window;
    bool mouse_detected;
    float time_gap;

    bool action[N_ACTIONS];

    TextShader *text_shader;
    RectangleShader *rectangle_shader;

    std::size_t title_font;
    std::size_t leftside_font;

    std::size_t font_size;
    float half_pause_scene_font_size;

    bool is_pausing;

    unsigned int fbo; // a frame buffer of current scene output
    unsigned int bto; // buffered texture
    unsigned int rbo;

private:
    int _lvl;
    int _fps;
    std::string _title;

    App &operator=(App const &) = delete;
    App &operator=(App &&) = delete;

    // gui related
    bool _full_screen;
    bool _on_resume, _on_restart, _on_option, _on_quit;
    bool _on_option_screen;

    std::atomic<bool> _game_stop_request;
    std::atomic<bool> _game_gen_request;
    std::thread *_game_gen_thread;
};

#endif
