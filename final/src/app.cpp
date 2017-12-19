#include "app.hpp"
#include "item.hpp"

#ifndef NDEBUG
#   include <iostream>
#endif
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace px;

const int App::WIN_HEIGHT = 480;
const int App::WIN_WIDTH  = 640;
const char *App::WIN_TITLE = "OpenGL 3D Game Demo - by Pei Xu";
App *App::instance = nullptr;

const unsigned char TITLE_FONT_DATA[] = {
#include "font/North_to_South.dat"
};
const unsigned char LEFTSIDE_PROMPT_FONT_DATA[] = {
#include "font/The_Brooklyn_Bold_Demo.dat"
};


App* App::getInstance()
{
    if (instance == nullptr) instance = new App;
    return instance;
}

App::App()
        : opt(), scene(&opt), menu(&opt),
          window(nullptr),
          _height(WIN_HEIGHT), _width(WIN_WIDTH), _title(WIN_TITLE)
{}

App::~App()
{
    glfwDestroyWindow(window);
}

[[noreturn]]
void App::err(std::string const &msg)
{
    throw WindowError("App Error: " + msg);
}

void App::setSize(int width, int height)
{
    if (height < 1 || width < 1)
        err("Unable to set window size as a non-positive value.");

    _height = height;
    _width = width;

    if (!_full_screen)
    {
        glfwSetWindowSize(window, _width, _height);
        updateWindowSize();
        updateFrameBufferSize();
    }
}

void App::setTitle(std::string const &title)
{
    _title = title;
    if (window)
        glfwSetWindowTitle(window, title.data());
}

void App::restart()
{
    std::memset(action, 0, sizeof(action));
    if (scene.state == Scene::State::Win)
        scene.lvl += 1;
    else scene.lvl = 1;
    scene.gen(std::min(20+4*scene.lvl, 39), std::min(5+1*scene.lvl, 10));
    opt.setDamageAmp(scene.lvl*1.2f);
    if (state == State::Pausing)
        togglePause();
    timer.restart();
}

void App::close()
{
    glfwSetWindowShouldClose(window, 1);
}

void App::updateWindowSize()
{
    if (window)
    {
        int w, h;
        glfwGetWindowSize(window, &w, &h);
        _center_x = w * 0.5f;
        _center_y = h * 0.5f;
    }
}

void App::updateFrameBufferSize()
{
    if (window)
    {
        glfwGetFramebufferSize(window, &scene.cam.width, &scene.cam.height);
        glViewport(0, 0, scene.cam.width, scene.cam.height);

        scene.cam.updateProj();
        scene.resize();

        menu.setFrameCenter(scene.cam.width * 0.5f, scene.cam.height * 0.5f);
    }
}

void App::updateTimeGap()
{
    static decltype(glfwGetTime()) last_time;
    static decltype(glfwGetTime()) last_count_time;
    static int frames;

    if (time_gap == -1)
    {
        last_time = glfwGetTime();
        time_gap = 0;
        _fps = 0;
        frames = 0;
        last_count_time = last_time;
    }
    else
    {
        auto current_time = glfwGetTime();
        time_gap = current_time - last_time;
        last_time = current_time;

        if (current_time - last_count_time >= 1.0)
        {
            _fps = frames;
            frames = 0;
            last_count_time = current_time;
        }
        else
            frames += 1;
    }
}

void App::togglePause()
{
    if (state == State::Pausing)
    {
        state = State::Running;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        mouse_detected = false;
        time_gap = -1;
        timer.resume();
    }
    else
    {
        timer.pause();
        state = State::Pausing;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void App::toggleFullscreen()
{
    auto m = glfwGetWindowMonitor(window);
    if (m == nullptr) m = glfwGetPrimaryMonitor();
    auto v = glfwGetVideoMode(m);

    if (_full_screen)
    {
        glfwSetWindowMonitor(window, nullptr,
                             (v->width - _width)/2, (v->height - _height)/2,
                             _width, _height, GL_DONT_CARE);
        glfwSetWindowSize(window, _width, _height);
    }
    else
    {
        glfwSetWindowMonitor(window, m, 0, 0, v->width, v->height, GL_DONT_CARE);
    }

    _full_screen = !_full_screen;
    updateWindowSize();
    updateFrameBufferSize();
}

void App::processEvents()
{
    glfwPollEvents();

    if (state == State::Pausing)
        return;

    if (action[static_cast<int>(Action::Run)])
        scene.character.activate(Action::Run, true);
    else
        scene.character.activate(Action::Run, false);

    static auto head_light_key_pressed = false;
    if (action[static_cast<int>(Action::ToggleHeadLight)] == false && head_light_key_pressed == true)
        scene.character.activate(Action::ToggleHeadLight, true);
    head_light_key_pressed = action[static_cast<int>(Action::ToggleHeadLight)];

    if (action[static_cast<int>(Action::MoveForward)] &&
        action[static_cast<int>(Action::MoveBackward)] == false)
        scene.character.activate(Action::MoveForward, true);
    else if (action[static_cast<int>(Action::MoveBackward)] &&
             action[static_cast<int>(Action::MoveForward)] == false)
        scene.character.activate(Action::MoveBackward, true);

    if (action[static_cast<int>(Action::MoveLeft)] &&
        action[static_cast<int>(Action::MoveRight)] == false)
        scene.character.activate(Action::MoveLeft, true);
    else if (action[static_cast<int>(Action::MoveRight)] &&
             action[static_cast<int>(Action::MoveLeft)] == false)
        scene.character.activate(Action::MoveRight, true);

//    if (action[static_cast<int>(Action::TurnLeft)] &&
//        action[static_cast<int>(Action::TurnRight)] == false)
//        scene.activate(Action::TurnLeft, true);
//    else if (action[static_cast<int>(Action::TurnRight)] &&
//             action[static_cast<int>(Action::TurnLeft)] == false)
//        scene.activate(Action::TurnRight, true);

    if (action[static_cast<int>(Action::Jump)])
        scene.character.activate(Action::Jump, true);
}

void App::scroll(float, float y_offset)
{
    scene.cam.zoom(y_offset);
}

void App::cursor(float x_pos, float y_pos)
{
    if (state == State::Pausing)
    {
        menu.cursor(x_pos * menu.frame_center_x / _center_x,
                     y_pos * menu.frame_center_y / _center_y);
        return;
    }

    if (mouse_detected)
    {
        auto x_offset = opt.mouseSensitivity() * (x_pos - _center_x);
        auto y_offset = opt.mouseSensitivity() * (y_pos - _center_y);
        scene.cam.yaw(x_offset).pitch(opt.invertY() ? y_offset : -y_offset);
    }
    else
    {
        mouse_detected = true;
    }
    glfwSetCursorPos(window, _center_x, _center_y);
}

void App::click(int button, int action)
{
    static auto button_state = GLFW_RELEASE;

    if (state == State::Pausing)
        menu.click(this, button, button_state, action);
    else if (scene.state == Scene::State::Running)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                scene.character.activate(Action::Shoot, true);
            }
            else
            {
                scene.character.activate(Action::Shoot, false);
            }
        }
    }

    button_state = action;
}
void App::windowSizeCallback(GLFWwindow *, int width, int height)
{
    instance->updateWindowSize();
}
void App::frameBufferSizeCallback(GLFWwindow *, int width, int height)
{
    instance->updateFrameBufferSize();
}

void App::scrollCallback(GLFWwindow *, double x_offset, double y_offset)
{
    instance->scroll(float(x_offset), float(y_offset));
}

void App::cursorPosCallback(GLFWwindow *, double x_pos, double y_pos)
{
    instance->cursor(float(x_pos), float(y_pos));
}

void App::mouseCallback(GLFWwindow *, int button, int action, int mods)
{
    instance->click(button ,action);
}

void App::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
#define KEY_CALLBACK(Action)                            \
    if (key == instance->opt.shortcuts[Action])         \
        instance->action[static_cast<int>(Action)] =    \
            action != GLFW_RELEASE;

    if (key == instance->opt.shortcuts[Action::Pause] && action == GLFW_PRESS)
    {
        if (instance->menu.state != Menu::State::Pause)
            instance->menu.state = Menu::State::Pause;
        else if (instance->state == State::Pausing || instance->scene.state == Scene::State::Running)
            instance->togglePause();
    }
    else KEY_CALLBACK(Action::MoveForward)
    else KEY_CALLBACK(Action::MoveBackward)
    else KEY_CALLBACK(Action::MoveLeft)
    else KEY_CALLBACK(Action::MoveRight)
    else KEY_CALLBACK(Action::TurnLeft)
    else KEY_CALLBACK(Action::TurnRight)
    else KEY_CALLBACK(Action::Jump)
    else KEY_CALLBACK(Action::Run)
    else KEY_CALLBACK(Action::ToggleHeadLight)

#undef KEY_CALLBACK
}

void App::init(bool fullscreen)
{
    if (window) glfwDestroyWindow(window);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
#ifndef DISABLE_MULTISAMPLE
    glfwWindowHint(GLFW_SAMPLES, 4);
#endif

    // init window
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
    _full_screen = !fullscreen;
    if (!window) err("Failed to initialize window.");

    // init OpenGL
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) err("Failed to initialize GLEW.");
    glEnable(GL_DEPTH_TEST);
#ifndef DISABLE_MULTISAMPLE
    glEnable(GL_MULTISAMPLE);
#endif

    // setup callback fns
    glfwSetKeyCallback(window, &App::keyCallback);
    glfwSetMouseButtonCallback(window, &App::mouseCallback);
    glfwSetCursorPosCallback(window, &App::cursorPosCallback);
//    glfwSetScrollCallback(window, &App::scrollCallback);
    glfwSetWindowSizeCallback(window, &App::windowSizeCallback);
    glfwSetFramebufferSizeCallback(window, &App::frameBufferSizeCallback);

    // load game resource
    scene.init();
    menu.init();

    // show window
    state = State::Running;
    togglePause();
    glfwShowWindow(window);
    toggleFullscreen();

    restart();
}

bool App::run()
{
    updateTimeGap();
    processEvents();

    if (glfwWindowShouldClose(window))
        return false;

    switch (state)
    {
        case State::Running:
        if (scene.run(timeGap()))
        {
            scene.render();
            gameGUI();
            break;
        }

        default:
            state = State::Pausing;
            menu.render(this);
    }

    glfwSwapBuffers(window);
    return true;
}

void App::gameGUI()
{
    menu.text_shader->render("FPS: " + std::to_string(_fps),
                        10, 10, 0.4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    auto t = timer.count<std::chrono::seconds>();
    std::stringstream ss;
    ss << "time: " << std::setfill('0') << std::setw(2) << t/60 << ":" << std::setw(2) << t%60;
    menu.text_shader->render(ss.str(),
                        10, 40, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    menu.text_shader->render("LVL:" + std::to_string(scene.lvl),
                        10, 70, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    menu.text_shader->render("Life:" + std::to_string(static_cast<int>(scene.character.characterHp())),
                        10, 100, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    auto p = 130;
    for (auto const &i: scene.character.items)
    {
        auto & item = Item::lookup(i.first);
        if (item.id() != 0)
        {
            menu.text_shader->render(item.name,
                                10, p, .4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::LeftTop);
            p += 30;
        }
    }
}