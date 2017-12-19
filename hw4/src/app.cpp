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
    : _height(WIN_HEIGHT), _width(WIN_WIDTH),
      opt(), scene(opt),
      window(nullptr), text_shader(nullptr), rectangle_shader(nullptr),
      font_size(40), half_pause_scene_font_size(8.0f),
      fbo(0), bto(0), rbo(0),
      _title(WIN_TITLE),
      _game_stop_request(false), _game_gen_request(false),
      _game_gen_thread(new std::thread([&](){
            while (_game_stop_request == false)
            {
                if (_game_gen_request)
                {
                    scene.reset(std::min(20+4*_lvl, 39), std::min(5+1*_lvl, 10));
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    _game_gen_request = false;
                }
            }
        }))
{}

App::~App()
{
    delete text_shader;
    delete rectangle_shader;
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &bto);
    glDeleteRenderbuffers(1, &rbo);
    glfwDestroyWindow(window);
    _game_stop_request = true;
    if (_game_gen_thread != nullptr && _game_gen_thread->joinable())
        _game_gen_thread->join();
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

void App::setLvl(int lvl)
{
    _lvl = std::max(1, lvl);
}

void App::restart()
{
    is_pausing = true;
    togglePause();
    scene.setState(Scene::State::Over);
    std::memset(action, 0, sizeof(action));
    _lvl = 1;
    _game_gen_request = true;
}

void App::initShaders()
{
    if (text_shader == nullptr)
    {
        text_shader = new TextShader;
        text_shader->setFontHeight(font_size);
        title_font = text_shader->addFont(TITLE_FONT_DATA,
                                          sizeof(TITLE_FONT_DATA));
        leftside_font = text_shader->addFont(LEFTSIDE_PROMPT_FONT_DATA,
                                          sizeof(LEFTSIDE_PROMPT_FONT_DATA));
    }
    if (rectangle_shader == nullptr)
        rectangle_shader = new RectangleShader;
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
        glfwGetFramebufferSize(window,
                               &scene.character.cam.width,
                               &scene.character.cam.height);
        glViewport(0, 0, scene.character.cam.width, scene.character.cam.height);

        scene.character.cam.updateProjMat();
        _frame_center_x = scene.character.cam.width * 0.5f;
        _frame_center_y = scene.character.cam.height * 0.5f;
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
    if (is_pausing)
    {
        is_pausing = false;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        mouse_detected = false;
        time_gap = -1;
        timer.resume();
    }
    else
    {
        timer.pause();
        _on_resume = false; _on_restart = false; _on_option = false; _on_quit = false;
        _on_resume = false; _on_restart = false; _on_option = false; _on_quit = false;
        is_pausing = true;
        renderScene();
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        pauseScene();
    }
    _on_option_screen = false;
}

void App::toggleFullscreen()
{
    if (_full_screen)
    {
        auto m = glfwGetWindowMonitor(window);
        auto v = glfwGetVideoMode(m);
        glfwSetWindowMonitor(window, nullptr,
                             (v->width - _width)/2, (v->height - _height)/2,
                             _width, _height, GL_DONT_CARE);
        glfwSetWindowSize(window, _width, _height);
        _full_screen = false;
    }
    else
    {
        auto m = glfwGetPrimaryMonitor();
        auto v = glfwGetVideoMode(m);
        glfwSetWindowMonitor(window, m, 0, 0, v->width, v->height, GL_DONT_CARE);
        _full_screen = true;
    }

    updateWindowSize();
    updateFrameBufferSize();
}

void App::processEvents()
{
    glfwPollEvents();

    if (is_pausing)
        return;

    if (action[static_cast<int>(Action::Run)])
        scene.character.activateAction(Action::Run, true);
    else
        scene.character.activateAction(Action::Run, false);

    static auto head_light_key_pressed = false;
    if (action[static_cast<int>(Action::ToggleHeadLight)] == false && head_light_key_pressed == true)
        scene.character.activateAction(Action::ToggleHeadLight, true);
    head_light_key_pressed = action[static_cast<int>(Action::ToggleHeadLight)];

    if (action[static_cast<int>(Action::MoveForward)] &&
            action[static_cast<int>(Action::MoveBackward)] == false)
        scene.character.activateAction(Action::MoveForward, true);
    else if (action[static_cast<int>(Action::MoveBackward)] &&
        action[static_cast<int>(Action::MoveForward)] == false)
        scene.character.activateAction(Action::MoveBackward, true);

    if (action[static_cast<int>(Action::MoveLeft)] &&
            action[static_cast<int>(Action::MoveRight)] == false)
        scene.character.activateAction(Action::MoveLeft, true);
    else if (action[static_cast<int>(Action::MoveRight)] &&
             action[static_cast<int>(Action::MoveLeft)] == false)
        scene.character.activateAction(Action::MoveRight, true);

//    if (action[static_cast<int>(Action::TurnLeft)] &&
//        action[static_cast<int>(Action::TurnRight)] == false)
//        scene.activateAction(Action::TurnLeft, true);
//    else if (action[static_cast<int>(Action::TurnRight)] &&
//             action[static_cast<int>(Action::TurnLeft)] == false)
//        scene.activateAction(Action::TurnRight, true);

    if (action[static_cast<int>(Action::Jump)])
        scene.character.activateAction(Action::Jump, true);
}

void App::scroll(float, float y_offset)
{
    scene.character.cam.zoom(y_offset);
}

void App::cursor(float x_pos, float y_pos)
{
    if (is_pausing || _game_gen_request)
    {   // gui mode

        // for H-DPI monitor
        if (_center_x != _frame_center_x)
            x_pos *= _frame_center_x / _center_x;
        if (_center_y != _frame_center_y)
            y_pos *= _frame_center_y / _center_y;

        if (is_pausing)
        {
            _on_resume = false; _on_restart = false; _on_option = false; _on_quit = false;
            _on_resume = false; _on_restart = false; _on_option = false; _on_quit = false;
            if (x_pos > _frame_center_x - half_pause_scene_font_size*(_on_option_screen ? 12.0f : 6.0f) &&
                x_pos < _frame_center_x + half_pause_scene_font_size*(_on_option_screen ? 12.0f : 6.0f))
            {
                if (y_pos > _frame_center_y - 20 - half_pause_scene_font_size && y_pos < _frame_center_y - 20 + half_pause_scene_font_size)
                {
                    _on_resume = true;
                }
                else if (y_pos > _frame_center_y + 20 - half_pause_scene_font_size && y_pos < _frame_center_y + 20 + half_pause_scene_font_size)
                {
                    _on_restart = true;
                }
                else if (y_pos > _frame_center_y + 60 - half_pause_scene_font_size && y_pos < _frame_center_y + 60 + half_pause_scene_font_size)
                {
                    _on_option = true;
                }
                else if (y_pos > _frame_center_y + 100 - half_pause_scene_font_size && y_pos < _frame_center_y + 100 + half_pause_scene_font_size)
                {
                    _on_quit = true;
                }
            }
        }
        else // if (_game_gen_request)
        {
            _on_quit = false;
            if (y_pos > _frame_center_y + 100 - half_pause_scene_font_size && y_pos < _frame_center_y + 100 + half_pause_scene_font_size)
            {
                _on_quit = true;
            }
        }
        return;
    }

    if (mouse_detected)
    {
        auto x_offset = opt.mouseSensitivity() * (x_pos - _center_x);
        auto y_offset = opt.mouseSensitivity() * (y_pos - _center_y);
        scene.character.cam.updateAng(x_offset, opt.invertY() ? -y_offset : y_offset);
    }
    else
    {
        mouse_detected = true;
    }
    glfwSetCursorPos(window, _center_x, _center_y);
}

void App::click(int button, int action)
{
    static auto state = GLFW_RELEASE;
    if (is_pausing)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS && action == GLFW_RELEASE)
        {
            if (_on_option_screen)
            {
                if (_on_resume) toggleFullscreen();
                else if (_on_restart) opt.setInvertY(!opt.invertY());
                else if (_on_option) _on_option_screen = false;
            }
            else if (scene.gameState() == Scene::State::Running)
            {
                if (_on_resume) togglePause();
                else if (_on_restart) restart();
                else if (_on_option) _on_option_screen = true;
                else if (_on_quit) glfwSetWindowShouldClose(window, 1);
            }
            else if (_on_resume)
            {
                if (scene.gameState() == Scene::State::Win)
                {
                    genNextLvl();
                    is_pausing = false;
                }
                else
                    restart();
            }
        }
    }
    else if (_game_gen_request)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS && action == GLFW_RELEASE)
        {
            if (_on_quit) glfwSetWindowShouldClose(window, 1);
        }
    }
    state = action;
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
            action == GLFW_RELEASE ? false : true;

    if (key == instance->opt.shortcuts[Action::Pause] && action == GLFW_PRESS)
    {
        if (instance->_on_option_screen)
            instance->_on_option_screen = false;
        else if (instance->scene.gameState() == Scene::State::Running)
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

void App::init(bool window_mode)
{
    if (window) glfwDestroyWindow(window);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
#ifndef DISABLE_MULTISAMPLE
    glfwWindowHint(GLFW_SAMPLES, 4);
#endif

    // init window
    if (window_mode)
    {
        window = glfwCreateWindow(_width, _height, _title.data(), nullptr, nullptr);
        _full_screen = false;
    }
    else
    {
        auto m = glfwGetPrimaryMonitor();
        auto v = glfwGetVideoMode(m);
        window = glfwCreateWindow(v->width, v->height, _title.data(), m, nullptr);
        _full_screen = true;
    }
    if (!window) err("Failed to initialize window.");

    // init OpenGL
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) err("Failed to initialize GLEW.");
    glEnable(GL_DEPTH_TEST);
#ifndef DISABLE_MULTISAMPLE
    glEnable(GL_MULTISAMPLE);
#endif

    updateWindowSize();
    updateFrameBufferSize();

    // setup callback fns
//    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetKeyCallback(window, &App::keyCallback);
    glfwSetMouseButtonCallback(window, &App::mouseCallback);
    glfwSetCursorPosCallback(window, &App::cursorPosCallback);
//    glfwSetScrollCallback(window, &App::scrollCallback);
    glfwSetWindowSizeCallback(window, &App::windowSizeCallback);
    glfwSetFramebufferSizeCallback(window, &App::frameBufferSizeCallback);

    initShaders();
    launchScreen();
    scene.init();
    restart();
}

bool App::run()
{
    updateTimeGap();
    processEvents();

    if (glfwWindowShouldClose(window))
        return false;

    if (_game_gen_request)
    {
        loadingScreen();
    }
    else if (is_pausing)
    {
        pauseScene();
    }
    else
    {
        if (scene.run(timeGap()))
        {
            scene.render();
            gameGUI();
            glfwSwapBuffers(window);
        }
        else
        {
            togglePause();
        }
    }
    return true;
}

void App::pauseScene()
{
    rectangle_shader->render(-1.0f, -1.0f, 2.0f, 2.0f,
                             glm::vec4(0.0f, 0.0f, 0.0f, 0.75f), bto);

    text_shader->activateFont(title_font);
    if (scene.gameState() == Scene::State::Running)
    {
        // show pause scene
        if (_on_option_screen)
        {
            text_shader->render("option",
                                _frame_center_x, _frame_center_y-100, 1.0f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("fullscreen",
                                _frame_center_x, _frame_center_y-20, _on_resume ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render(std::string("Y axis: ") + (opt.invertY() ? "non-inverted" : "inverted"),
                                _frame_center_x, _frame_center_y+20, _on_restart ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("back",
                                _frame_center_x, _frame_center_y+60, _on_option ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
        }
        else
        {
            text_shader->render("pausss...iiing",
                                _frame_center_x, _frame_center_y-100, 1.0f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("resume",
                                _frame_center_x, _frame_center_y-20, _on_resume ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("restart",
                                _frame_center_x, _frame_center_y+20, _on_restart ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("option",
                                _frame_center_x, _frame_center_y+60, _on_option ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
            text_shader->render("quit",
                                _frame_center_x, _frame_center_y+100, _on_quit ? 0.6f : 0.4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::Center);
        }
    }
    else
    {   // show game over scene
        text_shader->render(scene.gameState() == Scene::State::Win ? "win" : "loooooose",
                            _frame_center_x, _frame_center_y-100, 1.0f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::Center);
        text_shader->render("continue",
                            _frame_center_x, _frame_center_y-20, _on_resume ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::Center);
    }
    glfwSwapBuffers(window);
}

void App::loadingScreen()
{
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    static int count = 0;
    static float gap = 0;
    gap += timeGap();

    if (gap > 1)
    {
        gap -= 1;
        if (count > 5)
            count = 0;
        else
            ++count;
    }
    std::stringstream ss;
    ss << ". . .";
    for (auto i = 0; i < count; ++i)
        ss << " .";
    text_shader->render("load",
                        _frame_center_x, _frame_center_y-20, 0.6f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);
    text_shader->render(ss.str(),
                        _frame_center_x, _frame_center_y+20, 0.6f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);
    text_shader->render("quit",
                        _frame_center_x, _frame_center_y+100, _on_quit ? 0.6f : 0.4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);
    glfwSwapBuffers(window);

    if (_game_gen_request == false && scene.needUploadData())
    {
        scene.uploadData();
        timer.restart();
    }
}

void App::launchScreen()
{
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    text_shader->render("a  portal  to",
                        _frame_center_x, _frame_center_y-20, 0.6f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);
    text_shader->render("........",
                        _frame_center_x, _frame_center_y+20, 0.6f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);
    text_shader->render("somewhere",
                        _frame_center_x, _frame_center_y+60, 0.4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::Center);

    glfwSwapBuffers(window);
}

void App::gameGUI()
{
    text_shader->activateFont(leftside_font);
    text_shader->render("FPS: " + std::to_string(_fps),
                        10, 10, 0.4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    auto t = timer.count<std::chrono::seconds>();
    std::stringstream ss;
    ss << "time: " << std::setfill('0') << std::setw(2) << t/60 << ":" << std::setw(2) << t%60;
    text_shader->render(ss.str(),
                        10, 40, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    text_shader->render("LVL:" + std::to_string(_lvl),
                        10, 70, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    text_shader->render("Life:" + std::to_string(static_cast<int>(scene.character.characterHp())),
                        10, 100, .4f,
                        glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                        Anchor::LeftTop);
    auto p = 130;
    for (auto const &i: scene.character.items)
    {
        auto & item = Item::lookup(i.first);
        if (item.id() != 0)
        {
            text_shader->render(item.name,
                                10, p, .4f,
                                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                                Anchor::LeftTop);
            p += 30;
        }
    }
}

void App::genGameScene()
{
    _game_gen_request = true;
}

void App::genNextLvl()
{
    _lvl += 1;
    opt.setDamageAmp(_lvl*1.2f);
    genGameScene();
}

void App::renderScene()
{
    if (fbo == 0)
    {
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &bto);
        glGenRenderbuffers(1, &rbo);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, bto);
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB, scene.character.cam.width, scene.character.cam.height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bto, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, scene.character.cam.width, scene.character.cam.height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        err("Failed to generate frame buffer");

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    scene.render();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
