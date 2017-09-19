#include "app.hpp"

#include <random>
#include <exception>
#include <algorithm>
#include <limits>

#define FOR_EACH_SHAPE(fn, ...)                         \
    for (auto &s : shapes)                              \
        s->fn(__VA_ARGS__);

App* App::getAppInstance()
{
    static App instance;
    return &instance;
}

App::App()
    : shader_programs{{AutoChangeableShader::create(),
                      LocationBasedShader::create(),
                      TextureShader::create(),
                      UniformShader::create()}},
      action(Action::None),
      selected(-1),
      clicked_pos(0, 0),
      mouse_down(false),
      screen_width(SCREEN_DEF_WIDTH),
      screen_height(SCREEN_DEF_HEIGHT),
      aspect(static_cast<decltype(aspect)>(SCREEN_DEF_WIDTH)/SCREEN_DEF_HEIGHT),
      window(nullptr),
      last_time(0)
{

    for (std::size_t i = 0; i < shader_programs.size(); ++i)
    {
        if (i < 2)
            shapes.at(i) = Triangle<Precision>::create(i%2 == 0 ? -0.5 : 0.5,
                                                       0.5,
                                                       0.25,
                                                       0);
        else
            shapes.at(i) = Square<Precision>::create(i%2 == 0 ? -0.5 : 0.5,
                                                     -0.5,
                                                     0.25,
                                                     0);

        depth.at(i) = i * DEPTH_OFFSET;
        shapes.at(i)->updateVertices(aspect);
    }
}

App::~App()
{
    for (auto &s : shapes)
        delete s;
    for (auto &s : shader_programs)
    {
        s->clean();
        delete s;
    }
    clean();
}

[[ noreturn ]]
void App::err(std::string const &err_msg)
{
    throw std::runtime_error(err_msg);
}

void App::init()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        throw std::runtime_error(std::string("Failed to initialize SDL: ").append(SDL_GetError()));

    window = SDL_CreateWindow(WIN_TITLE,
                              WIN_INIT_POS_X, WIN_INIT_POS_Y,
                              screen_width, screen_height,
                              SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL);
    if (!window)
        err("Failed to create window");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    context = SDL_GL_CreateContext(window);

    SDL_ERROR_CHECK

    if (gladLoadGLLoader(SDL_GL_GetProcAddress))
    {
        std::cout << "OpenGL loaded\n"
                     "Vendor:   " << glGetString(GL_VENDOR)   << "\n"
                     "Renderer: " << glGetString(GL_RENDERER) << "\n"
                     "Version:  " << glGetString(GL_VERSION)  << std::endl;
    }
    else
    {
        err("Failed to initialize OpenGL context");
    }

    try
    {
        loadShaders();
    }
    catch (...)
    {
        clean();
        std::rethrow_exception(std::current_exception());
    }

    // glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}

void App::clean()
{
    SDL_GL_DeleteContext(context);
    SDL_FreeCursor(cursor);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void App::loadShaders()
{
    for (std::size_t i = 0; i < shader_programs.size(); ++i)
    {
        shader_programs[i]->compile();
        shader_programs[i]->setVertices(shapes.at(i)->vertices.data(),
                                        shapes.at(i)->num_vertices);
        shader_programs[i]->setDepth(depth.data() + i);
        shader_programs[i]->setTimeGap(&delta_time);
    }

}

bool App::run()
{
    auto current_time = SDL_GetTicks();
    delta_time = current_time - last_time;
    last_time  = current_time;

    if (!processInput())
        return false;

    processAction();

    refreshCursor(cursor_pos);
    renderShapes();

    SDL_GL_SwapWindow(window);

    return true;
}

bool App::processInput()
{
    auto new_selected = selected;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT ||
                (event.type == SDL_KEYUP &&
                 event.key.keysym.sym == SDLK_ESCAPE))
        {
            over();
            return false;
        }
        if (event.type == SDL_KEYUP &&
               (event.key.keysym.sym == SDLK_f ||
                event.key.keysym.sym == SDLK_F11))
        {
            fullscreen();
        }
        else if (event.type == SDL_KEYDOWN)
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_w:
            case SDLK_UP:
                action = Action::MoveUp;
                break;
            case SDLK_s:
            case SDLK_DOWN:
                action = Action::MoveDown;
                break;
            case SDLK_q:
                action = Action::MoveLeft;
                break;
            case SDLK_e:
                action = Action::MoveRight;
                break;
            case SDLK_a:
            case SDLK_LEFT:
                action = Action::RotateLeft;
                break;
            case SDLK_d:
            case SDLK_RIGHT:
                action = Action::RotateRight;
                break;
            case SDLK_z:
            case SDLK_PLUS:
            case SDLK_EQUALS:
                action = Action::ZoomIn;
                break;
            case SDLK_x:
            case SDLK_MINUS:
            case SDLK_UNDERSCORE:
                action = Action::ZoomOut;
                break;
            case SDLK_r:
                action = Action::Reset;
                break;
            case SDLK_0:
            case SDLK_BACKQUOTE:
                new_selected = -1;
                break;
            case SDLK_1:
                new_selected = 0;
                break;
            case SDLK_2:
                if (shader_programs.size() > 1)
                    new_selected = 1;
                break;
            case SDLK_3:
                if (shader_programs.size() > 2)
                    new_selected = 2;
                break;
            case SDLK_4:
                if (shader_programs.size() > 3)
                    new_selected = 3;
                break;
            default:
                break;
            }
            std::cout << "Key " << SDL_GetKeyName(event.key.keysym.sym) << " was pressed" << std::endl;
        }
        else if (event.type == SDL_MOUSEMOTION)
        {
            cursor_pos.x = static_cast<decltype(cursor_pos.x)>(2.0)*event.motion.x/screen_width - 1;
            cursor_pos.y = 1 - static_cast<decltype(cursor_pos.y)>(2.0)*event.motion.y/screen_height;
        }
        else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED)
        {
            updateScreenSize(event.window.data1, event.window.data2);
            updateVertices();
        }
    }
    if (new_selected != selected)
        updateDepth(new_selected);
    return true;
}

void App::processAction()
{
    if (SDL_GetMouseState(0, 0) & SDL_BUTTON(SDL_BUTTON_LEFT))
    {
        if (mouse_down)
        {
            mouseDragged(cursor_pos, clicked_pos);
        }
        else
        {
            mouseClicked(cursor_pos);
            mouse_down = true;
        }
        clicked_pos = cursor_pos;
    }
    else
    {
        mouse_down = false;
        if (action != Action::None)
            keyPressed();
    }

    if (action == Action::AutoDrift)
    {
        if (selected >= 0)
        {
            if (shapes.at(selected)->x != clicked_pos.x ||
				shapes.at(selected)->y != clicked_pos.y)
            {

                auto displacement = delta_time /
                        static_cast<decltype(MOTION_SPEED)>(20) * MOTION_SPEED;

#define AUTO_DRIFT_HELPER(shape)                            \
    if (std::abs(shape->x - clicked_pos.x) < displacement)  \
        shape->x = clicked_pos.x;                           \
    else                                                    \
        shape->x += shape->x < clicked_pos.x ?              \
                    displacement : -displacement;           \
    if (std::abs(shape->y - clicked_pos.y) < displacement)  \
        shape->y = clicked_pos.y;                           \
    else                                                    \
        shape->y += shape->y < clicked_pos.y ?              \
                    displacement : -displacement;           \
    shape->updateVertices(aspect);

                AUTO_DRIFT_HELPER(shapes.at(selected))
                std::cout << "Target to (" << clicked_pos << "; "
                             "Animation Action: " << getActionName(action) << "; "
                             "FPS: " << 1000/delta_time << std::endl;
            }
            else
            {
                action = Action::None;
            }
        }
        else
        {
            auto displacement = delta_time /
                    static_cast<decltype(MOTION_SPEED)>(20) * MOTION_SPEED;
            bool finished = true;
            for (auto & s: shapes)
            {
                if (s->x != clicked_pos.x || s->y != clicked_pos.y)
                {
                    finished = false;
                    AUTO_DRIFT_HELPER(s)
                }
            }
            if (finished)
                action = Action::None;
            else
                std::cout << "Target to (" << clicked_pos << "; "
                             "Animation Action: " << getActionName(action) << "; "
                             "FPS: " << 1000/delta_time << std::endl;
        }
    }
    else if (mouse_down == false && action != Action::None)
    {
        if (selected > 0)
            shapes.at(selected)->updateVertices(aspect);
        else
            updateVertices();
        action = Action::None;
    }
}

void App::refreshCursor(Point<Precision> const &cursor_pos)
{
    switch (action == Action::None ? getAvailableAction(cursor_pos) : action)
    {
    case Action::Trans:
        cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_SIZEALL);
        break;
    case Action::Rotate:
        cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_CROSSHAIR);
        break;
    case Action::Scale:
        cursor = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_HAND);
        break;
    default:
        cursor = SDL_GetDefaultCursor();
        break;
    }

    SDL_SetCursor(cursor);
}

void App::renderShapes()
{
    glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (std::size_t i = 0; i < shapes.size(); ++i)
    {
        shader_programs[i]->activate();
        shader_programs[i]->updateVertices();
        shader_programs[i]->updateDepth();
        shader_programs[i]->updateColor();
        shader_programs[i]->draw(GL_TRIANGLE_STRIP);
    }

}

void App::over()
{
    clean();
    std::cout << "Quit" << std::endl;
}

void App::fullscreen()
{
    SDL_SetWindowFullscreen(window,
                            SDL_GetWindowFlags(window) == SDL_WINDOW_FULLSCREEN ? 0 : SDL_WINDOW_FULLSCREEN
                            );
    std::cout << "Toggle Full Screen Mode" << std::endl;
}

void App::setWindowSize(std::size_t const &w, std::size_t const &h)
{
    SDL_SetWindowSize(window, w, h);
    updateScreenSize(w, h);
}

void App::mouseClicked(Point<Precision> const &cursor_pos)
{
    auto new_selected = selected;
    action = getAvailableAction(cursor_pos, new_selected);
    updateDepth(new_selected);

    std::cout << "Clicked at (" << cursor_pos << "); "
                 "Action: " << getActionName(action) << "; "
                 "FPS: " << 1000/delta_time << std::endl;
}

void App::updateDepth(int const &new_selected)
{
    if (new_selected < 0)
    {
        selected = new_selected;
        std::cout << "All shapes are selected" << std::endl;
    }
    else if (new_selected != selected)
    {
        auto tar_depth = depth.at(new_selected);
        std::for_each(depth.begin(), depth.end(),
                      [tar_depth](Precision &d)
                      {
                        if (d <= tar_depth)
                            d += DEPTH_OFFSET;
                      }
        );
        depth.at(new_selected) = 0;
        selected = new_selected;
        std::cout << "Shape " << new_selected + 1 << " is selected" << std::endl;
    }
}

void App::mouseDragged(Point<Precision> const &cursor_pos, Point<Precision> const &clicked_pos)
{
    switch (action)
    {
    case Action::Rotate:
        shapes[selected]->rotate(cursor_pos, clicked_pos);
        break;
    case Action::Scale:
        shapes[selected]->scale(std::sqrt(
                           (cursor_pos.x-shapes[selected]->x)*(cursor_pos.x-shapes[selected]->x) +
                           (cursor_pos.y-shapes[selected]->y)*(cursor_pos.y-shapes[selected]->y)
                       ) - std::sqrt(
                           (clicked_pos.x-shapes[selected]->x)*(clicked_pos.x-shapes[selected]->x) +
                           (clicked_pos.y-shapes[selected]->y)*(clicked_pos.y-shapes[selected]->y)
                       )
                );
        break;
    case Action::Trans:
        shapes[selected]->move(cursor_pos.x - clicked_pos.x, cursor_pos.y - clicked_pos.y);
        break;
    default:
        return;
    }

    shapes[selected]->updateVertices(aspect);

    std::cout << "Dragged at (" << cursor_pos << "); "
                 "Action: " << getActionName(action) << "; "
                 "FPS: " << 1000/delta_time << std::endl;
}

void App::keyPressed()
{
#define KEYBOARD_ACTION_HELPER(fn, ...)             \
    {                                               \
        if (selected >= 0)                          \
            shapes.at(selected)->fn(__VA_ARGS__);   \
        else                                        \
            FOR_EACH_SHAPE(fn, __VA_ARGS__);        \
    }

    auto displacement = delta_time / static_cast<decltype(MOTION_SPEED)>(20) * MOTION_SPEED;
    switch (action)
    {
    case Action::MoveUp:
        KEYBOARD_ACTION_HELPER(move, 0, displacement)
        break;
    case Action::MoveDown:
        KEYBOARD_ACTION_HELPER(move, 0, -displacement);
        break;
    case Action::MoveLeft:
        KEYBOARD_ACTION_HELPER(move, -displacement, 0);
        break;
    case Action::MoveRight:
        KEYBOARD_ACTION_HELPER(move, displacement, 0);
        break;
    case Action::RotateLeft:
        KEYBOARD_ACTION_HELPER(rotate, PI*displacement);
        break;
    case Action::RotateRight:
        KEYBOARD_ACTION_HELPER(rotate, -PI*displacement);
        break;
    case Action::ZoomIn:
        KEYBOARD_ACTION_HELPER(scale, displacement);
        break;
    case Action::ZoomOut:
        KEYBOARD_ACTION_HELPER(scale, -displacement);
        break;
    case Action::Reset:
        KEYBOARD_ACTION_HELPER(reset);
    default:
        break;
    }
    if (static_cast<int>(action) > 3)
        std::cout << "Keyboard Action: " << getActionName(action) << "; "
                     "FPS: " << 1000/delta_time << std::endl;
}

void App::updateScreenSize(std::size_t const &width, std::size_t const &height)
{
    screen_width = width;
    screen_height = height;

    glViewport(0, 0, screen_width, screen_height);
    aspect = static_cast<decltype(aspect)>(screen_width) / screen_height;
    std::cout << "Window resized to (" << screen_width << ", " << screen_height << ")" << std::endl;
}

void App::updateVertices()
{
    FOR_EACH_SHAPE(updateVertices, aspect);
}

#define APP_GET_ACTION_HELPER(injection_cmd)					\
    auto act = Action::AutoDrift;								\
    Precision z = 1.1;											\
    for (std::size_t i = 0; i < depth.size(); ++i)				\
    {															\
        if (depth[i] < z)										\
        {														\
            switch (cursor_pos.relativeTo(shapes[i], aspect))	\
            {													\
            case RelativePos::Corner:							\
                act = Action::Rotate;							\
                break;											\
            case RelativePos::Inner:							\
                act = Action::Trans;							\
                break;											\
            case RelativePos::Border:							\
                act = Action::Scale;							\
                break;											\
            default:											\
                continue;										\
            }													\
            z = depth[i];										\
            injection_cmd										\
        }														\
    }															\
    return act;

App::Action App::getAvailableAction(Point<Precision> const &cursor_pos,
                                    int &tar_shape) const
{
    APP_GET_ACTION_HELPER({tar_shape = static_cast<int>(i);})
}

App::Action App::getAvailableAction(Point<Precision> const &cursor_pos) const
{
	APP_GET_ACTION_HELPER({})
}

std::string App::getActionName(Action const &action) const
{
    switch (action)
    {
    case Action::AutoDrift:
        return "Auto Drift";
    case Action::Trans:
        return "Translate";
    case Action::Rotate:
        return "Rotate";
    case Action::MoveUp:
        return "Move Up";
    case Action::MoveDown:
        return "Move Down";
    case Action::MoveLeft:
        return "Move Left";
    case Action::MoveRight:
        return "Move Right";
    case Action::RotateLeft:
        return "Rotate Left";
    case Action::RotateRight:
        return "Rotate Right";
    case Action::ZoomIn:
        return "Zoom In";
    case Action::ZoomOut:
        return "Zoom Out";
    case Action::Reset:
        return "Reset";
    default:
        return "";
    }
}
