#ifndef PX_CG_WINDOW_HPP
#define PX_CG_WINDOW_HPP

#include "scene.hpp"

#include <SDL2/SDL.h>

#include <memory>
#include <string>

namespace px
{
class Window;
}

class px::Window
{
public:
    void setScene(std::shared_ptr<Scene> const &scene);
    void setTitle(std::string const &title);
    void render();
    bool run();

    Window(std::shared_ptr<Scene> const &scene);
    ~Window();

protected:
    SDL_Window *window;
    SDL_Renderer *renderer;

    std::shared_ptr<Scene> scene;

    bool _need_update;
};
#endif // PX_CG_WINDOW_HPP