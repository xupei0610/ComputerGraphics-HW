#ifndef PX_CG_WINDOW_HPP
#define PX_CG_WINDOW_HPP

#include "scene.hpp"

#include <SDL2/SDL.h>

#include <memory>

namespace px
{
class Window;
}

class px::Window
{
public:
    static Window *getInstance(std::shared_ptr<Scene> const &scene);

    void setScene(std::shared_ptr<Scene> const &scene);
    void render();
    bool run();

protected:
    Window(std::shared_ptr<Scene> const &scene);
    ~Window();

protected:
    SDL_Window *window;
    SDL_Renderer *renderer;
    // SDL_Texture *texture;

    std::shared_ptr<Scene> scene;
};
#endif // PX_CG_WINDOW_HPP