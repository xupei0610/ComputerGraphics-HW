#include "window.hpp"

using namespace px;

#define WIN_TITLE "HW3 Demo - by Pei Xu"
#define WIN_INIT_WIDTH  500
#define WIN_INIT_HEIGHT 500
#define WIN_INIT_POS_X  SDL_WINDOWPOS_CENTERED
#define WIN_INIT_POS_Y  SDL_WINDOWPOS_CENTERED

Window::Window(std::shared_ptr<Scene> const &scene)
    : window(nullptr),
      renderer(nullptr),
      scene(nullptr),
      _need_update(true)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        throw std::runtime_error(std::string("Failed to initialize SDL: ").append(SDL_GetError()));

    window = SDL_CreateWindow(WIN_TITLE,
                                    WIN_INIT_POS_X, WIN_INIT_POS_Y,
                                    WIN_INIT_WIDTH, WIN_INIT_HEIGHT,
                                    0);
    renderer = SDL_CreateRenderer(window, -1, 0);

    setScene(scene);
}

Window::~Window()
{
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Window::setScene(std::shared_ptr<Scene> const &scene)
{
    this->scene = scene;
}

void Window::setTitle(std::string const &title)
{
    SDL_SetWindowTitle(window, (title + " - by Pei Xu").data());
}

void Window::render()
{
    if (scene == nullptr || scene->is_rendering)
        return;

    if (scene->is_rendering)
        scene->stopRendering();

    while (scene->is_rendering);

    scene->render();
}

bool Window::run()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
            return false;
        if (event.type == SDL_WINDOWEVENT)
        {
            switch (event.window.event)
            {
                case SDL_WINDOWEVENT_EXPOSED:
                    _need_update = true;
                    break;
            }
        }
    }

    if (scene->is_rendering)
    {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        if (w < 100 || h < 100)
        {
            if (w < 100)
                w = 100;
            if (h < 100)
                h = 100;
            SDL_SetWindowSize(window, w, h);
        }

        if (scene->mode == Scene::ComputationMode::CPU)
        {
            SDL_Rect progressbar = {30, h/2-10, static_cast<int>(scene->renderingProgress() * 1.0/ scene->dimension * (w - 60)), 20};
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
            SDL_RenderFillRect(renderer, &progressbar);
        }
        else
        {
            SDL_Rect progressbar_bg = {30, h/2-10, w-60, 20};
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderFillRect(renderer, &progressbar_bg);

            current_time = SDL_GetTicks();

            last_pos = (current_time - last_time) * 0.1 + last_pos;
            if (last_pos > w-110)
                last_pos = 0;

            SDL_Rect progressbar_fg = {static_cast<int>(30 + last_pos), h/2-10, 50, 20};
            SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
            SDL_RenderFillRect(renderer, &progressbar_fg);

            last_time = current_time;

        }

        SDL_RenderPresent(renderer);

        _need_update = true;
    }
    else if (_need_update == true)
    {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        if (w != scene->width || h != scene->height)
            SDL_SetWindowSize(window, scene->width, scene->height);

        for (auto r = 0; r < scene->height; ++r)
        {
            for (auto c = 0; c < scene->width; ++c)
            {
                SDL_SetRenderDrawColor(renderer,
                                       scene->pixels.color[r*scene->width + c].r,
                                       scene->pixels.color[r*scene->width + c].g,
                                       scene->pixels.color[r*scene->width + c].b,
                                       255);
                SDL_RenderDrawPoint(renderer, c, r);
            }
        }
        _need_update = false;
        SDL_RenderPresent(renderer);
    }

    return true;
}
