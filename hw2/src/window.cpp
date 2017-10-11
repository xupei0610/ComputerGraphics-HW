#include "window.hpp"

using namespace px;

#define WIN_TITLE "HW1 Demo - by Pei Xu"
#define WIN_INIT_WIDTH  500
#define WIN_INIT_HEIGHT 500
#define WIN_INIT_POS_X  SDL_WINDOWPOS_CENTERED
#define WIN_INIT_POS_Y  SDL_WINDOWPOS_CENTERED

Window* Window::getInstance(std::shared_ptr<Scene> const &scene)
{
    static Window instance(scene);
    return &instance;
}

Window::Window(std::shared_ptr<Scene> const &scene)
    : window(nullptr),
      renderer(nullptr),
    //   texture(nullptr),
      scene(nullptr)
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
    // SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Window::setScene(std::shared_ptr<Scene> const &scene)
{
    this->scene = scene;
}

void Window::render()
{
    if (scene == nullptr)
        return;

    scene->render();

    // Failed to render by texture
    // need to check further
    // SDL_DestroyTexture(texture);
    // texture  = SDL_CreateTexture(renderer,
    //                              SDL_PIXELFORMAT_RGB888,
    //                              SDL_TEXTUREACCESS_STATIC,
    //                              scene->width, scene->height);
    // SDL_UpdateTexture(texture,
    //                   nullptr,
    //                   scene->pixels.data,
    //                   scene->width*3);
    // SDL_RenderClear(renderer);
    // SDL_RenderCopy(renderer, texture, nullptr, nullptr);
}

bool Window::run()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
            return false;

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
        SDL_RenderPresent(renderer);
    }
    return true;
}
