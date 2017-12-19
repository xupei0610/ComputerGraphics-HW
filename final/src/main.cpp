#include "glfw.hpp"
#include "app.hpp"

int main()
{
    px::glfw::init();

    auto w = px::App::getInstance();
    w->init(false);

    while(w->run());

    px::glfw::terminate();
    return 0;
}
