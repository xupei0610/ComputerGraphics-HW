#include "glfw.hpp"
#include "scene.hpp"
#include "app.hpp"

//#ifndef __APPLE__
//#include <atomic>
//#include <thread>
//#include <csignal>
//#endif

std::atomic<bool> stop_request;

int main()
{
//#ifndef __APPLE__
//    stop_request = false;
//    std::signal(SIGINT, [](int signal) {
//        stop_request = true;
//    });
//#endif

    px::glfw::init();

//#ifndef __APPLE__
//    auto t = std::thread([&]{
//#endif
        auto w = px::App::getInstance();
        w->init(false);

        while(w->run() && stop_request == false);

//#ifndef __APPLE__
//    });
//
//    t.join();
//#endif

    px::glfw::terminate();
    return 0;
}
