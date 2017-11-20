#include "glfw.hpp"
#include "scene.hpp"
#include "app.hpp"

#include <atomic>
#include <thread>
#include <csignal>

#include <iostream>

std::atomic<bool> stop_request;

int main()
{
    stop_request = false;
    std::signal(SIGINT, [](int signal) {
        stop_request = true;
    });

    px::glfw::init();

    auto t = std::thread([&]{
        auto w = px::App::getInstance();
        w->init();

        while(w->run() && stop_request == false);
    });

    t.join();

    px::glfw::terminate();
    return 0;
}
