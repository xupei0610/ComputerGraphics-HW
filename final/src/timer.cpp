#include "timer.hpp"

using namespace px;

void Timer::restart()
{
    elapsed = start_time - start_time;
    start_time = std::chrono::system_clock::now();
    paused = false;
}

void Timer::pause()
{
    if (paused == false)
    {
        elapsed += std::chrono::system_clock::now() - start_time;
        paused = true;
    }
}

void Timer::resume()
{
    paused = false;
    start_time = std::chrono::system_clock::now();
}


