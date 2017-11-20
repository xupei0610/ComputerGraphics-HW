#ifndef PX_CG_TIMER_HPP
#define PX_CG_TIMER_HPP

#include <chrono>

namespace px
{
class Timer;
}

class px::Timer
{

protected:
    std::chrono::system_clock::time_point start_time;
    decltype(start_time - start_time) elapsed;
    bool paused;

public:
    Timer() = default;
    ~Timer() = default;

    void restart();

    void pause();
    void resume();

    template <typename T>
    auto count() -> decltype(std::chrono::duration_cast<T>(elapsed).count())
    {
        if (paused == false)
        {
            auto c = std::chrono::system_clock::now();
            elapsed += c - start_time;
            start_time = c;
        }

        return std::chrono::duration_cast<T>(elapsed).count();
    }
};
#endif
