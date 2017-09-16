#include "app.hpp"

int main(int argc, char *argv[])
{
    auto app = App::getAppInstance();
    app->init();
    while (app->run());

    return 0;
}
