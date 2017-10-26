#include "parser.hpp"

#ifdef USE_GUI
  #include "window.hpp"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "util/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image_write.h"

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <csignal>
#include <iomanip>
#include <thread>

using namespace px;

bool stop_request;

void help(const char* const &exe_name)
{
    std::cout << "This is a ray tracing program developed by Pei Xu.\n\n"
                 "Usage:\n"
                 "  " << exe_name << " <scene_file>" << std::endl;
}

void outputImg(std::unordered_map<std::string, IMAGE_FORMAT> const &outputs,
               std::shared_ptr<Scene> const &scene);

int main(int argc, char *argv[])
{

    std::string static const DEFAULT_SCENE = {
#include "default_scene.dat"
    };

    std::string f;
    if (argc == 1)
    {
        f = DEFAULT_SCENE;
        std::cout << "[Info] Default scene loaded." << std::endl;
    }
    else if (argc == 2)
    {
        std::ifstream file(argv[1]);
        if (!file.is_open())
            throw std::invalid_argument("[Error] Failed to open scene file `" + std::string(argv[1]) + "`.");
        try
        {
            f.resize(file.seekg(0, std::ios::end).tellg());
            file.seekg(0, std::ios::beg).read(&f[0], static_cast<std::streamsize>(f.size()));
        }
        catch (std::exception)
        {
            throw std::invalid_argument("[Error] Failed to read scene file `" + std::string(argv[1]) + "`.");
        }
        std::cout << "[Info] Loaded scene file `" << argv[1] << "`" << std::endl;
    }
    else
    {
        help(argv[0]);
        return 1;
    }

    stop_request = false;
    auto scene = std::make_shared<Scene>();

    std::signal(SIGINT, [](int signal){
        stop_request = true;
    });

    auto outputs = Parser::parse(f, scene);

#ifdef USE_GUI
    Window w(scene);
    auto t = std::thread([&]
    {
        w.render();
        if (stop_request == false)
        {
            std::cout << "\033[1K\r[Info] Process time: " << scene->renderingTime() << "ms" << std::endl;
            outputImg(outputs, scene);
        }
    });

    while (w.run() && stop_request == false);


    if (stop_request == false)
        std::cout << "\033[1K\r[Info] Process time: " << scene->renderingTime() << "ms" << std::endl;

    if (t.joinable())
    {
        if (scene->is_rendering)
            std::cout << "[Info] Stop rendering..." << std::endl;
        scene->stopRendering();
        t.join();
    }
#else
    bool started_rendering = false;
    auto t = std::thread([&]
    {
        started_rendering = true;
	    scene->render();
        if (stop_request == false)
            outputImg(outputs, scene);
    });

    while (started_rendering == false)
        std::cout << "\r[Info] Begin rendering..." << std::flush;
    while (scene->is_rendering && stop_request == false)
        std::cout << "\r[Info] Rendering: "
                  << scene->renderingProgress() << " / " << scene->dimension
                  << " (" << std::setprecision(2)
                  << (scene->renderingProgress() * 100.0 / scene->dimension)
                  << "%)" << std::flush;

#ifdef NDEBUG
    std::cout << "\r\n";
#endif

    if (stop_request == false)
    {
        std::cout << "\033[1K\r[Info] Process time: " << scene->renderingTime() << "ms" << std::endl;
    }

    if (t.joinable())
    {
        if (scene->is_rendering)
            std::cout << "[Info] Stop rendering..." << std::endl;
        scene->stopRendering();
        t.join();
    }

    if (stop_request == false)
    {
	    std::cout << "\nPress enter key to exit...";
	    std::cin.ignore();
    }
#endif

    return 0;
}

void outputImg(std::unordered_map<std::string, IMAGE_FORMAT> const &outputs,
               std::shared_ptr<Scene> const &scene)
{
    if (scene->pixels.data == nullptr)
    {
        std::cout << "[Warn] Ignore to output empty image" << std::endl;
        return;
    }

    if (outputs.empty())
    {
        stbi_write_bmp("raytraced.bmp", scene->width, scene->height,
                       3, scene->pixels.data);
        std::cout << "[Info] Write output image into `raytraced.bmp`" << std::endl;
    }
    else
    {
        for (const auto &o : outputs)
        {
            switch (o.second)
            {
                case IMAGE_FORMAT::BMP:
                    stbi_write_bmp(o.first.data(), scene->width, scene->height,
                                   3, scene->pixels.data);
                    break;
                case IMAGE_FORMAT::JPG:
                    stbi_write_jpg(o.first.data(), scene->width, scene->height,
                                   3, scene->pixels.data, 100);
                    break;
                case IMAGE_FORMAT::TGA:
                    stbi_write_tga(o.first.data(), scene->width, scene->height,
                                   3, scene->pixels.data);
                    break;
                default:
                    stbi_write_png(o.first.data(), scene->width, scene->height,
                                   3, scene->pixels.data, scene->width * 3);
                    break;
            }
            std::cout << "[Info] Write output image into `" << o.first << "`" << std::endl;
        }
    }
}