#include "object.hpp"
#include "scene.hpp"
#ifdef USE_GUI
  #include "window.hpp"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <memory>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace px;

enum class IMAGE_FORMAT
{
    PNG,
    JPG,
    BMP,
    TGA
};

void help(const char* const &exe_name)
{
    std::cout << "This is a ray tracing program using only CPU developed by Pei Xu.\n\n"
                 "Usage:\n"
                 "  " << exe_name << " <scene_file>" << std::endl;
}

std::unordered_map<std::string, IMAGE_FORMAT> parse(
        std::string const &script_str,
        std::shared_ptr<Scene> const &scene);

std::string static const DEFAULT_SCENE = {
#include "default_scene.dat"
};

int main(int argc, char *argv[])
{
    std::string f;
    if (argc == 1)
    {
        f = DEFAULT_SCENE;
        std::cout << "Default scene loaded." << std::endl;
    }
    else if (argc == 2)
    {
        std::ifstream file(argv[1]);
        if (!file.is_open())
            throw std::invalid_argument("Failed to open scene file `" + std::string(argv[1]) + "`.");
        try
        {
            f.resize(file.seekg(0, std::ios::end).tellg());
            file.seekg(0, std::ios::beg).read(&f[0], static_cast<std::streamsize>(f.size()));
        }
        catch (std::exception)
        {
            throw std::invalid_argument("Failed to read scene file `" + std::string(argv[1]) + "`.");
        }
    }
    else
    {
        help(argv[0]);
        return 1;
    }

    auto s = std::make_shared<Scene>();
    auto outputs = parse(f, s);
#ifdef USE_GUI
    auto w = px::Window::getInstance(s);
    w->render();
#else
	s->render();
#endif
    if (outputs.empty())
    {
        stbi_write_bmp("raytraced.bmp", s->width, s->height, 3, s->pixels.data);
    }
    else
    {
        for (const auto &o : outputs)
        {
            switch (o.second)
            {
                case IMAGE_FORMAT::BMP:
                    stbi_write_bmp(o.first.data(), s->width, s->height, 3, s->pixels.data);
                    break;
                case IMAGE_FORMAT::JPG:
                    stbi_write_jpg(o.first.data(), s->width, s->height, 3, s->pixels.data, 100);
                    break;
                case IMAGE_FORMAT::TGA:
                    stbi_write_tga(o.first.data(), s->width, s->height, 3, s->pixels.data);
                    break;
                default:
                    stbi_write_png(o.first.data(), s->width, s->height, 3, s->pixels.data, s->width*3);
                    break;
            }
			std::cout << "Write image into " << o.first << std::endl;
        }
    }
#ifdef USE_GUI
    while (w->run());
#else
	std::cout << "Enter any key to exit. ";
	std::cin.ignore();

#endif
    return 0;
}

std::unordered_map<std::string, IMAGE_FORMAT> parse(
        std::string const &script_str,
        std::shared_ptr<Scene> const &scene)
{
    std::unordered_map<std::string, IMAGE_FORMAT> outputs;
    std::vector<Point> vertices;
    std::vector<Direction> normals;

    std::istringstream buffer(script_str);

    auto material = UniformMaterial::create();
    std::string line;
    int ln = 0;

#define S2D(var) std::stod(var)
#define S2I(var) std::stoi(var)
#define PARAM_CHECK(name, param_size, param)                                                \
    if (param.size() != param_size + 1)                                                     \
        throw std::invalid_argument("Failed to parse `"                                     \
                                    name                                                    \
                                    "` (" +                                                 \
                                    std::to_string(param_size) + " parameters expected, " + \
                                    std::to_string(param.size()-1) + "provided.");

    while (std::getline(buffer, line, '\n'))
    {
        ++ln;
        auto cmd_s = std::find_if_not(line.begin(), line.end(), isspace);
        if (cmd_s == line.end() || *cmd_s == '#')
            continue;

        auto buf = std::istringstream(line);
        std::istringstream end;

        std::vector<std::string> param;
        std::copy(
                std::istream_iterator<std::string>(buf),
                std::istream_iterator<std::string>(),
                std::back_inserter(param));

        if (param[0].size() > 17)
            throw std::invalid_argument("Failed to parse script with an unsupported property `" + param[0] + "`.");

        std::transform(param[0].begin(), param[0].end(), param[0].begin(), tolower); // case non-sensitive

        if (param[0] == "material")
        {
            PARAM_CHECK("material", 14, param)
            material = UniformMaterial::create(
                    {S2D(param[1]), S2D(param[2]), S2D(param[3])},
                    {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                    {S2D(param[7]), S2D(param[8]), S2D(param[9])}, S2D(param[10]),
                    {S2D(param[11]), S2D(param[12]), S2D(param[13])}, S2D(param[14]));
        }
        else if (param[0] == "sphere")
        {
            PARAM_CHECK("sphere", 4, param)
            scene->objects.insert(Sphere::create({S2D(param[1]), S2D(param[2]), S2D(param[3])}, S2D(param[4]), material));
        }
        else if (param[0] == "vertex")
        {
            PARAM_CHECK("vertex", 3, param)
            vertices.emplace_back(S2D(param[0]), S2D(param[1]), S2D(param[2]));
        }
        else if (param[0] == "triangle")
        {
            PARAM_CHECK("triangle", 3, param)
            scene->objects.insert(Triangle::create(vertices.at(S2I(param[1])),
                                               vertices.at(S2I(param[2])),
                                               vertices.at(S2I(param[3])),
                                               material));
        }
        else if (param[0] == "normal")
        {
            PARAM_CHECK("normal", 3, param)
            normals.emplace_back(S2D(param[0]), S2D(param[1]), S2D(param[2]));
        }
        else if (param[0] == "normal_triangle")
        {
            PARAM_CHECK("triangle", 6, param)
            scene->objects.insert(NormalTriangle::create(vertices.at(S2I(param[1])), normals.at(S2I(param[2])),
                                                     vertices.at(S2I(param[3])), normals.at(S2I(param[4])),
                                                     vertices.at(S2I(param[5])), normals.at(S2I(param[6])),
                                                     material));
        }
        else if (param[0] == "directional_light")
        {
            PARAM_CHECK("directional_light", 6, param)
            scene->lights.insert(DirectionalLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                      {S2D(param[4]), S2D(param[5]), S2D(param[6])}));
        }
        else if (param[0] == "point_light")
        {
            PARAM_CHECK("point_light", 6, param)
            scene->lights.insert(PointLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                {S2D(param[4]), S2D(param[5]), S2D(param[6])}));
        }
        else if (param[0] == "spot_light")
        {
            PARAM_CHECK("spot_light", 11, param)
            scene->lights.insert(SpotLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                               {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                               {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                               S2D(param[10])*DEG2RAD, S2D(param[11])*DEG2RAD,
                                               1));
        }
        else if (param[0] == "ambient_light")
        {
            PARAM_CHECK("ambient_light", 3, param)
            scene->setAmbientLight({S2D(param[1]), S2D(param[2]), S2D(param[3])});
        }
        else if (param[0] == "camera")
        {
            PARAM_CHECK("camera", 10, param)
            scene->setCamera(Camera::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                        {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                        {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                        S2D(param[10])*DEG2RAD));
        }
        else if (param[0] == "film_resolution")
        {
            PARAM_CHECK("film_resolution", 2, param)
            scene->setSceneSize(S2I(param[1]), S2I(param[2]));
        }
        else if (param[0] == "background")
        {
            PARAM_CHECK("background", 3, param)
            scene->setBackground(S2D(param[1]), S2D(param[2]), S2D(param[3]));
        }
        else if (param[0] == "max_depth")
        {
            PARAM_CHECK("max_depth", 1, param)
            scene->setRecursionDepth(S2I(param[1]));
        }
        else if (param[0] == "output_image")
        {
            PARAM_CHECK("output_image", 1, param)

            auto idx = param[1].rfind(".");
            if (idx == std::string::npos)
            {
                outputs.emplace(param[1], IMAGE_FORMAT::PNG);
                std::cout << "[Warn] Failed to parse output file extension. Set to PNG.\n";
            }
            else
            {
                auto extension = param[1].substr(idx+1);
                std::transform(extension.begin(), extension.end(), extension.begin(), tolower);
                if (extension == "bmp")
                    outputs.emplace(param[1], IMAGE_FORMAT::BMP);
                else if (extension == "jpg" || extension == "jpeg")
                    outputs.emplace(param[1], IMAGE_FORMAT::JPG);
                else if (extension == "png")
                    outputs.emplace(param[1], IMAGE_FORMAT::PNG);
                else
                {
                    outputs.emplace(param[1], IMAGE_FORMAT::PNG);
                    std::cout << "[Warn] Failed to parse output file extension `<< extension <<`. Set to PNG.\n";
                }
            }
        }
        else if (param[0] == "max_vertices")
        {
            PARAM_CHECK("max_vertices", 1, param)
//          setMaxVertices(S2I(param[1]));
        }
        else if (param[0] == "max_normals")
        {
            PARAM_CHECK("max_normals", 1, param)
//          setMaxNormals(S2I(param[1]));
        }
        else if (param[0] == "sampling_radius")
        {
            PARAM_CHECK("sampling_radius", 1, param)
            scene->setSamplingRadius(S2I(param[1]));
        }
        else
        {
            throw std::invalid_argument("Failed to parse script with an unsupported property `" + param[0] + "`.");
        }

//        std::cout << ln << ": " << line << std::endl;
//        for (auto & s : param)
//        {
//            std::cout << s << " ";
//        }
//        std::cout << std::endl;
    }

    return outputs;
}