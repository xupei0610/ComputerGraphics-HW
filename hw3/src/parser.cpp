#include "parser.hpp"
#include "object.hpp"

#include <vector>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace px;

std::unordered_map<std::string, IMAGE_FORMAT> Parser::parse(
        std::string const &script_str,
        std::shared_ptr<Scene> const &scene)
{
    std::unordered_map<std::string, IMAGE_FORMAT> outputs;
    std::vector<Point> vertices;
    std::vector<Direction> normals;

    std::vector<BoundBox *> bound_box;
    std::vector<std::shared_ptr<Transformation> > transform{nullptr};
//    std::shared_ptr<BumpMapping> bump_mapping = nullptr;

    std::istringstream buffer(script_str);

    auto material = UniformMaterial::create();
    std::string line;
    int ln = 0;

#define S2D(var) std::stof(var)
#define S2I(var) std::stoi(var)
#define PARAM_CHECK(name, param_size, param, line)                                          \
    if (param.size() < param_size + 1 ||                                                    \
        (param.size() > param_size + 1 && param[param_size+1][0]!='#'))                     \
    {                                                                                       \
        throw std::invalid_argument("[Error] Failed to parse `" name                        \
                                    "` at line " +                                          \
                                    std::to_string(line) + " (" +                           \
                                    std::to_string(param_size) + " parameters expected, " + \
                                    std::to_string(param.size() - 1) + " provided)");       \
    }
#define PARSE_TRY(cmd, name, line)                                   \
    try                                                              \
    {                                                                \
        cmd;                                                         \
    }                                                                \
    catch (...)                                                      \
    {                                                                \
        throw std::invalid_argument("[Error] Failed to parse `" name \
                                    "` at line " +                   \
                                    std::to_string(line));           \
    }

    bool use_bb = false;
    auto addObj = [&](std::shared_ptr<BaseGeometry> const &obj) {
        if (bound_box.empty())
            scene->addGeometry(obj);//addObj(obj);
        else
            bound_box.back()->addObj(obj);
    };

    while (std::getline(buffer, line, '\n'))
    {
        ++ln;
        auto cmd_s = std::find_if_not(line.begin(), line.end(), isspace);
        if (cmd_s == line.end() || *cmd_s == '#')
            continue;

        auto buf = std::istringstream(line);

        std::vector<std::string> param;
        std::copy(
                std::istream_iterator<std::string>(buf),
                std::istream_iterator<std::string>(),
                std::back_inserter(param));

        if (param[0].size() > 23)
            throw std::invalid_argument("[Error] Failed to parse script with an unsupported property `" + param[0] + "`.");

        std::transform(param[0].begin(), param[0].end(), param[0].begin(), tolower); // case non-sensitive

        if (param[0] == "material")
        {
            PARAM_CHECK("material", 14, param, ln)
            PARSE_TRY(material = UniformMaterial::create(
                    {S2D(param[1]), S2D(param[2]), S2D(param[3])},
                    {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                    {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                    S2D(param[10]),
                    {S2D(param[11]), S2D(param[12]), S2D(param[13])},
                    S2D(param[14])),
                      "material", ln);
        }
        else if (param[0] == "sphere")
        {
            PARAM_CHECK("sphere", 4, param, ln)
            PARSE_TRY(addObj(Sphere::create({S2D(param[1]), S2D(param[2]), S2D(param[3])}, S2D(param[4]), material,
                                            transform.back())),
                      "sphere", ln)
        }
        else if (param[0] == "vertex")
        {
            PARAM_CHECK("vertex", 3, param, ln)
            PARSE_TRY(vertices.emplace_back(S2D(param[1]), S2D(param[2]), S2D(param[3])),
                      "vertex", ln)
        }
        else if (param[0] == "triangle")
        {
            PARAM_CHECK("triangle", 3, param, ln)
            PARSE_TRY(addObj(Triangle::create(vertices.at(S2I(param[1])),
                                              vertices.at(S2I(param[2])),
                                              vertices.at(S2I(param[3])),
                                              material,
                                              transform.back())),
                      "triangle", ln)
        }
        else if (param[0] == "normal")
        {
            PARAM_CHECK("normal", 3, param, ln)
            PARSE_TRY(normals.emplace_back(S2D(param[1]), S2D(param[2]), S2D(param[3])),
                      "normal", ln)
        }
        else if (param[0] == "normal_triangle")
        {
            PARAM_CHECK("triangle", 6, param, ln)
            PARSE_TRY(addObj(NormalTriangle::create(vertices.at(S2I(param[1])), normals.at(S2I(param[4])),
                                                    vertices.at(S2I(param[2])), normals.at(S2I(param[5])),
                                                    vertices.at(S2I(param[3])), normals.at(S2I(param[6])),
                                                    material,
                                                    transform.back())),
                      "normal_triangle", ln)
        }
        else if (param[0] == "directional_light")
        {
            PARAM_CHECK("directional_light", 6, param, ln)
            PARSE_TRY(scene->addLight(DirectionalLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                                    {S2D(param[4]), S2D(param[5]), S2D(param[6])})),
                      "directional_light", ln)
        }
        else if (param[0] == "point_light")
        {
            PARAM_CHECK("point_light", 6, param, ln)
            PARSE_TRY(scene->addLight(PointLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                              {S2D(param[4]), S2D(param[5]), S2D(param[6])})),
                      "point_light", ln)
        }
        else if (param[0] == "spot_light")
        {
            PARAM_CHECK("spot_light", 11, param, ln)
            PARSE_TRY(scene->addLight(SpotLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                             {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                                             {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                                             S2D(param[10]) * DEG2RAD, S2D(param[11]) * DEG2RAD,
                                                             1)),
                      "spot_light", ln)
        }
        else if (param[0] == "ambient_light")
        {
            PARAM_CHECK("ambient_light", 3, param, ln)
            PARSE_TRY(scene->addAmbientLight({S2D(param[1]), S2D(param[2]), S2D(param[3])}),
                      "ambient_light", ln)
        }
        else if (param[0] == "camera")
        {
            PARAM_CHECK("camera", 10, param, ln)
            PARSE_TRY(scene->setCamera(Camera::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                      {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                                      {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                                      S2D(param[10]) * DEG2RAD)),
                      "camera_light", ln)
        }
        else if (param[0] == "film_resolution")
        {
            PARAM_CHECK("film_resolution", 2, param, ln)
            PARSE_TRY(scene->setSceneSize(S2I(param[1]), S2I(param[2])),
                      "film_resolution", ln)
        }
        else if (param[0] == "background")
        {
            PARAM_CHECK("background", 3, param, ln)
            PARSE_TRY(scene->setBackground(S2D(param[1]), S2D(param[2]),
                                           S2D(param[3])),
                      "background", ln)
        }
        else if (param[0] == "max_depth")
        {
            PARAM_CHECK("max_depth", 1, param, ln)
            PARSE_TRY(scene->setRecursionDepth(S2I(param[1])),
                      "max_depth", ln)
        }
        else if (param[0] == "output_image")
        {
            PARAM_CHECK("output_image", 1, param, ln)

            auto idx = param[1].rfind(".");
            if (idx == std::string::npos)
            {
                outputs.emplace(param[1], IMAGE_FORMAT::PNG);
                std::cout << "[Warn] Failed to parse output file extension. Set to PNG.\n";
            }
            else
            {
                auto extension = param[1].substr(idx + 1);
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
            PARAM_CHECK("max_vertices", 1, param, ln)
            PARSE_TRY(vertices.reserve(S2I(param[1])),
                      "max_vertices", ln)
        }
        else if (param[0] == "max_normals")
        {
            PARAM_CHECK("max_normals", 1, param, ln)
            PARSE_TRY(normals.reserve(S2I(param[1])),
                      "max_normals", ln)
        }
        else if (param[0] == "sampling_radius")
        {
            PARAM_CHECK("sampling_radius", 1, param, ln)
            PARSE_TRY(scene->setSamplingRadius(S2I(param[1])),
                      "sampling_radius", ln)
        }
        else if (param[0] == "plane")
        {
            if (param.size() == 7)
                PARSE_TRY(addObj(Plane::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                               {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                               material,
                                               transform.back())),
                          "plane", ln)
            else if (param.size() == 5)
                PARSE_TRY(addObj(Plane::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                               normals.at(S2I(param[4])),
                                               material,
                                               transform.back())),
                          "plane", ln)
            else
                throw std::invalid_argument("[Error] Failed to parse `plane` at line " +
                                            std::to_string(ln) + " (6 or 4 parameters expected, " +
                                            std::to_string(param.size() - 1) + " provided)");
        }
        else if (param[0] == "disk")
        {
            if (param.size() == 8)
                PARSE_TRY(addObj(Disk::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                              {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                              S2D(param[7]),
                                              material,
                                              transform.back())),
                          "disk", ln)

            else if (param.size() == 6)
                PARSE_TRY(addObj(Disk::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                              normals.at(S2I(param[4])),
                                              S2D(param[5]),
                                              material,
                                              transform.back())),
                          "disk", ln)
            else
                throw std::invalid_argument("[Error] Failed to parse `disk` at line " +
                                            std::to_string(ln) + " (7 or 5 parameters expected, " +
                                            std::to_string(param.size() - 1) + " provided)");
        }
        else if (param[0] == "ring")
        {
            if (param.size() == 9)
                PARSE_TRY(addObj(Ring::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                              {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                              S2D(param[7]), S2D(param[8]),
                                              material,
                                              transform.back())),
                          "ring", ln)

            else if (param.size() == 7)
                PARSE_TRY(addObj(Ring::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                              normals.at(S2I(param[4])),
                                              S2D(param[5]), S2D(param[6]),
                                              material,
                                              transform.back())),
                          "ring", ln)
            else
                throw std::invalid_argument("[Error] Failed to parse `ring` at line " +
                                            std::to_string(ln) + " (8 or 6 parameters expected, " +
                                            std::to_string(param.size() - 1) + " provided)");
        }
        else if (param[0] == "box")
        {
            if (param.size() == 3)
                PARSE_TRY(addObj(Box::create(vertices.at(S2I(param[1])), vertices.at(S2I(param[2])),
                                             material,
                                             transform.back())),
                          "box", ln)
            else if (param.size() == 7)
            PARSE_TRY(addObj(Box::create(S2D(param[1]), S2D(param[2]),
                                         S2D(param[3]), S2D(param[4]),
                                         S2D(param[5]), S2D(param[6]),
                                         material,
                                         transform.back())),
                      "box", ln)
            else
                throw std::invalid_argument("[Error] Failed to parse `box` at line " +
                                            std::to_string(ln) + " (6 or 2 parameters expected, " +
                                            std::to_string(param.size() - 1) + " provided)");
        }
        else if (param[0] == "quadric")
        {
            PARAM_CHECK("quadric", 19, param, ln)
            PARSE_TRY(addObj(Quadric::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                             S2D(param[4]), S2D(param[5]),
                                             S2D(param[6]), S2D(param[7]),
                                             S2D(param[8]), S2D(param[9]),
                                             S2D(param[10]), S2D(param[11]),
                                             S2D(param[12]), S2D(param[13]),
                                             S2D(param[14]), S2D(param[15]),
                                             S2D(param[16]), S2D(param[17]),
                                             S2D(param[18]), S2D(param[19]),
                                             material,
                                             transform.back())),
                      "quadric", ln)
        }
        else if (param[0] == "ellipsoid")
        {
            PARAM_CHECK("ellipsoid", 6, param, ln)
            PARSE_TRY(addObj(Ellipsoid::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                               S2D(param[4]),
                                               S2D(param[5]),
                                               S2D(param[6]),
                                               material,
                                               transform.back())),
                      "ellipsoid", ln)
        }
        else if (param[0] == "cylinder")
        {
            PARAM_CHECK("cylinder", 6, param, ln)
            PARSE_TRY(addObj(Cylinder::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                              S2D(param[4]),
                                              S2D(param[5]),
                                              S2D(param[6]),
                                              material,
                                              transform.back())),
                      "cylinder", ln)
        }
        else if (param[0] == "cone")
        {
            PARAM_CHECK("cone", 7, param, ln)
            PARSE_TRY(addObj(Cone::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                          S2D(param[4]), S2D(param[5]),
                                          S2D(param[6]), S2D(param[7]),
                                          material,
                                          transform.back())),
                      "cone", ln)
        }
        else if (param[0] == "texture")
        {
            PARAM_CHECK("texture", 18, param, ln)
            if (param[16] == "rgb")
                PARSE_TRY(material = TextureMaterial::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                             {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                                             {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                                             S2D(param[10]),
                                                             {S2D(param[11]), S2D(param[12]), S2D(param[13])},
                                                             S2D(param[14]),
                                                             Texture::create(param[15], Texture::Format::RGB,
                                                                             S2D(param[17]), S2D(param[18]))),
                          "texture", ln)
            else if (param[16] == "rgba")
                PARSE_TRY(material = TextureMaterial::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                             {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                                             {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                                                             S2D(param[10]),
                                                             {S2D(param[11]), S2D(param[12]), S2D(param[13])},
                                                             S2D(param[14]),
                                                             Texture::create(param[15], Texture::Format::RGBA,
                                                                             S2D(param[17]), S2D(param[18]))),
                          "texture", ln)
            else
                throw std::invalid_argument("[Error] Failed to parse `texture` parameter at line " +
                                            std::to_string(ln) + " (only support RGB or RGBA format)");
        }
        else if (param[0] == "brick_material")
        {
            PARAM_CHECK("brick_material", 31, param, ln)
            PARSE_TRY(material = BrickMaterial::create(
                    {S2D(param[1]), S2D(param[2]), S2D(param[3])},
                    {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                    {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                    S2D(param[10]),
                    {S2D(param[11]), S2D(param[12]), S2D(param[13])},
                    S2D(param[14]),
                    {S2D(param[15]), S2D(param[16]), S2D(param[17])},
                    {S2D(param[18]), S2D(param[19]), S2D(param[20])},
                    {S2D(param[21]), S2D(param[22]), S2D(param[23])},
                    S2D(param[24]),
                    {S2D(param[25]), S2D(param[26]), S2D(param[27])},
                    S2D(param[28]),
                    S2D(param[29]), S2D(param[30]), S2D(param[31])),
                      "brick_material", ln)
        }
        else if (param[0] == "checkerboard_material")
        {
            PARAM_CHECK("checkerboard_material", 16, param, ln)
            PARSE_TRY(material = CheckerboardMaterial::create(
                    {S2D(param[1]), S2D(param[2]), S2D(param[3])},
                    {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                    {S2D(param[7]), S2D(param[8]), S2D(param[9])},
                    S2D(param[10]),
                    {S2D(param[11]), S2D(param[12]), S2D(param[13])},
                    S2D(param[14]),
                    S2D(param[15]), S2D(param[16])),
                      "checkerboard_material", ln);
        }
        else if (param[0] == "area_light")
        {
            PARAM_CHECK("area_light", 7, param, ln)
            PARSE_TRY(scene->addLight(AreaLight::create({S2D(param[1]), S2D(param[2]), S2D(param[3])},
                                                             {S2D(param[4]), S2D(param[5]), S2D(param[6])},
                                                             S2D(param[7]))),
                      "area_light", ln)
        }
        else if (param[0] == "area_light_sampling")
        {
            PARAM_CHECK("area_light_sampling", 1, param, ln)
            PARSE_TRY(scene->setAreaLightSampling(S2I(param[1])),
                      "area_light_sampling", ln)
        }
        else if (param[0] == "diffuse_sampling")
        {
            PARAM_CHECK("diffuse_sampling", 2, param, ln)
            PARSE_TRY(scene->setDiffuseSampling(S2I(param[1]), S2I(param[2])),
                      "diffuse_sampling", ln)
        }
        else if (param[0] == "transform")
        {
            if (param.size() > 1)
            {
                std::transform(param[1].begin(), param[1].end(), param[1].begin(), tolower); // case non-sensitive
                if (param[1] == "end")
                {
                    if (transform.size() == 1)
                        throw std::invalid_argument("[Error] Failed to parse unmatched `transform` parameter at line " +
                                                    std::to_string(ln));
                    transform.pop_back();
                }
                else
                {
                    PARAM_CHECK("transform", 9, param, ln)
                    transform.push_back(Transformation::create(S2D(param[1]),
                                                               S2D(param[2]),
                                                               S2D(param[3]),
                                                               S2D(param[4]) * DEG2RAD,
                                                               S2D(param[5]) * DEG2RAD,
                                                               S2D(param[6]) * DEG2RAD,
                                                               S2D(param[7]),
                                                               S2D(param[8]),
                                                               S2D(param[9]),
                                                               transform.back()));
                }
            }
            else
                throw std::invalid_argument("[Error] Failed to parse `transform` parameter at line " +
                                            std::to_string(ln) + " (6 parameters or `end` must be provided)");
        }
        else if (param[0] == "group")
        {
            PARAM_CHECK("group", 1, param, ln)
            std::transform(param[1].begin(), param[1].end(), param[1].begin(), tolower); // case non-sensitive
            if (param[1] == "begin")
            {
                bound_box.push_back(new BoundBox(transform.back()));
                if (bound_box.size() > 2)
                    use_bb = true;
            }
            else if (param[1] == "end")
            {
                if (bound_box.size() == 0)
                {
                    throw std::invalid_argument("[Error] Failed to parse unmatched `group` parameter at line " +
                                                std::to_string(ln));
                }
                else
                {
                    auto bb = std::shared_ptr<BaseGeometry>(bound_box.back());
                    bound_box.pop_back();
                    addObj(bb);
                }
            }
            else
                throw std::invalid_argument("[Error] Failed to parse `group` at line " +
                                            std::to_string(ln) + " (`begin` or `end` expected, `" +
                                            param[1] + "` provided)");
        }
        else if (param[0] == "mode")
        {
            PARAM_CHECK("mode", 1, param, ln)
            std::transform(param[1].begin(), param[1].end(), param[1].begin(), tolower); // case non-sensitive
            if (param[1] == "gpu")
#ifdef USE_CUDA
                scene->setComputationMode(Scene::ComputationMode::GPU);
#else
                std::cout << "[Warn] No CUDA is supported. Ignore command `mode` at line " << ln << std::endl;
#endif
            else if (param[1] == "cpu")
                scene->setComputationMode(Scene::ComputationMode::CPU);
            else
                throw std::invalid_argument("[Error] Failed to parse `mode` at line " +
                                            std::to_string(ln) + " (`cpu` or `gpu` expected, `" +
                                            param[1] + "` provided)");
        }
        else
        {
            std::cout << "[Warn] Failed to parse script with an unsupported property `" +
                                        param[0] + "` at line " + std::to_string(ln) + " (Skipped)" << std::endl;
        }
    }

    if (!bound_box.empty())
        throw std::invalid_argument("[Error] Failed to parse unmatched `group` parameter at line " +
                                    std::to_string(ln));

    if (transform.size() != 1)
        throw std::invalid_argument("[Error] Failed to parse unmatched `transform` parameter at line " +
                                    std::to_string(ln));

    if (scene->mode == Scene::ComputationMode::GPU && use_bb == true)
        std::cout << "[Warn] \033[41mBound boxes are not fully supported in GPU!!!\033[0m" << std::endl;

    return outputs;
}
