#include "scene.hpp"
#include "app.hpp"

#include "global.hpp"
#include "camera.hpp"
#include "shader/base_shader.hpp"
#include "item/light_ball.hpp"
#include "item/key.hpp"
#include "item/door.hpp"
#include "util/random.hpp"

#include "soil/SOIL.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace px;


std::vector<unsigned char *> floor_textures;
std::vector<std::pair<int, int> >  floor_texture_dim(8, {0, 0});
std::vector<unsigned char *> wall_textures;
std::vector<std::pair<int, int> >  wall_texture_dim(8, {0, 0});

const char *Scene::GEO_VS =
#include "shader/glsl/scene_geo_shader.vs"
;
const char *Scene::GEO_FS =
#include "shader/glsl/scene_geo_shader.fs"
;
const char *Scene::LIGHT_VS =
#include "shader/glsl/scene_light_shader.vs"
;
const char *Scene::LIGHT_FS =
#include "shader/glsl/scene_light_shader.fs"
;

float floor_v[] = {
        // coordinates     texture    norm            tangent
        // x    y    z     u    v     x    y    z     x    y    z
        0.f, 0.f, 1.f,  0.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        0.f, 0.f, 0.f,  0.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 0.f,  1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,

        0.f, 0.f, 1.f,  0.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 0.f,  1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 1.f,  1.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
};
float cube_v[11*6*4] = {0};

std::vector<float> wall_v;;
int n_wall_v;

Scene::Scene(Option *opt)
        : opt(opt), cam(), character(&cam, this), state(State::Over),
          geo_shader(nullptr), light_shader(nullptr), skybox(nullptr),
          texture{0}, vao{0}, vbo{0}, fbo(0), bto{0}, rbo(0),
          need_upload_data(false),
          keys{item::MetalKey::create(), item::WoodKey::create(), item::WaterKey::create(),
               item::FireKey::create(), item::EarthKey::create()},
          doors{item::MetalDoor::create(), item::WoodDoor::create(), item::WaterDoor::create(),
                item::FireDoor::create(), item::EarthDoor::create()}
{}

Scene::~Scene()
{
    glDeleteVertexArrays(3, vao);
    glDeleteBuffers(3, vbo);
    glDeleteTextures(8, texture);

    glDeleteFramebuffers(1, &fbo);

    delete geo_shader;
    delete light_shader;
    delete skybox;
}

[[noreturn]]
void Scene::err(std::string const &msg)
{
    throw WindowError("App Error: " + msg);
}

void Scene::init()
{
    if (geo_shader == nullptr)
        geo_shader = new Shader(GEO_VS, GEO_FS);
    if (light_shader == nullptr)
        light_shader = new Shader(LIGHT_VS, LIGHT_FS);

    if (vao[0] == 0)
    {
        glGenVertexArrays(3, vao);
        glGenBuffers(3, vbo);
        glGenTextures(8, texture);

        glGenFramebuffers(1, &fbo);
        glGenTextures(5, bto);
        glGenRenderbuffers(1, &rbo);
    }

    geo_shader->use();
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    ATTRIB_BIND_HELPER_WITH_TANGENT
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    ATTRIB_BIND_HELPER_WITH_TANGENT

    for (auto &i : keys)
        i->init(geo_shader);
    for (auto &i : doors)
        i->init(geo_shader);

    if (floor_textures.empty())
    {
        int ch;
#define TEXTURE_LOAD_HELPER(filename_prefix, ext, width, height, target_container)   \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_d" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_n" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_s" ext, &width, &height, &ch, SOIL_LOAD_RGB));  \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_h" ext, &width, &height, &ch, SOIL_LOAD_RGB));

#define FLOOR_TEXTURE_LOAD_HELPER(file, i)                                                                           \
        TEXTURE_LOAD_HELPER(file, ".png", floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures)
#define WALL_TEXTURE_LOAD_HELPER(file, i)                                                                           \
        TEXTURE_LOAD_HELPER(file, ".png", wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures)

        FLOOR_TEXTURE_LOAD_HELPER("floor6", 0)
        WALL_TEXTURE_LOAD_HELPER("wall1", 0)

#undef TEXTURE_LOAD_HELPER
#undef WALL_TEXTURE_LOAD_HELPER
#undef FLOOR_TEXTURE_LOAD_HELPER

    }

    glBindFragDataLocation(light_shader->pid(), 0, "color");
    glBindVertexArray(vao[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void *)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    float vertices[] = {
            // x     y       u     v
            -1.0f,  1.0f,   0.0f, 1.0f,
            -1.0f, -1.0f,   0.0f, 0.0f,
             1.0f,  1.0f,   1.0f, 1.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    light_shader->use();
    glUniform1i(glGetUniformLocation(light_shader->pid(), "gPosition"), 0);
    glUniform1i(glGetUniformLocation(light_shader->pid(), "gView"), 1);
    glUniform1i(glGetUniformLocation(light_shader->pid(), "gNormal"), 2);
    glUniform1i(glGetUniformLocation(light_shader->pid(), "gDiffuse"), 3);
    glUniform1i(glGetUniformLocation(light_shader->pid(), "gSpecular"), 4);

    if (skybox == nullptr)
        skybox = new SkyBox(ASSET_PATH "/texture/skybox/right.jpg",
                            ASSET_PATH "/texture/skybox/left.jpg",
                            ASSET_PATH "/texture/skybox/top.jpg",
                            ASSET_PATH "/texture/skybox/bottom.jpg",
                            ASSET_PATH "/texture/skybox/back.jpg",
                            ASSET_PATH "/texture/skybox/front.jpg");

}

template<typename ...ARGS>
void Scene::gen(ARGS &&...args)
{
    maze.reset(std::forward<ARGS>(args)...);
    objs.clear();

    auto threshold = 2 * (1 - 30.f / ((maze.height - 1) * (maze.width - 1) * 0.25f)) - 1;

    auto h = static_cast<float>(maze.height);
    auto w = static_cast<float>(maze.width);

    auto u = w;
    auto v = h;
    auto ws = (opt->cellSize() + opt->wallThickness()) * (w-1)/2 + opt->wallThickness();
    auto hs = (opt->cellSize() + opt->wallThickness()) * (h-1)/2 + opt->wallThickness();

    floor_v[25] =  u; floor_v[47] =  u; floor_v[58] =  u;
    floor_v[4]  =  v; floor_v[37] =  v; floor_v[59] =  v;
    floor_v[22] = ws; floor_v[44] = ws; floor_v[55] = ws;
    floor_v[2]  = hs; floor_v[35] = hs; floor_v[57] = hs;

    wall_v.clear();
    wall_v.reserve(w*h * 150);
    auto ch = opt->cellHeight();
    auto dl = opt->cellSize()+opt->wallThickness();
    for (auto i = 0; i < h; ++i)
    {
        auto y0 = i/2*dl;
        if (i%2 == 1) y0 += opt->wallThickness();
        auto y1 = y0 + (i%2 == 0 ? opt->wallThickness() : opt->cellSize());

        for (auto j = 0; j < w; ++j)
        {

            auto x0 = j/2*dl; if (j%2 == 1) x0 += opt->wallThickness();
            auto x1 = x0 + (j%2 == 0 ? opt->wallThickness() : opt->cellSize());

            auto e = maze.at(j, i);
            if (Maze::isKey(e))
            {
                auto index = -1;
                if (e == Maze::METAL_KEY) index = 0;
                else if (e == Maze::WOOD_KEY) index = 1;
                else if (e == Maze::WATER_KEY) index = 2;
                else if (e == Maze::FIRE_KEY) index = 3;
                else if (e == Maze::EARTH_KEY) index = 4;
                if (index > -1)
                {
                    auto x = 0.5f * (static_cast<float>(j + 1)*dl - opt->cellSize());
                    auto y = 0.5f * (static_cast<float>(i + 1)*dl - opt->cellSize());
                    keys[index]->place(glm::vec3(x, character.characterHeight(), y));
                }
                continue;
            }
            else if (Maze::isDoor(e))
            {
                auto index = -1;
                if (e == Maze::METAL_DOOR) index = 0;
                else if (e == Maze::WOOD_DOOR) index = 1;
                else if (e == Maze::WATER_DOOR) index = 2;
                else if (e == Maze::FIRE_DOOR) index = 3;
                else if (e == Maze::EARTH_DOOR) index = 4;
                if (index > -1)
                {
                    auto x = 0.5f * (static_cast<float>(j + 1)*dl - opt->cellSize());
                    auto y = 0.5f * (static_cast<float>(i + 1)*dl - opt->cellSize());
                    auto wid = x1-x0;
                    auto hei = y1-y0;
                    if (j == 0)
                    {
                        wid *= 0.25f;
                        x -= 2.f*wid;
                    }
                    else if (j == w-1)
                    {
                        wid *= 0.25f;
                        x += 2.f*wid;
                    }
                    else if (i == 0)
                    {
                        hei *= 0.25f;
                        y -= 2.f*hei;
                    }
                    else if (i == h-1)
                    {
                        hei *= 0.25f;
                        y += 2.f*hei;
                    }
                    doors[index]->place(glm::vec3(x, ch*0.5f, y));
                    doors[index]->setHalfSize(glm::vec3(wid*0.5f, ch*0.5f, hei*0.5f));
                }
                continue;
            }
            else if (e == Maze::END_POINT)
            {
                continue;
            }

            if (!maze.isWall(e))
            {
                if (rnd() > threshold)
                {
                    auto light = new item::LightBall(glm::vec3((x1+x0)*0.5f, opt->cellHeight()*0.5f, (y1+y0)*0.5f),
                                                       glm::vec3(0.025f),
                                                       glm::vec3((x1-x0)*0.75f, opt->cellHeight()*0.25f, (y1-y0)*0.75f),
                                                       1.f);
                    light->init(geo_shader);
                    objs.emplace_back(light);
                }
                continue;
            }

            auto count = 0;
            if (!maze.isWall(j-1, i))
            {   // render left side
                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;
            }
            if (!maze.isWall(j+1, i))
            {   // render right side
                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;
            }
            if (!maze.isWall(j, i-1))
            {   //render up/backward side
                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;
            }
            if (!maze.isWall(j, i+1))
            {   // render down/forward side
                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;
            }
            wall_v.insert(wall_v.end(), cube_v, cube_v + count);
        }
    }

    n_wall_v = wall_v.size()/11;

    auto x = (static_cast<float>(maze.player_x) + 1)*dl - opt->cellSize();
    auto y = (static_cast<float>(maze.player_y) + 1)*dl - opt->cellSize();

    std::vector<int> d;
    d.reserve(4);
    auto to_l = maze.isWall(maze.player_x-1, maze.player_y) ? false : (d.push_back(1), true);
    auto to_r = maze.isWall(maze.player_x+1, maze.player_y) ? false : (d.push_back(2), true);
    auto to_u = maze.isWall(maze.player_x, maze.player_y-1) ? false : (d.push_back(3), true);
    auto to_d = maze.isWall(maze.player_x, maze.player_y+1) ? false : (d.push_back(4), true);
    if (to_l && to_u) d.push_back(5);
    if (to_l && to_d) d.push_back(6);
    if (to_r && to_u) d.push_back(7);
    if (to_r && to_d) d.push_back(8);

    std::random_device rd;
    auto i =  rd() % static_cast<int>(d.size());
    if (d[i] == 1)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -180.f);
    else if (d[i] == 2)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 0.f);
    else if (d[i] == 3)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -90.f);
    else if (d[i] == 4)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 90.f);
    else if (d[i] == 5)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -135.0f);
    else if (d[i] == 6)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 135.0f);
    else if (d[i] == 7)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -45.0f);
    else //if (d[i] == 8)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 45.f);

    character.clearBag();
    character.setCharacterHp(500.f);

    for (auto & k : keys)
        objs.push_back(k);
    for (auto & d : doors)
        objs.push_back(d);

    need_upload_data = true;

    state = State::Running;

#ifndef NDEBUG
    std::cout << "\n" << maze.map << std::endl;
#endif
}

template void Scene::gen(Map const &);
template void Scene::gen(Maze const &);
template void Scene::gen(std::size_t const &, std::size_t const &);
template void Scene::gen(int const &, int const &);

void Scene::upload()
{
    if (!need_upload_data)
        return;

    auto i = static_cast<int>(std::floor((rnd() + 1) * 0.5f * (floor_textures.size()/4)));
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_v), floor_v, GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[0], 0, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4],   geo_shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[1], 1, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+1], geo_shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[2], 2, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+2], geo_shader->pid(), "material.specular")
    TEXTURE_BIND_HELPER(texture[3], 3, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+3], geo_shader->pid(), "material.displace")

    i = static_cast<int>(std::floor((rnd() + 1) * 0.5f * (wall_textures.size()/4)));
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*wall_v.size(), wall_v.data(), GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[4], 0, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4],   geo_shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[5], 1, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+1], geo_shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[6], 2, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+2], geo_shader->pid(), "material.specular");
    TEXTURE_BIND_HELPER(texture[7], 3, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+3], geo_shader->pid(), "material.displace")

    geo_shader->use();
    geo_shader->set("headlight.ambient", glm::vec3(.6f, .5f, .3f));
    geo_shader->set("headlight.diffuse", glm::vec3(.5f, .4f, .25f));
    geo_shader->set("headlight.specular", glm::vec3(1.f, 1.f, 1.f));
    geo_shader->set("headlight.coef_a0", 1.f);
    geo_shader->set("headlight.coef_a1", .09f);
    geo_shader->set("headlight.coef_a2", .032f);
    geo_shader->set("global_ambient", glm::vec3(.1f, .1f, .1f));

    glClearColor(.2f, .3f, .3f, 1.f);

    need_upload_data = false;
}

void Scene::turnOffHeadLight()
{
    geo_shader->use();
    geo_shader->set("headlight.cutoff_outer", std::numeric_limits<float>::max());
    geo_shader->set("headlight.cutoff_diff", std::numeric_limits<float>::max());
}

void Scene::turnOnHeadLight()
{
    geo_shader->use();
    geo_shader->set("headlight.cutoff_outer", 0.9537f);
    geo_shader->set("headlight.cutoff_diff", 0.0226f);
}

void Scene::render()
{
//    glEnable(GL_CULL_FACE);
//    glCullFace(GL_BACK);
//    glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDepthFunc(GL_LESS);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    geo_shader->use();
    upload();

    if (character.headLight())
        turnOnHeadLight();
    else
        turnOffHeadLight();

    geo_shader->set("view", cam.viewMat());
    geo_shader->set("proj", cam.projMat());
    geo_shader->set("model", Camera::IDENTITY_MAT4);
    geo_shader->set("cam_pos", cam.eye);
    auto x = cam.eye;
    x.y += character.characterHeight()*0.2f;
    geo_shader->set("headlight.pos", x);
    geo_shader->set("headlight.dir", cam.camDir());

    geo_shader->set("use_tangent", 1);
    geo_shader->set("material.parallel_height", 0.f);
    geo_shader->set("material.shininess", 32.f);
    geo_shader->set("material.ambient", glm::vec3(1.f, 1.f, 1.f));
    geo_shader->set("material.displace_amp", 0.f);
    geo_shader->set("material.displace_mid", 0.5f);

    // render normal objects, would be influenced by lighting
    glBindVertexArray(vao[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[3]);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(vao[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[4]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[5]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[6]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[7]);
    glDrawArrays(GL_TRIANGLES, 0, n_wall_v);
    for (auto & o : objs)
    {
        if (o->preRender())
            o->render(geo_shader, cam.viewMat(), cam.projMat());
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // render lighting
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    light_shader->use();
    glBindVertexArray(vao[2]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, bto[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, bto[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, bto[2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, bto[3]);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, bto[4]);
    auto idx = 0;
    light_shader->set("tot_light", 100);
    for (auto & o : objs)
    {
        if (o->lighting())
        {
            auto & light = o->light();
            light_shader->set("light[" + std::to_string(idx) + "].pos",      o->pos());
            light_shader->set("light[" + std::to_string(idx) + "].ambient",  light.ambient);
            light_shader->set("light[" + std::to_string(idx) + "].diffuse",  light.diffuse);
            light_shader->set("light[" + std::to_string(idx) + "].specular", light.specular);
            light_shader->set("light[" + std::to_string(idx) + "].coef_a0",  light.coef.x);
            light_shader->set("light[" + std::to_string(idx) + "].coef_a1",  light.coef.y);
            light_shader->set("light[" + std::to_string(idx) + "].coef_a2",  light.coef.z);
            ++idx;
            if (idx == 100)
            {
                std::cout << "[Warn] More than 100 light sources are found. Ignore." << std::endl;
                break;
            }
        }
    }
    light_shader->set("tot_light", idx);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


    // render objects that would not be influenced by lighting
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, cam.width, cam.height, 0, 0, cam.width, cam.height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    for (auto & o : objs)
    {
        if (o->postRender())
            o->render(cam.viewMat(), cam.projMat());
    }
    skybox->render(cam.viewMat(), cam.projMat());
}

void Scene::resize()
{
    if (fbo != 0)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glBindTexture(GL_TEXTURE_2D, bto[0]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, cam.width, cam.height, 0, GL_RGB, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bto[0], 0);

        for (auto i = 1; i < 5; ++i)
        {
            glBindTexture(GL_TEXTURE_2D, bto[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, cam.width, cam.height, 0, GL_RGBA, GL_FLOAT, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, bto[i], 0);
        }

        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, cam.width, cam.height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

        GLuint attach[5] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                            GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
        glDrawBuffers(5, attach);

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            err("Failed to generate frame buffer");
    }
}
bool Scene::run(float dt)
{
    if (state == State::Win)
    {
        return winScreen(dt);
    }

    auto field_height = 0.f;

    auto movement = moveWithCollisionCheck(cam.eye,
                                              glm::vec3(character.characterHalfSize(), character.characterHalfHeight(), character.characterHalfSize()),
                                              character.makeAction(dt), nullptr);


    if (!character.isAscending())
    {
        if (cam.eye.y - character.characterHeight() > field_height)
            movement.y -= character.dropSpeed() * dt;

        auto lowest = field_height + character.characterHeight();
        if (cam.eye.y + movement.y <= lowest)
        {
            movement.y = lowest - cam.eye.y;
            character.disableDropping();
        }
        else
            character.enableDropping();
    }
    cam.eye += movement;

    for (auto & o : objs)
    {
        o->update(dt);
        if (o->canMove())
        {
            auto &movement = o->moveSpan();
            auto pos = o->pos();
            auto size = o->halfSize();

            if (o->isRigidBody())
                movement = moveWithCollisionCheck(pos, size, movement, o.get());

            if (o->mass() != 0)
            {
                auto lowest = field_height + size.y;
                movement.y -= o->mass() * dt;
                if (pos.y + movement.y <= lowest)
                {
                    movement.y = lowest - pos.y;
                    o->hit(pos);
                }
            }
            o->makeMove();
        }
        else if (o->mass() != 0)
        {
            auto pos = o->pos();
            auto size = o->halfSize();
            auto lowest = field_height + size.y;
            auto &movement = o->moveSpan();
            movement.x = 0;
            movement.z = 0;
            movement.y = - o->mass() * dt;
            if (pos.y + movement.y < lowest)
            {
                movement.y = lowest - pos.y;
                o->hit(pos);
            }
            o->makeMove();
        }
    }

    cam.updateView();

    if (maze.canWin(maze.player_x, maze.player_y))
    {
        state = State::Win;
        return true;
    }
    else
    {
        character.setCharacterHp(character.characterHp() - dt * opt->damageAmp());
        if (character.characterHp() < 0)
        {
            state = State::Lose;
            return false;
        }
    }

    std::cout << "\rLocation: ("
              << cam.eye.x << ", "
              << cam.eye.y << ", "
              << cam.eye.z << "); Look at: ("
              << cam.camDir().x << ", "
              << cam.camDir().y << ", "
              << cam.camDir().z << ")"
              << std::flush;
    return true;
}

glm::vec3 Scene::moveWithCollisionCheck(glm::vec3 const &pos, glm::vec3 const &half_size,
                                        glm::vec3 span, Item *player)
{
    auto dl = opt->cellSize()+opt->wallThickness();

    // cell coordinate
    int x, y;
    if (player == nullptr)
    {
        x = maze.player_x;
        y = maze.player_y;
    }
    else
    {
        auto tmp = static_cast<int>(pos.x / dl);
        x = tmp*2 + ((pos.x - tmp * dl > opt->wallThickness()) ? 1 : 2);
        tmp = static_cast<int>(pos.z / dl);
        y = tmp*2 + ((pos.z - tmp * dl > opt->wallThickness()) ? 1 : 2);
    }

    // character location
    auto new_pos_x = pos.x;
    auto new_pos_z = pos.z;
    auto pos_x = new_pos_x;
    auto pos_z = new_pos_z;

    while (span.x != 0.f || span.z != 0.f)
    {
        
        // bound of current cell
        auto x0 = x/2*dl; if (x%2 == 1) x0 += opt->wallThickness();
        auto x1 = x0 + (x%2 == 0 ? opt->wallThickness() : opt->cellSize());
        auto z0 = y/2*dl; if (y%2 == 1) z0 += opt->wallThickness();
        auto z1 = z0 + (y%2 == 0 ? opt->wallThickness() : opt->cellSize());


        // if unable to move the adj. cell, then move at most to the edge of the adj. cell
        if (span.x > 0.f && maze.isWall(x+1, y))
        {
            new_pos_x = std::min(x1 - character.characterHalfSize(),
                                 new_pos_x + span.x);
            span.x = 0.f;
        }
        if (span.x < 0.f && maze.isWall(x-1, y))
        {
            new_pos_x = std::max(x0 + character.characterHalfSize(),
                                 new_pos_x + span.x);
            span.x = 0.f;
        }
        if (span.z > 0.f && maze.isWall(x, y+1))
        {
            new_pos_z = std::min(z1 - character.characterHalfSize(),
                                 new_pos_z + span.z);
            span.z = 0.f;
        }
        if (span.z < 0.f && maze.isWall(x, y-1))
        {
            new_pos_z = std::max(z0 + character.characterHalfSize(),
                                 new_pos_z + span.z);
            span.z = 0.f;
        }

        auto will_move_left = false, will_move_right = false;
        auto will_move_up   = false, will_move_down  = false;
        if (span.x > 0.f)
        {   // can and is moving toward the right cell
            if (new_pos_x + span.x < x1)
            {   // if just move inside the current cell
                new_pos_x += span.x; span.x = 0.f;
            }
            else
            {   // move to the edge of the adj. cell
                will_move_right = true;
                span.x += new_pos_x - x1;
                new_pos_x = x1;
            }

        }
        else if (span.x < 0.f)
        {
            if (new_pos_x + span.x > x0)
            {
                new_pos_x += span.x; span.x = 0.f;
            }
            else
            {
                will_move_left = true;
                span.x += new_pos_x - x0;
                new_pos_x = x0;
            }
        }
        if (span.z > 0.f)
        {
            if (new_pos_z + span.z < z1)
            {
                new_pos_z += span.z; span.z = 0.f;
            }
            else
            {
                will_move_down = true;
                span.z += new_pos_z - z1;
                new_pos_z = z1;
            }

        }
        else if (span.z < 0.f)
        {
            if (new_pos_z + span.z > z0)
            {
                new_pos_z += span.z;
                span.z = 0.f;
            }
            else
            {
                will_move_up = true;
                span.z += new_pos_z - z0;
                new_pos_z = z0;
            }
        }

        bool move_false = false;
        for (auto it = objs.begin(); it != objs.end();)
        {
            auto item_pos = (*it)->pos();
            auto item_size = (*it)->halfSize();

            if (std::abs(item_pos.x - new_pos_x) > std::abs(item_size.x + half_size.x) ||
                std::abs(item_pos.y - pos.y)     > std::abs(item_size.y + half_size.y) ||
                std::abs(item_pos.z - new_pos_z) > std::abs(item_size.z + half_size.z))
            {
                ++it;
                continue;
            }

            // TODO collision would fail if the character moves toooo far during one trick
            //      but normally, this should not happen

            if (player == nullptr && (*it)->attribute.collectible)
            {
                auto id = (*it)->attribute.id();
                character.collectItem(id, 1);
                if (id == item::MetalKey::itemInfo().id())
                {
                    maze.collect(Maze::METAL_KEY);
                    static_cast<item::Door*>(doors[0].get())->enlight();
                }
                else if (id == item::WoodKey::itemInfo().id())
                {
                    maze.collect(Maze::WOOD_KEY);
                    static_cast<item::Door*>(doors[1].get())->enlight();
                }
                else if (id == item::WaterKey::itemInfo().id())
                {
                    maze.collect(Maze::WATER_KEY);
                    static_cast<item::Door*>(doors[2].get())->enlight();
                }
                else if (id == item::FireKey::itemInfo().id())
                {
                    maze.collect(Maze::FIRE_KEY);
                    static_cast<item::Door*>(doors[3].get())->enlight();
                }
                else if (id == item::EarthKey::itemInfo().id())
                {
                    maze.collect(Maze::EARTH_KEY);
                    static_cast<item::Door*>(doors[4].get())->enlight();
                }

                it = objs.erase(it);
                continue;
            }
            else if (it->get() != player && item_size.y >= half_size.y * 0.2f)
                move_false = true;

            ++it;
        }
        if (move_false)
        {
            break;
        }

        pos_x = new_pos_x;
        pos_z = new_pos_z;

        // update character cell
        if (will_move_left)
            --x;
        else if (will_move_right)
            ++x;

        if (will_move_up)
            --y;
        else if (will_move_down)
            ++y;
    }

    if (player == nullptr)
    {
        if (x != maze.player_x || y != maze.player_y)
        {
            maze.portal(x, y);
            std::cout << "\n" << maze.map << std::endl;
        }
    }

    return {pos_x - pos.x, span.y, pos_z - pos.z};
}

bool Scene::winScreen(float dt)
{
    geo_shader->use();
    geo_shader->set("global_ambient", glm::vec3(1.f, 1.f, 1.f));
    turnOffHeadLight();

    auto e = maze.at(maze.player_x, maze.player_y);
    auto index = -1;
    if (e == Maze::METAL_DOOR) index = 0;
    else if (e == Maze::WOOD_DOOR) index = 1;
    else if (e == Maze::WATER_DOOR) index = 2;
    else if (e == Maze::FIRE_DOOR) index = 3;
    else if (e == Maze::EARTH_DOOR) index = 4;

    if (index == -1)
        return false;

    auto dl = opt->cellSize()+opt->wallThickness();

    cam.setFov(60.f);

    auto px = maze.player_x;
    auto py = maze.player_y;
    if (maze.player_x == 0)
    {
        px = 1;
        cam.setAng(-90, -30);
    }
    else if (maze.player_x == maze.width-1)
    {
        px = maze.width - 2;
        cam.setAng(90, -30);
    }
    else if (maze.player_y == 0)
    {
        py = 1;
        cam.setAng(0, -30);
    }
    else if (maze.player_y == maze.height-1)
    {
        py = maze.height - 2;
        cam.setAng(180, -30);
    }

    cam.eye.x =
            0.5*((static_cast<float>(px) + 1)*dl - opt->cellSize());
    cam.eye.z =
            0.5*((static_cast<float>(py) + 1)*dl - opt->cellSize());

    cam.freeze(true);

    auto pos = doors[index]->pos();
    pos.y -= opt->cellHeight()*0.005f;

    if (pos.y + opt->cellHeight()*0.5f < 0)
    {
        cam.freeze(false);
        return false;
    }
    cam.updateProj();
    cam.updateView();

    doors[index]->place(pos);
    doors[index]->update(dt);

    return true;
}