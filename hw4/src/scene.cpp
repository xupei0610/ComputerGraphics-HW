#include "scene.hpp"
#include "global.hpp"
#include "camera.hpp"
#include "shader/base_shader.hpp"
#include "maze.hpp"
#include "item/key.hpp"
#include "item/door.hpp"
#include "util/random.hpp"

#include "soil/SOIL.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef NDEBUGE
#    include <iostream>
#endif

using namespace px;

const float Scene::DEFAULT_DISPLACE_AMP = 0.f;
const float Scene::DEFAULT_DISPLACE_MID = .5f;

std::vector<unsigned char *> Scene::wall_textures;
std::vector<std::pair<int, int> >  Scene::wall_texture_dim(5, {0, 0});
std::vector<unsigned char *> Scene::floor_textures;
std::vector<std::pair<int, int> >  Scene::floor_texture_dim(8, {0, 0});

const char *Scene::VS =
#include "shader/glsl/scene_shader.vs"
;
const char *Scene::FS =
#include "shader/glsl/scene_shader.fs"
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

Scene::Scene(Option &opt)
        : opt(opt), state(State::Over),
          keys{item::MetalKey::create(), item::WoodKey::create(), item::WaterKey::create(),
                item::FireKey::create(), item::EarthKey::create()},
          doors{item::MetalDoor::create(), item::WoodDoor::create(), item::WaterDoor::create(),
                item::FireDoor::create(), item::EarthDoor::create()},
          shader(nullptr), skybox(nullptr),
          texture{0}, vao{0}, vbo{0}, need_upload_data(false)
{}

Scene::~Scene()
{
    glDeleteVertexArrays(2, vao);
    glDeleteBuffers(2, vbo);
    glDeleteTextures(8, texture);
    delete shader;
    delete skybox;
}

void Scene::setState(State s)
{
    state = s;
}

void Scene::init()
{
    if (shader == nullptr)
        shader = new Shader(VS, FS);

    if (vao[0] == 0)
    {
        glGenVertexArrays(3, vao);
        glGenBuffers(3, vbo);
        glGenTextures(8, texture);
    }

    shader->use();
    glBindFragDataLocation(shader->pid(), 0, "color");
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    ATTRIB_BIND_HELPER_WITH_TANGENT

    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    ATTRIB_BIND_HELPER_WITH_TANGENT

    for (auto &i : keys)
        i->init(shader);
    for (auto &i : doors)
        i->init(shader);

    if (wall_textures.empty())
    {
        int ch;
#define TEXTURE_LOAD_HELPER(filename_prefix, ext, width, height, target_container)   \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_d" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_n" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_s" ext, &width, &height, &ch, SOIL_LOAD_RGB));  \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_h" ext, &width, &height, &ch, SOIL_LOAD_RGB));

#define WALL_TEXTURE_LOAD_HELPER(file, i)                                                                           \
        TEXTURE_LOAD_HELPER(file, ".png", wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures)
#define FLOOR_TEXTURE_LOAD_HELPER(file, i)                                                                           \
        TEXTURE_LOAD_HELPER(file, ".png", floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures)

        WALL_TEXTURE_LOAD_HELPER("wall1", 0)
        WALL_TEXTURE_LOAD_HELPER("wall2", 1)
        WALL_TEXTURE_LOAD_HELPER("wall4", 2)
        WALL_TEXTURE_LOAD_HELPER("wall5", 3)
        WALL_TEXTURE_LOAD_HELPER("wall6", 4)

        FLOOR_TEXTURE_LOAD_HELPER("floor2", 0)
        FLOOR_TEXTURE_LOAD_HELPER("floor3", 1)
        FLOOR_TEXTURE_LOAD_HELPER("floor4", 2)
        FLOOR_TEXTURE_LOAD_HELPER("floor5", 3)
        FLOOR_TEXTURE_LOAD_HELPER("floor6", 4)
        FLOOR_TEXTURE_LOAD_HELPER("floor7", 5)
        FLOOR_TEXTURE_LOAD_HELPER("floor8", 6)

#undef TEXTURE_LOAD_HELPER
#undef WALL_TEXTURE_LOAD_HELPER
#undef FLOOR_TEXTURE_LOAD_HELPER

    }


    if (skybox == nullptr)
        skybox = new SkyBox(ASSET_PATH "/texture/skybox/right.jpg",
                            ASSET_PATH "/texture/skybox/left.jpg",
                            ASSET_PATH "/texture/skybox/top.jpg",
                            ASSET_PATH "/texture/skybox/bottom.jpg",
                            ASSET_PATH "/texture/skybox/back.jpg",
                            ASSET_PATH "/texture/skybox/front.jpg");

}

template<typename ...ARGS>
void Scene::reset(ARGS &&...args)
{
    maze.reset(std::forward<ARGS>(args)...);

    auto h = static_cast<float>(maze.height);
    auto w = static_cast<float>(maze.width);

    auto u = w;
    auto v = h;
    auto ws = (opt.cellSize() + opt.wallThickness()) * (w-1)/2 + opt.wallThickness();
    auto hs = (opt.cellSize() + opt.wallThickness()) * (h-1)/2 + opt.wallThickness();

    floor_v[25] =  u; floor_v[47] =  u; floor_v[58] =  u;
    floor_v[4]  =  v; floor_v[37] =  v; floor_v[59] =  v;
    floor_v[22] = ws; floor_v[44] = ws; floor_v[55] = ws;
    floor_v[2]  = hs; floor_v[35] = hs; floor_v[57] = hs;

    wall_v.clear();
    wall_v.reserve(w*h * 150);
    auto ch = opt.cellHeight();
    auto dl = opt.cellSize()+opt.wallThickness();
    for (auto i = 0; i < h; ++i)
    {
        auto y0 = i/2*dl;
        if (i%2 == 1) y0 += opt.wallThickness();
        auto y1 = y0 + (i%2 == 0 ? opt.wallThickness() : opt.cellSize());

        for (auto j = 0; j < w; ++j)
        {

            auto x0 = j/2*dl; if (j%2 == 1) x0 += opt.wallThickness();
            auto x1 = x0 + (j%2 == 0 ? opt.wallThickness() : opt.cellSize());


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
                    auto x = 0.5f * (static_cast<float>(j + 1)*dl - opt.cellSize());
                    auto y = 0.5f * (static_cast<float>(i + 1)*dl - opt.cellSize());
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
                    auto x = 0.5f * (static_cast<float>(j + 1)*dl - opt.cellSize());
                    auto y = 0.5f * (static_cast<float>(i + 1)*dl - opt.cellSize());
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
                continue;

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

    auto x = (static_cast<float>(maze.player_x) + 1)*dl - opt.cellSize();
    auto y = (static_cast<float>(maze.player_y) + 1)*dl - opt.cellSize();

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
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 45.0f);

    character.clearBag();
    character.setCharacterHp(500.f);

    need_upload_data = true;

    std::list<Item*> tmp;
    for (auto & k : keys)
        tmp.push_front(k);
    for (auto & d : doors)
        tmp.push_back(d);
    interact_objs.swap(tmp);

    state = State::Running;
#ifndef NDEBUG
    std::cout << "\n" << maze.map << std::endl;
#endif
}

void Scene::uploadData()
{
    if (!need_upload_data)
        return;

    auto i = static_cast<int>(std::floor((rnd() + 1) * 0.5f * (floor_textures.size()/4)));
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_v), floor_v, GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[0], 0, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4],   shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[1], 1, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+1], shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[2], 2, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+2], shader->pid(), "material.specular")
    TEXTURE_BIND_HELPER(texture[6], 3, GL_RGB,
                        GL_REPEAT, GL_LINEAR, floor_texture_dim[i].first, floor_texture_dim[i].second, floor_textures[i*4+3], shader->pid(), "material.displace")

    i = static_cast<int>(std::floor((rnd() + 1) * 0.5f * (wall_textures.size()/4)));
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*wall_v.size(), wall_v.data(), GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[3], 0, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4],   shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[4], 1, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+1], shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[5], 2, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+2], shader->pid(), "material.specular");
    TEXTURE_BIND_HELPER(texture[7], 3, GL_RGB,
                        GL_REPEAT, GL_LINEAR, wall_texture_dim[i].first, wall_texture_dim[i].second, wall_textures[i*4+3], shader->pid(), "material.displace")

    shader->use();
    shader->set("headlight.ambient", glm::vec3(.6f, .5f, .3f));
    shader->set("headlight.diffuse", glm::vec3(.5f, .4f, .25f));
    shader->set("headlight.specular", glm::vec3(1.f, 1.f, 1.f));
    shader->set("headlight.coef_a0", 1.f);
    shader->set("headlight.coef_a1", .09f);
    shader->set("headlight.coef_a2", .032f);
    shader->set("global_ambient", glm::vec3(.1f, .1f, .1f));

    float x, y, z;
    keys[0]->position(x, y, z);
    shader->set("pointlight[0].pos", glm::vec3(x, y, z));
    shader->set("pointlight[0].ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[0].diffuse", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[0].specular", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[0].coef_a0", 0.f);
    shader->set("pointlight[0].coef_a0", 0.f);
    shader->set("pointlight[0].coef_a1", 0.f);
    shader->set("pointlight[0].coef_a2", 2.f);

    keys[1]->position(x, y, z);
    shader->set("pointlight[1].pos", glm::vec3(x, y, z));
    shader->set("pointlight[1].ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[1].coef_a0", 0.f);
    shader->set("pointlight[1].coef_a1", 0.f);
    shader->set("pointlight[1].coef_a2", 2.f);

    keys[2]->position(x, y, z);
    shader->set("pointlight[2].pos", glm::vec3(x, y, z));
    shader->set("pointlight[2].ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[2].coef_a0", 0.f);
    shader->set("pointlight[2].coef_a1", 0.f);
    shader->set("pointlight[2].coef_a2", 2.f);

    keys[3]->position(x, y, z);
    shader->set("pointlight[3].pos", glm::vec3(x, y, z));
    shader->set("pointlight[3].ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[3].coef_a0", 0.f);
    shader->set("pointlight[3].coef_a1", 0.f);
    shader->set("pointlight[3].coef_a2", 2.f);

    keys[4]->position(x, y, z);
    shader->set("pointlight[4].pos", glm::vec3(x, y, z));
    shader->set("pointlight[4].ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("pointlight[4].coef_a0", 0.f);
    shader->set("pointlight[4].coef_a1", 0.f);
    shader->set("pointlight[4].coef_a2", 2.f);

    need_upload_data = false;
}

void Scene::turnOffHeadLight()
{
    shader->use();
    shader->set("headlight.cutoff_outer", std::numeric_limits<float>::max());
    shader->set("headlight.cutoff_diff", std::numeric_limits<float>::max());
}

void Scene::turnOnHeadLight()
{
    shader->use();
    shader->set("headlight.cutoff_outer", 0.9537f);
    shader->set("headlight.cutoff_diff", 0.0226f);
}
template void Scene::reset(Map const &);
template void Scene::reset(Maze const &);
template void Scene::reset(std::size_t const &, std::size_t const &);
template void Scene::reset(int const &, int const &);

void Scene::render()
{
//    glEnable(GL_CULL_FACE);
//    glCullFace(GL_BACK);
//    glFrontFace(GL_CCW);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClearColor(.2f, .3f, .3f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    uploadData();

    shader->use();

    if (character.headLight())
        turnOnHeadLight();
    else
        turnOffHeadLight();

    shader->set("view", character.cam.viewMat());
    shader->set("proj", character.cam.projMat());
    shader->set("model", Item::IDENTITY_MODEL_MAT);
    shader->set("cam_pos", character.cam.cam_pos);
    auto x = character.cam.cam_pos;
    x.y += character.characterHeight()*0.2f;
    shader->set("headlight.pos", x);
    shader->set("headlight.dir", character.cam.cam_dir);

    shader->set("use_tangent", 1);
    shader->set("material.parallel_height", 0.0025f);
    shader->set("material.shininess", 32.f);
    shader->set("material.ambient", glm::vec3(1.f, 1.f, 1.f));
    shader->set("material.displace_amp", DEFAULT_DISPLACE_AMP);
    shader->set("material.displace_mid", DEFAULT_DISPLACE_MID);

    // render floor
    glBindVertexArray(vao[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[6]);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // render walls
    glBindVertexArray(vao[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[3]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[4]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[5]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[7]);
    glDrawArrays(GL_TRIANGLES, 0, n_wall_v);

    shader->set("material.parallel_height", 0.f);

//     render objs
    for (auto &k : interact_objs)
        k->render(shader, character.cam.viewMat(), character.cam.projMat());

    skybox->render(character.cam.viewMat(), character.cam.projMat());
}

bool Scene::run(float dt)
{


#ifndef NDEBUG
    bool moved = false;
#endif
    if (state == State::Win)
    {
        shader->use();
        shader->set("global_ambient", glm::vec3(1.f, 1.f, 1.f));
        shader->set("pointlight[0].coef_a2", 0.f);
        shader->set("pointlight[1].coef_a2", 0.f);
        shader->set("pointlight[2].coef_a2", 0.f);
        shader->set("pointlight[3].coef_a2", 0.f);
        shader->set("pointlight[4].coef_a2", 0.f);
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

        auto dl = opt.cellSize()+opt.wallThickness();

        character.cam.setFov(45.f);

        auto px = maze.player_x;
        auto py = maze.player_y;
        if (maze.player_x == 0)
        {
            px = 1;
            character.cam.setAng(-180, 20);
        }
        else if (maze.player_x == maze.width-1)
        {
            px = maze.width - 2;
            character.cam.setAng(0, 20);
        }
        else if (maze.player_y == 0)
        {
            py = 1;
            character.cam.setAng(-90, 20);
        }
        else if (maze.player_y == maze.height-1)
        {
            py = maze.height - 2;
            character.cam.setAng(90, 20);
        }

        character.cam.cam_pos.x =
                0.5*((static_cast<float>(px) + 1)*dl - opt.cellSize());
        character.cam.cam_pos.z =
                0.5*((static_cast<float>(py) + 1)*dl - opt.cellSize());

        character.cam.freeze(true);

        float _, y, __;
        doors[index]->position(_, y, __);

        if (y + opt.cellHeight()*0.5 < 0)
        {
            character.cam.freeze(false);
            return false;
        }
        character.cam.updateViewMat();

        doors[index]->place(glm::vec3(_, y + -opt.cellHeight()*0.005f, __));
    }
    else
    {
        character.setCharacterHp(
                character.characterHp() - dt * opt.damageAmp());
#ifndef NDEBUG
        moved =
#endif
                moveWithCollisionCheck(character, character.makeAction(dt));

        character.cam.updateViewMat();
        if (character.characterHp() < 0)
        {
            state = State::Lose;
            return false;
        }
    }
    for (auto &k : interact_objs)
        k->update(dt);

#ifndef NDEBUG
    if (moved)
        std::cout << "\n" << maze.map << std::endl;
    std::cout << "\rLocation: ("
              << character.cam.cam_pos.x << ", "
              << character.cam.cam_pos.y << ", "
              << character.cam.cam_pos.z << "); Look at: ("
              << character.cam.cam_dir.x << ", "
              << character.cam.cam_dir.y << ", "
              << character.cam.cam_dir.z << ")"
              << std::flush;
#endif

    return true;
}

bool Scene::moveWithCollisionCheck(Character &character, glm::vec3 span)
{
    auto need_reload_map = false;
    auto dl = opt.cellSize()+opt.wallThickness();
    while (span.x != 0.f || span.z != 0.f)
    {
        if (maze.isEndPoint(maze.player_x, maze.player_y))
        {
            state = State::Win;
            break;
        }

        // cell coordinate
        auto x = maze.player_x;
        auto y = maze.player_y;
        // bound of current cell
        auto x0 = x/2*dl; if (x%2 == 1) x0 += opt.wallThickness();
        auto x1 = x0 + (x%2 == 0 ? opt.wallThickness() : opt.cellSize());
        auto z0 = y/2*dl; if (y%2 == 1) z0 += opt.wallThickness();
        auto z1 = z0 + (y%2 == 0 ? opt.wallThickness() : opt.cellSize());

        // character location
        auto new_pos_x = character.cam.cam_pos.x;
        auto new_pos_z = character.cam.cam_pos.z;

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
        float item_x, item_y, item_z, size_x, size_y, size_z;
        auto player_y = character.cam.cam_pos.y - character.characterHalfHeight();
        for (auto it = interact_objs.begin(); it != interact_objs.end();)
        {
            (*it)->position(item_x, item_y, item_z);
            (*it)->halfSize(size_x, size_y, size_z);

            if (std::abs(item_x - new_pos_x) > std::abs(size_x + character.characterHalfSize()) ||
                std::abs(item_y - player_y) > std::abs(size_y + character.characterHalfHeight()) ||
                std::abs(item_z - new_pos_z) > std::abs(size_z + character.characterHalfSize()))
            {
                ++it;
                continue;
            }

            // TODO collision would fail if the character moves toooo far during one trick
            //      but normally, this should not happen

            if ((*it)->attribute.collectible)
            {
                auto id = (*it)->attribute.id();
                character.collectItem(id, 1);
                if (id == item::MetalKey::itemInfo().id())
                {
                    maze.collect(Maze::METAL_KEY);
                    int x, y;
                    float dx, dy, dz;
                    doors[0]->position(dx, dy, dz);
                    maze.getLoc(Maze::METAL_DOOR, x, y);
                    if (x == 0) dx += opt.wallThickness();
                    else if (x == maze.width-1) dx -= opt.wallThickness();
                    else if (y == 0) dz += opt.wallThickness();
                    else dz -= opt.wallThickness();
                    shader->use();
                    shader->set("pointlight[0].pos", glm::vec3(dx, dy, dz));
                    shader->set("pointlight[0].coef_a2", 0.075f);
                    need_reload_map = true;
                }
                else if (id == item::WoodKey::itemInfo().id())
                {
                    maze.collect(Maze::WOOD_KEY);
                    int x, y;
                    float dx, dy, dz;
                    doors[1]->position(dx, dy, dz);
                    maze.getLoc(Maze::WOOD_DOOR, x, y);
                    if (x == 0) dx += opt.wallThickness();
                    else if (x == maze.width) dx -= opt.wallThickness();
                    else if (y == 0) dz += opt.wallThickness();
                    else dz -= opt.wallThickness();
                    shader->use();
                    shader->set("pointlight[1].pos", glm::vec3(dx, dy, dz));
                    shader->set("pointlight[1].coef_a2", 0.075f);
                    need_reload_map = true;
                }
                else if (id == item::WaterKey::itemInfo().id())
                {
                    maze.collect(Maze::WATER_KEY);
                    int x, y;
                    float dx, dy, dz;
                    doors[2]->position(dx, dy, dz);
                    maze.getLoc(Maze::WATER_DOOR, x, y);
                    if (x == 0) dx += opt.wallThickness();
                    else if (x == maze.width-1) dx -= opt.wallThickness();
                    else if (y == 0) dz += opt.wallThickness();
                    else dz -= opt.wallThickness();
                    shader->use();
                    shader->set("pointlight[2].pos", glm::vec3(dx, dy, dz));
                    shader->set("pointlight[2].coef_a2", 0.075f);
                    need_reload_map = true;
                }
                else if (id == item::FireKey::itemInfo().id())
                {
                    maze.collect(Maze::FIRE_KEY);
                    int x, y;
                    float dx, dy, dz;
                    doors[3]->position(dx, dy, dz);
                    maze.getLoc(Maze::FIRE_DOOR, x, y);
                    if (x == 0) dx += opt.wallThickness();
                    else if (x == maze.width-1) dx -= opt.wallThickness();
                    else if (y == 0) dz += opt.wallThickness();
                    else dz -= opt.wallThickness();
                    shader->use();
                    shader->set("pointlight[3].pos", glm::vec3(dx, dy, dz));
                    shader->set("pointlight[3].coef_a2", 0.075f);
                    need_reload_map = true;
                }
                else if (id == item::EarthKey::itemInfo().id())
                {
                    maze.collect(Maze::EARTH_KEY);
                    int x, y;
                    float dx, dy, dz;
                    doors[4]->position(dx, dy, dz);
                    maze.getLoc(Maze::EARTH_DOOR, x, y);
                    if (x == 0) dx += opt.wallThickness();
                    else if (x == maze.width-1) dx -= opt.wallThickness();
                    else if (y == 0) dz += opt.wallThickness();
                    else dz -= opt.wallThickness();
                    shader->use();
                    shader->set("pointlight[4].pos", glm::vec3(dx, dy, dz));
                    shader->set("pointlight[4].coef_a2", 0.075f);
                    need_reload_map = true;
                }

                it = interact_objs.erase(it);
                continue;
            }
            move_false = true;

            ++it;
        }

        if (move_false)
        {

            break;
        }
        // update character pos
        character.cam.cam_pos.x = new_pos_x;
        character.cam.cam_pos.z = new_pos_z;
        if (maze.canWin(x, y))
        {
            state = State::Win;
            break;
        }

        // update character cell
        if (will_move_left)
        {
            need_reload_map = true;
            maze.moveLeft();
        }
        else if (will_move_right)
        {
            need_reload_map = true;
            maze.moveRight();
        }

        if (will_move_up)
        {
            need_reload_map = true;
            maze.moveUp();
        }
        else if (will_move_down)
        {
            need_reload_map = true;
            maze.moveDown();
        }

    }

    return need_reload_map;
}