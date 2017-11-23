#include "item/door.hpp"
#include "global.hpp"
#include "scene.hpp"

#include "soil/SOIL.h"

#include <glm/gtc/matrix_transform.hpp>

using namespace px;

ItemInfo item::MetalDoor::ITEM_INFO("Metal Door", "", 0, false, false, false);
std::vector<unsigned char *> item::MetalDoor::textures;
std::vector<std::pair<int, int> >  item::MetalDoor::texture_dim(1, {0, 0});
ItemInfo item::WoodDoor::ITEM_INFO("Wood Door", "", 0, false, false, false);
std::vector<unsigned char *> item::WoodDoor::textures;
std::vector<std::pair<int, int> >  item::WoodDoor::texture_dim(1, {0, 0});
ItemInfo item::WaterDoor::ITEM_INFO("Water Door", "", 0, false, false, false);
std::vector<unsigned char *> item::WaterDoor::textures;
std::vector<std::pair<int, int> >  item::WaterDoor::texture_dim(1, {0, 0});
ItemInfo item::FireDoor::ITEM_INFO("Fire Door", "", 0, false, false, false);
std::vector<unsigned char *> item::FireDoor::textures;
std::vector<std::pair<int, int> >  item::FireDoor::texture_dim(1, {0, 0});
ItemInfo item::EarthDoor::ITEM_INFO("Earth Door", "", 0, false, false, false);
std::vector<unsigned char *> item::EarthDoor::textures;
std::vector<std::pair<int, int> >  item::EarthDoor::texture_dim(1, {0, 0});


#define TEXTURE_LOAD_HELPER(filename_prefix, ext, width, height, target_container)   \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_d" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_n" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_s" ext, &width, &height, &ch, SOIL_LOAD_RGB));  \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_h" ext, &width, &height, &ch, SOIL_LOAD_RGB));

#define ITEM_DOOR_TEXTURE_BIND_HELPER                                                    \
    auto i = 0;                                                                         \
    glGenTextures(4, texture);                                                          \
    TEXTURE_BIND_HELPER(texture[0], 0, GL_RGB, GL_REPEAT, GL_LINEAR,                    \
                        texture_dim[i].first, texture_dim[i].second, textures[i*4],     \
                        shader->pid(), "material.diffuse");                             \
    TEXTURE_BIND_HELPER(texture[1], 1, GL_RGB, GL_REPEAT, GL_LINEAR,                    \
                        texture_dim[i].first, texture_dim[i].second, textures[i*4+1],   \
                        shader->pid(), "material.normal");                              \
    TEXTURE_BIND_HELPER(texture[2], 2, GL_RGB, GL_REPEAT, GL_LINEAR,                    \
                        texture_dim[i].first, texture_dim[i].second, textures[i*4+2],   \
                        shader->pid(), "material.specular")                             \
    TEXTURE_BIND_HELPER(texture[3], 3, GL_RGB, GL_REPEAT, GL_LINEAR,                    \
                        texture_dim[i].first, texture_dim[i].second, textures[i*4+3],   \
                        shader->pid(), "material.displace")

item::Door::Door(std::size_t item_id)
        : Item(item_id),
          rot(0.f, 0.f, 0.f), scale(1.f, 1.f, 1.f),
          mesh(nullptr)
{}

item::Door::~Door()
{
    delete mesh;
}

void item::Door::place(glm::vec3 const &p)
{
    pos = p;
}

void item::Door::rotate(float &x_deg, float &y_deg, float &z_deg)
{
    rot.x = x_deg;
    rot.y = y_deg;
    rot.z = z_deg;
}

void item::Door::position(float &x, float &y, float &z)
{
    x = pos.x;
    y = pos.y;
    z = pos.z;
}
void item::Door::halfSize(float &x, float &y, float &z)
{
    x = scale.x;
    y = scale.y;
    z = scale.z;
}
void item::Door::setHalfSize(glm::vec3 const &r)
{
    scale.x = r.x;
    scale.y = r.y;
    scale.z = r.z;
}

void item::Door::render(Shader *shader,
                        glm::mat4 const &view,
                        glm::mat4 const &proj)
{
    shader->use();
    shader->set("material.ambient", ambient);
    shader->set("material.displace_mid", .5f);
    shader->set("material.displace_amp", .01f * glm::length(scale));
    shader->set("model", model);

    for (auto & e: mesh->entries)
    {
        glBindVertexArray(e->vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture[1]);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, texture[2]);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, texture[3]);

        e->render();
    }

    shader->set("displace_amp", 0.f);
    shader->set("model", Item::IDENTITY_MODEL_MAT);
}

void item::Door::update(float dt)
{
    model = glm::translate(IDENTITY_MODEL_MAT, pos);
#ifdef GLM_FORCE_RADIANS
    model = glm::rotate(model, glm::radians(rot.x), X_AXIS);
    model = glm::rotate(model, glm::radians(rot.y), Y_AXIS);
    model = glm::rotate(model, glm::radians(rot.z), Z_AXIS);
#else
    model = glm::rotate(model, rot.x, X_AXIS);
    model = glm::rotate(model, rot.y, Y_AXIS);
    model = glm::rotate(model, rot.z, Z_AXIS);
#endif
    model = glm::scale(model, scale);
}


void item::MetalDoor::init(Shader *shader)
{
    int ch;
    TEXTURE_LOAD_HELPER("metal", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    mesh = new Mesh(ASSET_PATH "/model/cube.obj");
    ITEM_DOOR_TEXTURE_BIND_HELPER
}

item::MetalDoor::MetalDoor()
        : Door(MetalDoor::regItem())
{
    ambient = glm::vec3(0.8f, 0.6f, 0);
}

ItemInfo const &item::MetalDoor::itemInfo()
{
    return MetalDoor::ITEM_INFO;
}

Item *item::MetalDoor::create()
{
    regItem();
    return static_cast<Item*>(new MetalDoor);
}

std::size_t item::MetalDoor::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(MetalDoor::ITEM_INFO, item::MetalDoor::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}


void item::WoodDoor::init(Shader *shader)
{
    int ch;
    TEXTURE_LOAD_HELPER("wood", ".png", texture_dim[0].first, texture_dim[0].second, textures)

    mesh = new Mesh(ASSET_PATH "/model/cube.obj");
    ITEM_DOOR_TEXTURE_BIND_HELPER
}

item::WoodDoor::WoodDoor()
        : Door(WoodDoor::regItem())
{
    ambient = glm::vec3(0.8f, 0.4f, 0);
}

ItemInfo const &item::WoodDoor::itemInfo()
{
    return WoodDoor::ITEM_INFO;
}

Item *item::WoodDoor::create()
{
    regItem();
    return static_cast<Item*>(new WoodDoor);
}

std::size_t item::WoodDoor::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(WoodDoor::ITEM_INFO, item::WoodDoor::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

void item::WaterDoor::init(Shader *shader)
{
    int ch;
    TEXTURE_LOAD_HELPER("water", ".png", texture_dim[0].first, texture_dim[0].second, textures)

    mesh = new Mesh(ASSET_PATH "/model/cube.obj");
    ITEM_DOOR_TEXTURE_BIND_HELPER

}

item::WaterDoor::WaterDoor()
        : Door(WaterDoor::regItem())
{
    ambient = glm::vec3(0.2f, 0.6f, 1.0f);
}

ItemInfo const &item::WaterDoor::itemInfo()
{
    return WaterDoor::ITEM_INFO;
}

Item *item::WaterDoor::create()
{
    regItem();
    return static_cast<Item*>(new WaterDoor);
}

std::size_t item::WaterDoor::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(WaterDoor::ITEM_INFO, item::WaterDoor::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}


void item::FireDoor::init(Shader *shader)
{
    int ch;
    TEXTURE_LOAD_HELPER("fire", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    mesh = new Mesh(ASSET_PATH "/model/cube.obj");
    ITEM_DOOR_TEXTURE_BIND_HELPER

}

item::FireDoor::FireDoor()
        : Door(FireDoor::regItem())
{
    ambient = glm::vec3(1.0f, 0.45f, 0.f);
}

ItemInfo const &item::FireDoor::itemInfo()
{
    return FireDoor::ITEM_INFO;
}

Item *item::FireDoor::create()
{
    regItem();
    return static_cast<Item*>(new FireDoor);
}

std::size_t item::FireDoor::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(FireDoor::ITEM_INFO, item::FireDoor::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}


void item::EarthDoor::init(Shader *shader)
{
    int ch;
    TEXTURE_LOAD_HELPER("earth", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    mesh = new Mesh(ASSET_PATH "/model/cube.obj");
    ITEM_DOOR_TEXTURE_BIND_HELPER

}

item::EarthDoor::EarthDoor()
        : Door(EarthDoor::regItem())
{
    ambient = glm::vec3(.6f, .4f, .2f);
}

ItemInfo const &item::EarthDoor::itemInfo()
{
    return EarthDoor::ITEM_INFO;
}

Item *item::EarthDoor::create()
{
    regItem();
    return static_cast<Item*>(new EarthDoor);
}

std::size_t item::EarthDoor::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(EarthDoor::ITEM_INFO, item::EarthDoor::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

#undef TEXTURE_LOAD_HELPER
#undef ITEM_DOOR_TEXTURE_BIND_HELPER