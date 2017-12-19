#include "item/key.hpp"
#include "global.hpp"
#include "scene.hpp"

#include "soil/SOIL.h"

#include <glm/gtc/matrix_transform.hpp>

using namespace px;

ItemInfo item::MetalKey::ITEM_INFO("Metal Key", "", 0, true, true, true);
std::vector<unsigned char *> item::MetalKey::textures;
std::vector<std::pair<int, int> >  item::MetalKey::texture_dim(1, {0, 0});
ItemInfo item::WoodKey::ITEM_INFO("Wood Key", "", 0, true, true, true);
std::vector<unsigned char *> item::WoodKey::textures;
std::vector<std::pair<int, int> >  item::WoodKey::texture_dim(1, {0, 0});
ItemInfo item::WaterKey::ITEM_INFO("Water Key", "", 0, true, true, true);
std::vector<unsigned char *> item::WaterKey::textures;
std::vector<std::pair<int, int> >  item::WaterKey::texture_dim(1, {0, 0});
ItemInfo item::FireKey::ITEM_INFO("Fire Key", "", 0, true, true, true);
std::vector<unsigned char *> item::FireKey::textures;
std::vector<std::pair<int, int> >  item::FireKey::texture_dim(1, {0, 0});
ItemInfo item::EarthKey::ITEM_INFO("Earth Key", "", 0, true, true, true);
std::vector<unsigned char *> item::EarthKey::textures;
std::vector<std::pair<int, int> >  item::EarthKey::texture_dim(1, {0, 0});

#define TEXTURE_LOAD_HELPER(filename_prefix, ext, width, height, target_container)   \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_d" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_n" ext, &width, &height, &ch, SOIL_LOAD_RGB)); \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_s" ext, &width, &height, &ch, SOIL_LOAD_RGB));  \
        target_container.push_back(SOIL_load_image(ASSET_PATH "/texture/" filename_prefix "_h" ext, &width, &height, &ch, SOIL_LOAD_RGB));
#define ITEM_KEY_TEXTURE_BIND_HELPER                                                    \
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

item::Key::Key(std::size_t item_id)
    : Item(item_id), displace_amp(0.05f), rot(0),
      scale(0.25f, 0.25f, 0.25f), half_scale(0.125f, 0.125f, 0.125f),
      mesh(nullptr)
{
    light_source.ambient  = glm::vec3(1.f, 1.f, 1.f);
    light_source.diffuse  = glm::vec3(1.f, 1.f, 1.f);
    light_source.specular = glm::vec3(1.f, 1.f, 1.f);
    light_source.coef     = glm::vec3(0.f, 0.f, 2.f);
}

item::Key::~Key()
{
    delete mesh;
}

void item::Key::place(glm::vec3 const &p)
{
    position = p;
}

glm::vec3 item::Key::pos()
{
    return position;
}
glm::vec3 item::Key::halfSize()
{
    return half_scale;
}

void item::Key::render(Shader *shader, glm::mat4 const &view, glm::mat4 const &proj)
{
    shader->use();
    shader->set("use_tangent", 1);
    shader->set("material.ambient", ambient*0.2f);
    shader->set("material.shininess", 50.f);
    shader->set("material.displace_mid", .5f);
    shader->set("material.displace_amp", displace_amp);
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
    shader->set("model", IDENTITY_MODEL_MAT);
}

void item::Key::update(float dt)
{
    model = glm::translate(IDENTITY_MODEL_MAT, position);

    rot += dt*.5f;
    if (rot > 360) rot-=360;
    else if (rot < -360) rot += 360;
    model = glm::rotate(model, rot, Y_AXIS);

    model = glm::scale(model, scale);
}

bool item::Key::lighting()
{
    return true;
}

const Light &item::Key::light()
{
    return light_source;
}

void item::Key::init(Shader *scene_shader)
{
    mesh = new Mesh(ASSET_PATH "/model/sphere2.obj");
}

void item::MetalKey::init(Shader *shader)
{
    Key::init(shader);

    int ch;
    TEXTURE_LOAD_HELPER("metal", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    ITEM_KEY_TEXTURE_BIND_HELPER
}

item::MetalKey::MetalKey()
        : Key(MetalKey::regItem())
{
    ambient = glm::vec3(0.8f, 0.6f, 0);
}

ItemInfo const &item::MetalKey::itemInfo()
{
    return MetalKey::ITEM_INFO;
}

std::shared_ptr<Item> item::MetalKey::create()
{
    return std::shared_ptr<Item>(new MetalKey);
}

std::size_t item::MetalKey::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(MetalKey::ITEM_INFO, item::MetalKey::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}


void item::WoodKey::init(Shader *shader)
{
    Key::init(shader);

    int ch;
    TEXTURE_LOAD_HELPER("wood", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    ITEM_KEY_TEXTURE_BIND_HELPER

}

item::WoodKey::WoodKey()
        : Key(WoodKey::regItem())
{
    ambient = glm::vec3(0.8f, 0.4f, 0);
}

ItemInfo const &item::WoodKey::itemInfo()
{
    return WoodKey::ITEM_INFO;
}

std::shared_ptr<Item> item::WoodKey::create()
{
    return std::shared_ptr<Item>(new WoodKey);
}

std::size_t item::WoodKey::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(WoodKey::ITEM_INFO, item::WoodKey::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

void item::WaterKey::init(Shader *shader)
{
    Key::init(shader);

    int ch;
    TEXTURE_LOAD_HELPER("water", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    ITEM_KEY_TEXTURE_BIND_HELPER

}

item::WaterKey::WaterKey()
        : Key(WaterKey::regItem())
{
    ambient = glm::vec3(0.2f, 0.6f, 1.0f);
}

ItemInfo const &item::WaterKey::itemInfo()
{
    return WaterKey::ITEM_INFO;
}

std::shared_ptr<Item> item::WaterKey::create()
{
    return std::shared_ptr<Item>(new WaterKey);
}

std::size_t item::WaterKey::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(WaterKey::ITEM_INFO, item::WaterKey::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}


void item::FireKey::init(Shader *shader)
{
    Key::init(shader);

    int ch;
    TEXTURE_LOAD_HELPER("fire", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    ITEM_KEY_TEXTURE_BIND_HELPER
}

item::FireKey::FireKey()
        : Key(FireKey::regItem())
{
    ambient = glm::vec3(1.0f, 0.45f, 0.f);
}

ItemInfo const &item::FireKey::itemInfo()
{
    return FireKey::ITEM_INFO;
}

std::shared_ptr<Item> item::FireKey::create()
{
    return std::shared_ptr<Item>(new FireKey);
}

std::size_t item::FireKey::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(FireKey::ITEM_INFO, item::FireKey::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

void item::EarthKey::init(Shader *shader)
{
    Key::init(shader);

    int ch;
    TEXTURE_LOAD_HELPER("earth", ".png", texture_dim[0].first, texture_dim[0].second, textures)
    ITEM_KEY_TEXTURE_BIND_HELPER

}

item::EarthKey::EarthKey()
        : Key(EarthKey::regItem())
{
    ambient = glm::vec3(1.f, .8f, .2f);
}

ItemInfo const &item::EarthKey::itemInfo()
{
    return EarthKey::ITEM_INFO;
}

std::shared_ptr<Item> item::EarthKey::create()
{
    return std::shared_ptr<Item>(new EarthKey);
}

std::size_t item::EarthKey::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(EarthKey::ITEM_INFO, item::EarthKey::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

#undef TEXTURE_LOAD_HELPER
#undef ITEM_KEY_TEXTURE_BIND_HELPER