#ifndef PX_CG_ITEM_METAL_KEY_HPP
#define PX_CG_ITEM_METAL_KEY_HPP

#include "item.hpp"
#include "mesh.hpp"

namespace px { namespace item
{
class Key;
class MetalKey;
class WoodKey;
class WaterKey;
class FireKey;
class EarthKey;
}}

class px::item::Key : public Item
{
public:
    Light light_source;

    void place(glm::vec3 const &p) override;
    glm::vec3 pos() override;
    glm::vec3 halfSize() override;
    void render(Shader *scene_shader, glm::mat4 const &view, glm::mat4 const &proj) override;
    void update(float dt) override;

    bool lighting() override;

    const Light & light() override;
    void init(Shader *scene_shader) override;

    explicit Key(std::size_t item_id);
    ~Key();
    Key &operator=(Key const &) = delete;
    Key &operator=(Key &&) = delete;

protected:
    unsigned int texture[4];

    glm::vec3 ambient;
    float displace_amp;
    float rot;
    glm::vec3 scale;
    glm::vec3 half_scale;
    glm::mat4 model;

    Mesh *mesh;
};

class px::item::MetalKey : public Key
{
private:
    static ItemInfo ITEM_INFO;
protected:
    static std::vector<unsigned char *> textures;
    static std::vector<std::pair<int, int> >  texture_dim;
public:
    static std::shared_ptr<Item> create();
    static ItemInfo const &itemInfo();
    static std::size_t regItem();
    explicit MetalKey();
    void init(Shader *scene_shader) override;
};

class px::item::WoodKey : public Key
{
private:
    static ItemInfo ITEM_INFO;
protected:
    static std::vector<unsigned char *> textures;
    static std::vector<std::pair<int, int> >  texture_dim;
public:
    static std::shared_ptr<Item> create();
    static ItemInfo const &itemInfo();
    static std::size_t regItem();
    explicit WoodKey();
    void init(Shader *scene_shader) override;
};

class px::item::WaterKey : public Key
{
private:
    static ItemInfo ITEM_INFO;
protected:
    static std::vector<unsigned char *> textures;
    static std::vector<std::pair<int, int> >  texture_dim;
public:
    static std::shared_ptr<Item> create();
    static ItemInfo const &itemInfo();
    static std::size_t regItem();
    explicit WaterKey();
    void init(Shader *scene_shader) override;
};

class px::item::FireKey : public Key
{
private:
    static ItemInfo ITEM_INFO;
protected:
    static std::vector<unsigned char *> textures;
    static std::vector<std::pair<int, int> >  texture_dim;
public:
    static std::shared_ptr<Item> create();
    static ItemInfo const &itemInfo();
    static std::size_t regItem();
    explicit FireKey();
    void init(Shader *scene_shader) override;
};

class px::item::EarthKey : public Key
{
private:
    static ItemInfo ITEM_INFO;
protected:
    static std::vector<unsigned char *> textures;
    static std::vector<std::pair<int, int> >  texture_dim;
public:
    static std::shared_ptr<Item> create();
    static ItemInfo const &itemInfo();
    static std::size_t regItem();
    explicit EarthKey();
    void init(Shader *scene_shader) override;
};

#endif
