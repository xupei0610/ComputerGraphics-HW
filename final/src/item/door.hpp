#ifndef PX_CG_ITEM_METAL_DOOR_HPP
#define PX_CG_ITEM_METAL_DOOR_HPP

#include "item.hpp"
#include "mesh.hpp"

namespace px { namespace item
{
class Door;
class MetalDoor;
class WoodDoor;
class WaterDoor;
class FireDoor;
class EarthDoor;
}}

class px::item::Door : public Item
{
public:
    Light light_source;

    void place(glm::vec3 const &p) override;
    void rotate(float &x_deg, float &y_deg, float &z_deg) override;
    glm::vec3 pos() override;
    glm::vec3 halfSize() override;
    void setHalfSize(glm::vec3 const &r) override;
    void render(Shader *scene_shader, glm::mat4 const &view, glm::mat4 const &proj) override;
    void update(float dt) override;

    void enlight();
    bool lighting() override;
    const Light &light() override;

    explicit Door(std::size_t item_id);
    ~Door();
    Door &operator=(Door const &) = delete;
    Door &operator=(Door &&) = delete;

protected:
    unsigned int texture[4];

    glm::vec3 ambient;
    glm::vec3 rot;
    glm::vec3 scale;

    glm::mat4 model;

    Mesh *mesh;

    bool is_lighting;
};


class px::item::MetalDoor : public Door
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
    explicit MetalDoor();
    void init(Shader *scene_shader) override;
};

class px::item::WoodDoor : public Door
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
    explicit WoodDoor();
    void init(Shader *scene_shader) override;
};

class px::item::WaterDoor : public Door
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
    explicit WaterDoor();
    void init(Shader *scene_shader) override;
};

class px::item::FireDoor : public Door
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
    explicit FireDoor();
    void init(Shader *scene_shader) override;
};

class px::item::EarthDoor : public Door
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
    explicit EarthDoor();
    void init(Shader *scene_shader) override;
};

#endif
