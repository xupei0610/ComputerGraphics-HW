#ifndef PX_CG_ITEM_HPP
#define PX_CG_ITEM_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

#include "shader/base_shader.hpp"

namespace px
{
struct Light
{
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 coef;
};

class Item;

typedef std::shared_ptr<Item> (*ItemGenFn)();

struct ItemInfo
{
private:
    std::size_t _id;

public:
    const std::string name;
    const std::string description;
    const int weight;
    const bool stackable;
    const bool collectible;
    const bool placeable;

    ItemInfo(std::string const &name, std::string const &description,
             int weight,
             bool stackable, bool collectible, bool placeable);

    inline const std::size_t &id() const noexcept {return _id;}

    friend Item;
};

class ItemError;
}

class px::Item
{
private:
    static std::vector<ItemInfo> _items;
    static std::vector<ItemGenFn> _item_gen_fn;
public:
    static const glm::mat4 IDENTITY_MODEL_MAT;
    static const Light LIGHT_STUB;
    static const glm::vec3 X_AXIS;
    static const glm::vec3 Y_AXIS;
    static const glm::vec3 Z_AXIS;

    static std::size_t reg(ItemInfo &item_info,  ItemGenFn fn);
    static const ItemInfo &lookup(std::size_t const &index);
    static std::shared_ptr<Item> gen(std::size_t const &index);

    const ItemInfo & attribute;

    bool operator==(const Item &rhs) const
    {
        return attribute.id() == rhs.attribute.id();
    }

    [[noreturn]]
    static void err(std::string const &msg);

    virtual void place(glm::vec3 const &p) {position = p;};
    virtual void rotate(float &x_deg, float &y_deg, float &z_deg) {};
    virtual void setHalfSize(glm::vec3 const &r) {}

    virtual glm::vec3 pos() { return glm::vec3(0, 0, 0);}
    virtual glm::vec3 halfSize() {return glm::vec3(0, 0, 0);}

    virtual bool lighting() {return false;}
    virtual const Light & light(){ return LIGHT_STUB; };

    virtual bool preRender() {return true;}
    virtual bool postRender() {return false;}

    virtual void update(float dt) {}

    virtual bool isRigidBody() {return false;}

    virtual float mass() {return 0.f;}
    virtual bool canMove() {return false;}
    virtual glm::vec3 & moveSpan() {return movement;}
    virtual void makeMove() { position = position + movement;}
    virtual void hit(glm::vec3 const &at) {}

    virtual void init(Shader *scene_shader) {}
    // pre-render, rendering with scene, rendering into buffer
    virtual void render(Shader *scene_shader, glm::mat4 const &view, glm::mat4 const &proj) {}
    // post-render, rendering after lighting rendering, rendering directly for output
    virtual void render(glm::mat4 const &view, glm::mat4 const &proj) {}

protected:
    Item(std::size_t _id);

    glm::vec3 movement;
    glm::vec3 position;
};

class px::ItemError : public std::exception
{

public:
    ItemError(const std::string &msg, const int code=0)
            : msg(msg), err_code(code)
    {}
    const char *what() const noexcept override
    {
        return msg.data();
    }
    inline int code() const
    {
        return err_code;
    }

protected:
    std::string msg;
    int err_code;
};

namespace std
{
template<>
struct hash<px::Item>
{
    std::size_t operator()(const px::Item &item) const
    {
        return item.attribute.id();
    }
};
}

#endif
