#ifndef PX_CG_ITEM_HPP
#define PX_CG_ITEM_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include "shader/base_shader.hpp"

namespace px
{

class Item;

typedef Item* (*ItemGenFn)();

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
    static const glm::vec3 X_AXIS;
    static const glm::vec3 Y_AXIS;
    static const glm::vec3 Z_AXIS;

    static std::size_t reg(ItemInfo &item_info,  ItemGenFn fn);
    static const ItemInfo &lookup(std::size_t const &index);
    static Item *gen(std::size_t const &index);

    const ItemInfo & attribute;

    bool operator==(const Item &rhs) const
    {
        return attribute.id() == rhs.attribute.id();
    }

    [[noreturn]]
    static void err(std::string const &msg);

    virtual void place(glm::vec3 const &p) {};
    virtual void rotate(float &x_deg, float &y_deg, float &z_deg) {};
    virtual void setHalfSize(glm::vec3 const &r) {}

    virtual void position(float &x, float &y, float &z){}
    virtual void halfSize(float &x, float &y, float &z){x=0; y=0; z=0;}

    virtual void update(float dt) {}

    virtual void init(Shader *scene_shader) {}
    virtual void render(Shader *scene_shader, glm::mat4 const &view, glm::mat4 const &proj) {}

protected:
    Item(std::size_t _id);
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
