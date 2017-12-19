#include "item.hpp"
#include "global.hpp"

using namespace px;

const glm::mat4 Item::IDENTITY_MODEL_MAT;
const Light Item::LIGHT_STUB{};
const glm::vec3 Item::X_AXIS(1.f, 0.f, 0.f);
const glm::vec3 Item::Y_AXIS(0.f, 1.f, 0.f);
const glm::vec3 Item::Z_AXIS(0.f, 0.f, 1.f);

std::vector<ItemInfo> Item::_items;
std::vector<ItemGenFn> Item::_item_gen_fn;

static std::shared_ptr<Item> emptyItemGenFn()
{
    return nullptr;
}

ItemInfo::ItemInfo(std::string const &name, std::string const &description,
                   int weight, bool stackable, bool collectible, bool placeable)
    : name(name), description(description), weight(weight),
      stackable(stackable), collectible(collectible), placeable(placeable)
{}

std::size_t Item::reg(ItemInfo &item, ItemGenFn fn)
{
    auto s = _items.size();
    if (s == 0)
    {
        _items.reserve(ITEM_REGISTER_CAPACITY);
        _item_gen_fn.reserve(ITEM_REGISTER_CAPACITY);
        _items.push_back(ItemInfo("", "", 0, false, false, false));
        _items.back()._id = 0;
        _item_gen_fn.push_back(emptyItemGenFn);
        ++s;
    }
    item._id = s;
    _items.push_back(item);
    _item_gen_fn.push_back(fn);
    return s;
}

ItemInfo const &Item::lookup(std::size_t const &index)
{
    if (_items.size() > index)
        return _items[index];
    else
        return _items[0];
}

std::shared_ptr<Item> Item::gen(std::size_t const &index)
{
    if (_items.size() > index)
        return _item_gen_fn[index]();
    else
        return nullptr;
}

Item::Item(std::size_t id)
    : attribute(Item::lookup(id))
{
    if (id == 0)
        err("Invalid Item id");
}

[[noreturn]]
void Item::err(std::string const &msg)
{
    throw ItemError(msg);
}
