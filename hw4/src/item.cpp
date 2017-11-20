#include <vector>
#include "item.hpp"

using namespace px;

std::vector<Item> Bag::items = {Item(0, "", "", 0, false, false)};

std::size_t Bag::regItem(std::string const &name,
                           std::string const &description,
                           int weight,
                           bool stackable, bool collectible)
{
    auto s = items.size();
    items.push_back(Item(s, name, description, weight, stackable, collectible));
    return s;
}
const Item &Bag::searchItem(std::size_t const &index)
{
    if (items.size() < index)
        return items[0];
    else
        return items[index];
}
