#ifndef PX_CG_ITEM_HPP
#define PX_CG_ITEM_HPP

#include <string>

namespace px
{
struct Item
{
    std::size_t id;
    std::string name;
    std::string description;
    int weight;
    bool stackable;
    bool collectible;

    Item(std::size_t const &id, std::string const &name, std::string const &description, int weight,
         bool stackable, bool collectible)
            : id(id), name(name), description(description), weight(weight),
              stackable(stackable), collectible(collectible)
    {}

    bool operator==(const Item &rhs) const
    {
        return id == rhs.id;
    }
};

class Bag
{
protected:
    static std::vector<Item> items;
public:
    static std::size_t regItem(std::string const &name,
                               std::string const &description,
                               int weight,
                               bool stackable, bool collectible);
    static const Item &searchItem(std::size_t const &index);
};

}

namespace std
{
template<>
struct hash<px::Item>
{
    std::size_t operator()(const px::Item &item) const
    {
        return item.id;
    }
};
}

#endif
