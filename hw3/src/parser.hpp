#ifndef PX_CG_PARSER_HPP
#define PX_CG_PARSER_HPP

#include <unordered_map>
#include <string>
#include <memory>

#include <vector>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "scene.hpp"

namespace px
{

enum class IMAGE_FORMAT
{
    PNG,
    JPG,
    BMP,
    TGA
};

class Parser
{
public:
    static
    std::unordered_map<std::string, IMAGE_FORMAT>
    parse(std::string const &script_str,
          std::shared_ptr<Scene> const &scene);
};

}

#endif // PX_CG_PARSER_HPP
