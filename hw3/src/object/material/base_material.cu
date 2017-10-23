#include "object/material/base_material.hpp"

using namespace px;

BaseMaterial::BaseMaterial(const BumpMapping * const &bump_mapping)
        : _bump_mapping(bump_mapping)
{}
