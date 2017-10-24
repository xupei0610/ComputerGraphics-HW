#ifndef PX_CG_OBJECT_STRUCTURE_BASE_STRUCTURE_HPP
#define PX_CG_OBJECT_STRUCTURE_BASE_STRUCTURE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Strcture;
}

class px::Structure : public BaseGeometry
{
protected:
    Structure(const BaseMaterial * const &material,
              const Transformation * const &trans,
              int const &n_vertices)
            : BaseGeometry(material, trans, n_vertices)
    {}
    ~Structure() = default;

    Structure &operator=(Structure const &) = delete;
    Structure &operator=(Structure &&) = delete;
};

#endif // PX_CG_OBJECT_STRUCTURE_BASE_STRUCTURE_HPP
