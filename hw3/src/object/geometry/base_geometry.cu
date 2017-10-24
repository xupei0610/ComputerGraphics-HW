#include "object/geometry/base_geometry.hpp"

using namespace px;

BaseGeometry::BaseGeometry(const BaseMaterial * const &material,
                           const Transformation * const &trans,
                           int const &n_vertices)
        : _material(material),
          _transformation(trans),
          _n_vertices(n_vertices),
          _raw_vertices(nullptr)
{
    _raw_vertices = new Point[n_vertices];
}

BaseGeometry::~BaseGeometry()
{
    delete [] _raw_vertices;
}

PX_CUDA_CALLABLE
const BaseGeometry *BaseGeometry::hit(Ray const &ray,
                                double const &range_start,
                                double const &range_end,
                                double &hit_at) const
{
    // TODO bump mapping
    if (_transformation == nullptr)
    {
        return hitCheck(ray, range_start, range_end, hit_at);
    }

    Ray trans_ray(_transformation->point(ray.original),
                  _transformation->direction(ray.direction));

    return hitCheck(trans_ray, range_start, range_end, hit_at);
}

PX_CUDA_CALLABLE
Direction BaseGeometry::normal(double const &x,
                               double const &y,
                               double const &z) const
{
    // TODO bump mapping
    if (_transformation == nullptr)
        return normalVec(x, y, z);
    return _transformation->normal(normalVec(_transformation->point(x, y, z)));
}

PX_CUDA_CALLABLE
Vec3<double> BaseGeometry::textureCoord(double const &x,
                                        double const &y,
                                        double const &z) const
{
    if (_transformation == nullptr)
        return getTextureCoord(x, y, z);
    auto p = _transformation->point(x, y, z);
    return getTextureCoord(p.x, p.y, p.z);
}