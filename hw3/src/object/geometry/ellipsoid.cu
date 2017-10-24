#include "object/geometry/ellipsoid.hpp"

using namespace px;

BaseEllipsoid::BaseEllipsoid(const BaseMaterial *const &material,
                           const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BaseEllipsoid::hitCheck(Ray const &ray,
                                             double const &t_start,
                                             double const &t_end,
                                             double &hit_at) const
{
    auto xo = ray.original.x - _center.x;
    auto yo = ray.original.y - _center.y;
    auto zo = ray.original.z - _center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y +
              _c * ray.direction.z * ray.direction.z;
    auto B =  2 * _a * xo * ray.direction.x +
              2 * _b * yo * ray.direction.y +
              2 * _c * zo * ray.direction.z;
    auto C =  _a * xo * xo +
              _b * yo * yo +
              _c * zo * zo +
              -1;

    if (A == 0)
    {
        if (B == 0) return nullptr;

        auto tmp = - C / B;
        if (tmp > t_start && tmp < t_end)
        {
            hit_at = tmp;
            return this;
        }
        return nullptr;
    }

    auto discriminant = B * B - 4 * A * C;
    if (discriminant < 0)
        return nullptr;

    discriminant = std::sqrt(discriminant);
    auto tmp1 = (-B - discriminant)/ (2.0 * A);
    auto tmp2 = (-B + discriminant)/ (2.0 * A);
    if (tmp1 > tmp2)
    {
        auto tmp = tmp1;
        tmp1 = tmp2;
        tmp2 = tmp;
    }
    if (tmp1 > t_start && tmp1 < t_end)
    {
        hit_at = tmp1;
        return this;
    }
    if (tmp2 > t_start && tmp2 < t_end)
    {
        hit_at = tmp2;
        return this;
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseEllipsoid::normalVec(double const &x, double const &y, double const &z) const
{
    return {_a * (x - _center.x),
            _b * (y - _center.y),
            _c * (z - _center.z)};
}

PX_CUDA_CALLABLE
Vec3<double> BaseEllipsoid::getTextureCoord(double const &x, double const &y,
                                           double const &z) const
{
    auto dx = x - _center.x;
    auto dy = y - _center.y;
    auto dz = z - _center.z;

    return {(1 + std::atan2(dz, dx) / PI) * 0.5,
            std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PI,
            0};;
}

std::shared_ptr<BaseGeometry> Ellipsoid::create(Point const &center,
                                                double const &radius_x, double const &radius_y,
                                                double const &radius_z,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Ellipsoid(center,
                                                      radius_x, radius_y, radius_z,
                                                      material, trans));
}

Ellipsoid::Ellipsoid(Point const &center,
                     double const &radius_x, double const &radius_y,
                     double const &radius_z,
                     std::shared_ptr<BaseMaterial> const &material,
                     std::shared_ptr<Transformation> const &trans)
        : BaseEllipsoid(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setParams(center,
              radius_x, radius_y, radius_z);
}

Ellipsoid::~Ellipsoid()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Ellipsoid::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseEllipsoid)));

        _material = _material_ptr == nullptr ? nullptr : _material_ptr->up2Gpu();
        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseEllipsoid*>(this),
                                 sizeof(BaseEllipsoid),
                                 cudaMemcpyHostToDevice));

        _material = _material_ptr.get();
        _transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void Ellipsoid::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_transformation_ptr.use_count() == 1)
        _transformation_ptr->clearGpuData();
    if (_material_ptr.use_count() == 1)
        _material_ptr->clearGpuData();

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void Ellipsoid::setParams(Point const &center,
                          double const &radius_x,
                          double const &radius_y,
                          double const &radius_z)
{
    _center = center;
    _radius_x = radius_x;
    _radius_y = radius_y;
    _radius_y = radius_z;

    _a = 1.0 / (radius_x*radius_x);
    _b = 1.0 / (radius_y*radius_y);
    _c = 1.0 / (radius_z*radius_z);

    _raw_vertices[4].x = _center.x - radius_x;
    _raw_vertices[4].y = _center.y - radius_y;
    _raw_vertices[4].z = _center.x - radius_x;
    _raw_vertices[5].x = _center.x - radius_x;
    _raw_vertices[5].y = _center.y + radius_y;
    _raw_vertices[5].z = _center.x - radius_x;
    _raw_vertices[6].x = _center.x + radius_x;
    _raw_vertices[6].y = _center.y + radius_y;
    _raw_vertices[6].z = _center.x - radius_x;
    _raw_vertices[7].x = _center.x + radius_x;
    _raw_vertices[7].y = _center.y - radius_y;
    _raw_vertices[7].z = _center.x - radius_x;

    _raw_vertices[0].x = _center.x - radius_x;
    _raw_vertices[0].y = _center.y + radius_y;
    _raw_vertices[0].z = _center.z - radius_z;
    _raw_vertices[1].x = _center.x + radius_x;
    _raw_vertices[1].y = _center.y + radius_y;
    _raw_vertices[1].z = _center.z - radius_z;
    _raw_vertices[2].x = _center.x + radius_x;
    _raw_vertices[2].y = _center.y - radius_y;
    _raw_vertices[2].z = _center.z - radius_z;
    _raw_vertices[3].x = _center.x + radius_x;
    _raw_vertices[3].y = _center.y - radius_y;
    _raw_vertices[3].z = _center.z - radius_z;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}
