#include "object/geometry/plane.hpp"

#include <cfloat>

using namespace px;

BasePlane::BasePlane(Point const &pos,
                     const BaseMaterial * const &material,
                     const Transformation * const &trans)
        : BaseGeometry(material, trans, 4),
          _position(pos)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BasePlane::hitCheck(Ray const &ray,
                                   double const &t_start,
                                   double const &t_end,
                                   double &hit_at) const
{
    auto tmp = (_p_dot_n - ray.original.dot(_norm_vec)) / ray.direction.dot(_norm_vec);
    return (tmp > t_start && tmp < t_end) ? (hit_at = tmp, this) : nullptr;
}

PX_CUDA_CALLABLE
Direction BasePlane::normalVec(double const &x, double const &y, double const &z) const
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<double> BasePlane::getTextureCoord(double const &x, double const &y,
                                    double const &z) const
{
    return {x - _position.x,
            -_norm_vec.z*(y - _position.y) + _norm_vec.y*(z - _position.z),
            (x - _position.x)*_norm_vec.x + (y - _position.y)*_norm_vec.y + (z - _position.z)*_norm_vec.z};
}

std::shared_ptr<BaseGeometry> Plane::create(Point const &position,
                                            Direction const &norm_vec,
                                            std::shared_ptr<BaseMaterial> const &material,
                                            std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Plane(position, norm_vec, material, trans));
}

Plane::Plane(Point const &position,
             Direction const &norm_vec,
             std::shared_ptr<BaseMaterial> const &material,
             std::shared_ptr<Transformation> const &trans)
        : BasePlane(position, material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setNormVec(norm_vec);
}

Plane::~Plane()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Plane::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BasePlane)));

        _material = _material_ptr == nullptr ? nullptr : _material_ptr->up2Gpu();
        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BasePlane*>(this),
                                 sizeof(BasePlane),
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

void Plane::clearGpuData()
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

void Plane::setPosition(Point const &position)
{
    _position = position;
    _p_dot_n = position.dot(_norm_vec);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Plane::setNormVec(Direction const &norm_vec)
{
    _norm_vec = norm_vec;
    _p_dot_n = _position.dot(norm_vec);

    if ((_norm_vec.x == 1 || _norm_vec.x == -1) && _norm_vec.y == 0 && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = 0;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x = 0;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = 0;
        _raw_vertices[2].y = -FLT_MAX;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x = 0;
        _raw_vertices[3].y =  FLT_MAX;
        _raw_vertices[3].z = -FLT_MAX;
    }
    else if (_norm_vec.x == 0 && (_norm_vec.y == 1 || _norm_vec.y == -1) && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = 0;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y = 0;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y = 0;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = 0;
        _raw_vertices[3].z = -FLT_MAX;
    }
    else if (_norm_vec.x == 0 && _norm_vec.y == 0 && (_norm_vec.z == 1 || _norm_vec.z == -1))
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = 0;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z = 0;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y =  FLT_MAX;
        _raw_vertices[2].z = 0;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = -FLT_MAX;
        _raw_vertices[3].z = 0;
    }
    else if (_norm_vec.x == 0 && _norm_vec.y == 0 && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = 0;
        _raw_vertices[0].y = 0;
        _raw_vertices[0].z = 0;
        _raw_vertices[1].x = 0;
        _raw_vertices[1].y = 0;
        _raw_vertices[1].z = 0;
        _raw_vertices[2].x = 0;
        _raw_vertices[2].y = 0;
        _raw_vertices[2].z = 0;
        _raw_vertices[3].x = 0;
        _raw_vertices[3].y = 0;
        _raw_vertices[3].z = 0;
    }
    else
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y =  FLT_MAX;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = -FLT_MAX;
        _raw_vertices[3].z = -FLT_MAX;
    }
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
