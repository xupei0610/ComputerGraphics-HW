#include "object/geometry/sphere.hpp"

using namespace px;

BaseSphere::BaseSphere(Point const &pos,
                       double const &radius,
                       const BaseMaterial *const &material,
                       const Transformation *const &trans)
        : BaseGeometry(material, trans, 8),
          _center(pos),
          _radius(radius),
          _radius2(radius*radius)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BaseSphere::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at) const
{
    auto oc = Vec3<double>(ray.original.x - _center.x,
                           ray.original.y - _center.y,
                           ray.original.z - _center.z);
    auto a = ray.direction.dot(ray.direction);
    auto b = ray.direction.dot(oc);
    auto c = oc.dot(oc) - _radius2;
    auto discriminant = b*b - a*c;
    if (discriminant > 0)
    {
        auto tmp = -std::sqrt(discriminant)/a;
        auto b_by_a = -b/a;
        tmp += b_by_a;
        if (tmp > t_start && tmp < t_end)
        {
            hit_at = tmp;
            return this;
        }
        else
        {
            tmp = 2*b_by_a-tmp;
            if (tmp > t_start && tmp < t_end)
            {
                hit_at = tmp;
                return this;
            }
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseSphere::normalVec(double const &x, double const &y, double const &z) const
{
    return {x - _center.x, y - _center.y, z - _center.z};
}

PX_CUDA_CALLABLE
Vec3<double> BaseSphere::getTextureCoord(double const &x, double const &y,
                                       double const &z) const
{
    return {(1 + std::atan2(z - _center.z, x - _center.x) / PI) * 0.5,
            std::acos((y - _center.y) / _radius2) / PI,
            0};
}

std::shared_ptr<BaseGeometry> Sphere::create(Point const &position,
                                             double const &radius,
                                             std::shared_ptr<BaseMaterial> const &material,
                                             std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Sphere(position, radius,
                                                    material, trans));
}

Sphere::Sphere(Point const &position,
               double const &radius,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseSphere(position, radius, material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    updateVertices();
}

Sphere::~Sphere()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Sphere::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseSphere)));

        _material = _material_ptr == nullptr ? nullptr : _material_ptr->up2Gpu();
        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseSphere*>(this),
                                 sizeof(BaseSphere),
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

void Sphere::clearGpuData()
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


void Sphere::setCenter(Point const &center)
{
    _center = center;

    updateVertices();
}

void Sphere::setRadius(double const &r)
{
    _radius = r;
    _radius2 = r*r;

    updateVertices();
}

void Sphere::updateVertices()
{
    _raw_vertices[0].x = _center.x + _radius;
    _raw_vertices[0].y = _center.y + _radius;
    _raw_vertices[0].z = _center.z + _radius;

    _raw_vertices[1].x = _center.x - _radius;
    _raw_vertices[1].y = _center.y + _radius;
    _raw_vertices[1].z = _center.z + _radius;

    _raw_vertices[2].x = _center.x + _radius;
    _raw_vertices[2].y = _center.y - _radius;
    _raw_vertices[2].z = _center.z + _radius;

    _raw_vertices[3].x = _center.x + _radius;
    _raw_vertices[3].y = _center.y + _radius;
    _raw_vertices[3].z = _center.z - _radius;

    _raw_vertices[4].x = _center.x - _radius;
    _raw_vertices[4].y = _center.y - _radius;
    _raw_vertices[4].z = _center.z + _radius;

    _raw_vertices[5].x = _center.x - _radius;
    _raw_vertices[5].y = _center.y + _radius;
    _raw_vertices[5].z = _center.z - _radius;

    _raw_vertices[6].x = _center.x + _radius;
    _raw_vertices[6].y = _center.y - _radius;
    _raw_vertices[6].z = _center.z - _radius;

    _raw_vertices[7].x = _center.x - _radius;
    _raw_vertices[7].y = _center.y - _radius;
    _raw_vertices[7].z = _center.z - _radius;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
