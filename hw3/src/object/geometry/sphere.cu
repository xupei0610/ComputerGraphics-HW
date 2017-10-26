#include "object/geometry/sphere.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseSphere::BaseSphere(Point const &pos,
                       PREC const &radius,
                       const BaseMaterial *const &material,
                       const Transformation *const &trans)
        : BaseGeometry(material, trans, 8),
          _center(pos),
          _radius(radius),
          _radius2(radius*radius)
{
    updateVertices();
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseSphere::hitCheck(Ray const &ray,
                                  PREC const &t_start,
                                  PREC const &t_end,
                                  PREC &hit_at) const
{
    auto oc = Vec3<PREC>(ray.original.x - _center.x,
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
Direction BaseSphere::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return {x - _center.x, y - _center.y, z - _center.z};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseSphere::getTextureCoord(PREC const &x, PREC const &y,
                                       PREC const &z) const
{
    return {(1 + std::atan2(z - _center.z, x - _center.x) / PI) *PREC(0.5),
            std::acos((y - _center.y) / _radius2) / PI,
            0};
}

std::shared_ptr<Geometry> Sphere::create(Point const &position,
                                             PREC const &radius,
                                             std::shared_ptr<Material> const &material,
                                             std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Sphere(position, radius,
                                                    material, trans));
}

Sphere::Sphere(Point const &position,
               PREC const &radius,
           std::shared_ptr<Material> const &material,
           std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseSphere(position, radius, material->obj(), trans.get())),
          _base_obj(_obj),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Sphere::~Sphere()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &Sphere::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Sphere::devPtr()
{
    return _dev_ptr;
}

void Sphere::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseGeometry **)));

        if (_material_ptr != nullptr)
            _material_ptr->up2Gpu();
        if (_transformation_ptr != nullptr)
            _transformation_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::Sphere(_dev_ptr,
                           _obj->_center, _obj->_radius,
                           _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                           _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
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

    GpuCreator::destroy(_dev_ptr);
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void Sphere::setCenter(Point const &center)
{
    _obj->_center = center;
    _obj->updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Sphere::setRadius(PREC const &r)
{
    _obj->_radius = r;
    _obj->_radius2 = r*r;
    _obj->updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void BaseSphere::updateVertices()
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
}
