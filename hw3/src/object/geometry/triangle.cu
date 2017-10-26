#include "object/geometry/triangle.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

#include <cfloat>

using namespace px;

PX_CUDA_CALLABLE
BaseTriangle::BaseTriangle(Point const &a,
                           Point const &b,
                           Point const &c,
                           const BaseMaterial *const &material,
                           const Transformation *const &trans)
        : BaseGeometry(material, trans, 3)
{
    setVertices(a, b, c);
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseTriangle::hitCheck(Ray const &ray,
                                  PREC const &t_start,
                                  PREC const &t_end,
                                  PREC &hit_at) const
{
    auto pvec = ray.direction.cross(_ca);
    auto det = pvec.dot(_ba);
    if (det < FLT_MIN && det > -FLT_MIN)
        return nullptr;

    auto tvec = ray.original - _raw_vertices[0];
    auto u = tvec.dot(pvec) / det;
    if (u < 0 || u > 1) return nullptr;

    auto qvec = tvec.cross(_ba);
    auto v = qvec.dot(ray.direction) / det;
    if (v < 0 || v + u > 1) return nullptr;

    auto tmp = (_ca).dot(qvec) / det;
    return (tmp > t_start && tmp < t_end) ? (hit_at = tmp, this) : nullptr;

//    auto n_dot_d = ray.direction.dot(_norm_vec);
//    if (n_dot_d < 1e-12 && n_dot_d > -1e-12)
//        return false;
//
//    auto tmp = (_v1_dot_n - ray.original.dot(_norm_vec)) / n_dot_d;
//    if (tmp > t_start && tmp < t_end)
//    {
//      auto p = ray[tmp];
//      if (_cb.cross(ray[t,-b).dot(_norm_vec) >= 0 &&
//        _ca.cross(c-p).dot(_norm_vec) >= 0 &&
//        _ba.cross(p-a).dot(_norm_vec) >= 0)
//      {
//        hit_at = tmp;
//        return this;
//      }
//    }
}

PX_CUDA_CALLABLE
Direction BaseTriangle::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseTriangle::getTextureCoord(PREC const &x, PREC const &y,
                                         PREC const &z) const
{
    return {x - _center.x,
            -_norm_vec.z*(y - _center.y) + _norm_vec.y*(z - _center.z),
            (x - _center.x)*_norm_vec.x + (y - _center.y)*_norm_vec.y + (z - _center.z)*_norm_vec.z};
}

std::shared_ptr<Geometry> Triangle::create(Point const &a,
                                               Point const &b,
                                               Point const &c,
                                               std::shared_ptr<Material> const &material,
                                               std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Triangle(a, b, c,
                                                      material, trans));
}

Triangle::Triangle(Point const &a,
                   Point const &b,
                   Point const &c,
                   std::shared_ptr<Material> const &material,
                   std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseTriangle(a, b, c, material->obj(), trans.get())),
          _base_obj(_obj),
          _a(a), _b(b), _c(c),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Triangle::~Triangle()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry * const &Triangle::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Triangle::devPtr()
{
    return _dev_ptr;
}

void Triangle::up2Gpu()
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

        GpuCreator::Triangle(_dev_ptr,
                             _a, _b, _c,
                             _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                             _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void Triangle::clearGpuData()
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

void BaseTriangle::setVertices(Point const &a, Point const &b, Point const &c)
{
    _raw_vertices[0] = a;
    _raw_vertices[1] = b;
    _raw_vertices[2] = c;

    _ba = _raw_vertices[1] - _raw_vertices[0];
    _cb = _raw_vertices[2] - _raw_vertices[1];
    _ca = _raw_vertices[2] - _raw_vertices[0];

    _norm_vec = _ba.cross(_ca);

    _v1_dot_n = _raw_vertices[0].dot(_norm_vec);

    _center = _raw_vertices[0];
    _center += _raw_vertices[1];
    _center += _raw_vertices[2];
    _center /= 3.0;
}

void Triangle::setVertices(Point const &a,
                           Point const &b,
                           Point const &c)
{
    _a = a;
    _b = b;
    _c = c;
    _obj->setVertices(a, b, c);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

