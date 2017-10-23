#include "object/geometry/triangle.hpp"

using namespace px;

BaseTriangle::BaseTriangle(const BaseMaterial *const &material,
                           const Transformation *const &trans)
        : BaseGeometry(material, trans, 3)
{}

PX_CUDA_CALLABLE
BaseGeometry * BaseTriangle::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at)
{
    auto pvec = ray.direction.cross(_ca);
    auto det = pvec.dot(_ba);
    if (det < 1e-12 && det > -1e-12)
        return nullptr;

    auto tvec = ray.original - _raw_vertices[0];
    auto u = tvec.dot(pvec) / det;
    if (u < 0 || u > 1) return nullptr;

    auto qvec = tvec.cross(_ba);
    auto v = qvec.dot(ray.direction) / det;
    if (v < 0 || v + u > 1) return nullptr;

    auto tmp = (_ca).dot(qvec) / det;
    if (tmp > t_start && tmp < t_end)
    {
        hit_at = tmp;
        return this;
    }

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
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseTriangle::normalVec(double const &x, double const &y, double const &z)
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<double> BaseTriangle::getTextureCoord(double const &x, double const &y,
                                       double const &z)
{
    return {x - _center.x,
            -_norm_vec.z*(y - _center.y) + _norm_vec.y*(z - _center.z),
            (x - _center.x)*_norm_vec.x + (y - _center.y)*_norm_vec.y + (z - _center.z)*_norm_vec.z};
}

std::shared_ptr<BaseGeometry> Triangle::create(Point const &a,
                                               Point const &b,
                                               Point const &c,
                                               std::shared_ptr<BaseMaterial> const &material,
                                               std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Triangle(a, b, c,
                                                      material, trans));
}

Triangle::Triangle(Point const &a,
                   Point const &b,
                   Point const &c,
                   std::shared_ptr<BaseMaterial> const &material,
                   std::shared_ptr<Transformation> const &trans)
        : BaseTriangle(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setVertices(a, b, c);
}

Triangle::~Triangle()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Triangle::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseTriangle)));

        material = _material_ptr->up2Gpu();
        transformation = _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseTriangle*>(this),
                                 sizeof(BaseTriangle),
                                 cudaMemcpyHostToDevice));

        material = _material_ptr.get();
        transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void Triangle::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void Triangle::setVertices(Point const &a, Point const &b, Point const &c)
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

#ifdef USE_CUDA
    _need_upload = true;
#endif
}
