#include "object/geometry/normal_triangle.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseNormalTriangle::BaseNormalTriangle(Point const &vertex1, Direction const &normal1,
                                       Point const &vertex2, Direction const &normal2,
                                       Point const &vertex3, Direction const &normal3,
                                       const BaseMaterial *const &material,
                                       const Transformation *const &trans)
        : BaseGeometry(material, trans, 3),
          _na(normal1), _nb(normal2), _nc(normal3)
{
    setVertices(vertex1, vertex2, vertex3);
}

#include <cassert>

PX_CUDA_CALLABLE
const BaseGeometry * BaseNormalTriangle::hitCheck(Ray const &ray,
                                  PREC const &t_start,
                                  PREC const &t_end,
                                  PREC &hit_at) const
{
    auto pvec = ray.direction.cross(_ca);
    auto det = pvec.dot(_ba);
    if (det < EPSILON && det > -EPSILON)
        return nullptr;

    auto tvec = ray.original - _raw_vertices[0];
    auto u = tvec.dot(pvec) / det;
    if (u < 0 || u > 1) return nullptr;

    pvec = tvec.cross(_ba);
    auto v = pvec.dot(ray.direction) / det;
    if (v < 0 || v + u > 1) return nullptr;

    det = (_ca).dot(pvec) / det;
    return (det > t_start && det < t_end) ? (hit_at = det, this) : nullptr;
}

PX_CUDA_CALLABLE
Direction BaseNormalTriangle::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    auto u = _cb.cross(Vec3<PREC>(x-_raw_vertices[1].x, y-_raw_vertices[1].y, z-_raw_vertices[1].z)).dot(_n)/_n_norm;
    auto v = _ca.cross(Vec3<PREC>(_raw_vertices[2].x-x, _raw_vertices[2].y-y, _raw_vertices[2].z-z)).dot(_n)/_n_norm;

    return {_na.x * u + _nb.x * v + _nc.x * (1 - u - v),
            _na.y * u + _nc.y * v + _nc.y * (1 - u - v),
            _na.z * u + _nb.z * v + _nc.z * (1 - u - v)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseNormalTriangle::getTextureCoord(PREC const &x, PREC const &y,
                                               PREC const &z) const
{
    auto u = _cb.cross(Vec3<PREC>(x-_raw_vertices[1].x, y-_raw_vertices[1].y, z-_raw_vertices[1].z)).dot(_n)/_n_norm;
    auto v = _ca.cross(Vec3<PREC>(_raw_vertices[2].x-x, _raw_vertices[2].y-y, _raw_vertices[2].z-z)).dot(_n)/_n_norm;

    Direction norm_vec(_na.x * u + _nb.x * v + _nc.x * (1 - u - v),
                       _na.y * u + _nc.y * v + _nc.y * (1 - u - v),
                       _na.z * u + _nb.z * v + _nc.z * (1 - u - v));
    return {x - _center.x,
            -norm_vec.z*(y - _center.y) + norm_vec.y*(z - _center.z),
            (x - _center.x)*norm_vec.x + (y - _center.y)*norm_vec.y + (z - _center.z)*norm_vec.z};
}

std::shared_ptr<Geometry>
NormalTriangle::create(Point const &vertex1, Direction const &normal1,
                       Point const &vertex2, Direction const &normal2,
                       Point const &vertex3, Direction const &normal3,
                       std::shared_ptr<Material> const &material,
                       std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new NormalTriangle(vertex1, normal1,
                                                            vertex2, normal2,
                                                            vertex3, normal3,
                                                            material, trans));
}

NormalTriangle::NormalTriangle(Point const &vertex1, Direction const &normal1,
                               Point const &vertex2, Direction const &normal2,
                               Point const &vertex3, Direction const &normal3,
                               std::shared_ptr<Material> const &material,
                               std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseNormalTriangle(vertex1, normal1, vertex2, normal2, vertex3, normal3,
                             material->obj(), trans.get())),
          _base_obj(_obj),
          _a(vertex1), _b(vertex2), _c(vertex3),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

NormalTriangle::~NormalTriangle()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &NormalTriangle::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **NormalTriangle::devPtr()
{
    return _dev_ptr;
}

void NormalTriangle::up2Gpu()
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

        GpuCreator::NormalTriangle(_dev_ptr, _a, _obj->_na, _b, _obj->_nb, _c, _obj->_nc,
                                   _material_ptr == nullptr ? nullptr
                                                            : _material_ptr->devPtr(),
                                   _transformation_ptr == nullptr ? nullptr
                                                                  : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void NormalTriangle::clearGpuData()
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

void BaseNormalTriangle::setVertices(Point const &a, Point const &b, Point const &c)
{
    _raw_vertices[0] = a;
    _raw_vertices[1] = b;
    _raw_vertices[2] = c;

    _ba = _raw_vertices[1] - _raw_vertices[0];
    _cb = _raw_vertices[2] - _raw_vertices[1];
    _ca = _raw_vertices[2] - _raw_vertices[0];

    _center = _raw_vertices[0];
    _center += _raw_vertices[1];
    _center += _raw_vertices[2];
    _center /= 3.0;

    auto n = _ba.cross(_ca);
    _n_norm = n.norm();
    _n = n;
}

void BaseNormalTriangle::setNormals(Direction const &na, Direction const &nb,
                                    Direction const &nc)
{
    _na = na;
    _nb = nb;
    _nc = nc;
}

void NormalTriangle::setVertices(Point const &a,
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

void NormalTriangle::setNormals(Direction const &na,
                                Direction const &nb,
                                Direction const &nc)
{
    _obj->setNormals(na, nb, nc);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}