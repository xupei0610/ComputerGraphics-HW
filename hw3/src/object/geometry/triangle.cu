#include "object/geometry/triangle.hpp"

using namespace px;

BaseTriangle::BaseTriangle(Point const &a,
                           Point const &b,
                           Point const &c)
    : _dev_obj(nullptr)
{
    setVertices(a, b, c);
}

PX_CUDA_CALLABLE
GeometryObj *BaseTriangle::hitCheck(void * const &obj,
                            Ray const &ray,
                            PREC const &t_start,
                            PREC const &t_end,
                            PREC &hit_at)
{
    auto o = reinterpret_cast<BaseTriangle*>(obj);

    auto pvec = ray.direction.cross(o->_ca);
    auto det = pvec.dot(o->_ba);

    if (det < EPSILON && det > -EPSILON)
        return nullptr;

    auto tvec = ray.original - o->_a;
    auto u = tvec.dot(pvec) / det;
    if (u < 0 || u > 1) return nullptr;

    pvec = tvec.cross(o->_ba);
    auto v = pvec.dot(ray.direction) / det;
    if (v < 0 || v + u > 1) return nullptr;

    det = (o->_ca).dot(pvec) / det;
    return (det > t_start && det < t_end) ? (hit_at = det, o->_dev_obj) : nullptr;

//    auto n_dot_d = ray.direction.dot(_norm_vec);
//    if (n_dot_d < EPSILON && n_dot_d > EPSILON)
//        return nullptr;
//
//    n_dot_d = (_v1_dot_n - ray.original.dot(_norm_vec)) / n_dot_d;
//    if (n_dot_d > t_start && n_dot_d < t_end)
//    {
//      auto p = ray[n_dot_d];
//      if (_cb.cross(p - _raw_vertices[1]).dot(_norm_vec) >= 0 &&
//          _ca.cross(_raw_vertices[2]-p).dot(_norm_vec) >= 0 &&
//          _ba.cross(p-_raw_vertices[0]).dot(_norm_vec) >= 0)
//      {
//        hit_at = n_dot_d;
//        return o->_dev_obj;
//      }
//    }
//    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseTriangle::normalVec(void * const &obj,
                                  PREC const &x, PREC const &y, PREC const &z)
{
    return reinterpret_cast<BaseTriangle*>(obj)->_norm_vec;
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseTriangle::getTextureCoord(void * const &obj,
                                         PREC const &x, PREC const &y,
                                         PREC const &z)
{
    auto o = reinterpret_cast<BaseTriangle*>(obj);
    return {x - o->_center.x,
            o->_norm_vec.y*(z - o->_center.z) - o->_norm_vec.z*(y - o->_center.y) ,
            (x - o->_center.x)*o->_norm_vec.x + (y - o->_center.y)*o->_norm_vec.y + (z - o->_center.z)*o->_norm_vec.z};
}

void BaseTriangle::setVertices(Point const &a, Point const &b, Point const &c)
{
    _a = a;
    _ba = b - a;
//    _cb = c - b;
    _ca = c - a;

    _norm_vec = _ba.cross(_ca);

//    _v1_dot_n = a.dot(_norm_vec);

    _center = a;
    _center += b;
    _center += b;
    _center /= 3.0;
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
        : BaseGeometry(material, trans, 3),
          _obj(new BaseTriangle(a, b, c)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices(a, b, c);
}

Triangle::~Triangle()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_triangle = BaseTriangle::hitCheck;
__device__ fnNormal_t __fn_normal_triangle = BaseTriangle::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_triangle = BaseTriangle::getTextureCoord;
#endif
void Triangle::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseTriangle)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_triangle, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_triangle, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_triangle, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseTriangle), cudaMemcpyHostToDevice));
        _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);

        GeometryObj tmp(_gpu_obj, fn_hit_h, fn_normal_h, fn_texture_coord_h,
                        mat == nullptr ? nullptr : mat->devPtr(),
                        trans == nullptr ? nullptr : trans->devPtr());

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(GeometryObj),
                                 cudaMemcpyHostToDevice))

        _need_upload = false;
    }
#endif
}

void Triangle::clearGpuData()
{
#ifdef USE_CUDA
    BaseGeometry::clearGpuData();
    if (_gpu_obj != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_obj = nullptr;
    }
    _need_upload = true;
#endif
}

void Triangle::setVertices(Point const &a,
                           Point const &b,
                           Point const &c)
{
    _obj->setVertices(a, b, c);
    _updateVertices(a, b, c);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Triangle::_updateVertices(Point const &a,
                               Point const &b,
                               Point const &c)
{
    raw_vertices[0] = a;
    raw_vertices[1] = b;
    raw_vertices[2] = c;
}

Vec3<PREC> Triangle::getTextureCoord(PREC const &x,
                                   PREC const &y,
                                   PREC const &z) const
{
    return BaseTriangle::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry * Triangle::hitCheck(Ray const &ray,
                                        PREC const &t_start,
                                        PREC const &t_end,
                                        PREC &hit_at) const
{
    return BaseTriangle::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Triangle::normalVec(PREC const &x, PREC const &y,
                              PREC const &z) const
{
    return BaseTriangle::normalVec(_obj, x, y, z);
}
