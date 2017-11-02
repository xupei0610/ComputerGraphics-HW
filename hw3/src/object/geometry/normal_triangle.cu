#include "object/geometry/normal_triangle.hpp"

using namespace px;

BaseNormalTriangle::BaseNormalTriangle(Point const &vertex1, Direction const &normal1,
                                       Point const &vertex2, Direction const &normal2,
                                       Point const &vertex3, Direction const &normal3)
        : _na(normal1), _nb(normal2), _nc(normal3), _dev_obj(nullptr)
{
    setVertices(vertex1, vertex2, vertex3);
}

PX_CUDA_CALLABLE
GeometryObj *BaseNormalTriangle::hitCheck(void * const &obj,
                                    Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at)
{
    auto o = reinterpret_cast<BaseNormalTriangle*>(obj);

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

    det = o->_ca.dot(pvec) / det;
    return (det > t_start && det < t_end) ? (hit_at = det, o->_dev_obj) : nullptr;
}

PX_CUDA_CALLABLE
Direction BaseNormalTriangle::normalVec(void * const &obj,
                                        PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseNormalTriangle*>(obj);
    auto u = o->_cb.cross(Vec3<PREC>(x-o->_b.x, y-o->_b.y, z-o->_b.z)).norm()/o->_n_norm;
    auto v = o->_ca.cross(Vec3<PREC>(o->_c.x-x, o->_c.y-y, o->_c.z-z)).norm()/o->_n_norm;

    return {o->_na.x * u + o->_nb.x * v + o->_nc.x * (1 - u - v),
            o->_na.y * u + o->_nc.y * v + o->_nc.y * (1 - u - v),
            o->_na.z * u + o->_nb.z * v + o->_nc.z * (1 - u - v)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseNormalTriangle::getTextureCoord(void * const &obj,
                                               PREC const &x, PREC const &y,
                                               PREC const &z)
{
    auto o = reinterpret_cast<BaseNormalTriangle*>(obj);

    auto u = o->_cb.cross(Vec3<PREC>(x-o->_b.x, y-o->_b.y, z-o->_b.z)).norm()/o->_n_norm;
    auto v = o->_ca.cross(Vec3<PREC>(o->_c.x-x, o->_c.y-y, o->_c.z-z)).norm()/o->_n_norm;

    Direction norm_vec(o->_na.x * u + o->_nb.x * v + o->_nc.x * (1 - u - v),
                       o->_na.y * u + o->_nc.y * v + o->_nc.y * (1 - u - v),
                       o->_na.z * u + o->_nb.z * v + o->_nc.z * (1 - u - v));
    return {x - o->_center.x,
            norm_vec.y*(z - o->_center.z) - norm_vec.z*(y - o->_center.y),
            (x - o->_center.x)*norm_vec.x + (y - o->_center.y)*norm_vec.y + (z - o->_center.z)*norm_vec.z};
}

void BaseNormalTriangle::setVertices(Point const &a, Point const &b, Point const &c)
{
    _a = a;
    _b = b;
    _c = c;

    _ba = _b - _a;
    _cb = _c - _b;
    _ca = _c - _a;

    _center = _a;
    _center += _b;
    _center += _c;
    _center /= 3.0;

    _n_norm = _ba.cross(_ca).norm();
}

void BaseNormalTriangle::setNormals(Direction const &na, Direction const &nb,
                                    Direction const &nc)
{
    _na = na;
    _nb = nb;
    _nc = nc;
}

std::shared_ptr<BaseGeometry>
NormalTriangle::create(Point const &vertex1, Direction const &normal1,
                       Point const &vertex2, Direction const &normal2,
                       Point const &vertex3, Direction const &normal3,
                       std::shared_ptr<BaseMaterial> const &material,
                       std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new NormalTriangle(vertex1, normal1,
                                                            vertex2, normal2,
                                                            vertex3, normal3,
                                                            material, trans));
}

NormalTriangle::NormalTriangle(Point const &vertex1, Direction const &normal1,
                               Point const &vertex2, Direction const &normal2,
                               Point const &vertex3, Direction const &normal3,
                               std::shared_ptr<BaseMaterial> const &material,
                               std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 3),
          _obj(new BaseNormalTriangle(vertex1, normal1, vertex2, normal2, vertex3, normal3)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

NormalTriangle::~NormalTriangle()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_normal_triangle = BaseNormalTriangle::hitCheck;
__device__ fnNormal_t __fn_normal_normal_triangle = BaseNormalTriangle::normalVec;
__device__ fnTextureCoord_t __fn_texture_normal_coord_triangle = BaseNormalTriangle::getTextureCoord;
#endif
void NormalTriangle::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseNormalTriangle)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_normal_triangle, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_normal_triangle, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_normal_coord_triangle, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseNormalTriangle), cudaMemcpyHostToDevice));
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

void NormalTriangle::clearGpuData()
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

void NormalTriangle::setVertices(Point const &a, Point const &b, Point const &c)
{
    _obj->setVertices(a, b, c);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void NormalTriangle::setNormals(Direction const &na, Direction const &nb,
                                    Direction const &nc)
{
    _obj->setNormals(na, nb, nc);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void NormalTriangle::_updateVertices()
{
    raw_vertices[0] = _obj->_a;
    raw_vertices[1] = _obj->_b;
    raw_vertices[2] = _obj->_c;
}

Vec3<PREC> NormalTriangle::getTextureCoord(PREC const &x,
                                    PREC const &y,
                                    PREC const &z) const
{
    return BaseNormalTriangle::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry * NormalTriangle::hitCheck(Ray const &ray,
                                        PREC const &t_start,
                                        PREC const &t_end,
                                        PREC &hit_at) const
{
    return BaseNormalTriangle::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction NormalTriangle::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return BaseNormalTriangle::normalVec(_obj, x, y, z);
}