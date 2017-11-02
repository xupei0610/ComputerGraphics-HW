#include "object/geometry/box.hpp"

using namespace px;

BaseBox::BaseBox(PREC const &x1, PREC const &x2,
                 PREC const &y1, PREC const &y2,
                 PREC const &z1, PREC const &z2)
    : _dev_obj(nullptr)
{
    setVertices(x1, x2, y1, y2, z1, z2);
}

PX_CUDA_CALLABLE
GeometryObj * BaseBox::hitCheck(void * const &obj,
                       Ray const &ray,
                       PREC const &t_start,
                       PREC const &t_end,
                       PREC &hit_at)
{
    auto o = reinterpret_cast<BaseBox*>(obj);

    auto tmin  = ((ray.direction.x < 0 ? o->_vertex_max.x : o->_vertex_min.x) - ray.original.x) / ray.direction.x;
    auto tmax  = ((ray.direction.x < 0 ? o->_vertex_min.x : o->_vertex_max.x) - ray.original.x) / ray.direction.x;
    auto tymin = ((ray.direction.y < 0 ? o->_vertex_max.y : o->_vertex_min.y) - ray.original.y) / ray.direction.y;
    auto tymax = ((ray.direction.y < 0 ? o->_vertex_min.y : o->_vertex_max.y) - ray.original.y) / ray.direction.y;

    if (tmin > tymax || tymin > tmax)
        return nullptr;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    tymin = ((ray.direction.z < 0 ? o->_vertex_max.z : o->_vertex_min.z) - ray.original.z) / ray.direction.z;
    tymax = ((ray.direction.z < 0 ? o->_vertex_min.z : o->_vertex_max.z) - ray.original.z) / ray.direction.z;

    if (tmin > tymax || tymin > tmax)
        return nullptr;

    if (tymin > tmin)
        tmin = tymin;

    if (tmin > t_start && tmin < t_end)
    {
        hit_at = tmin;
        return o->_dev_obj;
    }

    if (tymax < tmax)
        tmax = tymax;

    if (tmax > t_start && tmax < t_end)
    {
        hit_at = tmax;
        return o->_dev_obj;
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseBox::normalVec(void * const &obj,
                               PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseBox*>(obj);

    auto v1 = std::abs(x-o->_vertex_min.x);
    auto v2 = std::abs(x-o->_vertex_max.x);

    if (v1 < v2)
    {
        v2 = std::abs(y-o->_vertex_min.y);
        if (v1 < v2)
        {
            v2 = std::abs(y-o->_vertex_max.y);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_min.z);
                if (v1 < v2)
                {
                    v2 = std::abs(z-o->_vertex_max.z);
                    if (v1 < v2)
                        return {-1, 0, 0};
                    return {0, 0, 1};
                }
                v1 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {0, 0, 1};
                return {0, 0, -1};
            }
            v1 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {0, 0, -1};
                return {0, 0, 1};
            }
            v1 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {0, 0, 1};
            return {0, 1, 0};
        }
        v1 = std::abs(y-o->_vertex_max.y);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z - o->_vertex_max.z);
                if (v1 < v2)
                    return {0, 1, 0};
            }
            v1 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {0, 0, 1};
            return {0, 0, -1};
        }
        v1 =std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {0, 0, -1};
            return {0, 0, 1};
        }
        v1 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {0, 0, 1};
        return {0, -1, 0};
    }

    v1 = std::abs(y-o->_vertex_min.y);
    if (v1 < v2)
    {
        v2 = std::abs(y-o->_vertex_max.y);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {0, -1, 0};
                return {0, 0, 1};
            }
            v1 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {0, 0, 1};
            return {0, 0, -1};
        }
        v1 = std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {0, 0, -1};
            return {0, 0, 1};
        }
        v1 = std::abs(z-o->_vertex_max.z);
        if (v1 < v2)
            return {0, 0, 1};
        return {0, 1, 0};
    }
    v1 = std::abs(y-o->_vertex_max.y);
    if (v1 < v2)
    {
        v2 = std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {0, 1, 0};
        }
        v1 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {0, 0, 1};
        return {0, 0, -1};
    }
    v1 =std::abs(z-o->_vertex_min.z);
    if (v1 < v2)
    {
        v2 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {0, 0, -1};
        return {0, 0, 1};
    }
    v1 = std::abs(z - o->_vertex_max.z);
    if (v1 < v2)
        return {0, 0, 1};
    return {1, 0, 0};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseBox::getTextureCoord(void * const &obj,
                                      PREC const &x, PREC const &y,
                                      PREC const &z)
{
    auto o = reinterpret_cast<BaseBox*>(obj);

    auto v1 = std::abs(x-o->_vertex_min.x);
    auto v2 = std::abs(x-o->_vertex_max.x);
    if (v1 < v2)
    {
        v2 = std::abs(y-o->_vertex_min.y);
        if (v1 < v2)
        {
            v2 = std::abs(y-o->_vertex_max.y);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_min.z);
                if (v1 < v2)
                {
                    v2 = std::abs(z-o->_vertex_max.z);
                    if (v1 < v2)
                        return {o->_vertex_max.z - z, o->_side.z + o->_vertex_max.y - y, 0};
                    return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
                }
                v1 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
                return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
            }
            v1 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
                return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
            }
            v1 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
            return {o->_side.z + x - o->_vertex_min.x, o->_vertex_max.z - z, 0};
        }
        v1 = std::abs(y-o->_vertex_max.y);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z - o->_vertex_max.z);
                if (v1 < v2)
                    return {o->_side.z + x - o->_vertex_min.x, o->_vertex_max.z - z, 0};
            }
            v1 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
            return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
        }
        v1 =std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
            return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
        }
        v1 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
        return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_side.y + z - o->_vertex_min.z, 0};
    }

    v1 = std::abs(y-o->_vertex_min.y);
    if (v1 < v2)
    {
        v2 = std::abs(y-o->_vertex_max.y);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_min.z);
            if (v1 < v2)
            {
                v2 = std::abs(z-o->_vertex_max.z);
                if (v1 < v2)
                    return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_side.y + z - o->_vertex_min.z, 0};
                return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
            }
            v1 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
            return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
        }
        v1 = std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z-o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
            return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
        }
        v1 = std::abs(z-o->_vertex_max.z);
        if (v1 < v2)
            return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
        return {o->_side.z + x - o->_vertex_min.x, o->_vertex_max.z - z, 0};
    }
    v1 = std::abs(y-o->_vertex_max.y);
    if (v1 < v2)
    {
        v2 = std::abs(z-o->_vertex_min.z);
        if (v1 < v2)
        {
            v2 = std::abs(z - o->_vertex_max.z);
            if (v1 < v2)
                return {o->_side.z + x - o->_vertex_min.x, o->_vertex_max.z - z, 0};
        }
        v1 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
        return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
    }
    v1 =std::abs(z-o->_vertex_min.z);
    if (v1 < v2)
    {
        v2 = std::abs(z - o->_vertex_max.z);
        if (v1 < v2)
            return {o->_side.z + x - o->_vertex_min.x, o->_side.z + o->_vertex_max.y - y, 0};
        return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
    }
    v1 = std::abs(z - o->_vertex_max.z);
    if (v1 < v2)
        return {o->_side.z + o->_side.z + o->_side.z + o->_vertex_max.x - x, o->_side.z + o->_vertex_max.y - y, 0};
    return {o->_side.z + o->_side.x + z - o->_vertex_min.z, o->_side.z + o->_vertex_max.y - y, 0};
}

void BaseBox::setVertices(PREC const &x1, PREC const &x2,
                          PREC const &y1, PREC const &y2,
                          PREC const &z1, PREC const &z2)
{
    _vertex_min.x =
            x1 > x2 ? (_vertex_max.x = x1, x2) : (_vertex_max.x = x2, x1);
    _vertex_min.y =
            y1 > y2 ? (_vertex_max.y = y1, y2) : (_vertex_max.y = y2, y1);
    _vertex_min.z =
            z1 > z2 ? (_vertex_max.z = z1, z2) : (_vertex_max.z = z2, z1);

    _center.x = (_vertex_min.x + _vertex_max.x) * 0.5;
    _center.y = (_vertex_min.y + _vertex_max.y) * 0.5;
    _center.z = (_vertex_min.z + _vertex_max.z) * 0.5;

    _side.x = _vertex_max.x - _vertex_min.x;
    _side.y = _vertex_max.y - _vertex_min.y;
    _side.z = _vertex_max.z - _vertex_min.z;
}

std::shared_ptr<BaseGeometry> Box::create(PREC const &x1, PREC const &x2,
                                          PREC const &y1, PREC const &y2,
                                          PREC const &z1, PREC const &z2,
                                          std::shared_ptr<BaseMaterial> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Box(x1, x2,
                                             y1, y2,
                                             z1, z2,
                                             material, trans));
}

std::shared_ptr<BaseGeometry> Box::create(Point const &v1, Point const &v2,
                                          std::shared_ptr<BaseMaterial> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Box(v1.x, v2.x,
                                             v1.y, v2.y,
                                             v1.z, v2.z,
                                             material, trans));
}

Box::Box(PREC const &x1, PREC const &x2,
         PREC const &y1, PREC const &y2,
         PREC const &z1, PREC const &z2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseBox(x1, x2, y1, y2, z1, z2)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Box::~Box()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_box = BaseBox::hitCheck;
__device__ fnNormal_t __fn_normal_box = BaseBox::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_box = BaseBox::getTextureCoord;
#endif
void Box::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseBox)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_box, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_box, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_box, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseBox), cudaMemcpyHostToDevice));
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

void Box::clearGpuData()
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

void Box::setVertices(Point const &v1, Point const &v2)
{
    _obj->setVertices(v1.x, v2.x, v1.y, v2.y, v1.z, v2.z);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Box::setVertices(PREC const &x1, PREC const &x2,
                      PREC const &y1, PREC const &y2,
                      PREC const &z1, PREC const &z2)
{
    _obj->setVertices(x1, x2, y1, y2, z1, z2);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Box::_updateVertices()
{
    raw_vertices[0].x = _obj->_vertex_min.x;
    raw_vertices[0].y = _obj->_vertex_min.y;
    raw_vertices[0].z = _obj->_vertex_min.z;

    raw_vertices[1].x = _obj->_vertex_max.x;
    raw_vertices[1].y = _obj->_vertex_min.y;
    raw_vertices[1].z = _obj->_vertex_min.z;

    raw_vertices[2].x = _obj->_vertex_min.x;
    raw_vertices[2].y = _obj->_vertex_max.y;
    raw_vertices[2].z = _obj->_vertex_min.z;

    raw_vertices[3].x = _obj->_vertex_min.x;
    raw_vertices[3].y = _obj->_vertex_min.y;
    raw_vertices[3].z = _obj->_vertex_max.z;

    raw_vertices[4].x = _obj->_vertex_max.x;
    raw_vertices[4].y = _obj->_vertex_max.y;
    raw_vertices[4].z = _obj->_vertex_min.z;

    raw_vertices[5].x = _obj->_vertex_max.x;
    raw_vertices[5].y = _obj->_vertex_min.y;
    raw_vertices[5].z = _obj->_vertex_max.z;

    raw_vertices[6].x = _obj->_vertex_min.x;
    raw_vertices[6].y = _obj->_vertex_max.y;
    raw_vertices[6].z = _obj->_vertex_max.z;

    raw_vertices[7].x = _obj->_vertex_max.x;
    raw_vertices[7].y = _obj->_vertex_max.y;
    raw_vertices[7].z = _obj->_vertex_max.z;
}

Vec3<PREC> Box::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseBox::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Box::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseBox::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Box::normalVec(PREC const &x, PREC const &y,
                           PREC const &z) const
{
    return BaseBox::normalVec(_obj, x, y, z);
}
