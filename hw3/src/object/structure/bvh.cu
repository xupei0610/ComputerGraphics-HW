#include "object/structure/bvh.hpp"

using namespace px;

BaseBVH::BaseBVH(Point const &vertex_min, Point const &vertex_max)
        : _vertex_min(vertex_min), _vertex_max(vertex_max)
{}

PX_CUDA_CALLABLE
bool BaseBVH::hitBox(Point const &vertex_min,
                          Point const &vertex_max,
                          Ray const &ray,
                          PREC const &t_start,
                          PREC const &t_end)
{
    if (ray.original.x > vertex_min.x-DOUBLE_EPSILON && ray.original.x < vertex_max.x+DOUBLE_EPSILON &&
        ray.original.y > vertex_min.y-DOUBLE_EPSILON && ray.original.y < vertex_max.y+DOUBLE_EPSILON &&
        ray.original.z > vertex_min.z-DOUBLE_EPSILON && ray.original.z < vertex_max.z+DOUBLE_EPSILON)
        return true;

    auto tmin  = ((ray.direction.x < 0 ? vertex_max.x : vertex_min.x) - ray.original.x) / ray.direction.x;
    auto tmax  = ((ray.direction.x < 0 ? vertex_min.x : vertex_max.x) - ray.original.x) / ray.direction.x;
    auto tymin = ((ray.direction.y < 0 ? vertex_max.y : vertex_min.y) - ray.original.y) / ray.direction.y;
    auto tymax = ((ray.direction.y < 0 ? vertex_min.y : vertex_max.y) - ray.original.y) / ray.direction.y;

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    tymin = ((ray.direction.z < 0 ? vertex_max.z : vertex_min.z) - ray.original.z) / ray.direction.z;
    tymax = ((ray.direction.z < 0 ? vertex_min.z : vertex_max.z) - ray.original.z) / ray.direction.z;

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tmin > t_start && tmin < t_end)
        return true;

    if (tymax < tmax)
        tmax = tymax;

    if (tmax > t_start && tmax < t_end)
        return true;

    return false;
}

PX_CUDA_CALLABLE
GeometryObj * BaseBVH::hit(Ray const &ray,
                           PREC const &t_start,
                           PREC const &t_end,
                           Point &intersect) const
{
    if (BaseBVH::hitBox(_vertex_min, _vertex_max, ray, t_start, t_end))
    {
        GeometryObj *obj = nullptr;

        PREC end_range = t_end, hit_at;

        for (auto i = 0; i < _n; ++i)
        {
            auto tmp = _geos[i]->hit(ray, t_start, end_range, hit_at);
            if (tmp != nullptr)
            {
                end_range = hit_at;
                obj = tmp;
            }
        }
        if (obj)
        {
            intersect = ray.direction;
            intersect *= end_range;
            intersect += ray.original;
            return obj;
        }
    }
    return nullptr;
}

const BaseGeometry *BVH::hit(Ray const &ray,
                             PREC const &t_start,
                             PREC const &t_end,
                             Point &intersect) const
{
    if (BaseBVH::hitBox(_vertex_min, _vertex_max, ray, t_start, t_end))
    {
        const BaseGeometry *obj = nullptr, *tmp;

        PREC end_range = t_end, hit_at;
        for (const auto &g : _geos)
        {
            tmp = g->hit(ray, t_start, end_range, hit_at);
            if (tmp != nullptr)
            {
                end_range = hit_at;
                obj = tmp;
            }
        }
        if (obj)
        {
            intersect = ray.direction;
            intersect *= end_range;
            intersect += ray.original;
            return obj;
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
bool BaseBVH::hit(Ray const &ray,
                  PREC const &t_start,
                  PREC const &t_end) const
{
    if (BaseBVH::hitBox(_vertex_min, _vertex_max, ray, t_start, t_end))
    {
        PREC hit_at;
        for (auto i = 0; i < _n; ++i)
        {
            if (_geos[i]->hit(ray, t_start, t_end, hit_at))
                return true;
        }
    }
    return false;
}

bool BVH::hit(Ray const &ray,
                  PREC const &t_start,
                  PREC const &t_end) const
{
    if (BaseBVH::hitBox(_vertex_min, _vertex_max, ray, t_start, t_end))
    {
        PREC t;
        for (const auto &g : _geos)
        {
            if (g->hit(ray, t_start, t_end, t))
                return true;
        }
    }
    return false;
}

BVH::BVH()
        : _gpu_obj(nullptr), _gpu_geos(nullptr), _need_upload(true)
{}

BVH::~BVH()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

void BVH::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_gpu_obj == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseBVH)));
        }

        auto count = 0;
        for (const auto &g : _geos)
        {
            if (g == nullptr)
                continue;
            g->up2Gpu();
            ++count;
        }

        BaseBVH bb(_vertex_min, _vertex_max);

        GeometryObj *ptr[count];
        bb._n = count;
        for (const auto &g : _geos)
        {
            if (g == nullptr)
                continue;
            ptr[--count] = g->devPtr();
            if (count == 0)
                break;
        }
        if (_gpu_geos != nullptr)
        {
            PX_CUDA_CHECK(cudaFree(_gpu_geos));
            _gpu_geos = nullptr;
        }
        PX_CUDA_CHECK(cudaMalloc(&_gpu_geos, sizeof(GeometryObj*)*bb._n));
        PX_CUDA_CHECK(cudaMemcpy(_gpu_geos, ptr, sizeof(GeometryObj*)*bb._n, cudaMemcpyHostToDevice));
        bb._geos = _gpu_geos;

        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, &bb, sizeof(BaseBVH), cudaMemcpyHostToDevice));

        _need_upload = false;
    }
#endif
}

void BVH::clearGpuData()
{
#ifdef USE_CUDA
    if (_gpu_obj != nullptr)
    {
        for (const auto &g : _geos)
        {
            if (g.use_count() == 1)
                g->clearGpuData();
        }
        PX_CUDA_CHECK(cudaFree(_gpu_geos));
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_geos = nullptr;
        _gpu_obj = nullptr;
    }
    _need_upload = true;
#endif
}

void BVH::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _geos.emplace(obj);

    int n_vert;
    auto vert = obj->rawVertices(n_vert);

    if (_geos.size() == 1)
    {
        _vertex_min.x = vert[0].x;
        _vertex_max.x = vert[0].x;
        _vertex_min.y = vert[0].y;
        _vertex_max.y = vert[0].y;
        _vertex_min.z = vert[0].z;
        _vertex_max.z = vert[0].z;
    }

#define SET_VERT(v)                                     \
        if (v.x < _vertex_min.x) _vertex_min.x = v.x;   \
        if (v.x > _vertex_max.x) _vertex_max.x = v.x;   \
        if (v.y < _vertex_min.y) _vertex_min.y = v.y;   \
        if (v.y > _vertex_max.y) _vertex_max.y = v.y;   \
        if (v.z < _vertex_min.z) _vertex_min.z = v.z;   \
        if (v.z > _vertex_max.z) _vertex_max.z = v.z;

    if (obj->transform() == nullptr)
    {
        for (auto i = 0; i < n_vert; ++i)
        {
            SET_VERT(vert[i])
        }
    }
    else
    {
        for (auto i = 0; i < n_vert; ++i)
        {
            auto v = obj->transform()->pointFromObjCoord(vert[i]);
            SET_VERT(v)
        }
    }

#ifdef USE_CUDA
    _need_upload = true;
#endif
}