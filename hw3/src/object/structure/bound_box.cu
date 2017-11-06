#include "object/structure/bound_box.hpp"

using namespace px;

BaseBoundBox::BaseBoundBox(Point const &vertex_min, Point const &vertex_max)
    : _vertex_min(vertex_min), _vertex_max(vertex_max)
{}

PX_CUDA_CALLABLE
bool BaseBoundBox::hitBox(Point const &vertex_min,
                          Point const &vertex_max,
                          Ray const &ray,
                          PREC const &t_start,
                          PREC const &t_end)
{
    if (ray.original.x > vertex_min.x && ray.original.x < vertex_max.x &&
        ray.original.y > vertex_min.y && ray.original.y < vertex_max.y &&
        ray.original.z > vertex_min.z && ray.original.z < vertex_max.z)
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
GeometryObj * BaseBoundBox::hitCheck(void * const &obj,
                            Ray const &ray,
                            PREC const &t_start,
                            PREC const &t_end,
                            PREC &hit_at)
{
    auto o = reinterpret_cast<BaseBoundBox*>(obj);

    if (BaseBoundBox::hitBox(o->_vertex_min, o->_vertex_max, ray, t_start, t_end))
    {
        GeometryObj *obj = nullptr;

        PREC end_range = t_end;

        for (auto i = 0; i < o->_n; ++i)
        {
            auto tmp = o->_geos[i]->hit(ray, t_start, end_range, hit_at);
            if (tmp != nullptr)
            {
                end_range = hit_at;
                obj = tmp;
            }
        }
        return obj == nullptr ? nullptr : (hit_at = end_range, obj);
    }
    return nullptr;
}

BoundBox::BoundBox(std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(nullptr, trans, 8),
          _gpu_obj(nullptr), _gpu_geos(nullptr), _need_upload(true)
{}

BoundBox::~BoundBox()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_bound_box = BaseBoundBox::hitCheck;
#endif
void BoundBox::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseBoundBox)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }
        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_bound_box, sizeof(fnHit_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();
        if (trans != nullptr)
            trans->up2Gpu();

        auto count = 0;
        for (const auto &g : _geos)
        {
            if (g == nullptr)
                continue;
            g->up2Gpu();
            ++count;
        }

        BaseBoundBox bb(_vertex_min, _vertex_max);

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

        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, &bb, sizeof(BaseBoundBox), cudaMemcpyHostToDevice));

        GeometryObj tmp(_gpu_obj, fn_hit_h, fn_normal_h, fn_texture_coord_h,
                        mat == nullptr ? nullptr : mat->devPtr(),
                        trans == nullptr ? nullptr : trans->devPtr());

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(GeometryObj),
                                 cudaMemcpyHostToDevice))

        _need_upload = false;
    }
#endif
}

void BoundBox::clearGpuData()
{
#ifdef USE_CUDA
    BaseGeometry::clearGpuData();
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

void BoundBox::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _geos.push_back(obj);

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

    raw_vertices[0].x = _vertex_min.x;
    raw_vertices[0].y = _vertex_min.y;
    raw_vertices[0].z = _vertex_min.z;

    raw_vertices[1].x = _vertex_max.x;
    raw_vertices[1].y = _vertex_min.y;
    raw_vertices[1].z = _vertex_min.z;

    raw_vertices[2].x = _vertex_min.x;
    raw_vertices[2].y = _vertex_max.y;
    raw_vertices[2].z = _vertex_min.z;

    raw_vertices[3].x = _vertex_min.x;
    raw_vertices[3].y = _vertex_min.y;
    raw_vertices[3].z = _vertex_max.z;

    raw_vertices[4].x = _vertex_max.x;
    raw_vertices[4].y = _vertex_max.y;
    raw_vertices[4].z = _vertex_min.z;

    raw_vertices[5].x = _vertex_max.x;
    raw_vertices[5].y = _vertex_min.y;
    raw_vertices[5].z = _vertex_max.z;

    raw_vertices[6].x = _vertex_min.x;
    raw_vertices[6].y = _vertex_max.y;
    raw_vertices[6].z = _vertex_max.z;

    raw_vertices[7].x = _vertex_max.x;
    raw_vertices[7].y = _vertex_max.y;
    raw_vertices[7].z = _vertex_max.z;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

const BaseGeometry *BoundBox::hitCheck(Ray const &ray,
                                      PREC const &t_start,
                                      PREC const &t_end,
                                      PREC &hit_at) const
{
    if (BaseBoundBox::hitBox(_vertex_min, _vertex_max, ray, t_start, t_end))
    {
        const BaseGeometry *obj = nullptr, *tmp;

        PREC end_range = t_end;
        for (const auto &g : _geos)
        {
            tmp = g->hit(ray, t_start, end_range, hit_at);

            if (tmp == nullptr)
                continue;

            end_range = hit_at;
            obj = tmp;
        }
        return obj;
    }
    return nullptr;
}
