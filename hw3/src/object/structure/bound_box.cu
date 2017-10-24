#include "object/structure/bound_box.hpp"

using namespace px;

BaseBoundBox::BaseBoundBox(const Transformation *const &trans)
        : Structure(nullptr, trans, 8)
{}

PX_CUDA_CALLABLE
bool BaseBoundBox::hitBox(Ray const &ray,
                          double const &t_start,
                          double const &t_end) const
{

    auto tmin  = ((ray.direction.x < 0 ? _vertex_max.x : _vertex_min.x) - ray.original.x) / ray.direction.x;
    auto tmax  = ((ray.direction.x < 0 ? _vertex_min.x : _vertex_max.x) - ray.original.x) / ray.direction.x;
    auto tymin = ((ray.direction.y < 0 ? _vertex_max.y : _vertex_min.y) - ray.original.y) / ray.direction.y;
    auto tymax = ((ray.direction.y < 0 ? _vertex_min.y : _vertex_max.y) - ray.original.y) / ray.direction.y;

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    auto tzmin = ((ray.direction.z < 0 ? _vertex_max.z : _vertex_min.z) - ray.original.z) / ray.direction.z;
    auto tzmax = ((ray.direction.z < 0 ? _vertex_min.z : _vertex_max.z) - ray.original.z) / ray.direction.z;

    if (tmin > tzmax || tzmin > tmax)
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tmin < t_start || tmin < t_end)
    {
        if (tzmax < tmax)
            tmax = tzmax;

        if (tmax < t_start || tmax > t_end)
            return false;
    }

    return true;
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseBoundBox::hitCheck(Ray const &ray,
                                 double const &t_start,
                                 double const &t_end,
                                 double &hit_at) const
{
    if (hitBox(ray, t_start, t_end))
    {
        const BaseGeometry *obj = nullptr, *tmp;

        double end_range = t_end;
        for (auto i = 0; i < _n_objs; ++i)
        {
            tmp = _objs[i]->hit(ray, t_start, end_range, hit_at);

            if (tmp == nullptr)
                continue;

            end_range = hit_at;
            obj = tmp;
        }

        return obj == nullptr ? nullptr : (hit_at = end_range, obj);
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseBoundBox::normalVec(double const &x, double const &y, double const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<double> BaseBoundBox::getTextureCoord(double const &x, double const &y,
                                           double const &z) const
{
    return {};
}

BoundBox::BoundBox(std::shared_ptr<Transformation> const &trans)
        : BaseBoundBox(trans.get()),
          _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

BoundBox::~BoundBox()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *BoundBox::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseBoundBox)));

        auto i = 0;
        BaseGeometry *gpu_objs[_n_objs];
        for (auto &o : _objects_ptr)
            gpu_objs[i++] = o->up2Gpu();

        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMalloc(&_objs, sizeof(BaseGeometry*)*_n_objs));
        PX_CUDA_CHECK(cudaMemcpy(_objs, gpu_objs,
                                 sizeof(BaseGeometry*)*_n_objs,
                                 cudaMemcpyHostToDevice));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseBoundBox*>(this),
                                 sizeof(BaseBoundBox),
                                 cudaMemcpyHostToDevice));

        _transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void BoundBox::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    for (auto &o : _objects_ptr)
    {
        if (o.use_count() == 1)
            o->clearGpuData();
    }

    PX_CUDA_CHECK(cudaFree(_objs));
    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _n_objs = 0;
    _objs = nullptr;
    _need_upload = true;
#endif
}

void BoundBox::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _objects_ptr.insert(obj);
    _objects.add(obj.get());
    _n_objs++;
    int n_vert;
    auto vert = obj->rawVertices(n_vert);

#define SET_VERT(v)                                     \
        if (v.x < _vertex_min.x) _vertex_min.x = v.x;   \
        if (v.x > _vertex_max.x) _vertex_max.x = v.x;   \
        if (v.y < _vertex_min.y) _vertex_min.y = v.y;   \
        if (v.y > _vertex_max.y) _vertex_max.y = v.y;   \
        if (v.z < _vertex_min.z) _vertex_min.z = v.z;   \
        if (v.z > _vertex_max.z) _vertex_max.z = v.z;

    for (auto i = 0; i < n_vert; ++i)
    {
        if (transform() == nullptr)
        {
            SET_VERT(vert[i])
        }
        else
        {
            auto v = obj->transform()->point(vert[i]);
            SET_VERT(v)
        }
    }

    _raw_vertices[0].x = _vertex_min.x;
    _raw_vertices[0].y = _vertex_min.y;
    _raw_vertices[0].z = _vertex_min.z;

    _raw_vertices[1].x = _vertex_max.x;
    _raw_vertices[1].y = _vertex_min.y;
    _raw_vertices[1].z = _vertex_min.z;

    _raw_vertices[2].x = _vertex_min.x;
    _raw_vertices[2].y = _vertex_max.y;
    _raw_vertices[2].z = _vertex_min.z;

    _raw_vertices[3].x = _vertex_min.x;
    _raw_vertices[3].y = _vertex_min.y;
    _raw_vertices[3].z = _vertex_max.z;

    _raw_vertices[4].x = _vertex_max.x;
    _raw_vertices[4].y = _vertex_max.y;
    _raw_vertices[4].z = _vertex_min.z;

    _raw_vertices[5].x = _vertex_max.x;
    _raw_vertices[5].y = _vertex_min.y;
    _raw_vertices[5].z = _vertex_max.z;

    _raw_vertices[6].x = _vertex_min.x;
    _raw_vertices[6].y = _vertex_max.y;
    _raw_vertices[6].z = _vertex_max.z;

    _raw_vertices[7].x = _vertex_max.x;
    _raw_vertices[7].y = _vertex_max.y;
    _raw_vertices[7].z = _vertex_max.z;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
const BaseGeometry * BoundBox::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at) const
{
    if (hitBox(ray, t_start, t_end))
    {
        const BaseGeometry *obj = nullptr, *tmp;

        double end_range = t_end;
        auto node = _objects.start;
        while (node != nullptr)
        {
            tmp = node->data->hit(ray, t_start, end_range, hit_at);

            if (tmp == nullptr)
                continue;

            end_range = hit_at;
            obj = tmp;

            node = node->next;
        }

        return obj == nullptr ? nullptr : (hit_at = end_range, obj);
    }

    return nullptr;
}
