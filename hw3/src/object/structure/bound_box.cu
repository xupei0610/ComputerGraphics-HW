#include "object/structure/bound_box.hpp"

#include "gpu_creator.hpp"

using namespace px;

PX_CUDA_CALLABLE
BaseBoundBox::BaseBoundBox(const Transformation *const &trans)
        : BaseGeometry(nullptr, trans, 8)
{}

PX_CUDA_CALLABLE
bool BaseBoundBox::hitBox(Ray const &ray,
                          PREC const &t_start,
                          PREC const &t_end) const
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
                                 PREC const &t_start,
                                 PREC const &t_end,
                                 PREC &hit_at) const
{
    if (hitBox(ray, t_start, t_end))
    {
        const BaseGeometry *obj = nullptr, *tmp;

        PREC end_range = t_end;
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

PX_CUDA_CALLABLE
Direction BaseBoundBox::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseBoundBox::getTextureCoord(PREC const &x, PREC const &y,
                                           PREC const &z) const
{
    return {};
}

BoundBox::BoundBox(std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseBoundBox(trans.get())),
          _base_obj(_obj),
          _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

BoundBox::~BoundBox()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &BoundBox::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **BoundBox::devPtr()
{
    return _dev_ptr;
}

void BoundBox::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseGeometry**)));

        for (auto &o : _objects_ptr)
            o->up2Gpu();

        if (_transformation_ptr != nullptr)
            _transformation_ptr->up2Gpu();

        cudaDeviceSynchronize();

        auto i = 0;
        BaseGeometry **gpu_objs[_obj->_objects.n];
        for (auto &o : _objects_ptr)
            gpu_objs[i++] = o->devPtr();

        BaseGeometry ***tmp;

        PX_CUDA_CHECK(cudaMalloc(&tmp, sizeof(BaseGeometry **) * _obj->_objects.n));
        PX_CUDA_CHECK(cudaMemcpy(tmp, gpu_objs, sizeof(BaseGeometry **) * _obj->_objects.n,
                           cudaMemcpyHostToDevice));

        GpuCreator::BoundBox(_dev_ptr, tmp, _obj->_objects.n,
                             _transformation_ptr == nullptr ? nullptr
                                                            : _transformation_ptr->devPtr());

        _need_upload = false;
    }
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

    GpuCreator::destroy(_dev_ptr);
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void BoundBox::addObj(std::shared_ptr<Geometry> const &obj)
{
    if (obj == nullptr)
        return;

    _objects_ptr.insert(obj);
    _obj->addObj(obj->obj());

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void BaseBoundBox::addObj(BaseGeometry *const &obj)
{
    _objects.add(obj);

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
}
