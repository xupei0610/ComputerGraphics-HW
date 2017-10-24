#include "object/geometry/box.hpp"

using namespace px;

BaseBox::BaseBox(const BaseMaterial *const &material,
                 const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BaseBox::hitCheck(Ray const &ray,
                                       double const &t_start,
                                       double const &t_end,
                                       double &hit_at) const
{
    auto tmin  = ((ray.direction.x < 0 ? _vertex_max.x : _vertex_min.x) - ray.original.x) / ray.direction.x;
    auto tmax  = ((ray.direction.x < 0 ? _vertex_min.x : _vertex_max.x) - ray.original.x) / ray.direction.x;
    auto tymin = ((ray.direction.y < 0 ? _vertex_max.y : _vertex_min.y) - ray.original.y) / ray.direction.y;
    auto tymax = ((ray.direction.y < 0 ? _vertex_min.y : _vertex_max.y) - ray.original.y) / ray.direction.y;

    if (tmin > tymax || tymin > tmax)
        return nullptr;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    auto tzmin = ((ray.direction.z < 0 ? _vertex_max.z : _vertex_min.z) - ray.original.z) / ray.direction.z;
    auto tzmax = ((ray.direction.z < 0 ? _vertex_min.z : _vertex_max.z) - ray.original.z) / ray.direction.z;

    if (tmin > tzmax || tzmin > tmax)
        return nullptr;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tmin > t_start && tmin < t_end)
    {
        hit_at = tmin;
        return this;
    }

    if (tzmax < tmax)
        tmax = tzmax;

    if (tmax > t_start && tmax < t_end)
    {
        hit_at = tmax;
        return this;
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseBox::normalVec(double const &x, double const &y, double const &z) const
{
    if (std::abs(x-_vertex_min.x) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return {-1, 0, 0};
        }
    }
    else if (std::abs(x-_vertex_max.x) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return {1, 0, 0};
        }
    }
    else if (std::abs(y-_vertex_min.y) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {0, -1, 0};
        }
    }
    else if (std::abs(y-_vertex_max.y) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {0, 1, 0};
        }
    }
    else if (std::abs(z-_vertex_min.z) < 1e-12)
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {0, 0, -1};
        }
    }
    else if (std::abs(z-_vertex_max.z) < 1e-12)
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {0, 0, 1};
        }
    }
    return {x-_center.x, y-_center.y, z-_center.z}; // Undefined action
}

PX_CUDA_CALLABLE
Vec3<double> BaseBox::getTextureCoord(double const &x, double const &y,
                                            double const &z) const
{
    if (std::abs(x-_vertex_min.x) < 1e-12) // left side
    {
        if (!(z <= _vertex_min.z || z > _vertex_max.z || y < _vertex_min.y || y > _vertex_max.y))
        {
            return {_vertex_max.z - z, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(x-_vertex_max.x) < 1e-12) // right side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return {_side.z + _side.x + z - _vertex_min.z, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(y-_vertex_min.y) < 1e-12) // bottom side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _side.z + _side.y + z - _vertex_min.z, 0};
        }
    }
    else if (std::abs(y-_vertex_max.y) < 1e-12) // top side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _vertex_max.z - z, 0};
        }
    }
    else if (std::abs(z-_vertex_min.z) < 1e-12) // forward side
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(z-_vertex_max.z) < 1e-12) // backward side
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + _side.z + _side.z + _vertex_max.x - x, _side.z + _vertex_max.y - y, 0};
        }
    }
    return {x-_center.x, y-_center.y, z-_center.z}; // Undefined action
}

std::shared_ptr<BaseGeometry> Box::create(double const &x1, double const &x2,
                                          double const &y1, double const &y2,
                                          double const &z1, double const &z2,
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
    return std::shared_ptr<BaseGeometry>(new Box(v1, v2,
                                                 material, trans));
}

Box::Box(double const &x1, double const &x2,
         double const &y1, double const &y2,
         double const &z1, double const &z2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans)
        : BaseBox(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setVertices(x1, x2, y1, y2, z1, z2);
}

Box::Box(Point const &v1, Point const &v2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans)
        : BaseBox(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setVertices(v1, v2);
}


Box::~Box()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Box::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseBox)));

        _material = _material_ptr == nullptr ? nullptr : _material_ptr->up2Gpu();
        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseBox*>(this),
                                 sizeof(BaseBox),
                                 cudaMemcpyHostToDevice));

        _material = _material_ptr.get();
        _transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void Box::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_transformation_ptr.use_count() == 1)
            _transformation_ptr->clearGpuData();
    if (_material_ptr.use_count() == 1)
        _material_ptr->clearGpuData();

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void Box::setVertices(double const &x1, double const &x2,
                      double const &y1, double const &y2,
                      double const &z1, double const &z2)
{
    _vertex_min.x = x1 > x2 ? (_vertex_max.x = x1, x2) : (_vertex_max.x = x2, x1);
    _vertex_min.y = y1 > y2 ? (_vertex_max.y = y1, y2) : (_vertex_max.y = y2, y1);
    _vertex_min.z = z1 > z2 ? (_vertex_max.z = z1, z2) : (_vertex_max.z = z2, z1);

    _center.x = (_vertex_min.x + _vertex_max.x) * 0.5;
    _center.y = (_vertex_min.y + _vertex_max.y) * 0.5;
    _center.z = (_vertex_min.z + _vertex_max.z) * 0.5;

    _side.x = _vertex_max.x - _vertex_min.x;
    _side.y = _vertex_max.y - _vertex_min.y;
    _side.z = _vertex_max.z - _vertex_min.z;

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
