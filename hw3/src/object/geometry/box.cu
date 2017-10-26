#include "object/geometry/box.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseBox::BaseBox(PREC const &x1, PREC const &x2,
                 PREC const &y1, PREC const &y2,
                 PREC const &z1, PREC const &z2,
                 const BaseMaterial *const &material,
                 const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{
    setVertices(x1, x2, y1, y2, z1, z2);
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseBox::hitCheck(Ray const &ray,
                                       PREC const &t_start,
                                       PREC const &t_end,
                                       PREC &hit_at) const
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
Direction BaseBox::normalVec(PREC const &x, PREC const &y, PREC const &z) const
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
Vec3<PREC> BaseBox::getTextureCoord(PREC const &x, PREC const &y,
                                            PREC const &z) const
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

std::shared_ptr<Geometry> Box::create(PREC const &x1, PREC const &x2,
                                          PREC const &y1, PREC const &y2,
                                          PREC const &z1, PREC const &z2,
                                          std::shared_ptr<Material> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Box(x1, x2,
                                                 y1, y2,
                                                 z1, z2,
                                                 material, trans));
}

std::shared_ptr<Geometry> Box::create(Point const &v1, Point const &v2,
                                          std::shared_ptr<Material> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Box(v1.x, v2.x,
                                                 v1.y, v2.y,
                                                 v1.z, v2.z,
                                                 material, trans));
}

Box::Box(PREC const &x1, PREC const &x2,
         PREC const &y1, PREC const &y2,
         PREC const &z1, PREC const &z2,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseBox(x1, x2, y1, y2, z1, z2,
                  material->obj(), trans.get())),
          _base_obj(_obj),
          _v1_x(x1), _v1_y(y1), _v1_z(z1),
          _v2_x(x2), _v2_y(y2), _v2_z(z2),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Box::~Box()
{
    delete _obj;

#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &Box::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Box::devPtr()
{
    return _dev_ptr;
}

void Box::up2Gpu()
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

        GpuCreator::Box(_dev_ptr,
                        _v1_x, _v2_x,
                        _v1_y, _v2_y,
                        _v1_z, _v2_z,
                        _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                        _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
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

    GpuCreator::destroy(_dev_ptr);

    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void BaseBox::setVertices(PREC const &x1, PREC const &x2,
                      PREC const &y1, PREC const &y2,
                      PREC const &z1, PREC const &z2)
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
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
