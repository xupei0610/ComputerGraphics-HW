#include "object/geometry/plane.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

#include <cfloat>

using namespace px;

PX_CUDA_CALLABLE
BasePlane::BasePlane(Point const &pos,
                     Direction const &norm_vec,
                     const BaseMaterial * const &material,
                     const Transformation * const &trans)
        : BaseGeometry(material, trans, 4),
          _position(pos)
{
    setNormVec(norm_vec);
}

PX_CUDA_CALLABLE
const BaseGeometry * BasePlane::hitCheck(Ray const &ray,
                                   PREC const &t_start,
                                   PREC const &t_end,
                                   PREC &hit_at) const
{
    auto tmp = (_p_dot_n - ray.original.dot(_norm_vec)) / ray.direction.dot(_norm_vec);
    return (tmp > t_start && tmp < t_end) ? (hit_at = tmp, this) : nullptr;
}

PX_CUDA_CALLABLE
Direction BasePlane::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<PREC> BasePlane::getTextureCoord(PREC const &x, PREC const &y,
                                    PREC const &z) const
{
    return {x - _position.x,
            -_norm_vec.z*(y - _position.y) + _norm_vec.y*(z - _position.z),
            (x - _position.x)*_norm_vec.x + (y - _position.y)*_norm_vec.y + (z - _position.z)*_norm_vec.z};
}

std::shared_ptr<Geometry> Plane::create(Point const &position,
                                            Direction const &norm_vec,
                                            std::shared_ptr<Material> const &material,
                                            std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Plane(position, norm_vec, material, trans));
}

Plane::Plane(Point const &position,
             Direction const &norm_vec,
             std::shared_ptr<Material> const &material,
             std::shared_ptr<Transformation> const &trans)
        : _obj(new BasePlane(position, norm_vec, material->obj(), trans.get())),
          _base_obj(_obj),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Plane::~Plane()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &Plane::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Plane::devPtr()
{
    return _dev_ptr;
}

void Plane::up2Gpu()
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

        GpuCreator::Plane(_dev_ptr,
                          _obj->_position, _obj->_norm_vec,
                        _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                        _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void Plane::clearGpuData()
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

void Plane::setPosition(Point const &position)
{
    _obj->_position = position;
    _obj->_p_dot_n = position.dot(_obj->_norm_vec);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void BasePlane::setNormVec(Direction const &norm_vec)
{
    _norm_vec = norm_vec;
    _p_dot_n = _position.dot(norm_vec);

    if ((_norm_vec.x == 1 || _norm_vec.x == -1) && _norm_vec.y == 0 && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = 0;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x = 0;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = 0;
        _raw_vertices[2].y = -FLT_MAX;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x = 0;
        _raw_vertices[3].y =  FLT_MAX;
        _raw_vertices[3].z = -FLT_MAX;
    }
    else if (_norm_vec.x == 0 && (_norm_vec.y == 1 || _norm_vec.y == -1) && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = 0;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y = 0;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y = 0;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = 0;
        _raw_vertices[3].z = -FLT_MAX;
    }
    else if (_norm_vec.x == 0 && _norm_vec.y == 0 && (_norm_vec.z == 1 || _norm_vec.z == -1))
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = 0;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z = 0;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y =  FLT_MAX;
        _raw_vertices[2].z = 0;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = -FLT_MAX;
        _raw_vertices[3].z = 0;
    }
    else if (_norm_vec.x == 0 && _norm_vec.y == 0 && _norm_vec.z == 0)
    {
        _raw_vertices[0].x = 0;
        _raw_vertices[0].y = 0;
        _raw_vertices[0].z = 0;
        _raw_vertices[1].x = 0;
        _raw_vertices[1].y = 0;
        _raw_vertices[1].z = 0;
        _raw_vertices[2].x = 0;
        _raw_vertices[2].y = 0;
        _raw_vertices[2].z = 0;
        _raw_vertices[3].x = 0;
        _raw_vertices[3].y = 0;
        _raw_vertices[3].z = 0;
    }
    else
    {
        _raw_vertices[0].x = -FLT_MAX;
        _raw_vertices[0].y = -FLT_MAX;
        _raw_vertices[0].z = -FLT_MAX;
        _raw_vertices[1].x =  FLT_MAX;
        _raw_vertices[1].y =  FLT_MAX;
        _raw_vertices[1].z =  FLT_MAX;
        _raw_vertices[2].x = -FLT_MAX;
        _raw_vertices[2].y =  FLT_MAX;
        _raw_vertices[2].z =  FLT_MAX;
        _raw_vertices[3].x =  FLT_MAX;
        _raw_vertices[3].y = -FLT_MAX;
        _raw_vertices[3].z = -FLT_MAX;
    }
}

void Plane::setNormVec(Direction const &norm_vec)
{
    _obj->setNormVec(norm_vec);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}