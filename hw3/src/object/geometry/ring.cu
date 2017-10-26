#include "object/geometry/ring.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseRing::BaseRing(Point const &pos,
                   Direction const &norm_vec,
                   PREC const &radius1,
                   PREC const &radius2,
                   const BaseMaterial *const &material,
                   const Transformation *const &trans)
        : BaseGeometry(material, trans, 8),
          _center(pos),
          _inner_radius(radius1 < radius2 ? radius1 : radius2),
          _outer_radius(radius1 > radius2 ? radius1 : radius2),
          _inner_radius2(_inner_radius*_inner_radius),
          _outer_radius2(_outer_radius*_outer_radius),
          _p_dot_n(pos.dot(norm_vec))
{
    updateVertices();
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseRing::hitCheck(Ray const &ray,
                                  PREC const &t_start,
                                  PREC const &t_end,
                                  PREC &hit_at) const
{
    auto tmp = (_p_dot_n - ray.original.dot(_norm_vec)) / ray.direction.dot(_norm_vec);
    if (tmp > t_start && tmp < t_end)
    {
        auto intersect = ray[tmp];
        auto dist2 = (intersect.x - _center.x) * (intersect.x - _center.x) +
                     (intersect.y - _center.y) * (intersect.y - _center.y) +
                     (intersect.z - _center.z) * (intersect.z - _center.z);
        if (dist2 <= _outer_radius2 && dist2 >= _inner_radius2)
        {
            hit_at = tmp;
            return this;
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseRing::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseRing::getTextureCoord(PREC const &x, PREC const &y,
                                       PREC const &z) const
{
    return {x - _center.x,
            -_norm_vec.z*(y - _center.y) + _norm_vec.y*(z - _center.z),
            (x - _center.x)*_norm_vec.x + (y - _center.y)*_norm_vec.y + (z - _center.z)*_norm_vec.z};
}

std::shared_ptr<Geometry> Ring::create(Point const &position,
                                           Direction const &norm_vec,
                                           PREC const &radius1,
                                           PREC const &radius2,
                                           std::shared_ptr<Material> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Ring(position, norm_vec,
                                                  radius1, radius2,
                                                  material, trans));
}

Ring::Ring(Point const &position,
           Direction const &norm_vec,
           PREC const &radius1,
           PREC const &radius2,
           std::shared_ptr<Material> const &material,
           std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseRing(position, norm_vec, radius1, radius2, material->obj(), trans.get())),
          _base_obj(_obj),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Ring::~Ring()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry * const &Ring::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Ring::devPtr()
{
    return _dev_ptr;
}

void Ring::up2Gpu()
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

        GpuCreator::Ring(_dev_ptr,
                         _obj->_center, _obj->_norm_vec, _obj->_inner_radius, _obj->_outer_radius,
                        _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                        _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void Ring::clearGpuData()
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


void Ring::setCenter(Point const &center)
{
    _obj->_center = center;
    _obj->_p_dot_n = center.dot(_obj->_norm_vec);

    _obj->updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ring::setNormVec(Direction const &norm_vec)
{
    _obj->_norm_vec = norm_vec;
    _obj->_p_dot_n = _obj->_center.dot(norm_vec);

    _obj->updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ring::setRadius(PREC const &radius1, PREC const &radius2)
{
    _obj->_inner_radius = std::min(radius1, radius2);
    _obj->_outer_radius = std::max(radius1, radius2);
    _obj->_inner_radius2 = _obj->_inner_radius*_obj->_inner_radius;
    _obj->_outer_radius2 = _obj->_outer_radius*_obj->_outer_radius;

    _obj->updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void BaseRing::updateVertices()
{
    _raw_vertices[0].x = _center.x + _outer_radius;
    _raw_vertices[0].y = _center.y + _outer_radius;
    _raw_vertices[0].z = _center.z + _outer_radius;

    _raw_vertices[1].x = _center.x - _outer_radius;
    _raw_vertices[1].y = _center.y + _outer_radius;
    _raw_vertices[1].z = _center.z + _outer_radius;

    _raw_vertices[2].x = _center.x + _outer_radius;
    _raw_vertices[2].y = _center.y - _outer_radius;
    _raw_vertices[2].z = _center.z + _outer_radius;

    _raw_vertices[3].x = _center.x + _outer_radius;
    _raw_vertices[3].y = _center.y + _outer_radius;
    _raw_vertices[3].z = _center.z - _outer_radius;

    _raw_vertices[4].x = _center.x - _outer_radius;
    _raw_vertices[4].y = _center.y - _outer_radius;
    _raw_vertices[4].z = _center.z + _outer_radius;

    _raw_vertices[5].x = _center.x - _outer_radius;
    _raw_vertices[5].y = _center.y + _outer_radius;
    _raw_vertices[5].z = _center.z - _outer_radius;

    _raw_vertices[6].x = _center.x + _outer_radius;
    _raw_vertices[6].y = _center.y - _outer_radius;
    _raw_vertices[6].z = _center.z - _outer_radius;

    _raw_vertices[7].x = _center.x - _outer_radius;
    _raw_vertices[7].y = _center.y - _outer_radius;
    _raw_vertices[7].z = _center.z - _outer_radius;
}
