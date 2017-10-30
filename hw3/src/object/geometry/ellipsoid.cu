#include "object/geometry/ellipsoid.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseEllipsoid::BaseEllipsoid(Point const &center,
                             PREC const &radius_x,
                             PREC const &radius_y,
                             PREC const &radius_z,
                             const BaseMaterial *const &material,
                             const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{
    setParams(center,
              radius_x, radius_y, radius_z);
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseEllipsoid::hitCheck(Ray const &ray,
                                             PREC const &t_start,
                                             PREC const &t_end,
                                             PREC &hit_at) const
{
    auto xo = ray.original.x - _center.x;
    auto yo = ray.original.y - _center.y;
    auto zo = ray.original.z - _center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y +
              _c * ray.direction.z * ray.direction.z;
    auto B =  2 * _a * xo * ray.direction.x +
              2 * _b * yo * ray.direction.y +
              2 * _c * zo * ray.direction.z;
    auto C =  _a * xo * xo +
              _b * yo * yo +
              _c * zo * zo +
              -1;

    if (A == 0)
    {
        if (B == 0) return nullptr;

        C = - C / B;
        if (C > t_start && C < t_end)
        {
            hit_at = C;
            return this;
        }
        return nullptr;
    }

    C = B * B - 4 * A * C;
    if (C < 0)
        return nullptr;

    C = std::sqrt(C);
    xo = (-B - C)/ (2.0 * A);
    yo = (-B + C)/ (2.0 * A);
    if (xo > yo)
    {
        zo = yo;
        yo = xo;
        xo = zo;
    }
    if (xo > t_start && xo < t_end)
    {
        hit_at = xo;
        return this;
    }
    if (yo > t_start && yo < t_end)
    {
        hit_at = yo;
        return this;
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseEllipsoid::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return {_a * (x - _center.x),
            _b * (y - _center.y),
            _c * (z - _center.z)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseEllipsoid::getTextureCoord(PREC const &x, PREC const &y,
                                           PREC const &z) const
{
    auto dx = x - _center.x;
    auto dy = y - _center.y;
    auto dz = z - _center.z;

    return {(1 + std::atan2(dz, dx) / PI) * PREC(0.5),
            std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PI,
            0};;
}

std::shared_ptr<Geometry> Ellipsoid::create(Point const &center,
                                                PREC const &radius_x, PREC const &radius_y,
                                                PREC const &radius_z,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Ellipsoid(center,
                                                      radius_x, radius_y, radius_z,
                                                      material, trans));
}

Ellipsoid::Ellipsoid(Point const &center,
                     PREC const &radius_x, PREC const &radius_y,
                     PREC const &radius_z,
                     std::shared_ptr<Material> const &material,
                     std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseEllipsoid(center, radius_x, radius_y, radius_z,
                        material->obj(), trans.get())),
          _base_obj(_obj),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Ellipsoid::~Ellipsoid()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &Ellipsoid::obj() const noexcept
{
    return  _base_obj;
}

BaseGeometry **Ellipsoid::devPtr()
{
    return _dev_ptr;
}

void Ellipsoid::up2Gpu()
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

        GpuCreator::Ellipsoid(_dev_ptr, _obj->_center, _obj->_radius_x, _obj->_radius_y,
                              _obj->_radius_z,
                              _material_ptr == nullptr ? nullptr
                                                       : _material_ptr->devPtr(),
                              _transformation_ptr == nullptr ? nullptr
                                                             : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void Ellipsoid::clearGpuData()
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
void BaseEllipsoid::setParams(Point const &center,
                              PREC const &radius_x,
                              PREC const &radius_y,
                              PREC const &radius_z)
{
    _center = center;
    _radius_x = radius_x;
    _radius_y = radius_y;
    _radius_z = radius_z;

    _a = 1.0 / (radius_x*radius_x);
    _b = 1.0 / (radius_y*radius_y);
    _c = 1.0 / (radius_z*radius_z);

    _raw_vertices[4].x = _center.x - radius_x;
    _raw_vertices[4].y = _center.y - radius_y;
    _raw_vertices[4].z = _center.x - radius_x;
    _raw_vertices[5].x = _center.x - radius_x;
    _raw_vertices[5].y = _center.y + radius_y;
    _raw_vertices[5].z = _center.x - radius_x;
    _raw_vertices[6].x = _center.x + radius_x;
    _raw_vertices[6].y = _center.y + radius_y;
    _raw_vertices[6].z = _center.x - radius_x;
    _raw_vertices[7].x = _center.x + radius_x;
    _raw_vertices[7].y = _center.y - radius_y;
    _raw_vertices[7].z = _center.x - radius_x;

    _raw_vertices[0].x = _center.x - radius_x;
    _raw_vertices[0].y = _center.y + radius_y;
    _raw_vertices[0].z = _center.z - radius_z;
    _raw_vertices[1].x = _center.x + radius_x;
    _raw_vertices[1].y = _center.y + radius_y;
    _raw_vertices[1].z = _center.z - radius_z;
    _raw_vertices[2].x = _center.x + radius_x;
    _raw_vertices[2].y = _center.y - radius_y;
    _raw_vertices[2].z = _center.z - radius_z;
    _raw_vertices[3].x = _center.x + radius_x;
    _raw_vertices[3].y = _center.y - radius_y;
    _raw_vertices[3].z = _center.z - radius_z;
}

void Ellipsoid::setParams(Point const &center,
                          PREC const &radius_x,
                          PREC const &radius_y,
                          PREC const &radius_z)
{
    _obj->setParams(center, radius_x, radius_y, radius_z);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}