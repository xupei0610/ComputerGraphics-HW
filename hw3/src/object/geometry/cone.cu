#include "object/geometry/cone.hpp"

using namespace px;

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

PX_CUDA_CALLABLE
BaseCone::BaseCone(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &ideal_height,
                   PREC const &real_height,
                   const BaseMaterial *const &material,
                   const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{
    setParams(center_of_bottom_face,
              radius_x, radius_y,
              ideal_height, real_height);
}

PX_CUDA_CALLABLE
const BaseGeometry * BaseCone::hitCheck(Ray const &ray,
                                        PREC const &t_start,
                                        PREC const &t_end,
                                        PREC &hit_at) const
{
    auto xo = ray.original.x - _center.x;
    auto yo = ray.original.y - _center.y;
    auto zo = ray.original.z - _quadric_center_z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y +
              _c * ray.direction.z * ray.direction.z;
    auto B = 2 * _a * xo * ray.direction.x +
             2 * _b * yo * ray.direction.y +
             2 * _c * zo * ray.direction.z;
    auto C =  _a * xo * xo +
              _b * yo * yo +
              _c * zo * zo;

    auto hit_top = false;

    auto tmp1 = (_center.z - ray.original.z) / ray.direction.z;
    auto tmp_x = ray.original.x + ray.direction.x * tmp1;
    auto tmp_y = ray.original.y + ray.direction.y * tmp1;

    if (tmp1 >= t_start && tmp1 <= t_end &&
        _a * (tmp_x - _center.x) * (tmp_x - _center.x) +
        _b * (tmp_y - _center.y) * (tmp_y - _center.y) <= 1)
    {
        hit_top= true;
        hit_at = tmp1;
    }

    if (_real_height != _ideal_height)
    {
        auto tmp2 = (_top_z - ray.original.z) / ray.direction.z;

        if (tmp2 >= t_start && tmp2 <= t_end &&
            (hit_at == false || tmp2 < tmp1))
        {
            tmp_x = ray.original.x + ray.direction.x * tmp2;
            tmp_y = ray.original.y + ray.direction.y * tmp2;

            if ((tmp_x - _center.x) * (tmp_x - _center.x) / (_top_r_x*_top_r_x)+
                (tmp_y - _center.y) * (tmp_y - _center.y) / (_top_r_y*_top_r_y) <= 1)
            {
                hit_top = true;
                hit_at = tmp2;
            }
        }
    }

    if (A == 0)
    {
        if (B == 0) return nullptr;

        auto tmp = - C / B;
        if (tmp > t_start && tmp < t_end)
        {
            auto iz = ray.original.z + ray.direction.z*tmp;
            if (iz >= _z0 && iz<=_z1)
            {
                if (hit_top == false || hit_at > tmp)
                    hit_at = tmp;
                return this;
            }
        }
        return nullptr;
    }

    auto discriminant = B * B - 4 * A * C;
    if (discriminant < 0)
        return nullptr;

    discriminant = std::sqrt(discriminant);
    tmp1 = (-B - discriminant)/ (2.0 * A);
    auto tmp2 = (-B + discriminant)/ (2.0 * A);
    if (tmp1 > tmp2)
    {
        auto tmp = tmp1;
        tmp1 = tmp2;
        tmp2 = tmp;
    }
    if (tmp1 > t_start && tmp1 < t_end)
    {
        auto iz = ray.original.z + ray.direction.z*tmp1;
        if (iz >= _z0 && iz<=_z1)
        {
            if (hit_top == false || hit_at > tmp1)
                hit_at = tmp1;
            return this;
        }
    }
    if (tmp2 > t_start && tmp2 < t_end)
    {
        auto iz = ray.original.z + ray.direction.z*tmp2;
        if (iz >= _z0 && iz<=_z1)
        {
            if (hit_top == false || hit_at > tmp2)
                hit_at = tmp2;
            return this;
        }
    }

    return hit_top ? this : nullptr;
}

PX_CUDA_CALLABLE
Direction BaseCone::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    if (std::abs(z - _z0) < EPSILON)
        return {0, 0, -1};
    if (std::abs(z - _z1) < EPSILON)
        return {0, 0, 1};

    return {_a * (x - _center.x),
            _b * (y - _center.y),
            _c * (z - _center.z)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseCone::getTextureCoord(PREC const &x, PREC const &y,
                                       PREC const &z) const
{
    return {x - _center.x, y - _center.y, 0};
}

std::shared_ptr<Geometry> Cone::create(Point const &center_of_bottom_face,
                                           PREC const &radius_x,
                                           PREC const &radius_y,
                                           PREC const &ideal_height,
                                           PREC const &real_height,
                                           std::shared_ptr<Material> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new Cone(center_of_bottom_face,
                                                  radius_x, radius_y,
                                                  ideal_height, real_height,
                                                  material, trans));
}

Cone::Cone(Point const &center_of_bottom_face,
           PREC const &radius_x,
           PREC const &radius_y,
           PREC const &ideal_height,
           PREC const &real_height,
           std::shared_ptr<Material> const &material,
           std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseCone(center_of_bottom_face, radius_x, radius_y,
                   ideal_height, real_height,
                   material->obj(), trans.get())),
          _base_obj(_obj),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

Cone::~Cone()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}
BaseGeometry *const &Cone::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **Cone::devPtr()
{
    return _dev_ptr;
}

void Cone::up2Gpu()
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

        GpuCreator::Cone(_dev_ptr,
                         _obj->_center, _obj->_radius_x, _obj->_radius_y,
                         _obj->_ideal_height, _obj->_real_height,
                         _material_ptr == nullptr ? nullptr : _material_ptr->devPtr(),
                         _transformation_ptr == nullptr ? nullptr : _transformation_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void Cone::clearGpuData()
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
void BaseCone::setParams(Point const &center_of_bottom_face,
                     PREC const &radius_x, PREC const &radius_y,
                     PREC const &ideal_height,
                     PREC const &real_height)
{
    _center = center_of_bottom_face;
    _radius_x = radius_x;
    _radius_y = radius_y;

    if (ideal_height == real_height)
    {
        _ideal_height = ideal_height;
        _real_height = real_height;

        if (_n_vertices != 5)
        {
            delete [] _raw_vertices;
            _raw_vertices = new Point[5];
        }
        _raw_vertices[4].x = _center.x;
        _raw_vertices[4].y = _center.y;
        _raw_vertices[4].z = _center.z + ideal_height;
    }
    else
    {
        _ideal_height = std::abs(ideal_height);
        _real_height = std::abs(real_height);

        if (_ideal_height < _real_height)
        {
            auto tmp = _ideal_height;
            _ideal_height = _real_height;
            _real_height = tmp;
        }

        if (_n_vertices != 8)
        {
            delete [] _raw_vertices;
            _raw_vertices = new Point[8];
        }

        auto ratio = (_ideal_height - _real_height) / _ideal_height;
        _top_r_x = radius_x * ratio;
        _top_r_y = radius_y * ratio;

        if (ideal_height < 0)
        {
            _ideal_height *= -1;
            _real_height *= -1;
        }

        _raw_vertices[4].x = _center.x - _top_r_x;
        _raw_vertices[4].y = _center.y - _top_r_y;
        _raw_vertices[4].z = _center.z + _real_height;
        _raw_vertices[5].x = _center.x - _top_r_x;
        _raw_vertices[5].y = _center.y + _top_r_y;
        _raw_vertices[5].z = _center.z + _real_height;
        _raw_vertices[6].x = _center.x + _top_r_x;
        _raw_vertices[6].y = _center.y + _top_r_y;
        _raw_vertices[6].z = _center.z + _real_height;
        _raw_vertices[7].x = _center.x + _top_r_x;
        _raw_vertices[7].y = _center.y - _top_r_y;
        _raw_vertices[7].z = _center.z + _real_height;
    }

    _raw_vertices[0].x = _center.x - radius_x;
    _raw_vertices[0].y = _center.y + radius_y;
    _raw_vertices[0].z = _center.z;
    _raw_vertices[1].x = _center.x + radius_x;
    _raw_vertices[1].y = _center.y + radius_y;
    _raw_vertices[1].z = _center.z;
    _raw_vertices[2].x = _center.x + radius_x;
    _raw_vertices[2].y = _center.y - radius_y;
    _raw_vertices[2].z = _center.z;
    _raw_vertices[3].x = _center.x - radius_x;
    _raw_vertices[3].y = _center.y - radius_y;
    _raw_vertices[3].z = _center.z;

    _a =  1.0 / (radius_x*radius_x);
    _b =  1.0 / (radius_y*radius_y);
    _c = -1.0 / (_ideal_height*_ideal_height);

    _quadric_center_z = _center.z + _ideal_height;
    _z0 = _ideal_height < 0 ? (_z1 = _center.z, _center.z + _real_height)
                            : (_z1 = _center.z + _real_height, _center.z);

    _top_z = _center.z + _real_height;
}

void Cone::setParams(Point const &center_of_bottom_face, PREC const &radius_x,
                     PREC const &radius_y, PREC const &ideal_height,
                     PREC const &real_height)
{
    _obj->setParams(center_of_bottom_face, radius_x, radius_y,
                        ideal_height, real_height);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
