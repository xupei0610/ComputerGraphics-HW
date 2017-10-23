#include "object/geometry/cylinder.hpp"

using namespace px;

BaseCylinder::BaseCylinder(const BaseMaterial *const &material,
                   const Transformation *const &trans)
        : BaseGeometry(material, trans, 8)
{}

PX_CUDA_CALLABLE
BaseGeometry * BaseCylinder::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at)
{
    auto xo = ray.original.x - _center.x;
    auto yo = ray.original.y - _center.y;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y;
    auto B = 2 * _a * xo * ray.direction.x +
             2 * _b * yo * ray.direction.y;
    auto C =  _a * xo * xo +
              _b * yo * yo - 1;

    bool hit_top = false;

    auto tmp1 = (_z1 - ray.original.z) / ray.direction.z;
    auto tmp_x = ray.original.x + ray.direction.x * tmp1;
    auto tmp_y = ray.original.y + ray.direction.y * tmp1;

    if (tmp1 >= t_start && tmp1 <= t_end &&
        _a * (tmp_x - _center.x) * (tmp_x - _center.x) +
        _b * (tmp_y - _center.y) * (tmp_y - _center.y) <= 1)
    {
        hit_top = true;
        hit_at = tmp1;
    }

    auto tmp2 = (_z0 - ray.original.z) / ray.direction.z;

    if (tmp1 >= t_start && tmp1 <= t_end &&
        (hit_top == false || tmp2 < tmp1))
    {
        tmp_x = ray.original.x + ray.direction.x * tmp2;
        tmp_y = ray.original.y + ray.direction.y * tmp2;

        if (_a * (tmp_x - _center.x) * (tmp_x - _center.x) +
            _b * (tmp_y - _center.y) * (tmp_y - _center.y) <= 1)
        {
            hit_top = true;
            hit_at = tmp2;
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
    tmp2 = (-B + discriminant)/ (2.0 * A);
    if (tmp1 > tmp2)
        std::swap(tmp1, tmp2);
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
Direction BaseCylinder::normalVec(double const &x, double const &y, double const &z)
{
    if (std::abs(z - _z0) < 1e-12)
        return {0, 0, -1};
    if (std::abs(z - _z1) < 1e-12)
        return {0, 0, 1};

    return {_a * (x - _center.x),
            _b * (y - _center.y),
            0};
}

PX_CUDA_CALLABLE
Vec3<double> BaseCylinder::getTextureCoord(double const &x, double const &y,
                                       double const &z)
{
    if (std::abs(z - _z0) < 1e-12)
        return {x - _center.x,
                _radius_y + y - _center.y, 0};
    if (std::abs(z - _z1) < 1e-12)
        return {x - _center.x,
                _radius_y + _radius_y + _radius_y + _abs_height + y - _center.y, 0};

    auto dx = x - _center.x;
    auto dy = y - _center.y - _radius_y;

    return {((_a/3.0 * dx * dx * dx - dx) + _b/3.0 * dy * dy * dy),
            _radius_y + _radius_y + z - _center.z, 0};
}

std::shared_ptr<BaseGeometry> Cylinder::create(Point const &center_of_bottom_face,
                                               double const &radius_x, double const &radius_y,
                                               double const &height,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Cylinder(center_of_bottom_face,
                                                      radius_x, radius_y,
                                                      height,
                                                      material, trans));
}

Cylinder::Cylinder(Point const &center_of_bottom_face,
                   double const &radius_x, double const &radius_y,
                   double const &height,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseCylinder(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setParams(center_of_bottom_face,
              radius_x, radius_y,
              height);
}

Cylinder::~Cylinder()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Cylinder::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseCylinder)));

        material = _material_ptr->up2Gpu();
        transformation = _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseCylinder*>(this),
                                 sizeof(BaseCylinder),
                                 cudaMemcpyHostToDevice));

        material = _material_ptr.get();
        transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void Cylinder::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void Cylinder::setParams(Point const &center_of_bottom_face,
                         double const &radius_x, double const &radius_y,
                         double const &height)
{
    _center = center_of_bottom_face;
    _radius_x = std::abs(radius_x);
    _radius_y = std::abs(radius_y);
    _height = height;
    _abs_height = std::abs(height);

    auto top = _center.z + _height;;
    _raw_vertices[4].x = _center.x - radius_x;
    _raw_vertices[4].y = _center.y - radius_y;
    _raw_vertices[4].z = top;
    _raw_vertices[5].x = _center.x - radius_x;
    _raw_vertices[5].y = _center.y + radius_y;
    _raw_vertices[5].z = top;
    _raw_vertices[6].x = _center.x + radius_x;
    _raw_vertices[6].y = _center.y + radius_y;
    _raw_vertices[6].z = top;
    _raw_vertices[7].x = _center.x + radius_x;
    _raw_vertices[7].y = _center.y - radius_y;
    _raw_vertices[7].z = top;

    _raw_vertices[0].x = _center.x - radius_x;
    _raw_vertices[0].y = _center.y + radius_y;
    _raw_vertices[0].z = _center.z;
    _raw_vertices[1].x = _center.x + radius_x;
    _raw_vertices[1].y = _center.y + radius_y;
    _raw_vertices[1].z = _center.z;
    _raw_vertices[2].x = _center.x + radius_x;
    _raw_vertices[2].y = _center.y - radius_y;
    _raw_vertices[2].z = _center.z;
    _raw_vertices[3].x = _center.x + radius_x;
    _raw_vertices[3].y = _center.y - radius_y;
    _raw_vertices[3].z = _center.z;

    _a =  1.0 / (radius_x*radius_x);
    _b =  1.0 / (radius_y*radius_y);

    _z0 = height < 0 ? (_z1 = _center.z, top)
                     : (_z1 = top, _center.z);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}
