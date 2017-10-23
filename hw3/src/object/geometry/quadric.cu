#include "object/geometry/quadric.hpp"

using namespace px;

BaseQuadric::BaseQuadric(Point const &center,
                         const BaseMaterial *const &material,
                        const Transformation *const &trans)
        : BaseGeometry(material, trans, 8),
          _center(center)
{}

PX_CUDA_CALLABLE
BaseGeometry * BaseQuadric::hitCheck(Ray const &ray,
                                     double const &t_start,
                                     double const &t_end,
                                     double &hit_at)
{
    auto xo = ray.original.x - _center.x;
    auto yo = ray.original.y - _center.y;
    auto zo = ray.original.z - _center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y +
              _c * ray.direction.z * ray.direction.z +
              _d * ray.direction.x * ray.direction.y +
              _e * ray.direction.x * ray.direction.z +
              _f * ray.direction.y * ray.direction.z;
    auto B =  2 * _a * xo * ray.direction.x +
              2 * _b * yo * ray.direction.y +
              2 * _c * zo * ray.direction.z +
              _d * (xo * ray.direction.y + yo * ray.direction.x) +
              _e * (xo * ray.direction.z + zo * ray.direction.x) +
              _f * (yo * ray.direction.z + zo * ray.direction.y) +
              _g * ray.direction.x +
              _h * ray.direction.y +
              _i * ray.direction.z;
    auto C =  _a * xo * xo +
              _b * yo * yo +
              _c * zo * zo +
              _d * xo * yo +
              _e * xo * zo +
              _f * yo * zo +
              _g * xo +
              _h * yo +
              _i * zo +
              _j;

    if (A == 0)
    {
        if (B == 0) return nullptr;

        auto tmp = - C / B;
        if (tmp > t_start && tmp < t_end)
        {
            auto i_x =ray.original.x + ray.direction.x * tmp;
            if (i_x >_x0 && i_x < _x1)
            {
                auto i_y =ray.original.y + ray.direction.y * tmp;
                if (i_y >_y0 && i_y < _y1)
                {
                    auto i_z =ray.original.z + ray.direction.z * tmp;
                    if (i_z >_z0 && i_z < _z1)
                    {
                        hit_at = tmp;
                        return this;
                    }
                }
            }
        }
        return nullptr;
    }

    auto discriminant = B * B - 4 * A * C;
    if (discriminant < 0)
        return nullptr;

    discriminant = std::sqrt(discriminant);
    auto tmp1 = (-B - discriminant)/ (2.0 * A);
    auto tmp2 = (-B + discriminant)/ (2.0 * A);
    if (tmp1 > tmp2)
        std::swap(tmp1, tmp2);

    if (tmp1 > t_start && tmp1 < t_end)
    {
        auto i_x =ray.original.x + ray.direction.x * tmp1;
        if (i_x >_x0 && i_x < _x1)
        {
            auto i_y =ray.original.y + ray.direction.y * tmp1;
            if (i_y >_y0 && i_y < _y1)
            {
                auto i_z =ray.original.z + ray.direction.z * tmp1;
                if (i_z >_z0 && i_z < _z1)
                {
                    hit_at = tmp1;
                    return this;
                }
            }
        }
    }
    if (tmp2 > t_start && tmp2 < t_end)
    {
        auto i_x =ray.original.x + ray.direction.x * tmp2;
        if (i_x >_x0 && i_x < _x1)
        {
            auto i_y =ray.original.y + ray.direction.y * tmp2;
            if (i_y >_y0 && i_y < _y1)
            {
                auto i_z =ray.original.z + ray.direction.z * tmp2;
                if (i_z >_z0 && i_z < _z1)
                {
                    hit_at = tmp2;
                    return this;
                }
            }
        }
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseQuadric::normalVec(double const &x, double const &y, double const &z)
{
    auto dx = x - _center.x;
    auto dy = y - _center.y;
    auto dz = z - _center.z;

    return {2 * _a * dx + _d * dy + _e * dz + _g,
            2 * _b * dy + _d * dx + _f * dz + _h,
            2 * _c * dz + _e * dx + _f * dy + _i};
}

PX_CUDA_CALLABLE
Vec3<double> BaseQuadric::getTextureCoord(double const &x, double const &y,
                                       double const &z)
{
    // FIXME better way for quadric surface texture mapping

    if (_sym_o)
    {
        if (_a == 0 && _b == 0)
            return {x-_center.x, y-_center.y, 0};
        if (_a == 0 && _c == 0)
            return {z-_center.z, x-_center.x, 0};
        if (_b == 0 && _c == 0)
            return {y-_center.y, z-_center.z, 0};
        if (_c == 0)
        {
            auto discriminant = _h*_h - 4*_b*_j;
            auto cy = discriminant < 0 ? 0 : (std::sqrt(discriminant) - _h)/2.0/_b;

//            auto dx = x - _center.x;
//            auto dy = y - _center.y;
//            auto dz = z - _center.z;
//
//            auto du = (_a*dx*dx/2.0 + _c*dz*dz + _e*dx*dz/2.0 + _g*dx/2.0 + _i*dz + _j) * dx * (dy-cy) +
//                      (_d*dx*dx/2.0 + _f*dx*dz + _h*dx)/2.0 * (dy*dy - cy*cy) +
//                      _b*dx/3.0 * (dy*dy*dy - cy*cy*cy) ;

            return {x-_center.x > 0 ? y - cy : cy - y, z-_center.z, 0};
//            return {x-_center.x > 0 ? y - _center.y : _center.y - y, z-_center.z, 0};
//            return {du, z-_center.z, 0};
        }
        if (_b == 0)
        {
            auto discriminant = _g*_g - 4*_a*_j;
            auto cx = discriminant < 0 ? _center.x : ((std::sqrt(discriminant) - _g)/2.0/_a + _center.x);
            return {z-_center.z > 0 ? x - cx : cx - x, y-_center.y, 0};
        }
        if (_a == 0)
        {
            auto discriminant = _i*_i - 4*_c*_j;
            auto cz = discriminant < 0 ? _center.z : ((std::sqrt(discriminant) - _i)/2.0/_c + _center.z);
            return {y-_center.y > 0 ? z-cz : cz-z, x-_center.x, 0};
        }

        if (_a > 0 && _b > 0 && _c > 0)
        {
            auto dx = x - _center.x;
            auto dy = y - _center.y;
            auto dz = z - _center.z;

            return {(1 + std::atan2(dz, dx) / PI) * 0.5,
                    std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PI,
                    0};;
        }

        return {x - _center.x,
                y - _center.y,
                0};;
    }
    if (_sym_x)
    {
        return {y-_center.y, z-_center.z, 0};
    }
    if (_sym_y)
    {
        return {x-_center.x, z-_center.z, 0};
    }
    if (_sym_z)
    {
        return {x-_center.x, y-_center.y, 0};
    }

    if (_sym_xy)
        return {z-_center.z, x-_center.x, 0};
    if (_sym_yz)
        return {x-_center.x, y-_center.y, 0};
    if (_sym_xz)
        return {y-_center.y, x-_center.x, 0};

    return {x-_center.x, y-_center.y, 0};

}

std::shared_ptr<BaseGeometry> Quadric::create(Point const &center,
                                              double const &a,
                                              double const &b,
                                              double const &c,
                                              double const &d,
                                              double const &e,
                                              double const &f,
                                              double const &g,
                                              double const &h,
                                              double const &i,
                                              double const &j,
                                              double const &x0, double const &x1,
                                              double const &y0, double const &y1,
                                              double const &z0, double const &z1,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Quadric(center,
                                                     a, b, c, d, e, f, g, h, i,
                                                     j,
                                                     x0, x1, y0, y1, z0, z1,
                                                     material, trans));
}

Quadric::Quadric(Point const &center,
                 double const &a,
                 double const &b,
                 double const &c,
                 double const &d,
                 double const &e,
                 double const &f,
                 double const &g,
                 double const &h,
                 double const &i,
                 double const &j,
                 double const &x0, double const &x1,
                 double const &y0, double const &y1,
                 double const &z0, double const &z1,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseQuadric(center, material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setCoef(a, b, c, d, e, f, g, h, i, j, x0, x1, y0, y1, z0, z1);
}

Quadric::~Quadric()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Quadric::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseQuadric)));

        material = _material_ptr->up2Gpu();
        transformation = _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseQuadric*>(this),
                                 sizeof(BaseQuadric),
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

void Quadric::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}


void Quadric::setCenter(Point const &center)
{
    _center = center;
//    updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Quadric::setCoef(double const &a,
                      double const &b,
                      double const &c,
                      double const &d,
                      double const &e,
                      double const &f,
                      double const &g,
                      double const &h,
                      double const &i,
                      double const &j,
                      double const &x0, double const &x1,
                      double const &y0, double const &y1,
                      double const &z0, double const &z1)
{
    if (j < 0)
    {
        _a = -a, _b = -b, _c = -c, _d = -d;
        _e = -e, _f = -f, _g = -g, _h = -h;
        _i = -i, _j = -j;
    }
    else
    {
        _a = a, _b = b, _c = c, _d = d;
        _e = e, _f = f, _g = g, _h = h;
        _i = i, _j = j;
    }

    _sym_xy = ((e == 0) && (f == 0) && (i == 0));
    _sym_yz = ((d == 0) && (e == 0) && (g == 0));
    _sym_xz = ((d == 0) && (f == 0) && (h == 0));
    _sym_o = _sym_xy && _sym_yz && _sym_xz;
    _sym_z = _sym_xz && _sym_yz;
    _sym_x = _sym_xy && _sym_xz;
    _sym_y = _sym_xy && _sym_yz;

    _x0 = x0 < x1 ? (_x1 = x1, x0) : (_x1 = x0, x1);
    _y0 = y0 < y1 ? (_y1 = y1, y0) : (_y1 = y0, y1);
    _z0 = z0 < z1 ? (_z1 = z1, z0) : (_z1 = z0, z1);

    updateVertices();
}

void Quadric::updateVertices()
{
    // FIXME better way to find the bound of vertices of quadric surface
    _raw_vertices[0].x = _x0;
    _raw_vertices[0].y = _y0;
    _raw_vertices[0].z = _z0;

    _raw_vertices[1].x = _x1;
    _raw_vertices[1].y = _y0;
    _raw_vertices[1].z = _z0;

    _raw_vertices[2].x = _x0;
    _raw_vertices[2].y = _y1;
    _raw_vertices[2].z = _z0;

    _raw_vertices[3].x = _x0;
    _raw_vertices[3].y = _y0;
    _raw_vertices[3].z = _z1;

    _raw_vertices[4].x = _x1;
    _raw_vertices[4].y = _y1;
    _raw_vertices[4].z = _z0;

    _raw_vertices[5].x = _x1;
    _raw_vertices[5].y = _y0;
    _raw_vertices[5].z = _z1;

    _raw_vertices[6].x = _x0;
    _raw_vertices[6].y = _y1;
    _raw_vertices[6].z = _z1;

    _raw_vertices[7].x = _x1;
    _raw_vertices[7].y = _y1;
    _raw_vertices[7].z = _z1;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
