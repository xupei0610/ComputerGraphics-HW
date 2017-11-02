#include "object/geometry/quadric.hpp"

using namespace px;

BaseQuadric::BaseQuadric(Point const &center,
                         PREC const &a,
                         PREC const &b,
                         PREC const &c,
                         PREC const &d,
                         PREC const &e,
                         PREC const &f,
                         PREC const &g,
                         PREC const &h,
                         PREC const &i,
                         PREC const &j,
                         PREC const &x0, PREC const &x1,
                         PREC const &y0, PREC const &y1,
                         PREC const &z0, PREC const &z1)
        : _center(center), _dev_obj(nullptr)
{
    setCoef(a, b, c, d, e, f, g, h, i, j, x0, x1, y0, y1, z0, z1);
}

PX_CUDA_CALLABLE
GeometryObj * BaseQuadric::hitCheck(void * const &obj,
                           Ray const &ray,
                           PREC const &t_start,
                           PREC const &t_end,
                           PREC &hit_at)
{
    auto o = reinterpret_cast<BaseQuadric*>(obj);

    auto xo = ray.original.x - o->_center.x;
    auto yo = ray.original.y - o->_center.y;
    auto zo = ray.original.z - o->_center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A = o->_a * ray.direction.x * ray.direction.x +
             o->_b * ray.direction.y * ray.direction.y +
             o->_c * ray.direction.z * ray.direction.z +
             o->_d * ray.direction.x * ray.direction.y +
             o->_e * ray.direction.x * ray.direction.z +
             o->_f * ray.direction.y * ray.direction.z;
    auto B = 2 * o->_a * xo * ray.direction.x +
             2 * o->_b * yo * ray.direction.y +
             2 * o->_c * zo * ray.direction.z +
             o->_d * (xo * ray.direction.y + yo * ray.direction.x) +
             o->_e * (xo * ray.direction.z + zo * ray.direction.x) +
             o->_f * (yo * ray.direction.z + zo * ray.direction.y) +
             o->_g * ray.direction.x +
             o->_h * ray.direction.y +
             o->_i * ray.direction.z;
    auto C = o->_a * xo * xo +
             o->_b * yo * yo +
             o->_c * zo * zo +
             o->_d * xo * yo +
             o->_e * xo * zo +
             o->_f * yo * zo +
             o->_g * xo +
             o->_h * yo +
             o->_i * zo +
             o->_j;

    if (A == 0)
    {
        if (B == 0) return o->_dev_obj;

        C = - C / B;
        if (C > t_start && C < t_end)
        {
            B = ray.original.x + ray.direction.x * C;
            if (B > o->_x0 && B < o->_x1)
            {
                B = ray.original.y + ray.direction.y * C;
                if (B > o->_y0 && B < o->_y1)
                {
                    B = ray.original.z + ray.direction.z * C;
                    if (B > o->_z0 && B < o->_z1)
                    {
                        hit_at = C;
                        return o->_dev_obj;
                    }
                }
            }
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
        zo = xo;
        xo = yo;
        yo = zo;
    }

    if (xo > t_start && xo < t_end)
    {
        B = ray.original.x + ray.direction.x * xo;
        if (B > o->_x0 && B < o->_x1)
        {
            B = ray.original.y + ray.direction.y * xo;
            if (B > o->_y0 && B < o->_y1)
            {
                B =ray.original.z + ray.direction.z * xo;
                if (B > o->_z0 && B < o->_z1)
                {
                    hit_at = xo;
                    return o->_dev_obj;
                }
            }
        }
    }
    if (yo > t_start && yo < t_end)
    {
        B = ray.original.x + ray.direction.x * yo;
        if (B > o->_x0 && B < o->_x1)
        {
            B =ray.original.y + ray.direction.y * yo;
            if (B >o->_y0 && B < o->_y1)
            {
                B = ray.original.z + ray.direction.z * yo;
                if (B >o->_z0 && B < o->_z1)
                {
                    hit_at = yo;
                    return o->_dev_obj;
                }
            }
        }
    }

    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseQuadric::normalVec(void * const &obj,
                                 PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseQuadric*>(obj);

    auto dx = x - o->_center.x;
    auto dy = y - o->_center.y;
    auto dz = z - o->_center.z;

    return {2 * o->_a * dx + o->_d * dy + o->_e * dz + o->_g,
            2 * o->_b * dy + o->_d * dx + o->_f * dz + o->_h,
            2 * o->_c * dz + o->_e * dx + o->_f * dy + o->_i};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseQuadric::getTextureCoord(void * const &obj,
                                        PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseQuadric*>(obj);

    // FIXME better way for quadric surface texture mapping
    if (o->_sym_o)
    {
        if (o->_a == 0 && o->_b == 0)
            return {x-o->_center.x, y-o->_center.y, 0};
        if (o->_a == 0 && o->_c == 0)
            return {z-o->_center.z, x-o->_center.x, 0};
        if (o->_b == 0 && o->_c == 0)
            return {y-o->_center.y, z-o->_center.z, 0};
        if (o->_c == 0)
        {
            auto discriminant = o->_h*o->_h - 4*o->_b*o->_j;
            auto cy = discriminant < 0 ? 0 : (std::sqrt(discriminant) - o->_h)*PREC(0.5)/o->_b;

//            auto dx = x - o->_center.x;
//            auto dy = y - o->_center.y;
//            auto dz = z - o->_center.z;
//
//            auto du = (o->_a*dx*dx/2.0 + o->_c*dz*dz + o->_e*dx*dz/2.0 + o->_g*dx/2.0 + o->_i*dz + o->_j) * dx * (dy-cy) +
//                      (o->_d*dx*dx/2.0 + o->_f*dx*dz + o->_h*dx)/2.0 * (dy*dy - cy*cy) +
//                      o->_b*dx/3.0 * (dy*dy*dy - cy*cy*cy) ;

            return {x-o->_center.x > 0 ? y - cy : cy - y, z-o->_center.z, 0};
//            return {x-o->_center.x > 0 ? y - o->_center.y : o->_center.y - y, z-o->_center.z, 0};
//            return {du, z-o->_center.z, 0};
        }
        if (o->_b == 0)
        {
            auto discriminant = o->_g*o->_g - 4*o->_a*o->_j;
            auto cx = discriminant < 0 ? o->_center.x : ((std::sqrt(discriminant) - o->_g)*PREC(0.5)/o->_a + o->_center.x);
            return {z-o->_center.z > 0 ? x - cx : cx - x, y-o->_center.y, 0};
        }
        if (o->_a == 0)
        {
            auto discriminant = o->_i*o->_i - 4*o->_c*o->_j;
            auto cz = discriminant < 0 ? o->_center.z : ((std::sqrt(discriminant) - o->_i)*PREC(0.5)/o->_c + o->_center.z);
            return {y-o->_center.y > 0 ? z-cz : cz-z, x-o->_center.x, 0};
        }

        if (o->_a > 0 && o->_b > 0 && o->_c > 0)
        {
            auto dx = x - o->_center.x;
            auto dy = y - o->_center.y;
            auto dz = z - o->_center.z;

            return {(1 + std::atan2(dz, dx) / PREC(PI)) * PREC(0.5),
                    std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PREC(PI),
                    0};;
        }

        return {x - o->_center.x,
                y - o->_center.y,
                0};;
    }
    if (o->_sym_x)
    {
        return {y-o->_center.y, z-o->_center.z, 0};
    }
    if (o->_sym_y)
    {
        return {x-o->_center.x, z-o->_center.z, 0};
    }
    if (o->_sym_z)
    {
        return {x-o->_center.x, y-o->_center.y, 0};
    }

    if (o->_sym_xy)
        return {z-o->_center.z, x-o->_center.x, 0};
    if (o->_sym_yz)
        return {x-o->_center.x, y-o->_center.y, 0};
    if (o->_sym_xz)
        return {y-o->_center.y, x-o->_center.x, 0};

    return {x-o->_center.x, y-o->_center.y, 0};
}

void BaseQuadric::setCenter(Point const &center)
{
    _center = center;
}

void BaseQuadric::setCoef(PREC const &a,
                          PREC const &b,
                          PREC const &c,
                          PREC const &d,
                          PREC const &e,
                          PREC const &f,
                          PREC const &g,
                          PREC const &h,
                          PREC const &i,
                          PREC const &j,
                          PREC const &x0, PREC const &x1,
                          PREC const &y0, PREC const &y1,
                          PREC const &z0, PREC const &z1)
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
}

std::shared_ptr<BaseGeometry> Quadric::create(Point const &center,
                                              PREC const &a,
                                              PREC const &b,
                                              PREC const &c,
                                              PREC const &d,
                                              PREC const &e,
                                              PREC const &f,
                                              PREC const &g,
                                              PREC const &h,
                                              PREC const &i,
                                              PREC const &j,
                                              PREC const &x0, PREC const &x1,
                                              PREC const &y0, PREC const &y1,
                                              PREC const &z0, PREC const &z1,
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
                 PREC const &a,
                 PREC const &b,
                 PREC const &c,
                 PREC const &d,
                 PREC const &e,
                 PREC const &f,
                 PREC const &g,
                 PREC const &h,
                 PREC const &i,
                 PREC const &j,
                 PREC const &x0, PREC const &x1,
                 PREC const &y0, PREC const &y1,
                 PREC const &z0, PREC const &z1,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseQuadric(center, a, b, c, d, e, f, g, h, i, j, x0, x1, y0, y1, z0, z1)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Quadric::~Quadric()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_quadric= BaseQuadric::hitCheck;
__device__ fnNormal_t __fn_normal_quadric= BaseQuadric::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_quadric= BaseQuadric::getTextureCoord;
#endif
void Quadric::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseQuadric)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_quadric, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_quadric, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_quadric, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseQuadric), cudaMemcpyHostToDevice));
        _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);

        GeometryObj tmp(_gpu_obj, fn_hit_h, fn_normal_h, fn_texture_coord_h,
                        mat == nullptr ? nullptr : mat->devPtr(),
                        trans == nullptr ? nullptr : trans->devPtr());

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(GeometryObj),
                                 cudaMemcpyHostToDevice))

        _need_upload = false;
    }
#endif
}

void Quadric::clearGpuData()
{
#ifdef USE_CUDA
    BaseGeometry::clearGpuData();
    if (_gpu_obj != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_obj = nullptr;
    }
    _need_upload = true;
#endif
}

void Quadric::setCenter(Point const &center)
{
    _obj->setCenter(center);
//    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}


void Quadric::setCoef(PREC const &a,
                      PREC const &b,
                      PREC const &c,
                      PREC const &d,
                      PREC const &e,
                      PREC const &f,
                      PREC const &g,
                      PREC const &h,
                      PREC const &i,
                      PREC const &j,
                      PREC const &x0, PREC const &x1,
                      PREC const &y0, PREC const &y1,
                      PREC const &z0, PREC const &z1)
{
    _obj->setCoef(a, b, c, d, e, f, g, h, i, j, x0, x1, y0, y1, z0, z1);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Quadric::_updateVertices()
{
    // FIXME better way to find the bound of vertices of quadric surface
    raw_vertices[0].x = _obj->_x0;
    raw_vertices[0].y = _obj->_y0;
    raw_vertices[0].z = _obj->_z0;

    raw_vertices[1].x = _obj->_x1;
    raw_vertices[1].y = _obj->_y0;
    raw_vertices[1].z = _obj->_z0;

    raw_vertices[2].x = _obj->_x0;
    raw_vertices[2].y = _obj->_y1;
    raw_vertices[2].z = _obj->_z0;

    raw_vertices[3].x = _obj->_x0;
    raw_vertices[3].y = _obj->_y0;
    raw_vertices[3].z = _obj->_z1;

    raw_vertices[4].x = _obj->_x1;
    raw_vertices[4].y = _obj->_y1;
    raw_vertices[4].z = _obj->_z0;

    raw_vertices[5].x = _obj->_x1;
    raw_vertices[5].y = _obj->_y0;
    raw_vertices[5].z = _obj->_z1;

    raw_vertices[6].x = _obj->_x0;
    raw_vertices[6].y = _obj->_y1;
    raw_vertices[6].z = _obj->_z1;

    raw_vertices[7].x = _obj->_x1;
    raw_vertices[7].y = _obj->_y1;
    raw_vertices[7].z = _obj->_z1;
}

Vec3<PREC> Quadric::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseQuadric::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Quadric::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseQuadric::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Quadric::normalVec(PREC const &x, PREC const &y,
                           PREC const &z) const
{
    return BaseQuadric::normalVec(_obj, x, y, z);
}
