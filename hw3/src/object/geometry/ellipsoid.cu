#include "object/geometry/ellipsoid.hpp"

using namespace px;

BaseEllipsoid::BaseEllipsoid(Point const &center,
                             PREC const &radius_x,
                             PREC const &radius_y,
                             PREC const &radius_z)
        : _dev_obj(nullptr)
{
    setParams(center,
              radius_x, radius_y, radius_z);
}

PX_CUDA_CALLABLE
GeometryObj *BaseEllipsoid::hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &t_start,
                         PREC const &t_end,
                         PREC &hit_at)
{
    auto o = reinterpret_cast<BaseEllipsoid*>(obj);

    auto xo = ray.original.x - o->_center.x;
    auto yo = ray.original.y - o->_center.y;
    auto zo = ray.original.z - o->_center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  o->_a * ray.direction.x * ray.direction.x +
              o->_b * ray.direction.y * ray.direction.y +
              o->_c * ray.direction.z * ray.direction.z;
    auto B =  2 * o->_a * xo * ray.direction.x +
              2 * o->_b * yo * ray.direction.y +
              2 * o->_c * zo * ray.direction.z;
    auto C =  o->_a * xo * xo +
              o->_b * yo * yo +
              o->_c * zo * zo +
              -1;

    if (A == 0)
    {
        if (B == 0) return nullptr;

        C = - C / B;
        if (C > t_start && C < t_end)
        {
            hit_at = C;
            return o->_dev_obj;
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
        return o->_dev_obj;
    }
    if (yo > t_start && yo < t_end)
    {
        hit_at = yo;
        return o->_dev_obj;
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseEllipsoid::normalVec(void * const &obj,
                                   PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseEllipsoid*>(obj);

    return {o->_a * (x - o->_center.x),
            o->_b * (y - o->_center.y),
            o->_c * (z - o->_center.z)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseEllipsoid::getTextureCoord(void * const &obj,
                                          PREC const &x, PREC const &y,
                                          PREC const &z)
{
    auto o = reinterpret_cast<BaseEllipsoid*>(obj);

    auto dx = x - o->_center.x;
    auto dy = y - o->_center.y;
    auto dz = z - o->_center.z;

    return {(1 + std::atan2(dz, dx) / PREC(PI)) * PREC(0.5),
            std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PREC(PI),
            0};;
}


void BaseEllipsoid::setParams(Point const &center,
                          PREC const &radius_x,
                          PREC const &radius_y,
                          PREC const &radius_z)
{
    _center = center;
    _radius_x = radius_x;
    _radius_y = radius_y;
    _radius_z = radius_z;

    _a = 1.0 / (radius_x * radius_x);
    _b = 1.0 / (radius_y * radius_y);
    _c = 1.0 / (radius_z * radius_z);
}

std::shared_ptr<BaseGeometry> Ellipsoid::create(Point const &center,
                                                PREC const &radius_x, PREC const &radius_y,
                                                PREC const &radius_z,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Ellipsoid(center,
                                                      radius_x, radius_y, radius_z,
                                                      material, trans));
}

Ellipsoid::Ellipsoid(Point const &center,
                     PREC const &radius_x, PREC const &radius_y,
                     PREC const &radius_z,
                     std::shared_ptr<BaseMaterial> const &material,
                     std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseEllipsoid(center, radius_x, radius_y, radius_z)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Ellipsoid::~Ellipsoid()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}


#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_ellipsoid = BaseEllipsoid::hitCheck;
__device__ fnNormal_t __fn_normal_ellipsoid = BaseEllipsoid::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_ellipsoid = BaseEllipsoid::getTextureCoord;
#endif
void Ellipsoid::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseEllipsoid)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_ellipsoid, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_ellipsoid, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_ellipsoid, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseEllipsoid), cudaMemcpyHostToDevice));
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

void Ellipsoid::clearGpuData()
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

void Ellipsoid::setParams(Point const &center,
                              PREC const &radius_x,
                              PREC const &radius_y,
                              PREC const &radius_z)
{
    _obj->setParams(center, radius_x, radius_y, radius_z);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ellipsoid::_updateVertices()
{
    raw_vertices[4].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[4].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[4].z = _obj->_center.x - _obj->_radius_x;
    raw_vertices[5].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[5].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[5].z = _obj->_center.x - _obj->_radius_x;
    raw_vertices[6].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[6].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[6].z = _obj->_center.x - _obj->_radius_x;
    raw_vertices[7].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[7].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[7].z = _obj->_center.x - _obj->_radius_x;

    raw_vertices[0].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[0].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[0].z = _obj->_center.z - _obj->_radius_z;
    raw_vertices[1].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[1].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[1].z = _obj->_center.z - _obj->_radius_z;
    raw_vertices[2].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[2].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[2].z = _obj->_center.z - _obj->_radius_z;
    raw_vertices[3].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[3].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[3].z = _obj->_center.z - _obj->_radius_z;
}

Vec3<PREC> Ellipsoid::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseEllipsoid::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Ellipsoid::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseEllipsoid::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Ellipsoid::normalVec(PREC const &x, PREC const &y,
                           PREC const &z) const
{
    return BaseEllipsoid::normalVec(_obj, x, y, z);
}
