#include "object/geometry/sphere.hpp"

using namespace px;

BaseSphere::BaseSphere(Point const &pos,
                       PREC const &radius)
        : _center(pos),
          _radius(radius),
          _radius2(radius*radius),
          _dev_obj(nullptr)
{}

PX_CUDA_CALLABLE
GeometryObj *BaseSphere::hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &t_start,
                         PREC const &t_end,
                         PREC &hit_at)
{
    auto o = reinterpret_cast<BaseSphere*>(obj);

    auto oc = Vec3<PREC>(ray.original.x - o->_center.x,
                           ray.original.y - o->_center.y,
                           ray.original.z - o->_center.z);
    auto a = ray.direction.dot(ray.direction);
    auto b = ray.direction.dot(oc);
    auto c = oc.dot(oc) - o->_radius*o->_radius;
    auto discriminant = b*b - a*c;
    if (discriminant > 0)
    {
        auto tmp = -std::sqrt(discriminant)/a;
        auto b_by_a = -b/a;
        tmp += b_by_a;
        if (tmp > t_start && tmp < t_end)
        {
            hit_at = tmp;
            return o->_dev_obj;
        }
        else
        {
            tmp = b_by_a+b_by_a-tmp;
            if (tmp > t_start && tmp < t_end)
            {
                hit_at = tmp;
                return o->_dev_obj;
            }
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseSphere::normalVec(void * const &obj,
                               PREC const &x, PREC const &y, PREC const &z,
                                bool &double_face)
{
    double_face = false;
    auto o = reinterpret_cast<BaseSphere*>(obj);
    return {x - o->_center.x, y - o->_center.y, z - o->_center.z};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseSphere::getTextureCoord(void * const &obj,
                                       PREC const &x, PREC const &y,
                                       PREC const &z)
{
    auto o = reinterpret_cast<BaseSphere*>(obj);
    return {(1 + std::atan2(z - o->_center.z, x - o->_center.x) / PREC(PI)) *PREC(0.5),
            std::acos((y - o->_center.y) / o->_radius2) / PREC(PI),
            0};
}

void BaseSphere::setCenter(Point const &center)
{
    _center = center;
}

void BaseSphere::setRadius(PREC const &r)
{
    _radius = r;
    _radius2 = r*r;
}

std::shared_ptr<BaseGeometry> Sphere::create(Point const &position,
                                             PREC const &radius,
                                             std::shared_ptr<BaseMaterial> const &material,
                                             std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Sphere(position, radius,
                                                    material, trans));
}

Sphere::Sphere(Point const &position,
               PREC const &radius,
               std::shared_ptr<BaseMaterial> const &material,
               std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseSphere(position, radius)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Sphere::~Sphere()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_sphere = BaseSphere::hitCheck;
__device__ fnNormal_t __fn_normal_sphere = BaseSphere::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_sphere = BaseSphere::getTextureCoord;
#endif
void Sphere::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseSphere)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_sphere, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_sphere, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_sphere, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseSphere), cudaMemcpyHostToDevice));
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

void Sphere::clearGpuData()
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

void Sphere::setCenter(Point const &center)
{
    _obj->setCenter(center);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Sphere::setRadius(PREC const &r)
{
    _obj->setRadius(r);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Sphere::_updateVertices()
{
    raw_vertices[0].x = _obj->_center.x + _obj->_radius;
    raw_vertices[0].y = _obj->_center.y + _obj->_radius;
    raw_vertices[0].z = _obj->_center.z + _obj->_radius;

    raw_vertices[1].x = _obj->_center.x - _obj->_radius;
    raw_vertices[1].y = _obj->_center.y + _obj->_radius;
    raw_vertices[1].z = _obj->_center.z + _obj->_radius;

    raw_vertices[2].x = _obj->_center.x + _obj->_radius;
    raw_vertices[2].y = _obj->_center.y - _obj->_radius;
    raw_vertices[2].z = _obj->_center.z + _obj->_radius;

    raw_vertices[3].x = _obj->_center.x + _obj->_radius;
    raw_vertices[3].y = _obj->_center.y + _obj->_radius;
    raw_vertices[3].z = _obj->_center.z - _obj->_radius;

    raw_vertices[4].x = _obj->_center.x - _obj->_radius;
    raw_vertices[4].y = _obj->_center.y - _obj->_radius;
    raw_vertices[4].z = _obj->_center.z + _obj->_radius;

    raw_vertices[5].x = _obj->_center.x - _obj->_radius;
    raw_vertices[5].y = _obj->_center.y + _obj->_radius;
    raw_vertices[5].z = _obj->_center.z - _obj->_radius;

    raw_vertices[6].x = _obj->_center.x + _obj->_radius;
    raw_vertices[6].y = _obj->_center.y - _obj->_radius;
    raw_vertices[6].z = _obj->_center.z - _obj->_radius;

    raw_vertices[7].x = _obj->_center.x - _obj->_radius;
    raw_vertices[7].y = _obj->_center.y - _obj->_radius;
    raw_vertices[7].z = _obj->_center.z - _obj->_radius;
}

Vec3<PREC> Sphere::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseSphere::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Sphere::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseSphere::hitCheck(_obj, ray, t_start, t_end, hit_at) == nullptr ? nullptr : this;
}
Direction Sphere::normalVec(PREC const &x, PREC const &y,
                           PREC const &z,
                            bool &double_face) const
{
    return BaseSphere::normalVec(_obj, x, y, z, double_face);
}
