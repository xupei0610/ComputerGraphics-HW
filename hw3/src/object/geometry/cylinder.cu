#include "object/geometry/cylinder.hpp"

#include <cfloat>

using namespace px;

BaseCylinder::BaseCylinder(Point const &center_of_bottom_face,
                           PREC const &radius_x, PREC const &radius_y,
                           PREC const &height)
        : _dev_obj(nullptr)
{
    setParams(center_of_bottom_face,
              radius_x, radius_y,
              height);
}

PX_CUDA_CALLABLE
GeometryObj *BaseCylinder::hitCheck(void * const &obj,
                            Ray const &ray,
                            PREC const &t_start,
                            PREC const &t_end,
                            PREC &hit_at)
{
    auto o = reinterpret_cast<BaseCylinder*>(obj);

    auto xo = ray.original.x - o->_center.x;
    auto yo = ray.original.y - o->_center.y;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A = o->_a * ray.direction.x * ray.direction.x +
             o->_b * ray.direction.y * ray.direction.y;
    auto B = 2 * o->_a * xo * ray.direction.x +
             2 * o->_b * yo * ray.direction.y;
    auto C = o->_a * xo * xo +
             o->_b * yo * yo - 1;

    bool hit_top = false;

    auto tmp1 = (o->_z1 - ray.original.z) / ray.direction.z;
    auto tmp_x = ray.original.x + ray.direction.x * tmp1;
    auto tmp_y = ray.original.y + ray.direction.y * tmp1;

    if (tmp1 >= t_start && tmp1 <= t_end &&
            o->_a * (tmp_x - o->_center.x) * (tmp_x - o->_center.x) +
                    o->_b * (tmp_y - o->_center.y) * (tmp_y - o->_center.y) <= 1)
    {
        hit_top = true;
        hit_at = tmp1;
    }

    auto tmp2 = (o->_z0 - ray.original.z) / ray.direction.z;

    if (tmp1 >= t_start && tmp1 <= t_end &&
        (hit_top == false || tmp2 < tmp1))
    {
        tmp_x = ray.original.x + ray.direction.x * tmp2;
        tmp_y = ray.original.y + ray.direction.y * tmp2;

        if (o->_a * (tmp_x - o->_center.x) * (tmp_x - o->_center.x) +
                    o->_b * (tmp_y - o->_center.y) * (tmp_y - o->_center.y) <= 1)
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
            if (iz >= o->_z0 && iz<=o->_z1)
            {
                if (hit_top == false || hit_at > tmp)
                    hit_at = tmp;
                return o->_dev_obj;
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
    {
        auto tmp = tmp1;
        tmp1 = tmp2;
        tmp2 = tmp;
    }
    if (tmp1 > t_start && tmp1 < t_end)
    {
        auto iz = ray.original.z + ray.direction.z*tmp1;
        if (iz >= o->_z0 && iz <= o->_z1)
        {
            if (hit_top == false || hit_at > tmp1)
                hit_at = tmp1;
            return o->_dev_obj;
        }
    }
    if (tmp2 > t_start && tmp2 < t_end)
    {
        auto iz = ray.original.z + ray.direction.z*tmp2;
        if (iz >= o->_z0 && iz <= o->_z1)
        {
            if (hit_top == false || hit_at > tmp2)
                hit_at = tmp2;
            return o->_dev_obj;
        }
    }

    return hit_top ? o->_dev_obj : nullptr;
}

PX_CUDA_CALLABLE
Direction BaseCylinder::normalVec(void * const &obj,
                                  PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseCylinder*>(obj);

    if (std::abs(z - o->_z0) < EPSILON)
        return {0, 0, -1};
    if (std::abs(z - o->_z1) < EPSILON)
        return {0, 0, 1};

    return {o->_a * (x - o->_center.x),
            o->_b * (y - o->_center.y),
            0};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseCylinder::getTextureCoord(void * const &obj,
                                         PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseCylinder*>(obj);

    if (std::abs(z - o->_z0) < EPSILON)
        return {x - o->_center.x,
                o->_radius_y + y - o->_center.y, 0};
    if (std::abs(z - o->_z1) < EPSILON)
        return {x - o->_center.x,
                o->_radius_y + o->_radius_y + o->_radius_y + o->_abs_height + y - o->_center.y, 0};

    auto dx = x - o->_center.x;
    auto dy = y - o->_center.y - o->_radius_y;

    return {((o->_a/PREC(3.0) * dx * dx * dx - dx) + o->_b/PREC(3.0) * dy * dy * dy),
            o->_radius_y + o->_radius_y + z - o->_center.z, 0};
}

void BaseCylinder::setParams(Point const &center_of_bottom_face,
                             PREC const &radius_x, PREC const &radius_y,
                             PREC const &height)
{
    _center = center_of_bottom_face;
    _radius_x = std::abs(radius_x);
    _radius_y = std::abs(radius_y);
    _height = height;
    _abs_height = std::abs(height);

    _a =  1.0 / (radius_x*radius_x);
    _b =  1.0 / (radius_y*radius_y);

    _z0 = height < 0 ? (_z1 = _center.z, _center.z + _height)
                     : (_z1 = _center.z + _height, _center.z);
}

std::shared_ptr<BaseGeometry> Cylinder::create(Point const &center_of_bottom_face,
                                               PREC const &radius_x, PREC const &radius_y,
                                               PREC const &height,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Cylinder(center_of_bottom_face,
                                                      radius_x, radius_y,
                                                      height,
                                                      material, trans));
}

Cylinder::Cylinder(Point const &center_of_bottom_face,
                   PREC const &radius_x, PREC const &radius_y,
                   PREC const &height,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseCylinder(center_of_bottom_face, radius_x, radius_y, height)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Cylinder::~Cylinder()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_cylinder = BaseCylinder::hitCheck;
__device__ fnNormal_t __fn_normal_cylinder = BaseCylinder::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_cylinder = BaseCylinder::getTextureCoord;
#endif
void Cylinder::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseCylinder)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_cylinder, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_cylinder, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_cylinder, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseCylinder), cudaMemcpyHostToDevice));
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

void Cylinder::clearGpuData()
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

void Cylinder::setParams(Point const &center_of_bottom_face,
                         PREC const &radius_x, PREC const &radius_y,
                         PREC const &height)
{
    _obj->setParams(center_of_bottom_face,
                    radius_x, radius_y,
                    height);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Cylinder::_updateVertices()
{

    auto top = _obj->_center.z + _obj->_height;;
    raw_vertices[4].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[4].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[4].z = top;
    raw_vertices[5].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[5].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[5].z = top;
    raw_vertices[6].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[6].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[6].z = top;
    raw_vertices[7].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[7].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[7].z = top;

    raw_vertices[0].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[0].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[0].z = _obj->_center.z;
    raw_vertices[1].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[1].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[1].z = _obj->_center.z;
    raw_vertices[2].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[2].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[2].z = _obj->_center.z;
    raw_vertices[3].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[3].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[3].z = _obj->_center.z;
}

Vec3<PREC> Cylinder::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseCylinder::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Cylinder::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseCylinder::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Cylinder::normalVec(PREC const &x, PREC const &y,
                           PREC const &z) const
{
    return BaseCylinder::normalVec(_obj, x, y, z);
}
