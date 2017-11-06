#include "object/geometry/cone.hpp"

using namespace px;

BaseCone::BaseCone(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &ideal_height,
                   PREC const &real_height)
        : _dev_obj(nullptr)
{
    setParams(center_of_bottom_face,
              radius_x, radius_y,
              ideal_height, real_height);
}

PX_CUDA_CALLABLE
GeometryObj * BaseCone::hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &t_start,
                         PREC const &t_end,
                         PREC &hit_at)
{
    auto o = reinterpret_cast<BaseCone*>(obj);

    auto xo = ray.original.x - o->_center.x;
    auto yo = ray.original.y - o->_center.y;
    auto zo = ray.original.z - o->_quadric_center_z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  o->_a * ray.direction.x * ray.direction.x +
              o->_b * ray.direction.y * ray.direction.y +
              o->_c * ray.direction.z * ray.direction.z;
    auto B = 2 * o->_a * xo * ray.direction.x +
             2 * o->_b * yo * ray.direction.y +
             2 * o->_c * zo * ray.direction.z;
    auto C =  o->_a * xo * xo +
              o->_b * yo * yo +
              o->_c * zo * zo;

    auto hit_top = false;

    auto tmp1 = (o->_center.z - ray.original.z) / ray.direction.z;
    auto tmp_x = ray.original.x + ray.direction.x * tmp1;
    auto tmp_y = ray.original.y + ray.direction.y * tmp1;

    if (tmp1 >= t_start && tmp1 <= t_end &&
            o->_a * (tmp_x - o->_center.x) * (tmp_x - o->_center.x) +
                o->_b * (tmp_y - o->_center.y) * (tmp_y - o->_center.y) <= 1)
    {
        hit_top= true;
        hit_at = tmp1;
    }

    if (o->_real_height != o->_ideal_height)
    {
        auto tmp2 = (o->_top_z - ray.original.z) / ray.direction.z;

        if (tmp2 >= t_start && tmp2 <= t_end &&
            (hit_at == false || tmp2 < tmp1))
        {
            tmp_x = ray.original.x + ray.direction.x * tmp2;
            tmp_y = ray.original.y + ray.direction.y * tmp2;

            if ((tmp_x - o->_center.x) * (tmp_x - o->_center.x) / (o->_top_r_x*o->_top_r_x)+
                (tmp_y - o->_center.y) * (tmp_y - o->_center.y) / (o->_top_r_y*o->_top_r_y) <= 1)
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
        if (iz >= o->_z0 && iz<=o->_z1)
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
Direction BaseCone::normalVec(void * const &obj,
                              PREC const &x, PREC const &y, PREC const &z,
                              bool &double_face)
{
    double_face = false;
    auto o = reinterpret_cast<BaseCone*>(obj);

    if (std::abs(z - o->_z0) < EPSILON)
        return {0, 0, -1};
    if (std::abs(z - o->_z1) < EPSILON)
        return {0, 0, 1};

    return {o->_a * (x - o->_center.x),
            o->_b * (y - o->_center.y),
            o->_c * (z - o->_center.z)};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseCone::getTextureCoord(void * const &obj,
                                     PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BaseCone*>(obj);
    return {x - o->_center.x, y - o->_center.y, 0};
}

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

        auto ratio = (_ideal_height - _real_height) / _ideal_height;
        _top_r_x = radius_x * ratio;
        _top_r_y = radius_y * ratio;

        if (ideal_height < 0)
        {
            _ideal_height *= -1;
            _real_height *= -1;
        }
    }

    _a =  1.0 / (radius_x*radius_x);
    _b =  1.0 / (radius_y*radius_y);
    _c = -1.0 / (_ideal_height*_ideal_height);

    _quadric_center_z = _center.z + _ideal_height;
    _z0 = _ideal_height < 0 ? (_z1 = _center.z, _center.z + _real_height)
                            : (_z1 = _center.z + _real_height, _center.z);

    _top_z = _center.z + _real_height;
}

std::shared_ptr<BaseGeometry> Cone::create(Point const &center_of_bottom_face,
                                           PREC const &radius_x,
                                           PREC const &radius_y,
                                           PREC const &ideal_height,
                                           PREC const &real_height,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Cone(center_of_bottom_face,
                                                  radius_x, radius_y,
                                                  ideal_height, real_height,
                                                  material, trans));
}

Cone::Cone(Point const &center_of_bottom_face,
           PREC const &radius_x,
           PREC const &radius_y,
           PREC const &ideal_height,
           PREC const &real_height,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseCone(center_of_bottom_face, radius_x, radius_y,
                   ideal_height, real_height)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Cone::~Cone()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}


#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_cone = BaseCone::hitCheck;
__device__ fnNormal_t __fn_normal_cone = BaseCone::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_cone = BaseCone::getTextureCoord;
#endif
void Cone::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseCone)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_cone, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_cone, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_cone, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseCone), cudaMemcpyHostToDevice));
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

void Cone::clearGpuData()
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

void Cone::setParams(Point const &center_of_bottom_face,
                     PREC const &radius_x, PREC const &radius_y,
                     PREC const &ideal_height,
                     PREC const &real_height)
{
    _obj->setParams(center_of_bottom_face,
                    radius_x, radius_y, ideal_height, real_height);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Cone::_updateVertices()
{
    if (_obj->_ideal_height == _obj->_real_height)
    {
        if (n_vertices != 5)
            resetVertices(5);
        raw_vertices[4].x = _obj->_center.x;
        raw_vertices[4].y = _obj->_center.y;
        raw_vertices[4].z = _obj->_center.z + _obj->_ideal_height;
    }
    else
    {
        if (n_vertices != 8)
            resetVertices(8);

        raw_vertices[4].x = _obj->_center.x - _obj->_top_r_x;
        raw_vertices[4].y = _obj->_center.y - _obj->_top_r_y;
        raw_vertices[4].z = _obj->_center.z + _obj->_real_height;
        raw_vertices[5].x = _obj->_center.x - _obj->_top_r_x;
        raw_vertices[5].y = _obj->_center.y + _obj->_top_r_y;
        raw_vertices[5].z = _obj->_center.z + _obj->_real_height;
        raw_vertices[6].x = _obj->_center.x + _obj->_top_r_x;
        raw_vertices[6].y = _obj->_center.y + _obj->_top_r_y;
        raw_vertices[6].z = _obj->_center.z + _obj->_real_height;
        raw_vertices[7].x = _obj->_center.x + _obj->_top_r_x;
        raw_vertices[7].y = _obj->_center.y - _obj->_top_r_y;
        raw_vertices[7].z = _obj->_center.z + _obj->_real_height;
    }

    raw_vertices[0].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[0].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[0].z = _obj->_center.z;
    raw_vertices[1].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[1].y = _obj->_center.y + _obj->_radius_y;
    raw_vertices[1].z = _obj->_center.z;
    raw_vertices[2].x = _obj->_center.x + _obj->_radius_x;
    raw_vertices[2].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[2].z = _obj->_center.z;
    raw_vertices[3].x = _obj->_center.x - _obj->_radius_x;
    raw_vertices[3].y = _obj->_center.y - _obj->_radius_y;
    raw_vertices[3].z = _obj->_center.z;
}

Vec3<PREC> Cone::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseCone::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Cone::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseCone::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Cone::normalVec(PREC const &x,
                          PREC const &y,
                          PREC const &z,
                          bool &double_face) const
{
    return BaseCone::normalVec(_obj, x, y, z, double_face);
}
