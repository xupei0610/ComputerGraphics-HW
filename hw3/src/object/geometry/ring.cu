#include "object/geometry/ring.hpp"

using namespace px;

BaseRing::BaseRing(Point const &pos,
                   Direction const &norm_vec,
                   PREC const &radius1,
                   PREC const &radius2)
        : _center(pos), _norm(norm_vec),
          _inner_radius(radius1 < radius2 ? radius1 : radius2),
          _outer_radius(radius1 > radius2 ? radius1 : radius2),
          _inner_radius2(_inner_radius*_inner_radius),
          _outer_radius2(_outer_radius*_outer_radius),
          _p_dot_n(pos.dot(norm_vec)),
          _dev_obj(nullptr)
{}

PX_CUDA_CALLABLE
GeometryObj *BaseRing::hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &t_start,
                         PREC const &t_end,
                         PREC &hit_at)
{
    auto o = reinterpret_cast<BaseRing*>(obj);

    auto tmp = (o->_p_dot_n - ray.original.dot(o->_norm)) / ray.direction.dot(o->_norm);
    if (tmp > t_start && tmp < t_end)
    {
        auto intersect = ray[tmp];
        auto dist2 = (intersect.x - o->_center.x) * (intersect.x - o->_center.x) +
                     (intersect.y - o->_center.y) * (intersect.y - o->_center.y) +
                     (intersect.z - o->_center.z) * (intersect.z - o->_center.z);
        if (dist2 <= o->_outer_radius2 && dist2 >= o->_inner_radius2)
        {
            hit_at = tmp;
            return o->_dev_obj;
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseRing::normalVec(void * const &obj,
                              PREC const &x, PREC const &y, PREC const &z)
{
    return reinterpret_cast<BaseRing*>(obj)->_norm;
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseRing::getTextureCoord(void * const &obj,
                                     PREC const &x, PREC const &y,
                                     PREC const &z)
{
    auto o = reinterpret_cast<BaseRing*>(obj);
    return {x - o->_center.x,
            o->_norm.y*(z - o->_center.z) - o->_center.z*(y - o->_center.y) ,
            (x - o->_center.x)*o->_norm.x + (y - o->_center.y)*o->_norm.y + (z - o->_center.z)*o->_norm.z};
}


void BaseRing::setCenter(Point const &center)
{
    _center = center;
    _p_dot_n = center.dot(_norm);
}

void BaseRing::setNormal(Direction const &norm_vec)
{
    _norm = norm_vec;
    _p_dot_n = _center.dot(norm_vec);
}

void BaseRing::setRadius(PREC const &radius1, PREC const &radius2)
{
    _inner_radius = std::min(radius1, radius2);
    _outer_radius = std::max(radius1, radius2);
    _inner_radius2 = _inner_radius*_inner_radius;
    _outer_radius2 = _outer_radius*_outer_radius;
}


std::shared_ptr<BaseGeometry> Ring::create(Point const &position,
                                           Direction const &norm_vec,
                                           PREC const &radius1,
                                           PREC const &radius2,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Ring(position, norm_vec,
                                                  radius1, radius2,
                                                  material, trans));
}

Ring::Ring(Point const &position,
           Direction const &norm_vec,
           PREC const &radius1,
           PREC const &radius2,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseRing(position, norm_vec, radius1, radius2)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Ring::~Ring()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}


#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_ring = BaseRing::hitCheck;
__device__ fnNormal_t __fn_normal_ring = BaseRing::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_ring = BaseRing::getTextureCoord;
#endif
void Ring::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseRing)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_ring, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_ring, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_ring, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseRing), cudaMemcpyHostToDevice));
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

void Ring::clearGpuData()
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


void Ring::setCenter(Point const &center)
{
    _obj->setCenter(center);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ring::setNormal(Direction const &norm)
{
    _obj->setNormal(norm);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ring::setRadius(PREC const &radius1, PREC const &radius2)
{
    _obj->setRadius(radius1, radius2);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Ring::_updateVertices()
{
    raw_vertices[0].x = _obj->_center.x + _obj->_outer_radius;
    raw_vertices[0].y = _obj->_center.y + _obj->_outer_radius;
    raw_vertices[0].z = _obj->_center.z + _obj->_outer_radius;

    raw_vertices[1].x = _obj->_center.x - _obj->_outer_radius;
    raw_vertices[1].y = _obj->_center.y + _obj->_outer_radius;
    raw_vertices[1].z = _obj->_center.z + _obj->_outer_radius;

    raw_vertices[2].x = _obj->_center.x + _obj->_outer_radius;
    raw_vertices[2].y = _obj->_center.y - _obj->_outer_radius;
    raw_vertices[2].z = _obj->_center.z + _obj->_outer_radius;

    raw_vertices[3].x = _obj->_center.x + _obj->_outer_radius;
    raw_vertices[3].y = _obj->_center.y + _obj->_outer_radius;
    raw_vertices[3].z = _obj->_center.z - _obj->_outer_radius;

    raw_vertices[4].x = _obj->_center.x - _obj->_outer_radius;
    raw_vertices[4].y = _obj->_center.y - _obj->_outer_radius;
    raw_vertices[4].z = _obj->_center.z + _obj->_outer_radius;

    raw_vertices[5].x = _obj->_center.x - _obj->_outer_radius;
    raw_vertices[5].y = _obj->_center.y + _obj->_outer_radius;
    raw_vertices[5].z = _obj->_center.z - _obj->_outer_radius;

    raw_vertices[6].x = _obj->_center.x + _obj->_outer_radius;
    raw_vertices[6].y = _obj->_center.y - _obj->_outer_radius;
    raw_vertices[6].z = _obj->_center.z - _obj->_outer_radius;

    raw_vertices[7].x = _obj->_center.x - _obj->_outer_radius;
    raw_vertices[7].y = _obj->_center.y - _obj->_outer_radius;
    raw_vertices[7].z = _obj->_center.z - _obj->_outer_radius;
}

Vec3<PREC> Ring::getTextureCoord(PREC const &x,
                                 PREC const &y,
                                 PREC const &z) const
{
    return BaseRing::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Ring::hitCheck(Ray const &ray,
                                   PREC const &t_start,
                                   PREC const &t_end,
                                   PREC &hit_at) const
{
    return BaseRing::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Ring::normalVec(PREC const &x, PREC const &y,
                          PREC const &z) const
{
    return BaseRing::normalVec(_obj, x, y, z);
}
