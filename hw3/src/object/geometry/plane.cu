#include "object/geometry/plane.hpp"

#include <cfloat>

using namespace px;

BasePlane::BasePlane(Point const &pos,
                     Direction const &norm_vec)
        : _pos(pos), _dev_obj(nullptr)
{
    setNormal(norm_vec);
}

PX_CUDA_CALLABLE
GeometryObj *BasePlane::hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &t_start,
                         PREC const &t_end,
                         PREC &hit_at)
{
    auto o = reinterpret_cast<BasePlane*>(obj);
    auto tmp = (o->_p_dot_n - ray.original.dot(o->_norm)) / ray.direction.dot(o->_norm);
    return (tmp > t_start && tmp < t_end) ? (hit_at = tmp, o->_dev_obj) : nullptr;
}

PX_CUDA_CALLABLE
Direction BasePlane::normalVec(void * const &obj,
                               PREC const &x, PREC const &y, PREC const &z,
                               bool &double_face)
{
    double_face = true;
    return reinterpret_cast<BasePlane*>(obj)->_norm;
}

PX_CUDA_CALLABLE
Vec3<PREC> BasePlane::getTextureCoord(void * const &obj,
                                      PREC const &x, PREC const &y, PREC const &z)
{
    auto o = reinterpret_cast<BasePlane*>(obj);

    return {x - o->_pos.x,
            o->_norm.y*(z - o->_pos.z)-o->_norm.z*(y - o->_pos.y),
            (x - o->_pos.x)*o->_norm.x + (y - o->_pos.y)*o->_norm.y + (z - o->_pos.z)*o->_norm.z};
}

void BasePlane::setPos(Point const &position)
{
    _pos = position;
    _p_dot_n = position.dot(_norm);
}

void BasePlane::setNormal(Direction const &norm_vec)
{
    _norm = norm_vec;
    _p_dot_n = _pos.dot(norm_vec);
}

std::shared_ptr<BaseGeometry> Plane::create(Point const &position,
                                            Direction const &norm_vec,
                                            std::shared_ptr<BaseMaterial> const &material,
                                            std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Plane(position, norm_vec, material, trans));
}

Plane::Plane(Point const &position,
             Direction const &norm_vec,
             std::shared_ptr<BaseMaterial> const &material,
             std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 4),
          _obj(new BasePlane(position, norm_vec)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Plane::~Plane()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_plane = BasePlane::hitCheck;
__device__ fnNormal_t __fn_normal_plane = BasePlane::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_plane = BasePlane::getTextureCoord;
#endif
void Plane::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BasePlane)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_plane, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_plane, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_plane, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BasePlane), cudaMemcpyHostToDevice));
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

void Plane::clearGpuData()
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

void Plane::setPos(Point const &position)
{
    _obj->setPos(position);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Plane::setNormal(Direction const &norm_vec)
{
    _obj->setNormal(norm_vec);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Plane::_updateVertices()
{
    if ((_obj->_norm.x == 1 || _obj->_norm.x == -1) && _obj->_norm.y == 0 && _obj->_norm.z == 0)
    {
        raw_vertices[0].x = 0;
        raw_vertices[0].y = -FLT_MAX;
        raw_vertices[0].z = -FLT_MAX;
        raw_vertices[1].x = 0;
        raw_vertices[1].y =  FLT_MAX;
        raw_vertices[1].z =  FLT_MAX;
        raw_vertices[2].x = 0;
        raw_vertices[2].y = -FLT_MAX;
        raw_vertices[2].z =  FLT_MAX;
        raw_vertices[3].x = 0;
        raw_vertices[3].y =  FLT_MAX;
        raw_vertices[3].z = -FLT_MAX;
    }
    else if (_obj->_norm.x == 0 && (_obj->_norm.y == 1 || _obj->_norm.y == -1) && _obj->_norm.z == 0)
    {
        raw_vertices[0].x = -FLT_MAX;
        raw_vertices[0].y = 0;
        raw_vertices[0].z = -FLT_MAX;
        raw_vertices[1].x =  FLT_MAX;
        raw_vertices[1].y = 0;
        raw_vertices[1].z =  FLT_MAX;
        raw_vertices[2].x = -FLT_MAX;
        raw_vertices[2].y = 0;
        raw_vertices[2].z =  FLT_MAX;
        raw_vertices[3].x =  FLT_MAX;
        raw_vertices[3].y = 0;
        raw_vertices[3].z = -FLT_MAX;
    }
    else if (_obj->_norm.x == 0 && _obj->_norm.y == 0 && (_obj->_norm.z == 1 || _obj->_norm.z == -1))
    {
        raw_vertices[0].x = -FLT_MAX;
        raw_vertices[0].y = -FLT_MAX;
        raw_vertices[0].z = 0;
        raw_vertices[1].x =  FLT_MAX;
        raw_vertices[1].y =  FLT_MAX;
        raw_vertices[1].z = 0;
        raw_vertices[2].x = -FLT_MAX;
        raw_vertices[2].y =  FLT_MAX;
        raw_vertices[2].z = 0;
        raw_vertices[3].x =  FLT_MAX;
        raw_vertices[3].y = -FLT_MAX;
        raw_vertices[3].z = 0;
    }
    else if (_obj->_norm.x == 0 && _obj->_norm.y == 0 && _obj->_norm.z == 0)
    {
        raw_vertices[0].x = 0;
        raw_vertices[0].y = 0;
        raw_vertices[0].z = 0;
        raw_vertices[1].x = 0;
        raw_vertices[1].y = 0;
        raw_vertices[1].z = 0;
        raw_vertices[2].x = 0;
        raw_vertices[2].y = 0;
        raw_vertices[2].z = 0;
        raw_vertices[3].x = 0;
        raw_vertices[3].y = 0;
        raw_vertices[3].z = 0;
    }
    else
    {
        raw_vertices[0].x = -FLT_MAX;
        raw_vertices[0].y = -FLT_MAX;
        raw_vertices[0].z = -FLT_MAX;
        raw_vertices[1].x =  FLT_MAX;
        raw_vertices[1].y =  FLT_MAX;
        raw_vertices[1].z =  FLT_MAX;
        raw_vertices[2].x = -FLT_MAX;
        raw_vertices[2].y =  FLT_MAX;
        raw_vertices[2].z =  FLT_MAX;
        raw_vertices[3].x =  FLT_MAX;
        raw_vertices[3].y = -FLT_MAX;
        raw_vertices[3].z = -FLT_MAX;
    }
}

Vec3<PREC> Plane::getTextureCoord(PREC const &x,
                                     PREC const &y,
                                     PREC const &z) const
{
    return BasePlane::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Plane::hitCheck(Ray const &ray,
                                       PREC const &t_start,
                                       PREC const &t_end,
                                       PREC &hit_at) const
{
    return BasePlane::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Plane::normalVec(PREC const &x, PREC const &y,
                              PREC const &z,
                           bool &double_face) const
{
    return BasePlane::normalVec(_obj, x, y, z, double_face);
}
