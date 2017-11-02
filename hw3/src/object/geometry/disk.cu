#include "object/geometry/disk.hpp"

using namespace px;

BaseDisk::BaseDisk(Point const &pos,
                   Direction const &norm_vec,
                   PREC const &radius)
        : _center(pos), _norm(norm_vec),
          _radius(radius), _radius2(radius*radius),
          _p_dot_n(pos.dot(norm_vec)),
          _dev_obj(nullptr)
{}

PX_CUDA_CALLABLE
GeometryObj *BaseDisk::hitCheck(void * const &obj,
                        Ray const &ray,
                        PREC const &t_start,
                        PREC const &t_end,
                        PREC &hit_at)
{
    auto o = reinterpret_cast<BaseDisk*>(obj);

    auto tmp = (o->_p_dot_n - ray.original.dot(o->_norm)) / ray.direction.dot(o->_norm);
    if (tmp > t_start && tmp < t_end)
    {
        auto intersect = ray[tmp];
        if ((intersect.x - o->_center.x) * (intersect.x - o->_center.x) +
            (intersect.y - o->_center.y) * (intersect.y - o->_center.y) +
            (intersect.z - o->_center.z) * (intersect.z - o->_center.z) <= o->_radius2)
        {
            hit_at = tmp;
            return o->_dev_obj;
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseDisk::normalVec(void * const &obj,
                                  PREC const &x, PREC const &y, PREC const &z)
{
    return reinterpret_cast<BaseDisk*>(obj)->_norm;
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseDisk::getTextureCoord(void * const &obj,
                                         PREC const &x, PREC const &y,
                                         PREC const &z)
{
    auto o = reinterpret_cast<BaseDisk*>(obj);
    return {x - o->_center.x,
            o->_norm.y*(z - o->_center.z) - o->_norm.z*(y - o->_center.y) ,
            (x - o->_center.x)*o->_norm.x + (y - o->_center.y)*o->_norm.y + (z - o->_center.z)*o->_norm.z};
}

void BaseDisk::setCenter(Point const &center)
{
    _center = center;
    _p_dot_n = center.dot(_norm);
}

void BaseDisk::setNormal(Direction const &norm_vec)
{
    _norm = norm_vec;
    _p_dot_n = _center.dot(norm_vec);
}

void BaseDisk::setRadius(PREC const &radius)
{
    _radius = radius;
    _radius2 = radius*radius;
}

std::shared_ptr<BaseGeometry> Disk::create(Point const &position,
                                           Direction const &norm_vec,
                                           PREC const &radius,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Disk(position, norm_vec, radius,
                                                  material, trans));
}

Disk::Disk(Point const &position,
           Direction const &norm_vec,
           PREC const &radius,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          _obj(new BaseDisk(position, norm_vec, radius)),
          _gpu_obj(nullptr), _need_upload(true)
{
    _obj->_dev_obj = reinterpret_cast<GeometryObj*>(this);
    _updateVertices();
}

Disk::~Disk()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnHit_t __fn_hit_disk = BaseDisk::hitCheck;
__device__ fnNormal_t __fn_normal_disk = BaseDisk::normalVec;
__device__ fnTextureCoord_t __fn_texture_coord_disk = BaseDisk::getTextureCoord;
#endif
void Disk::up2Gpu()
{
#ifdef USE_CUDA
    static fnHit_t fn_hit_h = nullptr;
    static fnNormal_t fn_normal_h;
    static fnTextureCoord_t fn_texture_coord_h;

    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseDisk)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(GeometryObj)));
        }

        if (fn_hit_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_hit_h, __fn_hit_disk, sizeof(fnHit_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_normal_h, __fn_normal_disk, sizeof(fnNormal_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_texture_coord_h, __fn_texture_coord_disk, sizeof(fnTextureCoord_t)));
        }

        if (mat != nullptr)
            mat->up2Gpu();

        if (trans != nullptr)
            trans->up2Gpu();

        cudaDeviceSynchronize();

        _obj->_dev_obj = dev_ptr;
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseDisk), cudaMemcpyHostToDevice));
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

void Disk::clearGpuData()
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

void Disk::setCenter(Point const &center)
{
    _obj->setCenter(center);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Disk::setNormal(Direction const &norm_vec)
{
    _obj->setNormal(norm_vec);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Disk::setRadius(PREC const &radius)
{
    _obj->setRadius(radius);
    _updateVertices();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void Disk::_updateVertices()
{
    if (_obj->_norm.x == 0 && _obj->_norm.y == 0 && (_obj->_norm.z == 1 || _obj->_norm.z == -1))
    {
        if (n_vertices != 4)
            resetVertices(4);

        raw_vertices[0].x = _obj->_center.x + _obj->_radius;
        raw_vertices[0].y = _obj->_center.y + _obj->_radius;
        raw_vertices[0].z = _obj->_center.z;

        raw_vertices[1].x = _obj->_center.x - _obj->_radius;
        raw_vertices[1].y = _obj->_center.y + _obj->_radius;
        raw_vertices[1].z = _obj->_center.z;

        raw_vertices[2].x = _obj->_center.x + _obj->_radius;
        raw_vertices[2].y = _obj->_center.y - _obj->_radius;
        raw_vertices[2].z = _obj->_center.z;

        raw_vertices[3].x = _obj->_center.x - _obj->_radius;
        raw_vertices[3].y = _obj->_center.y - _obj->_radius;
        raw_vertices[3].z = _obj->_center.z;
    }
    else if (_obj->_norm.z == 0 && _obj->_norm.y == 0 && (_obj->_norm.x == 1 || _obj->_norm.x == -1))
    {
        if (n_vertices != 4)
            resetVertices(4);

        raw_vertices[0].x = _obj->_center.x;
        raw_vertices[0].y = _obj->_center.y + _obj->_radius;
        raw_vertices[0].z = _obj->_center.z + _obj->_radius;

        raw_vertices[1].x = _obj->_center.x;
        raw_vertices[1].y = _obj->_center.y + _obj->_radius;
        raw_vertices[1].z = _obj->_center.z - _obj->_radius;

        raw_vertices[2].x = _obj->_center.x;
        raw_vertices[2].y = _obj->_center.y - _obj->_radius;
        raw_vertices[2].z = _obj->_center.z + _obj->_radius;

        raw_vertices[3].x = _obj->_center.x;
        raw_vertices[3].y = _obj->_center.y - _obj->_radius;
        raw_vertices[3].z = _obj->_center.z - _obj->_radius;
    }
    else if (_obj->_norm.z == 0 && _obj->_norm.x == 0 && (_obj->_norm.y == 1 || _obj->_norm.y == -1))
    {
        if (n_vertices != 4)
            resetVertices(4);

        raw_vertices[0].x = _obj->_center.x + _obj->_radius;
        raw_vertices[0].y = _obj->_center.y;
        raw_vertices[0].z = _obj->_center.z + _obj->_radius;

        raw_vertices[1].x = _obj->_center.x - _obj->_radius;
        raw_vertices[1].y = _obj->_center.y;
        raw_vertices[1].z = _obj->_center.z + _obj->_radius;

        raw_vertices[2].x = _obj->_center.x + _obj->_radius;
        raw_vertices[2].y = _obj->_center.y;
        raw_vertices[2].z = _obj->_center.z - _obj->_radius;

        raw_vertices[3].x = _obj->_center.x - _obj->_radius;
        raw_vertices[3].y = _obj->_center.y;
        raw_vertices[3].z = _obj->_center.z - _obj->_radius;
    }
    else
    {
        if (n_vertices != 8)
            resetVertices(8);

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
}

Vec3<PREC> Disk::getTextureCoord(PREC const &x,
                                  PREC const &y,
                                  PREC const &z) const
{
    return BaseDisk::getTextureCoord(_obj, x, y, z);
}
const BaseGeometry *Disk::hitCheck(Ray const &ray,
                                    PREC const &t_start,
                                    PREC const &t_end,
                                    PREC &hit_at) const
{
    return BaseDisk::hitCheck(_obj, ray, t_start, t_end, hit_at) ? this : nullptr;
}
Direction Disk::normalVec(PREC const &x, PREC const &y,
                           PREC const &z) const
{
    return BaseDisk::normalVec(_obj, x, y, z);
}
