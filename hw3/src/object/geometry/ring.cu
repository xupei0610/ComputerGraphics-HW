#include "object/geometry/ring.hpp"

using namespace px;

BaseRing::BaseRing(Point const &pos,
                   double const &radius1,
                   double const &radius2,
                   const BaseMaterial *const &material,
                   const Transformation *const &trans)
        : BaseGeometry(material, trans, 8),
          _center(pos),
          _inner_radius(std::min(radius1, radius2)),
          _outer_radius(std::max(radius1, radius2)),
          _inner_radius2(_inner_radius*_inner_radius),
          _outer_radius2(_outer_radius*_outer_radius)
{}

PX_CUDA_CALLABLE
BaseGeometry * BaseRing::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at)
{
    auto tmp = (_p_dot_n - ray.original.dot(_norm_vec)) / ray.direction.dot(_norm_vec);
    if (tmp > t_start && tmp < t_end)
    {
        auto intersect = ray[tmp];
        auto dist2 = (intersect.x - _center.x) * (intersect.x - _center.x) +
                     (intersect.y - _center.y) * (intersect.y - _center.y) +
                     (intersect.z - _center.z) * (intersect.z - _center.z);
        if (dist2 <= _outer_radius2 && dist2 >= _inner_radius2)
        {
            hit_at = tmp;
            return this;
        }
    }
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseRing::normalVec(double const &x, double const &y, double const &z)
{
    return _norm_vec;
}

PX_CUDA_CALLABLE
Vec3<double> BaseRing::getTextureCoord(double const &x, double const &y,
                                       double const &z)
{
    return {x - _center.x,
            -_norm_vec.z*(y - _center.y) + _norm_vec.y*(z - _center.z),
            (x - _center.x)*_norm_vec.x + (y - _center.y)*_norm_vec.y + (z - _center.z)*_norm_vec.z};
}

std::shared_ptr<BaseGeometry> Ring::create(Point const &position,
                                           Direction const &norm_vec,
                                           double const &radius1,
                                           double const &radius2,
                                           std::shared_ptr<BaseMaterial> const &material,
                                           std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Ring(position, norm_vec,
                                                  radius1, radius2,
                                                  material, trans));
}

Ring::Ring(Point const &position,
           Direction const &norm_vec,
           double const &radius1,
           double const &radius2,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans)
        : BaseRing(position, radius1, radius2, material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{
    setNormVec(norm_vec);
}

Ring::~Ring()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *Ring::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseRing)));

        material = _material_ptr->up2Gpu();
        transformation = _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseRing*>(this),
                                 sizeof(BaseRing),
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

void Ring::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}


void Ring::setCenter(Point const &center)
{
    _center = center;
    _p_dot_n = center.dot(_norm_vec);

    updateVertices();
}

void Ring::setNormVec(Direction const &norm_vec)
{
    _norm_vec = norm_vec;
    _p_dot_n = _center.dot(norm_vec);

    updateVertices();
}

void Ring::setRadius(double const &radius1, double const &radius2)
{
    _inner_radius = std::min(radius1, radius2);
    _outer_radius = std::max(radius1, radius2);
    _inner_radius2 = _inner_radius*_inner_radius;
    _outer_radius2 = _outer_radius*_outer_radius;

    updateVertices();
}

void Ring::updateVertices()
{
    _raw_vertices[0].x = _center.x + _outer_radius;
    _raw_vertices[0].y = _center.y + _outer_radius;
    _raw_vertices[0].z = _center.z + _outer_radius;

    _raw_vertices[1].x = _center.x - _outer_radius;
    _raw_vertices[1].y = _center.y + _outer_radius;
    _raw_vertices[1].z = _center.z + _outer_radius;

    _raw_vertices[2].x = _center.x + _outer_radius;
    _raw_vertices[2].y = _center.y - _outer_radius;
    _raw_vertices[2].z = _center.z + _outer_radius;

    _raw_vertices[3].x = _center.x + _outer_radius;
    _raw_vertices[3].y = _center.y + _outer_radius;
    _raw_vertices[3].z = _center.z - _outer_radius;

    _raw_vertices[4].x = _center.x - _outer_radius;
    _raw_vertices[4].y = _center.y - _outer_radius;
    _raw_vertices[4].z = _center.z + _outer_radius;

    _raw_vertices[5].x = _center.x - _outer_radius;
    _raw_vertices[5].y = _center.y + _outer_radius;
    _raw_vertices[5].z = _center.z - _outer_radius;

    _raw_vertices[6].x = _center.x + _outer_radius;
    _raw_vertices[6].y = _center.y - _outer_radius;
    _raw_vertices[6].z = _center.z - _outer_radius;

    _raw_vertices[7].x = _center.x - _outer_radius;
    _raw_vertices[7].y = _center.y - _outer_radius;
    _raw_vertices[7].z = _center.z - _outer_radius;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}