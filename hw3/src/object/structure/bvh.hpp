#ifndef PX_CG_OBJECT_GEOMETRY_BVH_HPP
#define PX_CG_OBJECT_GEOMETRY_BVH_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class BVH;
class BaseBVH;
}

class px::BaseBVH
{
public:
    PX_CUDA_CALLABLE
    GeometryObj *hit(Ray const &ray,
                     PREC const &range_start,
                     PREC const &range_end,
                     Point &intersect) const;
    PX_CUDA_CALLABLE
    bool hit(Ray const &ray,
             PREC const &range_start,
             PREC const &range_end) const;

    PX_CUDA_CALLABLE
    static bool hitBox(Point const &vertex_min,
                       Point const &vertex_max,
                       Ray const &ray,
                       PREC const &t_start,
                       PREC const &t_end);

    ~BaseBVH() = default;
protected:
    BaseBVH(Point const &vertex_min, Point const &vertex_max);

    GeometryObj **_geos;
    int _n;

    Point _vertex_min;
    Point _vertex_max;

    BaseBVH &operator=(BaseBVH const &) = delete;
    BaseBVH &operator=(BaseBVH &&) = delete;

    friend class BVH;
};

class px::BVH
{
public:
    BVH();

    const BaseGeometry *hit(Ray const &ray,
                                 PREC const &range_start,
                                 PREC const &range_end,
                                 Point &intersect) const;
    bool hit(Ray const &ray,
             PREC const &range_start,
             PREC const &range_end) const;

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    inline BaseBVH *devPtr() const noexcept { return _gpu_obj; }
    void up2Gpu();
    void clearGpuData();

    ~BVH();
protected:
    std::unordered_set<std::shared_ptr<BaseGeometry> > _geos;
    Point _vertex_min;
    Point _vertex_max;

    BaseBVH *_gpu_obj;
    GeometryObj **_gpu_geos;

    bool _need_upload;

    void _updateVertices();

    BVH &operator=(BVH const &) = delete;
    BVH &operator=(BVH &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BVH_HPP
