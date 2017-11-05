#ifndef PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
#define PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP

#include "object/geometry/base_geometry.hpp"

#include <queue>

namespace px
{
class BoundBox;
class BaseBoundBox;
}

class px::BaseBoundBox
{
public:
    PX_CUDA_CALLABLE
    static GeometryObj *hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &range_start,
                         PREC const &range_end,
                         PREC &hit_at);
    PX_CUDA_CALLABLE
    static Vec3<PREC> getTextureCoord(void * const &obj,
                                      PREC const &x,
                                      PREC const &y,
                                      PREC const &z);
    PX_CUDA_CALLABLE
    static Direction normalVec(void * const &obj,
                               PREC const &x, PREC const &y, PREC const &z);

    PX_CUDA_CALLABLE
    static bool hitBox(Point const &vertex_min,
                       Point const &vertex_max,
                       Ray const &ray,
                       PREC const &t_start,
                       PREC const &t_end);

    ~BaseBoundBox() = default;
protected:
    BaseBoundBox(Point const &vertex_min, Point const &vertex_max);

    GeometryObj **_geos;
    int _n;

    Point _vertex_min;
    Point _vertex_max;

    BaseBoundBox &operator=(BaseBoundBox const &) = delete;
    BaseBoundBox &operator=(BaseBoundBox &&) = delete;

    friend class BoundBox;
};

class px::BoundBox : public BaseGeometry
{
public:
    BoundBox(std::shared_ptr<Transformation> const &trans);

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    void up2Gpu() override;
    void clearGpuData() override;

    ~BoundBox();
protected:
    std::deque<std::shared_ptr<BaseGeometry> > _geos;
    Point _vertex_min;
    Point _vertex_max;

    void *_gpu_obj;
    GeometryObj **_gpu_geos;

    bool _need_upload;

    void _updateVertices();

    Vec3<PREC> getTextureCoord(PREC const &x,
                               PREC const &y,
                               PREC const &z) const override;
    const BaseGeometry *hitCheck(Ray const &ray,
                                 PREC const &range_start,
                                 PREC const &range_end,
                                 PREC &hit_at) const override;
    Direction normalVec(PREC const &x, PREC const &y,
                        PREC const &z) const override;

    BoundBox &operator=(BoundBox const &) = delete;
    BoundBox &operator=(BoundBox &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
