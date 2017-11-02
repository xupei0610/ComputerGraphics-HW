#ifndef PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP
#define PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Ellipsoid;
class BaseEllipsoid;
}

class px::BaseEllipsoid
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

    void setParams(Point const &center,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &radius_z);

    ~BaseEllipsoid() = default;
protected:
    Point _center;
    PREC _radius_x;
    PREC _radius_y;
    PREC _radius_z;

    PREC _a;
    PREC _b;
    PREC _c;

    GeometryObj *_dev_obj;

    BaseEllipsoid(Point const &center,
                  PREC const &radius_x,
                  PREC const &radius_y,
                  PREC const &radius_z);

    BaseEllipsoid &operator=(BaseEllipsoid const &) = delete;
    BaseEllipsoid &operator=(BaseEllipsoid &&) = delete;

    friend class Ellipsoid;
};

class px::Ellipsoid : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                PREC const &radius_x,
                                                PREC const &radius_y,
                                                PREC const &radius_z,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &radius_z);

    ~Ellipsoid();
protected:
    BaseEllipsoid *_obj;
    void *_gpu_obj;
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

    Ellipsoid(Point const &center,
              PREC const &radius_x,
              PREC const &radius_y,
              PREC const &radius_z,
              std::shared_ptr<BaseMaterial> const &material,
              std::shared_ptr<Transformation> const &trans);

    Ellipsoid &operator=(Ellipsoid const &) = delete;
    Ellipsoid &operator=(Ellipsoid &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP
