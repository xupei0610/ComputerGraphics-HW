#ifndef PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP
#define PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Cylinder;
class BaseCylinder;
}

class px::BaseCylinder
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
                               PREC const &x, PREC const &y, PREC const &z,
                               bool &double_face);

    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &height);

    ~BaseCylinder() = default;

protected:
    Point _center;
    PREC _radius_x;
    PREC _radius_y;
    PREC _height;

    PREC _a, _b;
    PREC _z0, _z1;
    PREC _abs_height;

    GeometryObj *_dev_obj;

    BaseCylinder(Point const &center_of_bottom_face,
                 PREC const &radius_x,
                 PREC const &radius_y,
                 PREC const &height);

    BaseCylinder &operator=(BaseCylinder const &) = delete;
    BaseCylinder &operator=(BaseCylinder &&) = delete;

    friend class Cylinder;
};

class px::Cylinder : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center_of_bottom_face,
                                                PREC const &radius_x,
                                                PREC const &radius_y,
                                                PREC const &height,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &height);

    ~Cylinder();
protected:
    BaseCylinder *_obj;
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
                        PREC const &z,
                        bool &double_face) const override;

    Cylinder(Point const &center_of_bottom_face,
             PREC const &radius_x,
             PREC const &radius_y,
             PREC const &height,
             std::shared_ptr<BaseMaterial> const &material,
             std::shared_ptr<Transformation> const &trans);

    Cylinder &operator=(Cylinder const &) = delete;
    Cylinder &operator=(Cylinder &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP
