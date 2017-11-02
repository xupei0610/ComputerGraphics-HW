#ifndef PX_CG_OBJECT_GEOMETRY_CONE_HPP
#define PX_CG_OBJECT_GEOMETRY_CONE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Cone;
class BaseCone;
}

class px::BaseCone
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

    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &ideal_height,
                   PREC const &real_height);

    ~BaseCone() = default;
protected:
    Point _center;
    PREC _radius_x;
    PREC _radius_y;
    PREC _ideal_height;
    PREC _real_height;

    PREC _a, _b, _c;
    PREC _quadric_center_z;
    PREC _z0, _z1;
    PREC _top_z, _top_r_x, _top_r_y;

    GeometryObj *_dev_obj;

    BaseCone(Point const &center_of_bottom_face,
             PREC const &radius_x,
             PREC const &radius_y,
             PREC const &ideal_height,
             PREC const &real_height);

    BaseCone &operator=(BaseCone const &) = delete;
    BaseCone &operator=(BaseCone &&) = delete;

    friend class Cone;
};

class px::Cone : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center_of_bottom_face,
                                                PREC const &radius_x,
                                                PREC const &radius_y,
                                                PREC const &ideal_height,
                                                PREC const &real_height,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &ideal_height,
                   PREC const &real_height);

    ~Cone();
protected:
    BaseCone *_obj;
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

    Cone(Point const &center_of_bottom_face,
         PREC const &radius_x,
         PREC const &radius_y,
         PREC const &ideal_height,
         PREC const &real_height,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Cone &operator=(Cone const &) = delete;
    Cone &operator=(Cone &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CONE_HPP
