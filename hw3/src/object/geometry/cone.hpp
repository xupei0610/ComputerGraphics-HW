#ifndef PX_CG_OBJECT_GEOMETRY_CONE_HPP
#define PX_CG_OBJECT_GEOMETRY_CONE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Cone;
class BaseCone;
}

class px::BaseCone : public BaseGeometry
{
protected:
    PX_CUDA_CALLABLE
    const BaseGeometry * hitCheck(Ray const &ray,
                                  PREC const &range_start,
                                  PREC const &range_end,
                                  PREC &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<PREC> getTextureCoord(PREC const &x,
                                 PREC const &y,
                                 PREC const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(PREC const &x, PREC const &y, PREC const &z) const override;

public:
    PX_CUDA_CALLABLE
    BaseCone(Point const &center_of_bottom_face,
             PREC const &radius_x,
             PREC const &radius_y,
             PREC const &ideal_height,
             PREC const &real_height,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    PX_CUDA_CALLABLE
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

    PX_CUDA_CALLABLE
    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &ideal_height,
                   PREC const &real_height);

    BaseCone &operator=(BaseCone const &) = delete;
    BaseCone &operator=(BaseCone &&) = delete;

    friend class Cone;
};

class px::Cone : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &center_of_bottom_face,
                                                PREC const &radius_x,
                                                PREC const &radius_y,
                                                PREC const &ideal_height,
                                                PREC const &real_height,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
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
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Cone(Point const &center_of_bottom_face,
         PREC const &radius_x,
         PREC const &radius_y,
         PREC const &ideal_height,
         PREC const &real_height,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans);

    Cone &operator=(Cone const &) = delete;
    Cone &operator=(Cone &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CONE_HPP
