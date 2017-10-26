#ifndef PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP
#define PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Cylinder;
class BaseCylinder;
}

class px::BaseCylinder : public BaseGeometry
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
    BaseCylinder(Point const &center_of_bottom_face,
                 PREC const &radius_x,
                 PREC const &radius_y,
                 PREC const &height,
                 const BaseMaterial * const &material,
                 const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BaseCylinder() = default;
protected:
    Point _center;
    PREC _radius_x;
    PREC _radius_y;
    PREC _height;

    PREC _a, _b;
    PREC _z0, _z1;
    PREC _abs_height;

    PX_CUDA_CALLABLE
    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &height);

    BaseCylinder &operator=(BaseCylinder const &) = delete;
    BaseCylinder &operator=(BaseCylinder &&) = delete;

    friend class Cylinder;
};

class px::Cylinder : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &center_of_bottom_face,
                                                PREC const &radius_x,
                                                PREC const &radius_y,
                                                PREC const &height,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center_of_bottom_face,
                   PREC const &radius_x,
                   PREC const &radius_y,
                   PREC const &height);

    ~Cylinder();
protected:
    BaseCylinder *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Cylinder(Point const &center_of_bottom_face,
             PREC const &radius_x,
             PREC const &radius_y,
             PREC const &height,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans);

    Cylinder &operator=(Cylinder const &) = delete;
    Cylinder &operator=(Cylinder &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP
