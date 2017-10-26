#ifndef PX_CG_OBJECT_GEOMETRY_SPHERE_HPP
#define PX_CG_OBJECT_GEOMETRY_SPHERE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Sphere;
class BaseSphere;
}

class px::BaseSphere : public BaseGeometry
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
    BaseSphere(Point const &pos,
               PREC const &radius,
               const BaseMaterial * const &material,
               const Transformation * const &trans);

    PX_CUDA_CALLABLE
    ~BaseSphere() = default;

protected:
    Point _center;
    PREC _radius;
    PREC _radius2;

    PX_CUDA_CALLABLE
    void updateVertices();

    BaseSphere &operator=(BaseSphere const &) = delete;
    BaseSphere &operator=(BaseSphere &&) = delete;

    friend class Sphere;
};

class px::Sphere : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &center,
                                                PREC const &radius,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setRadius(PREC const &r);

    ~Sphere();
protected:
    BaseSphere *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Sphere(Point const &center,
           PREC const &radius,
           std::shared_ptr<Material> const &material,
           std::shared_ptr<Transformation> const &trans);

    Sphere &operator=(Sphere const &) = delete;
    Sphere &operator=(Sphere &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_SPHERE_HPP
