#ifndef PX_CG_OBJECT_GEOMETRY_PLANE_HPP
#define PX_CG_OBJECT_GEOMETRY_PLANE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Plane;
class BasePlane;
}

class px::BasePlane : public BaseGeometry
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
    BasePlane(Point const &pos,
              Direction const &norm_vec,
              const BaseMaterial * const &material,
              const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BasePlane() = default;
protected:
    Point _position;
    Direction _norm_vec;
    PREC _p_dot_n;

    PX_CUDA_CALLABLE
    void setNormVec(Direction const &norm_vec);

    BasePlane &operator=(BasePlane const &) = delete;
    BasePlane &operator=(BasePlane &&) = delete;

    friend class Plane;
};

class px::Plane : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setPosition(Point const &position);
    void setNormVec(Direction const &norm_vec);

    ~Plane();
protected:
    BasePlane *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Plane(Point const &position,
          Direction const &norm_vec,
          std::shared_ptr<Material> const &material,
          std::shared_ptr<Transformation> const &trans);

    Plane &operator=(Plane const &) = delete;
    Plane &operator=(Plane &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_PLANE_HPP
