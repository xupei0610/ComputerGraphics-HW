#ifndef PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP
#define PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class NormalTriangle;
class BaseNormalTriangle;
}

class px::BaseNormalTriangle : public BaseGeometry
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
    BaseNormalTriangle(Point const &vertex1, Direction const &normal1,
                       Point const &vertex2, Direction const &normal2,
                       Point const &vertex3, Direction const &normal3,
                       const BaseMaterial * const &material,
                       const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BaseNormalTriangle() = default;
protected:
    PX_CUDA_CALLABLE
    void setParam(Point const &vertex1, Direction const &normal1,
                  Point const &vertex2, Direction const &normal2,
                  Point const &vertex3, Direction const &normal3);

    BaseNormalTriangle &operator=(BaseNormalTriangle const &) = delete;
    BaseNormalTriangle &operator=(BaseNormalTriangle &&) = delete;

    friend class NormalTriangle;
};

class px::NormalTriangle : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &vertex1, Direction const &normal1,
                                                Point const &vertex2, Direction const &normal2,
                                                Point const &vertex3, Direction const &normal3,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setParam(Point const &vertex1, Direction const &normal1,
                  Point const &vertex2, Direction const &normal2,
                  Point const &vertex3, Direction const &normal3);

    ~NormalTriangle();
protected:
    BaseNormalTriangle *_obj;
    BaseGeometry *_base_obj;

    Point _a;
    Direction _na;
    Point _b;
    Direction _nb;
    Point _c;
    Direction _nc;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    NormalTriangle(Point const &vertex1, Direction const &normal1,
                   Point const &vertex2, Direction const &normal2,
                   Point const &vertex3, Direction const &normal3,
                   std::shared_ptr<Material> const &material,
                   std::shared_ptr<Transformation> const &trans);

    NormalTriangle &operator=(NormalTriangle const &) = delete;
    NormalTriangle &operator=(NormalTriangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP
