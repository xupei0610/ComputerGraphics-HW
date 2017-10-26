#ifndef PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP
#define PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Triangle;
class BaseTriangle;
}

class px::BaseTriangle : public BaseGeometry
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
    BaseTriangle(Point const &a,
                 Point const &b,
                 Point const &c,
                 const BaseMaterial * const &material,
                 const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BaseTriangle() = default;


    PX_CUDA_CALLABLE
    void setVertices(Point const &a,
                     Point const &b,
                     Point const &c);
protected:
    Point _center;
    Direction _norm_vec;
    Vec3<PREC> _ba;
    Vec3<PREC> _cb;
    Vec3<PREC> _ca;
    PREC _v1_dot_n;

    BaseTriangle &operator=(BaseTriangle const &) = delete;
    BaseTriangle &operator=(BaseTriangle &&) = delete;
};

class px::Triangle : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &a,
                                                Point const &b,
                                                Point const &c,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    BaseGeometry * const &obj() const noexcept override;

    void setVertices(Point const &a,
                     Point const &b,
                     Point const &c);

    ~Triangle();
protected:
    BaseTriangle *_obj;
    BaseGeometry *_base_obj;

    Point _a;
    Point _b;
    Point _c;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Triangle(Point const &a,
             Point const &b,
             Point const &c,
             std::shared_ptr<Material> const &material,
             std::shared_ptr<Transformation> const &trans);

    Triangle &operator=(Triangle const &) = delete;
    Triangle &operator=(Triangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP
