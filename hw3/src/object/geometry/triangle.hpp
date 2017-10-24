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
                                  double const &range_start,
                                  double const &range_end,
                                  double &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) const override;

    ~BaseTriangle() = default;
protected:
    Point _center;
    Direction _norm_vec;
    Vec3<double> _ba;
    Vec3<double> _cb;
    Vec3<double> _ca;
    double _v1_dot_n;

    BaseTriangle(const BaseMaterial * const &material,
                 const Transformation * const &trans);

    BaseTriangle &operator=(BaseTriangle const &) = delete;
    BaseTriangle &operator=(BaseTriangle &&) = delete;
};

class px::Triangle : public BaseTriangle
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &a,
                                                Point const &b,
                                                Point const &c,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setVertices(Point const &a,
                     Point const &b,
                     Point const &c);

    ~Triangle();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseTriangle * _dev_ptr;
    bool _need_upload;

    Triangle(Point const &a,
             Point const &b,
             Point const &c,
             std::shared_ptr<BaseMaterial> const &material,
             std::shared_ptr<Transformation> const &trans);

    Triangle &operator=(Triangle const &) = delete;
    Triangle &operator=(Triangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP
