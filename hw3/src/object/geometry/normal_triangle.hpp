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
                                  double const &range_start,
                                  double const &range_end,
                                  double &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) const override;

    ~BaseNormalTriangle() = default;
protected:
    BaseNormalTriangle(const BaseMaterial * const &material,
                       const Transformation * const &trans);

    BaseNormalTriangle &operator=(BaseNormalTriangle const &) = delete;
    BaseNormalTriangle &operator=(BaseNormalTriangle &&) = delete;
};

class px::NormalTriangle : public BaseNormalTriangle
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &vertex1, Direction const &normal1,
                                                Point const &vertex2, Direction const &normal2,
                                                Point const &vertex3, Direction const &normal3,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setParam(Point const &vertex1, Direction const &normal1,
                  Point const &vertex2, Direction const &normal2,
                  Point const &vertex3, Direction const &normal3);

    ~NormalTriangle();
protected:
    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseNormalTriangle * _dev_ptr;
    bool _need_upload;

    NormalTriangle(Point const &vertex1, Direction const &normal1,
                   Point const &vertex2, Direction const &normal2,
                   Point const &vertex3, Direction const &normal3,
                   std::shared_ptr<BaseMaterial> const &material,
                   std::shared_ptr<Transformation> const &trans);

    NormalTriangle &operator=(NormalTriangle const &) = delete;
    NormalTriangle &operator=(NormalTriangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP
