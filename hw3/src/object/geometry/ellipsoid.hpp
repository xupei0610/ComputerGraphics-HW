#ifndef PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP
#define PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Ellipsoid;
class BaseEllipsoid;
}

class px::BaseEllipsoid : public BaseGeometry
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

    ~BaseEllipsoid() = default;
protected:
    Point _center;
    double _radius_x;
    double _radius_y;
    double _radius_z;

    double _a;
    double _b;
    double _c;

    BaseEllipsoid(const BaseMaterial * const &material,
                 const Transformation * const &trans);

    BaseEllipsoid &operator=(BaseEllipsoid const &) = delete;
    BaseEllipsoid &operator=(BaseEllipsoid &&) = delete;
};

class px::Ellipsoid : public BaseEllipsoid
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                double const &radius_x,
                                                double const &radius_y,
                                                double const &radius_z,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center,
                   double const &radius_x,
                   double const &radius_y,
                   double const &radius_z);

    ~Ellipsoid();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseEllipsoid * _dev_ptr;
    bool _need_upload;

    Ellipsoid(Point const &center,
              double const &radius_x,
              double const &radius_y,
              double const &radius_z,
              std::shared_ptr<BaseMaterial> const &material,
              std::shared_ptr<Transformation> const &trans);

    Ellipsoid &operator=(Ellipsoid const &) = delete;
    Ellipsoid &operator=(Ellipsoid &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_ELLIPSOID_HPP
