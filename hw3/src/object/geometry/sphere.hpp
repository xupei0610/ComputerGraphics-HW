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
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) override;

    ~BaseSphere() = default;
protected:
    Point _center;
    double _radius;
    double _radius2;

    BaseSphere(Point const &pos,
               double const &radius,
               const BaseMaterial * const &material,
               const Transformation * const &trans);

    BaseSphere &operator=(BaseSphere const &) = delete;
    BaseSphere &operator=(BaseSphere &&) = delete;
};

class px::Sphere : public BaseSphere
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                double const &radius,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setRadius(double const &r);

    ~Sphere();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseSphere * _dev_ptr;
    bool _need_upload;

    Sphere(Point const &center,
           double const &radius,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans);

    void updateVertices();

    Sphere &operator=(Sphere const &) = delete;
    Sphere &operator=(Sphere &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_SPHERE_HPP
