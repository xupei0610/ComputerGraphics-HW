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

    ~BaseCylinder() = default;
protected:
    Point _center;
    double _radius_x;
    double _radius_y;
    double _height;

    double _a, _b;
    double _z0, _z1;
    double _abs_height;

    BaseCylinder(const BaseMaterial * const &material,
                 const Transformation * const &trans);

    BaseCylinder &operator=(BaseCylinder const &) = delete;
    BaseCylinder &operator=(BaseCylinder &&) = delete;
};

class px::Cylinder : public BaseCylinder
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center_of_bottom_face,
                                                double const &radius_x,
                                                double const &radius_y,
                                                double const &height,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center_of_bottom_face,
                   double const &radius_x,
                   double const &radius_y,
                   double const &height);

    ~Cylinder();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseCylinder * _dev_ptr;
    bool _need_upload;

    Cylinder(Point const &center_of_bottom_face,
             double const &radius_x,
             double const &radius_y,
             double const &height,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Cylinder &operator=(Cylinder const &) = delete;
    Cylinder &operator=(Cylinder &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CYLINDER_HPP
