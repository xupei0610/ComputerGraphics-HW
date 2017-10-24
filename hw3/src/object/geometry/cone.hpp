#ifndef PX_CG_OBJECT_GEOMETRY_CONE_HPP
#define PX_CG_OBJECT_GEOMETRY_CONE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Cone;
class BaseCone;
}

class px::BaseCone : public BaseGeometry
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

    ~BaseCone() = default;
protected:
    Point _center;
    double _radius_x;
    double _radius_y;
    double _ideal_height;
    double _real_height;

    double _a, _b, _c;
    double _quadric_center_z;
    double _z0, _z1;
    double _top_z, _top_r_x, _top_r_y;

    BaseCone(const BaseMaterial * const &material,
             const Transformation * const &trans);

    BaseCone &operator=(BaseCone const &) = delete;
    BaseCone &operator=(BaseCone &&) = delete;
};

class px::Cone : public BaseCone
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center_of_bottom_face,
                                                double const &radius_x,
                                                double const &radius_y,
                                                double const &ideal_height,
                                                double const &real_height,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setParams(Point const &center_of_bottom_face,
                   double const &radius_x,
                   double const &radius_y,
                   double const &ideal_height,
                   double const &real_height);

    ~Cone();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseCone * _dev_ptr;
    bool _need_upload;

    Cone(Point const &center_of_bottom_face,
         double const &radius_x,
         double const &radius_y,
         double const &ideal_height,
         double const &real_height,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Cone &operator=(Cone const &) = delete;
    Cone &operator=(Cone &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_CONE_HPP
