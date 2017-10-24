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
                                  double const &range_start,
                                  double const &range_end,
                                  double &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) const override;

    ~BasePlane() = default;
protected:
    Point _position;
    Direction _norm_vec;
    double _p_dot_n;

    BasePlane(Point const &pos,
              const BaseMaterial * const &material,
              const Transformation * const &trans);

    BasePlane &operator=(BasePlane const &) = delete;
    BasePlane &operator=(BasePlane &&) = delete;
};

class px::Plane : public BasePlane
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setPosition(Point const &position);
    void setNormVec(Direction const &norm_vec);

    ~Plane();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BasePlane * _dev_ptr;
    bool _need_upload;

    Plane(Point const &position,
          Direction const &norm_vec,
          std::shared_ptr<BaseMaterial> const &material,
          std::shared_ptr<Transformation> const &trans);

    Plane &operator=(Plane const &) = delete;
    Plane &operator=(Plane &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_PLANE_HPP
