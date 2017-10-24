#ifndef PX_CG_OBJECT_GEOMETRY_RING_HPP
#define PX_CG_OBJECT_GEOMETRY_RING_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Ring;
class BaseRing;
}

class px::BaseRing : public BaseGeometry
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

    ~BaseRing() = default;
protected:
    Point _center;
    Direction _norm_vec;
    double _inner_radius;
    double _outer_radius;
    double _inner_radius2;
    double _outer_radius2;
    double _p_dot_n;

    BaseRing(Point const &pos,
             double const &radius1,
             double const &radius2,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    BaseRing &operator=(BaseRing const &) = delete;
    BaseRing &operator=(BaseRing &&) = delete;
};

class px::Ring : public BaseRing
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                Direction const &norm_vec,
                                                double const &radius1,
                                                double const &radius2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setNormVec(Direction const &norm_vec);
    void setRadius(double const &radius1, double const &radius2);

    ~Ring();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseRing * _dev_ptr;
    bool _need_upload;

    Ring(Point const &center,
         Direction const &norm_vec,
         double const &radius1,
         double const &radius2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    void updateVertices();

    Ring &operator=(Ring const &) = delete;
    Ring &operator=(Ring &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_RING_HPP
