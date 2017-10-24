#ifndef PX_CG_OBJECT_GEOMETRY_DISK_HPP
#define PX_CG_OBJECT_GEOMETRY_DISK_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Disk;
class BaseDisk;
}

class px::BaseDisk : public BaseGeometry
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

    ~BaseDisk() = default;
protected:
    Point _center;
    Direction _norm_vec;
    double _radius;
    double _radius2;
    double _p_dot_n;

    BaseDisk(Point const &pos,
             double const &radius,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    BaseDisk &operator=(BaseDisk const &) = delete;
    BaseDisk &operator=(BaseDisk &&) = delete;
};

class px::Disk : public BaseDisk
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                double const &radius,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setNormVec(Direction const &norm_vec);
    void setRadius(double const &radius);

    ~Disk();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseDisk * _dev_ptr;
    bool _need_upload;

    Disk(Point const &position,
         Direction const &norm_vec,
         double const &radius,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    void updateVertices();

    Disk &operator=(Disk const &) = delete;
    Disk &operator=(Disk &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_DISK_HPP
