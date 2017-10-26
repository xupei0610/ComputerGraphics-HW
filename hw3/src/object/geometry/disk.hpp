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
    BaseDisk(Point const &pos,
             Direction const &norm_vec,
             PREC const &radius,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    PX_CUDA_CALLABLE
    ~BaseDisk() = default;
protected:
    Point _center;
    Direction _norm_vec;
    PREC _radius;
    PREC _radius2;
    PREC _p_dot_n;

    PX_CUDA_CALLABLE
    void updateVertices();

    BaseDisk &operator=(BaseDisk const &) = delete;
    BaseDisk &operator=(BaseDisk &&) = delete;

    friend class Disk;
};

class px::Disk : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                PREC const &radius,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setRadius(PREC const &radius);
    void setNormVec(Direction const &norm_vec);

    ~Disk();
protected:
    BaseDisk *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Disk(Point const &position,
         Direction const &norm_vec,
         PREC const &radius,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans);

    Disk &operator=(Disk const &) = delete;
    Disk &operator=(Disk &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_DISK_HPP
