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
    BaseRing(Point const &pos,
             Direction const &norm_vec,
             PREC const &radius1,
             PREC const &radius2,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    PX_CUDA_CALLABLE
    ~BaseRing() = default;
protected:
    Point _center;
    Direction _norm_vec;
    PREC _inner_radius;
    PREC _outer_radius;
    PREC _inner_radius2;
    PREC _outer_radius2;
    PREC _p_dot_n;

    PX_CUDA_CALLABLE
    void updateVertices();

    BaseRing &operator=(BaseRing const &) = delete;
    BaseRing &operator=(BaseRing &&) = delete;

    friend class Ring;
};

class px::Ring : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &center,
                                                Direction const &norm_vec,
                                                PREC const &radius1,
                                                PREC const &radius2,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry * const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setNormVec(Direction const &norm_vec);
    void setRadius(PREC const &radius1, PREC const &radius2);

    ~Ring();
protected:
    BaseRing *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Ring(Point const &center,
         Direction const &norm_vec,
         PREC const &radius1,
         PREC const &radius2,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans);

    Ring &operator=(Ring const &) = delete;
    Ring &operator=(Ring &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_RING_HPP
