#ifndef PX_CG_OBJECT_GEOMETRY_BOX_HPP
#define PX_CG_OBJECT_GEOMETRY_BOX_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Box;
class BaseBox;
}

class px::BaseBox : public BaseGeometry
{
public:
    PX_CUDA_CALLABLE
    ~BaseBox() = default;
    PX_CUDA_CALLABLE
    BaseBox(PREC const &x1, PREC const &x2,
            PREC const &y1, PREC const &y2,
            PREC const &z1, PREC const &z2,
            const BaseMaterial * const &material,
            const Transformation * const &trans);

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

protected:
    Point _vertex_min;
    Point _vertex_max;

    Point _center;
    Vec3<PREC> _side;

    PX_CUDA_CALLABLE
    void setVertices(PREC const &x1, PREC const &x2,
                     PREC const &y1, PREC const &y2,
                     PREC const &z1, PREC const &z2);

    BaseBox &operator=(BaseBox const &) = delete;
    BaseBox &operator=(BaseBox &&) = delete;

    friend class Box;
};

class px::Box : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(PREC const &x1, PREC const &x2,
                                                PREC const &y1, PREC const &y2,
                                                PREC const &z1, PREC const &z2,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    static std::shared_ptr<Geometry> create(Point const &v1, Point const &v2,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setVertices(Point const &v1, Point const &v2);
    void setVertices(PREC const &x1, PREC const &x2,
                     PREC const &y1, PREC const &y2,
                     PREC const &z1, PREC const &z2);

    ~Box();
protected:
    BaseBox *_obj;
    BaseGeometry *_base_obj;

    PREC _v1_x, _v1_y, _v1_z;
    PREC _v2_x, _v2_y, _v2_z;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Box(PREC const &x1, PREC const &x2,
        PREC const &y1, PREC const &y2,
        PREC const &z1, PREC const &z2,
        std::shared_ptr<Material> const &material,
        std::shared_ptr<Transformation> const &trans);

    Box &operator=(Box const &) = delete;
    Box &operator=(Box &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_BOX_HPP
