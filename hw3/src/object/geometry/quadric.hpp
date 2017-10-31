#ifndef PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
#define PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Quadric;
class BaseQuadric;
}
// Ax2 + By2 + Cz2 + Dxy+ Exz + Fyz + Gx + Hy + Iz + J = 0
class px::BaseQuadric : public BaseGeometry
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
    BaseQuadric(Point const &center,
                PREC const &a,
                PREC const &b,
                PREC const &c,
                PREC const &d,
                PREC const &e,
                PREC const &f,
                PREC const &g,
                PREC const &h,
                PREC const &i,
                PREC const &j,
                PREC const &x0, PREC const &x1,
                PREC const &y0, PREC const &y1,
                PREC const &z0, PREC const &z1,
                const BaseMaterial * const &material,
                const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BaseQuadric() = default;
protected:
    Point _center;
    PREC _a, _b, _c;
    PREC _d, _e, _f;
    PREC _g, _h, _i;
    PREC _j;

    bool _sym_xy, _sym_yz, _sym_xz, _sym_x, _sym_y, _sym_z, _sym_o;

    PREC _x0, _x1;
    PREC _y0, _y1;
    PREC _z0, _z1;

    PX_CUDA_CALLABLE
    void setCoef(PREC const &a,
                 PREC const &b,
                 PREC const &c,
                 PREC const &d,
                 PREC const &e,
                 PREC const &f,
                 PREC const &g,
                 PREC const &h,
                 PREC const &i,
                 PREC const &j,
                 PREC const &x0, PREC const &x1,
                 PREC const &y0, PREC const &y1,
                 PREC const &z0, PREC const &z1);
    PX_CUDA_CALLABLE
    void updateVertices();

    BaseQuadric &operator=(BaseQuadric const &) = delete;
    BaseQuadric &operator=(BaseQuadric &&) = delete;

    friend class Quadric;
};

class px::Quadric : public Geometry
{
public:
    static std::shared_ptr<Geometry> create(Point const &center,
                                                PREC const &a,
                                                PREC const &b,
                                                PREC const &c,
                                                PREC const &d,
                                                PREC const &e,
                                                PREC const &f,
                                                PREC const &g,
                                                PREC const &h,
                                                PREC const &i,
                                                PREC const &j,
                                                PREC const &x0, PREC const &x1,
                                                PREC const &y0, PREC const &y1,
                                                PREC const &z0, PREC const &z1,
                                                std::shared_ptr<Material> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    void setCoef(PREC const &a,
                 PREC const &b,
                 PREC const &c,
                 PREC const &d,
                 PREC const &e,
                 PREC const &f,
                 PREC const &g,
                 PREC const &h,
                 PREC const &i,
                 PREC const &j,
                 PREC const &x0, PREC const &x1,
                 PREC const &y0, PREC const &y1,
                 PREC const &z0, PREC const &z1);
    void setCenter(Point const &position);

    ~Quadric();
protected:
    BaseQuadric *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Material> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    Quadric(Point const &center,
            PREC const &a,
            PREC const &b,
            PREC const &c,
            PREC const &d,
            PREC const &e,
            PREC const &f,
            PREC const &g,
            PREC const &h,
            PREC const &i,
            PREC const &j,
            PREC const &x0, PREC const &x1,
            PREC const &y0, PREC const &y1,
            PREC const &z0, PREC const &z1,
         std::shared_ptr<Material> const &material,
         std::shared_ptr<Transformation> const &trans);

    Quadric &operator=(Quadric const &) = delete;
    Quadric &operator=(Quadric &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
