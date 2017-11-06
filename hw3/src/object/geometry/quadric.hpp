#ifndef PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
#define PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Quadric;
class BaseQuadric;
}
// Ax2 + By2 + Cz2 + Dxy+ Exz + Fyz + Gx + Hy + Iz + J = 0
class px::BaseQuadric
{
public:
    PX_CUDA_CALLABLE
    static GeometryObj *hitCheck(void * const &obj,
                         Ray const &ray,
                         PREC const &range_start,
                         PREC const &range_end,
                         PREC &hit_at);
    PX_CUDA_CALLABLE
    static Vec3<PREC> getTextureCoord(void * const &obj,
                                      PREC const &x,
                                      PREC const &y,
                                      PREC const &z);
    PX_CUDA_CALLABLE
    static Direction normalVec(void * const &obj,
                               PREC const &x, PREC const &y, PREC const &z,
                               bool &double_face);

    void setCenter(Point const &position);
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

    GeometryObj *_dev_obj;

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
                PREC const &z0, PREC const &z1);

    BaseQuadric &operator=(BaseQuadric const &) = delete;
    BaseQuadric &operator=(BaseQuadric &&) = delete;

    friend class Quadric;
};

class px::Quadric : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
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
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
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

    ~Quadric();
protected:
    BaseQuadric *_obj;
    void *_gpu_obj;
    bool _need_upload;

    void _updateVertices();

    Vec3<PREC> getTextureCoord(PREC const &x,
                               PREC const &y,
                               PREC const &z) const override;
    const BaseGeometry *hitCheck(Ray const &ray,
                                 PREC const &range_start,
                                 PREC const &range_end,
                                 PREC &hit_at) const override;
    Direction normalVec(PREC const &x, PREC const &y,
                        PREC const &z,
                        bool &double_face) const override;

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
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Quadric &operator=(Quadric const &) = delete;
    Quadric &operator=(Quadric &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
