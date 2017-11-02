#ifndef PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP
#define PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Triangle;
class BaseTriangle;
}

class px::BaseTriangle
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
                               PREC const &x, PREC const &y, PREC const &z);

    void setVertices(Point const &a,
                     Point const &b,
                     Point const &c);

    ~BaseTriangle() = default;
protected:
    Point _a;
    Point _center;
    Direction _norm_vec;
    Vec3<PREC> _ba;
//    Vec3<PREC> _cb;
    Vec3<PREC> _ca;
//    PREC _v1_dot_n;

    GeometryObj *_dev_obj;

    BaseTriangle(Point const &a,
                 Point const &b,
                 Point const &c);

    BaseTriangle &operator=(BaseTriangle const &) = delete;
    BaseTriangle &operator=(BaseTriangle &&) = delete;

    friend class Triangle;
};

class px::Triangle : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &a,
                                                Point const &b,
                                                Point const &c,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setVertices(Point const &a,
                     Point const &b,
                     Point const &c);

    ~Triangle();
protected:
    BaseTriangle *_obj;
    void *_gpu_obj;
    bool _need_upload;

    void _updateVertices(Point const &a,
                         Point const &b,
                         Point const &c);

    Vec3<PREC> getTextureCoord(PREC const &x,
                                       PREC const &y,
                                       PREC const &z) const override;
    const BaseGeometry * hitCheck(Ray const &ray,
                                          PREC const &range_start,
                                          PREC const &range_end,
                                          PREC &hit_at) const override;
    Direction normalVec(PREC const &x, PREC const &y,
                                PREC const &z) const override;

    Triangle(Point const &a,
             Point const &b,
             Point const &c,
             std::shared_ptr<BaseMaterial> const &material,
             std::shared_ptr<Transformation> const &trans);

    Triangle &operator=(Triangle const &) = delete;
    Triangle &operator=(Triangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_TRIANGLE_HPP
