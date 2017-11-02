#ifndef PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP
#define PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class NormalTriangle;
class BaseNormalTriangle;
}

class px::BaseNormalTriangle
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
    void setNormals(Direction const &na,
                    Direction const &nb,
                    Direction const &nc);

    ~BaseNormalTriangle() = default;
protected:
    Point _center;
    Point _a, _b, _c;
    Vec3<PREC> _ba, _cb, _ca;
    Direction _na, _nb, _nc;
    PREC _n_norm;

    GeometryObj *_dev_obj;

    BaseNormalTriangle(Point const &vertex1, Direction const &normal1,
                       Point const &vertex2, Direction const &normal2,
                       Point const &vertex3, Direction const &normal3);

    BaseNormalTriangle &operator=(BaseNormalTriangle const &) = delete;
    BaseNormalTriangle &operator=(BaseNormalTriangle &&) = delete;

    friend class NormalTriangle;
};

class px::NormalTriangle : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &vertex1, Direction const &normal1,
                                                Point const &vertex2, Direction const &normal2,
                                                Point const &vertex3, Direction const &normal3,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setVertices(Point const &a,
                        Point const &b,
                        Point const &c);
    void setNormals(Direction const &na,
                    Direction const &nb,
                    Direction const &nc);

    ~NormalTriangle();
protected:
    BaseNormalTriangle *_obj;
    void *_gpu_obj;
    bool _need_upload;

    void _updateVertices();

    Vec3<PREC> getTextureCoord(PREC const &x,
                                PREC const &y,
                                PREC const &z) const override;
    const BaseGeometry * hitCheck(Ray const &ray,
                                    PREC const &range_start,
                                    PREC const &range_end,
                                    PREC &hit_at) const override;
    Direction normalVec(PREC const &x, PREC const &y, PREC const &z) const override;


    NormalTriangle(Point const &vertex1, Direction const &normal1,
                   Point const &vertex2, Direction const &normal2,
                   Point const &vertex3, Direction const &normal3,
                   std::shared_ptr<BaseMaterial> const &material,
                   std::shared_ptr<Transformation> const &trans);

    NormalTriangle &operator=(NormalTriangle const &) = delete;
    NormalTriangle &operator=(NormalTriangle &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_NORMAL_TRIANGLE_HPP
