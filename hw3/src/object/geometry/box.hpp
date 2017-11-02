#ifndef PX_CG_OBJECT_GEOMETRY_BOX_HPP
#define PX_CG_OBJECT_GEOMETRY_BOX_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Box;
class BaseBox;
}

class px::BaseBox
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

    void setVertices(PREC const &x1, PREC const &x2,
                     PREC const &y1, PREC const &y2,
                     PREC const &z1, PREC const &z2);

    ~BaseBox() = default;
protected:
    Point _vertex_min;
    Point _vertex_max;

    Point _center;
    Vec3<PREC> _side;

    GeometryObj *_dev_obj;

    BaseBox(PREC const &x1, PREC const &x2,
            PREC const &y1, PREC const &y2,
            PREC const &z1, PREC const &z2);

    BaseBox &operator=(BaseBox const &) = delete;
    BaseBox &operator=(BaseBox &&) = delete;

    friend class Box;
};

class px::Box : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(PREC const &x1, PREC const &x2,
                                                PREC const &y1, PREC const &y2,
                                                PREC const &z1, PREC const &z2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    static std::shared_ptr<BaseGeometry> create(Point const &v1, Point const &v2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setVertices(Point const &v1, Point const &v2);
    void setVertices(PREC const &x1, PREC const &x2,
                     PREC const &y1, PREC const &y2,
                     PREC const &z1, PREC const &z2);

    ~Box();
protected:
    BaseBox *_obj;
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
                        PREC const &z) const override;

    Box(PREC const &x1, PREC const &x2,
        PREC const &y1, PREC const &y2,
        PREC const &z1, PREC const &z2,
        std::shared_ptr<BaseMaterial> const &material,
        std::shared_ptr<Transformation> const &trans);

    Box &operator=(Box const &) = delete;
    Box &operator=(Box &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BOX_HPP
