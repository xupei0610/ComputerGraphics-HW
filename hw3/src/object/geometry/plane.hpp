#ifndef PX_CG_OBJECT_GEOMETRY_PLANE_HPP
#define PX_CG_OBJECT_GEOMETRY_PLANE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Plane;
class BasePlane;
}

class px::BasePlane
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

    void setPos(Point const &p);
    void setNormal(Direction const &n);

    ~BasePlane() = default;

protected:
    Point _pos;
    Direction _norm;
    PREC _p_dot_n;

    GeometryObj *_dev_obj;

    BasePlane(Point const &pos,
              Direction const &norm_vec);
    BasePlane &operator=(BasePlane const &) = delete;
    BasePlane &operator=(BasePlane &&) = delete;

    friend class Plane;
};

class px::Plane : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setPos(Point const &position);
    void setNormal(Direction const &norm_vec);

    ~Plane();
protected:
    BasePlane *_obj;
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

    Plane(Point const &position,
          Direction const &norm_vec,
          std::shared_ptr<BaseMaterial> const &material,
          std::shared_ptr<Transformation> const &trans);

    Plane &operator=(Plane const &) = delete;
    Plane &operator=(Plane &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_PLANE_HPP
