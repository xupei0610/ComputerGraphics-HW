#ifndef PX_CG_OBJECT_GEOMETRY_RING_HPP
#define PX_CG_OBJECT_GEOMETRY_RING_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Ring;
class BaseRing;
}

class px::BaseRing
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
    void setNormal(Direction const &norm_vec);
    void setRadius(PREC const &radius1, PREC const &radius2);

    ~BaseRing() = default;
protected:
    Point _center;
    Direction _norm;
    PREC _inner_radius;
    PREC _outer_radius;
    PREC _inner_radius2;
    PREC _outer_radius2;
    PREC _p_dot_n;

    GeometryObj *_dev_obj;

    BaseRing(Point const &pos,
             Direction const &norm_vec,
             PREC const &radius1,
             PREC const &radius2);

    BaseRing &operator=(BaseRing const &) = delete;
    BaseRing &operator=(BaseRing &&) = delete;

    friend class Ring;
};

class px::Ring : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                Direction const &norm_vec,
                                                PREC const &radius1,
                                                PREC const &radius2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setNormal(Direction const &norm_vec);
    void setRadius(PREC const &radius1, PREC const &radius2);

    ~Ring();
protected:
    BaseRing *_obj;
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
    Direction normalVec(PREC const &x, PREC const &y,
                        PREC const &z,
                        bool &double_face) const override;

    Ring(Point const &center,
         Direction const &norm_vec,
         PREC const &radius1,
         PREC const &radius2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Ring &operator=(Ring const &) = delete;
    Ring &operator=(Ring &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_RING_HPP
