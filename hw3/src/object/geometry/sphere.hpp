#ifndef PX_CG_OBJECT_GEOMETRY_SPHERE_HPP
#define PX_CG_OBJECT_GEOMETRY_SPHERE_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Sphere;
class BaseSphere;
}

class px::BaseSphere
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
    void setRadius(PREC const &r);

    ~BaseSphere() = default;
protected:
    Point _center;
    PREC _radius;
    PREC _radius2;

    GeometryObj *_dev_obj;

    BaseSphere(Point const &pos,
               PREC const &radius);

    BaseSphere &operator=(BaseSphere const &) = delete;
    BaseSphere &operator=(BaseSphere &&) = delete;

    friend class Sphere;
};

class px::Sphere : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                PREC const &radius,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setRadius(PREC const &r);

    ~Sphere();
protected:
    BaseSphere *_obj;
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

    Sphere(Point const &center,
           PREC const &radius,
           std::shared_ptr<BaseMaterial> const &material,
           std::shared_ptr<Transformation> const &trans);

    Sphere &operator=(Sphere const &) = delete;
    Sphere &operator=(Sphere &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_SPHERE_HPP
