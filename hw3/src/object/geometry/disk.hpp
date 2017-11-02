#ifndef PX_CG_OBJECT_GEOMETRY_DISK_HPP
#define PX_CG_OBJECT_GEOMETRY_DISK_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Disk;
class BaseDisk;
}

class px::BaseDisk
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
    
    void setCenter(Point const &position);
    void setRadius(PREC const &radius);
    void setNormal(Direction const &norm_vec);

    ~BaseDisk() = default;
protected:
    Point _center;
    Direction _norm;
    PREC _radius;
    PREC _radius2;
    PREC _p_dot_n;

    GeometryObj *_dev_obj;

    BaseDisk(Point const &pos,
             Direction const &norm_vec,
             PREC const &radius);

    BaseDisk &operator=(BaseDisk const &) = delete;
    BaseDisk &operator=(BaseDisk &&) = delete;

    friend class Disk;
};

class px::Disk : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &position,
                                                Direction const &norm_vec,
                                                PREC const &radius,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    void up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setRadius(PREC const &radius);
    void setNormal(Direction const &norm_vec);

    ~Disk();
protected:
    BaseDisk *_obj;
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
                                PREC const &z) const override;

    Disk(Point const &position,
         Direction const &norm_vec,
         PREC const &radius,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    Disk &operator=(Disk const &) = delete;
    Disk &operator=(Disk &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_DISK_HPP
