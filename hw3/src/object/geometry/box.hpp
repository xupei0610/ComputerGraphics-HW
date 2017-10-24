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
protected:
    PX_CUDA_CALLABLE
    const BaseGeometry * hitCheck(Ray const &ray,
                                  double const &range_start,
                                  double const &range_end,
                                  double &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) const override;

    ~BaseBox() = default;
protected:
    Point _vertex_min;
    Point _vertex_max;

    Point _center;
    Vec3<double> _side;

    BaseBox(const BaseMaterial * const &material,
            const Transformation * const &trans);

    BaseBox &operator=(BaseBox const &) = delete;
    BaseBox &operator=(BaseBox &&) = delete;
};

class px::Box : public BaseBox
{
public:
    static std::shared_ptr<BaseGeometry> create(double const &x1, double const &x2,
                                                double const &y1, double const &y2,
                                                double const &z1, double const &z2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    static std::shared_ptr<BaseGeometry> create(Point const &v1, Point const &v2,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    inline void setVertices(Point const &v1, Point const &v2)
    {
        setVertices(v1.x, v2.x, v1.y, v2.y, v1.z, v2.z);
    }
    void setVertices(double const &x1, double const &x2,
                     double const &y1, double const &y2,
                     double const &z1, double const &z2);

    ~Box();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseBox * _dev_ptr;
    bool _need_upload;

    Box(double const &x1, double const &x2,
        double const &y1, double const &y2,
        double const &z1, double const &z2,
        std::shared_ptr<BaseMaterial> const &material,
        std::shared_ptr<Transformation> const &trans);
    Box(Point const &v1, Point const &v2,
        std::shared_ptr<BaseMaterial> const &material,
        std::shared_ptr<Transformation> const &trans);

    Box &operator=(Box const &) = delete;
    Box &operator=(Box &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_BOX_HPP
