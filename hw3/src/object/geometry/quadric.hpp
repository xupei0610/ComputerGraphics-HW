#ifndef PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
#define PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class Quadric;
class BaseQuadric;
}

class px::BaseQuadric : public BaseGeometry
{
protected:
    PX_CUDA_CALLABLE
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) override;

    ~BaseQuadric() = default;
protected:
    Point _center;
    double _a, _b, _c;
    double _d, _e, _f;
    double _g, _h, _i;
    double _j;

    bool _sym_xy, _sym_yz, _sym_xz, _sym_x, _sym_y, _sym_z, _sym_o;

    double _x0, _x1;
    double _y0, _y1;
    double _z0, _z1;

    BaseQuadric(Point const &pos,
             const BaseMaterial * const &material,
             const Transformation * const &trans);

    BaseQuadric &operator=(BaseQuadric const &) = delete;
    BaseQuadric &operator=(BaseQuadric &&) = delete;
};

class px::Quadric : public BaseQuadric
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                double const &a,
                                                double const &b,
                                                double const &c,
                                                double const &d,
                                                double const &e,
                                                double const &f,
                                                double const &g,
                                                double const &h,
                                                double const &i,
                                                double const &j,
                                                double const &x0, double const &x1,
                                                double const &y0, double const &y1,
                                                double const &z0, double const &z1,
                                                std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans);
    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    void setCenter(Point const &position);
    void setCoef(double const &a,
                 double const &b,
                 double const &c,
                 double const &d,
                 double const &e,
                 double const &f,
                 double const &g,
                 double const &h,
                 double const &i,
                 double const &j,
                 double const &x0, double const &x1,
                 double const &y0, double const &y1,
                 double const &z0, double const &z1);

    ~Quadric();
protected:

    std::shared_ptr<BaseMaterial> _material_ptr;
    std::shared_ptr<Transformation> _transformation_ptr;

    BaseQuadric * _dev_ptr;
    bool _need_upload;

    Quadric(Point const &center,
            double const &a,
            double const &b,
            double const &c,
            double const &d,
            double const &e,
            double const &f,
            double const &g,
            double const &h,
            double const &i,
            double const &j,
            double const &x0, double const &x1,
            double const &y0, double const &y1,
            double const &z0, double const &z1,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans);

    void updateVertices();

    Quadric &operator=(Quadric const &) = delete;
    Quadric &operator=(Quadric &&) = delete;
};


#endif // PX_CG_OBJECT_GEOMETRY_QUADRIC_HPP
