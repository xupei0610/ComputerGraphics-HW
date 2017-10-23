#ifndef PX_CG_GEOMETRY_GEOMETRY_HPP
#define PX_CG_GEOMETRY_GEOMETRY_HPP

#include <unordered_set>
#include <limits>
#include <vector>
#include <list>

#include "object/base_object.hpp"
#include "object/material.hpp"

namespace px
{
class BaseGeometry;

// TODO Polygon
//class Polygon;

// Surface
class Quadric;

// 3D geometry
class Ellipsoid;
class Box;
// TODO Torus
//class Torus

// TODO NormalTriangle
class NormalTriangle;

// TODO Constructive Solid Geometry
// TODO Implicit Twisted Super Quadric
// TODO Procedurally generated terrain/heightfields

class BoundBox;
class BVH;
// TODO OctTree
}

class px::BaseGeometry
{
protected:
    const BaseMaterial * material;
    const Transformation * transformation;

    Point * _raw_vertices;
    int _n_vertices;

public:
    virtual BaseGeometry *up2Gpu() {return this;};
    virtual void clearGpuData() {};

    BaseGeometry * hit(Ray const &ray,
                       double const &range_start,
                       double const &range_end,
                       double &hit_at);
    Direction normal(double const &x,
                     double const &y,
                     double const &z);
    Vec3<double> textureCoord(double const &x, double const &y, double const &z);

    virtual Point * rawVertices(int &n_vertices) const noexcept
    {
        n_vertices = _n_vertices;
        return _raw_vertices;
    }

    virtual Light ambient(double const &x,
                          double const &y,
                          double const &z)
    {
        return material->ambient(textureCoord(x, y, z));
    }
    virtual Light diffuse(double const &x,
                          double const &y,
                          double const &z)
    {
        return material->diffuse(textureCoord(x, y, z));
    }
    virtual Light specular(double const &x,
                           double const &y,
                           double const &z)
    {
        return material->specular(textureCoord(x, y, z));
    }
    virtual Light transmissive(double const &x,
                               double const &y,
                               double const &z)
    {
        return material->transmissive(textureCoord(x, y, z));
    }
    virtual Vec3<double> textureCoord(Point const &p)
    {
        return textureCoord(p.x, p.y, p.z);
    }
    Direction normVec(Point const &p)
    {
        return normal(p.x, p.y, p.z);
    }
    virtual Light ambient(Point const &p)
    {
        return ambient(p.x, p.y, p.z);
    }
    virtual Light diffuse(Point const &p)
    {
        return diffuse(p.x, p.y, p.z);
    }
    virtual Light specular(Point const &p)
    {
        return specular(p.x, p.y, p.z);
    }
    virtual Light transmissive(Point const &p)
    {
        return transmissive(p.x, p.y, p.z);
    }
    virtual double refractiveIndex(Point const &p)
    {
        return material->refractiveIndex(p.x, p.y, p.z);
    }

    virtual ~BaseGeometry() = default;
protected:
    virtual Vec3<double> getTextureCoord(double const &x,
                                         double const &y,
                                         double const &z) = 0;
    inline Vec3<double> getTextureCoord(Point const &p)
    {
        return getTextureCoord(p.x, p.y, p.z);
    }
    virtual BaseGeometry * hitCheck(Ray const &ray,
                                    double const &range_start,
                                    double const &range_end,
                                    double &hit_at) = 0;
    virtual Direction normalVec(double const &x, double const &y,
                                double const &z) = 0;
    inline Direction normalVec(Point const &p)
    {
        return normalVec(p.x, p.y, p.z);
    }

    BaseGeometry(const BaseMaterial * const &material,
                 const Transformation * const &trans,
                 int const &n_vertices);

private:

    BaseGeometry &operator=(BaseGeometry const &) = delete;
    BaseGeometry &operator=(BaseGeometry &&) = delete;

};

class px::Ellipsoid : public BaseGeometry
{
public:
    Point const &center;
    double const &radius_x;
    double const &radius_y;
    double const &radius_z;

    static std::shared_ptr<BaseGeometry> create(Point const &center,
                                                double const &radius_x,
                                                double const &radius_y,
                                                double const &radius_z,
                                                std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);

    void setParams(Point const &center,
                   double const &radius_x,
                   double const &radius_y,
                   double const &radius_z);

    ~Ellipsoid() override = default;
protected:
    
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override;
    Direction normalVec(double const &x, double const &y, double const &z) override;

    Point _center;
    double _radius_x;
    double _radius_y;
    double _radius_z;

    double _a;
    double _b;
    double _c;

    Ellipsoid(Point const &center,
              double const &radius_x,
              double const &radius_y,
              double const &radius_z,
              std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);
};

class px::Box : public BaseGeometry, protected AbstractBox
{
public:
    Direction const static NORM001; // {0, 0, -1}
    Direction const static NORM002; // {0, 0,  1}
    Direction const static NORM010; // {0, -1, 0}
    Direction const static NORM020; // {0,  1, 0}
    Direction const static NORM100; // {-1, 0, 0}
    Direction const static NORM200; // { 1, 0, 0}

    static std::shared_ptr<BaseGeometry> create(double const &x1, double const &x2,
                                                double const &y1, double const &y2,
                                                double const &z1, double const &z2,
                                                std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);
    static std::shared_ptr<BaseGeometry> create(Point const &v1, Point const &v2,
                                                std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);

    std::vector<Point> const & rawVertices() const noexcept override;

    inline void setVertices(Point const &v1, Point const &v2)
    {
        setVertices(v1.x, v2.x, v1.y, v2.y, v1.z, v2.z);
    }
    void setVertices(double const &x1, double const &x2,
                     double const &y1, double const &y2,
                     double const &z1, double const &z2);

    ~Box() override = default;
protected:
    
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override;
    Direction normalVec(double const &x, double const &y, double const &z) override;

    Box(double const &x1, double const &x2,
        double const &y1, double const &y2,
        double const &z1, double const &z2,
        std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);
    Box(Point const &v1, Point const &v2,
        std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);

};

class px::BoundBox : public BaseGeometry, protected AbstractBox
{
public:
    BoundBox(std::shared_ptr<Transformation> const &trans);

    std::vector<Point> const & rawVertices() const noexcept override;

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    ~BoundBox() override = default;
protected:
    
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override; // always {0, 0, 0}
    Direction normalVec(double const &x, double const &y, double const &z) override; // always {0, 0, 0}

    std::unordered_set<std::shared_ptr<BaseGeometry> > _objs;
};

class px::BVH : public BaseGeometry
{
protected:
    class Extent
    {
    public:
        Extent(std::shared_ptr<BaseGeometry> const &obj);

        double lower_bound[7];
        double upper_bound[7];
        std::shared_ptr<BaseGeometry> obj;

        ~Extent() = default;
    };

public:
    Direction const static NORM_VEC[7];

    BVH();

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    ~BVH() override = default;
protected:
    
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override; // always {0, 0, 0}
    Direction normalVec(double const &x, double const &y, double const &z) override; // always {0, 0, 0}


    std::list<Extent> _extents;
};

class px::NormalTriangle : public BaseGeometry
{
public:
    static std::shared_ptr<BaseGeometry> create(Point const &vertex1, Direction const &normal1,
                                                Point const &vertex2, Direction const &normal2,
                                                Point const &vertex3, Direction const &normal3,
                                                std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);

    ~NormalTriangle() override = default;
protected:
    
    BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) override;
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) override;
    Direction normalVec(double const &x, double const &y, double const &z) override;

    NormalTriangle(Point const &vertex1, Direction const &normal1,
                   Point const &vertex2, Direction const &normal2,
                   Point const &vertex3, Direction const &normal3,
                   std::shared_ptr<BaseMaterial> const &material,
                 std::shared_ptr<Transformation> const &trans);
};
#endif // PX_CG_GEOMETRY_GEOMETRY_HPP
