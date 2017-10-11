#ifndef PX_CG_OBJECT_HPP
#define PX_CG_OBJECT_HPP

#include <exception>
#include <memory>
#include <cmath>
#include <iostream>

#include "util.hpp"

namespace px
{
class Light;

typedef Vec3<double> Point;
class Direction;
class Ray;

class BaseLight;
class DirectionalLight;
class PointLight;
class SpotLight;

class BaseMaterial;
class UniformMaterial;

class Camera;

class BaseObject;
class Sphere;
class Triangle;
class NormalTriangle;

}

class px::Light : public Vec3<double>
{
public:
    Light()
        : Vec3<double>(1, 2, 3)
    {}

    Light(double const &x, double const &y, double const &z)
        : Vec3<double>(x, y, z)
    {}

    template<typename T>
    Light(Vec3<T> const &v)
        : Vec3<double>(v.x, v.y, v.z)
    {}
};

class px::Direction : public Vec3<double>
{
public:
    Direction();
    Direction(Vec3<double> const &);
    Direction(double const &x, double const &y, double const &z);
    Direction &operator=(Direction const &rhs) = default;
    template<typename T>
    Direction &operator=(Vec3<T> const &rhs)
    {
        set(rhs.x, rhs.y, rhs.z);
        return *this;
    }
    void set(double const &x, double const &y, double const &z);
};

class px::Ray
{
public:
    Point original; // original
    Direction direction; // direction

    Ray(Point const &o, Direction const &d);

    Point operator[](double const &t) const noexcept
    {
        return Point(original.x + direction.x*t,
                     original.y + direction.y*t,
                     original.z + direction.z*t);
    }

    Ray &operator=(Ray const &r)
    {
        original = r.original;
        direction = r.direction;
        return *this;
    }
};


class px::BaseLight
{
public:
    double static constexpr MAX_LIGHT = 10000000;

    Light light;
    Point position;

    virtual double attenuate(double const &x, double const &y, double const &z) = 0;

    inline double attenuate(Point const & p)
    {
        return attenuate(p.x, p.y, p.z);
    }
    virtual ~BaseLight() = default;

protected:
    BaseLight(Light const &light, Point const &pos);
    BaseLight &operator=(BaseLight const &) = delete;
    BaseLight &operator=(BaseLight &&) = delete;
};

class px::DirectionalLight : public BaseLight
{
public:

    static std::shared_ptr<BaseLight> create(Light const &light, Point const &pos);

    double attenuate(double const &x, double const &y, double const &z);

    ~DirectionalLight() = default;
protected:
    DirectionalLight(Light const &light, Point const &pos);
};

class px::PointLight : public BaseLight
{
public:

    static std::shared_ptr<BaseLight> create(Light const &light, Point const &pos);

    double attenuate(double const &x, double const &y, double const &z);

    ~PointLight() = default;
protected:
    PointLight(Light const &light, Point const &pos);
};

class px::SpotLight : public BaseLight
{
public:
    Direction direction;
    double const &inner_half_angle;
    double const &outer_half_angle;
    double falloff;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &pos,
                                             Direction const &direction,
                                             double const &half_angle1,
                                             double const &half_angle2,
                                             double const &falloff);

    double attenuate(double const &x, double const &y, double const &z);

    void setAngles(double const &half_angle1, double const &half_angle2);

    ~SpotLight() = default;
protected:
    SpotLight(Light const &light,
              Point const &pos,
              Direction const &direction,
              double const &half_angle1,
              double const &half_angle2,
              double const &falloff);
    double _inner_ha;
    double _outer_ha;
    double _inner_ha_cosine;
    double _outer_ha_cosine;
    double _multiplier; // 1.0 / (_outer_ha_cosine - inner_ha_cosine)
};

class px::BaseMaterial
{
public:
    virtual Light ambient(double const &x, double const &y, double const &z) const = 0;
    virtual Light diffuse(double const &x, double const &y, double const &z) const = 0;
    virtual Light specular(double const &x, double const &y, double const &z) const = 0;
    virtual Light transmissive(double const &x, double const &y, double const &z) const = 0;
    virtual int specularExponent() const = 0;
    virtual double refractiveIndex() const = 0;

    inline Light ambient(Point const &p) const
    {
        return ambient(p.x, p.y, p.z);
    }
    inline Light diffuse(Point const &p) const
    {
        return diffuse(p.x, p.y, p.z);
    }
    inline Light specular(Point const &p) const
    {
        return specular(p.x, p.y, p.z);
    }
    inline Light transmissive(Point const &p) const
    {
        return transmissive(p.x, p.y, p.z);
    }

    virtual ~BaseMaterial() = default;
protected:
    BaseMaterial() = default;
    BaseMaterial &operator=(BaseMaterial const &) = delete;
    BaseMaterial &operator=(BaseMaterial &&) = delete;
};

class px::UniformMaterial : public BaseMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient = {0, 0, 0},
                                                Light const &diffuse = {1, 1, 1},
                                                Light const &specular = {0, 0, 0},
                                                int const &specular_exponent = 5,
                                                Light const &transmissive ={0, 0, 0},
                                                double const &refractive_index = 1.0);
    ~UniformMaterial() = default;

    UniformMaterial &operator=(UniformMaterial const &m) = default;

    Light ambient(double const &x, double const &y, double const &z) const override;
    Light diffuse(double const &x, double const &y, double const &z) const override;
    Light specular(double const &x, double const &y, double const &z) const override;
    Light transmissive(double const &x, double const &y, double const &z) const override;
    int specularExponent() const override;
    double refractiveIndex() const override;

protected:
    UniformMaterial(Light const &ambient,
             Light const &diffuse,
             Light const &specular,
             int const &specular_exponent,
             Light const &transmissive,
             double const &refractive_index);

    Light _ambient; // valid range [0, 255]
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    double _refractive_index;
};

class px::Camera
{
public:
    Point position;  // position
    Direction const &direction; // view direction
    Direction const &up_vector; // up direction
    Direction const &right_vector; // right direction = direction x up_vector
    double half_angle; // half of the height angle, rad

    static std::shared_ptr<Camera> create(Point const &pos = {0, 0, 0},
                                          Direction const &d = {0, 0, 1},
                                          Direction const &u = {0, 1, 0},
                                          double const &ha = PI_by_4);
    void setPosition(Point const &pos);
    void setDirection(Direction const &d, Direction const &u);
    void setHalfAngle(double const ha);
    inline void setAs(Camera const &c)
    {
        setPosition(c.position);
        setDirection(c.direction, c.up_vector);
        setHalfAngle(c.half_angle);
    }

    ~Camera() = default;

    Camera &operator=(Camera const &c)
    {
        setAs(c);
        return *this;
    }
protected:
    Camera(Point const &pos,
           Direction const &d,
           Direction const &u,
           double const &ha);

private:
    Direction _d;
    Direction _u;
    Direction _r;
};

class px::BaseObject
{
public:
    std::shared_ptr<BaseMaterial> material;

    virtual bool hit(Ray const &ray,
                     double const &range_start,
                     double const &range_end,
                     double &hit_at) = 0;
    virtual Direction normVec(double const &x,
                           double const &y,
                           double const &z) = 0;
    virtual Vec3<double> relativePos(double const &x,
                                     double const &y,
                                     double const &z) = 0;

    virtual bool contain(double const &x, double const &y, double const &z) = 0;

    virtual Light ambient(double const &x,
                          double const &y,
                          double const &z)
    {
        return material->ambient(relativePos(x, y, z));
    }
    virtual Light diffuse(double const &x,
                          double const &y,
                          double const &z)
    {
        return material->diffuse(relativePos(x, y, z));
    }
    virtual Light specular(double const &x,
                           double const &y,
                           double const &z)
    {
        return material->specular(relativePos(x, y, z));
    }
    virtual Light transmissive(double const &x,
                               double const &y,
                               double const &z)
    {
        return material->transmissive(relativePos(x, y, z));
    }
    virtual bool contain(Point const &p)
    {
        return contain(p.x, p.y, p.z);
    }
    virtual Vec3<double> relativePos(Point const &p)
    {
        return relativePos(p.x, p.y, p.z);
    }
    virtual Direction normVec(Point const &p)
    {
        return normVec(p.x, p.y, p.z);
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

    virtual ~BaseObject() = default;
protected:
    BaseObject(std::shared_ptr<BaseMaterial> const &material);

private:
    BaseObject &operator=(BaseObject const &) = delete;
    BaseObject &operator=(BaseObject &&) = delete;
};

class px::Sphere : public BaseObject
{
public:
    Point position;
    double radius;

    static std::shared_ptr<BaseObject> create(Point const &pos,
                                              double const &r,
                                              std::shared_ptr<BaseMaterial> const &material);
    bool hit(Ray const &ray,
             double const &range_start,
             double const &range_end,
             double &hit_at) override;
    bool contain(double const &x, double const &y, double const &z) override;
    Direction normVec(double const &x, double const &y, double const &z) override;
    Vec3<double> relativePos(double const &x, double const &y, double const &z) override;

    ~Sphere() = default;
protected:
    Sphere(Point const &pos,
           double const &r,
           std::shared_ptr<BaseMaterial> const &material);
};

class px::Triangle : public BaseObject
{
public:
    static std::shared_ptr<BaseObject> create(Point const &vertex1,
                                              Point const &vertex2,
                                              Point const &vertex3,
                                              std::shared_ptr<BaseMaterial> const &material);
    bool hit(Ray const &ray,
             double const &range_start,
             double const &range_end,
             double &hit_at) override;
    bool contain(double const &x, double const &y, double const &z) override;
    Direction normVec(double const &x, double const &y, double const &z) override;
    Vec3<double> relativePos(double const &x, double const &y, double const &z) override;

    ~Triangle() = default;
protected:
    Triangle(Point const &vertex1,
             Point const &vertex2,
             Point const &vertex3,
             std::shared_ptr<BaseMaterial> const &material);
};

class px::NormalTriangle : public BaseObject
{
public:
    static std::shared_ptr<BaseObject> create(Point const &vertex1, Direction const &normal1,
                                              Point const &vertex2, Direction const &normal2,
                                              Point const &vertex3, Direction const &normal3,
                                              std::shared_ptr<BaseMaterial> const &material);
    bool hit(Ray const &ray,
             double const &range_start,
             double const &range_end,
             double &hit_at) override;
    bool contain(double const &x, double const &y, double const &z) override;
    Direction normVec(double const &x, double const &y, double const &z) override;
    Vec3<double> relativePos(double const &x, double const &y, double const &z) override;

    ~NormalTriangle() = default;
protected:
    NormalTriangle(Point const &vertex1, Direction const &normal1,
                   Point const &vertex2, Direction const &normal2,
                   Point const &vertex3, Direction const &normal3,
                   std::shared_ptr<BaseMaterial> const &material);
};

#endif // PX_CG_OBJECT_HPP
