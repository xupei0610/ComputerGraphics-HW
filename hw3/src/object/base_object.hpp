#ifndef PX_CG_OBJECT_BASE_OBJECT_HPP
#define PX_CG_OBJECT_BASE_OBJECT_HPP

#include <exception>
#include <memory>
#include <cmath>
#include <vector>

#include "util.hpp"

namespace px
{
class Light;

typedef Vec3<double> Point;
class Direction;
class Ray;

class Camera;

class Transformation;
}

class px::Light : public Vec3<double>
{
public:
    PX_CUDA_CALLABLE
    Light()
        : Vec3<double>(0, 0, 0)
    {}

    PX_CUDA_CALLABLE
    Light(double const &x, double const &y, double const &z)
        : Vec3<double>(x, y, z)
    {}

    template<typename T>
    PX_CUDA_CALLABLE
    Light(Vec3<T> const &v)
        : Vec3<double>(v.x, v.y, v.z)
    {}
};

class px::Direction : public Vec3<double>
{
public:
    PX_CUDA_CALLABLE
    Direction();

    PX_CUDA_CALLABLE
    Direction(Vec3<double> const &);

    PX_CUDA_CALLABLE
    Direction(double const &x, double const &y, double const &z);

    PX_CUDA_CALLABLE
    Direction &operator=(Direction const &rhs) = default;

    template<typename T>
    PX_CUDA_CALLABLE
    Direction &operator=(Vec3<T> const &rhs)
    {
        set(rhs.x, rhs.y, rhs.z);
        return *this;
    }

    PX_CUDA_CALLABLE
    void set(double const &x, double const &y, double const &z);
};

class px::Ray
{
public:
    Point original; // original
    Direction direction; // direction

    PX_CUDA_CALLABLE
    Ray(Point const &o, Direction const &d);

    PX_CUDA_CALLABLE
    Point operator[](double const &t) const noexcept
    {
        return Point(original.x + direction.x*t,
                     original.y + direction.y*t,
                     original.z + direction.z*t);
    }
    PX_CUDA_CALLABLE
    Ray &operator=(Ray const &r)
    {
        original = r.original;
        direction = r.direction;
        return *this;
    }
};

class px::Camera
{
public:
    Point position;  // center
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
    inline void setAs(const Camera *const &c)
    {
        setPosition(c->position);
        setDirection(c->direction, c->up_vector);
        setHalfAngle(c->half_angle);
    }

    ~Camera() = default;

protected:
    Camera(Point const &pos,
           Direction const &d,
           Direction const &u,
           double const &ha);

    Camera &operator=(Camera const &c) = delete;
    Camera &operator=(Camera &&c) = delete;
private:
    Direction _d;
    Direction _u;
    Direction _r;
};

class px::Transformation
{
public:
    Transformation * up2Gpu();
    void clearGpuData();

    static std::shared_ptr<Transformation> create(double const &rotate_x,
                                                  double const &rotate_y,
                                                  double const &rotate_z,
                                                  double const &t_x,
                                                  double const &t_y,
                                                  double const &t_z,
                                                  std::shared_ptr<Transformation> const &parent = nullptr);

    PX_CUDA_CALLABLE
    Point point(double const &x, double const &y, double const &z) const noexcept;
    PX_CUDA_CALLABLE
    Direction direction(Direction const &d) const noexcept;
    PX_CUDA_CALLABLE
    Direction normal(Direction const &n) const noexcept;

    PX_CUDA_CALLABLE
    inline Point point(Point const &p) const noexcept
    {
        return point(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Direction direction(double const &x, double const &y, double const &z) const noexcept
    {
        return direction(Direction(x, y, z));
    }
    PX_CUDA_CALLABLE
    inline Direction normal(double const &x, double const &y, double const &z) const noexcept
    {
        return direction(Direction(x, y, z));
    }

    void setParams(double const &rotate_x,
                   double const &rotate_y,
                   double const &rotate_z,
                   double const &t_x,
                   double const &t_y,
                   double const &t_z,
                   std::shared_ptr<Transformation> const &parent = nullptr);

    inline double const & r00() const noexcept { return _r00; }
    inline double const & r01() const noexcept { return _r01; }
    inline double const & r02() const noexcept { return _r02; }
    inline double const & r10() const noexcept { return _r10; }
    inline double const & r11() const noexcept { return _r11; }
    inline double const & r12() const noexcept { return _r12; }
    inline double const & r20() const noexcept { return _r20; }
    inline double const & r21() const noexcept { return _r21; }
    inline double const & r22() const noexcept { return _r22; }
    inline double const & t0() const noexcept { return _t0; }
    inline double const & t1() const noexcept { return _t1; }
    inline double const & t2() const noexcept { return _t2; }

    ~Transformation();
protected:
    double _rotate_x, _rotate_y, _rotate_z;

    double _r00, _r01, _r02;
    double _r10, _r11, _r12;
    double _r20, _r21, _r22;
    double _t0,  _t1,  _t2;
    double _t00, _t01, _t02;

    Transformation * _dev_ptr;
    bool _need_upload;

    Transformation(double const &rotate_x,
                   double const &rotate_y,
                   double const &rotate_z,
                   double const &t_x,
                   double const &t_y,
                   double const &t_z,
                   std::shared_ptr<Transformation> const &parent);
};


#endif // PX_CG_OBJECT_BASE_OBJECT_HPP
