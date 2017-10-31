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

typedef Vec3<PREC> Point;
class Direction;
class Ray;

class Camera;

class Transformation;
}

class px::Light : public Vec3<PREC>
{
public:
    PX_CUDA_CALLABLE
    Light() = default;

    PX_CUDA_CALLABLE
    Light(PREC const &x, PREC const &y, PREC const &z)
        : Vec3<PREC>(x, y, z)
    {}

    template<typename T>
    PX_CUDA_CALLABLE
    Light(Vec3<T> const &v)
        : Vec3<PREC>(v.x, v.y, v.z)
    {}

    PX_CUDA_CALLABLE
    ~Light() = default;
};

class px::Direction : public Vec3<PREC>
{
public:
    PX_CUDA_CALLABLE
    Direction() = default;

    PX_CUDA_CALLABLE
    ~Direction() = default;

    PX_CUDA_CALLABLE
    Direction(Vec3<PREC> const &);

    PX_CUDA_CALLABLE
    Direction(PREC const &x, PREC const &y, PREC const &z);

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
    void set(PREC const &x, PREC const &y, PREC const &z);
};

class px::Ray
{
public:
    Point original; // original
    Direction direction; // direction

    PX_CUDA_CALLABLE
    Ray() = default;

    PX_CUDA_CALLABLE
    Ray(Point const &o, Direction const &d);

    PX_CUDA_CALLABLE
    Point operator[](PREC const &t) const noexcept
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
    PREC half_angle; // half of the height angle, rad

    static std::shared_ptr<Camera> create(Point const &pos = {0, 0, 0},
                                          Direction const &d = {0, 0, 1},
                                          Direction const &u = {0, 1, 0},
                                          PREC const &ha = PI_by_4);
    void setPosition(Point const &pos);
    void setDirection(Direction const &d, Direction const &u);
    void setHalfAngle(PREC const ha);
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
           PREC const &ha);

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
    Transformation * const &devPtr() const noexcept;
    void up2Gpu();
    void clearGpuData();

    static std::shared_ptr<Transformation> create(PREC const &s_x,
                                                  PREC const &s_y,
                                                  PREC const &s_z,
                                                  PREC const &rotate_x,
                                                  PREC const &rotate_y,
                                                  PREC const &rotate_z,
                                                  PREC const &t_x,
                                                  PREC const &t_y,
                                                  PREC const &t_z,
                                                  std::shared_ptr<Transformation> const &parent = nullptr);

    PX_CUDA_CALLABLE
    Point point2ObjCoord(PREC const &x, PREC const &y, PREC const &z) const noexcept;
    PX_CUDA_CALLABLE
    Point pointFromObjCoord(PREC const &x, PREC const &y, PREC const &z) const noexcept;
    PX_CUDA_CALLABLE
    Direction direction(Direction const &d) const noexcept;
    PX_CUDA_CALLABLE
    Direction normal(Direction const &n) const noexcept;

    PX_CUDA_CALLABLE
    inline Point point2ObjCoord(Point const &p) const noexcept
    {
        return point2ObjCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Point pointFromObjCoord(Point const &p) const noexcept
    {
        return pointFromObjCoord(p.x, p.y, p.z);
    }
    PX_CUDA_CALLABLE
    inline Direction direction(PREC const &x, PREC const &y, PREC const &z) const noexcept
    {
        return direction(Direction(x, y, z));
    }
    PX_CUDA_CALLABLE
    inline Direction normal(PREC const &x, PREC const &y, PREC const &z) const noexcept
    {
        return direction(Direction(x, y, z));
    }

    void setParams(PREC const &s_x,
                   PREC const &s_y,
                   PREC const &s_z,
                   PREC const &rotate_x,
                   PREC const &rotate_y,
                   PREC const &rotate_z,
                   PREC const &t_x,
                   PREC const &t_y,
                   PREC const &t_z,
                   std::shared_ptr<Transformation> const &parent = nullptr);

    inline PREC const & m00() const noexcept { return _m00; }
    inline PREC const & m01() const noexcept { return _m01; }
    inline PREC const & m02() const noexcept { return _m02; }
    inline PREC const & m10() const noexcept { return _m10; }
    inline PREC const & m11() const noexcept { return _m11; }
    inline PREC const & m12() const noexcept { return _m12; }
    inline PREC const & m20() const noexcept { return _m20; }
    inline PREC const & m21() const noexcept { return _m21; }
    inline PREC const & m22() const noexcept { return _m22; }
    inline PREC const & m03() const noexcept { return _m03; }
    inline PREC const & m13() const noexcept { return _m13; }
    inline PREC const & m23() const noexcept { return _m23; }

    ~Transformation();
protected:
    PREC _rotate_x, _rotate_y, _rotate_z;
    PREC _t0,  _t1,  _t2;

    PREC _m00, _m01, _m02, _m03;
    PREC _m10, _m11, _m12, _m13;
    PREC _m20, _m21, _m22, _m23;

    PREC _inv00, _inv01, _inv02, _inv03;
    PREC _inv10, _inv11, _inv12, _inv13;
    PREC _inv20, _inv21, _inv22, _inv23;

    Transformation * _dev_ptr;
    bool _need_upload;

    Transformation(PREC const &s_x,
                   PREC const &s_y,
                   PREC const &s_z,
                   PREC const &rotate_x,
                   PREC const &rotate_y,
                   PREC const &rotate_z,
                   PREC const &t_x,
                   PREC const &t_y,
                   PREC const &t_z,
                   std::shared_ptr<Transformation> const &parent);
};


#endif // PX_CG_OBJECT_BASE_OBJECT_HPP
