#include <limits>
#include "base_object.hpp"

using namespace px;

PX_CUDA_CALLABLE
Direction::Direction()
    : Vec3<double>(0, 0, 0)
{
    Vec3::normalize();
}

PX_CUDA_CALLABLE
Direction::Direction(Vec3<double> const &v)
    : Vec3<double>(v)
{
    Vec3::normalize();
}

PX_CUDA_CALLABLE
Direction::Direction(double const &x, double const &y, double const &z)
    : Vec3<double>(x, y, z)
{
    Vec3::normalize();
}

PX_CUDA_CALLABLE
void Direction::set(double const &x, double const &y, double const &z)
{
    Vec3<double>::x = x;
    Vec3<double>::y = y;
    Vec3<double>::z = z;
    Vec3::normalize();
}

PX_CUDA_CALLABLE
Ray::Ray(Point const &o, Direction const &d)
    : original(o), direction(d)
{}

std::shared_ptr<Camera> Camera::create(Point const &pos,
                                       Direction const &d,
                                       Direction const &u,
                                       double const &ha)
{
    return std::shared_ptr<Camera>(new Camera(pos, d, u, ha));
}

Camera::Camera(Point const &pos,
               Direction const &d,
               Direction const &u,
               double const &ha)
    : direction(_d), up_vector(_u), right_vector(_r)
{
    setPosition(pos);
    setHalfAngle(ha);
    setDirection(d, u);
}

void Camera::setPosition(Point const &pos)
{
    position = pos;
}

void Camera::setHalfAngle(double const ha)
{
    half_angle = ha;
}

void Camera::setDirection(Direction const &d, Direction const &u)
{
    _d = d;
    _u = u;
    _r = d.cross(u);
}

AbstractBox::AbstractBox()
    : _vertex_min(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
      _vertex_max(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()),
      _vertices(8, {0, 0, 0})
{}

AbstractBox::AbstractBox(Point const &v)
    : _vertex_min(v),
      _vertex_max(v),
      _vertices(8, {0, 0, 0})
{}

AbstractBox::AbstractBox(double const &x, double const &y, double const &z)
    : _vertex_min(Point(x, y, z)),
      _vertex_max(_vertex_min),
      _vertices(8, {0, 0, 0})
{}

void AbstractBox::addVertex(double const &x, double const &y, double const &z)
{
    if (x < _vertex_min.x)
        _vertex_min.x = x;
    if (x > _vertex_max.x)
        _vertex_max.x = x;

    if (y < _vertex_min.y)
        _vertex_min.y = y;
    if (y > _vertex_max.y)
        _vertex_max.y = y;

    if (z < _vertex_min.z)
        _vertex_min.z = z;
    if (z > _vertex_max.z)
        _vertex_max.z = z;

    updateVertices();
}

void AbstractBox::addVertex(std::vector<Point> const &vert)
{
    for (const auto &v : vert)
    {
        if (v.x < _vertex_min.x)
            _vertex_min.x = v.x;
        if (v.x > _vertex_max.x)
            _vertex_max.x = v.x;

        if (v.y < _vertex_min.y)
            _vertex_min.y = v.y;
        if (v.y > _vertex_max.y)
            _vertex_max.y = v.y;

        if (v.z < _vertex_min.z)
            _vertex_min.z = v.z;
        if (v.z > _vertex_max.z)
            _vertex_max.z = v.z;
    }
    updateVertices();
}

void AbstractBox::updateVertices()
{
    _vertices.at(0).x = _vertex_min.x;
    _vertices.at(0).y = _vertex_min.y;
    _vertices.at(0).z = _vertex_min.z;

    _vertices.at(1).x = _vertex_max.x;
    _vertices.at(1).y = _vertex_min.y;
    _vertices.at(1).z = _vertex_min.z;

    _vertices.at(2).x = _vertex_min.x;
    _vertices.at(2).y = _vertex_max.y;
    _vertices.at(2).z = _vertex_min.z;

    _vertices.at(3).x = _vertex_min.x;
    _vertices.at(3).y = _vertex_min.y;
    _vertices.at(3).z = _vertex_max.z;

    _vertices.at(4).x = _vertex_max.x;
    _vertices.at(4).y = _vertex_max.y;
    _vertices.at(4).z = _vertex_min.z;

    _vertices.at(5).x = _vertex_max.x;
    _vertices.at(5).y = _vertex_min.y;
    _vertices.at(5).z = _vertex_max.z;

    _vertices.at(6).x = _vertex_min.x;
    _vertices.at(6).y = _vertex_max.y;
    _vertices.at(6).z = _vertex_max.z;

    _vertices.at(7).x = _vertex_max.x;
    _vertices.at(7).y = _vertex_max.y;
    _vertices.at(7).z = _vertex_max.z;
}

bool AbstractBox::onForwardFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(z-_vertex_min.z) < 1e-12) &&
           (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x);
}

bool AbstractBox::onBackwardFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(z-_vertex_max.z) < 1e-12) &&
           (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x);
}

bool AbstractBox::onLeftFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(x-_vertex_min.x) < 1e-12) &&
           (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y);
}

bool AbstractBox::onRightFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(x-_vertex_max.x) < 1e-12) &&
           (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y);
}

bool AbstractBox::onTopFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(y-_vertex_max.y) < 1e-12) &&
           (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x);
}

bool AbstractBox::onBottomFace(double const &x, double const &y, double const &z) const
{
    return (std::abs(y-_vertex_min.y) < 1e-12) &&
           (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x);
}

#include <iostream>

bool AbstractBox::hit(Ray const &ray,
                       double const &t_start,
                       double const &t_end,
                       double &hit_at)
{
    auto tmin  = ((ray.direction.x < 0 ? _vertex_max.x : _vertex_min.x) - ray.original.x) / ray.direction.x;
    auto tmax  = ((ray.direction.x < 0 ? _vertex_min.x : _vertex_max.x) - ray.original.x) / ray.direction.x;
    auto tymin = ((ray.direction.y < 0 ? _vertex_max.y : _vertex_min.y) - ray.original.y) / ray.direction.y;
    auto tymax = ((ray.direction.y < 0 ? _vertex_min.y : _vertex_max.y) - ray.original.y) / ray.direction.y;

    if (tmin > tymax || tymin > tmax)
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    auto tzmin = ((ray.direction.z < 0 ? _vertex_max.z : _vertex_min.z) - ray.original.z) / ray.direction.z;
    auto tzmax = ((ray.direction.z < 0 ? _vertex_min.z : _vertex_max.z) - ray.original.z) / ray.direction.z;

    if (tmin > tzmax || tzmin > tmax)
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tmin > t_start && tmin < t_end)
    {
        hit_at = tmin;
        return true;
    }

    if (tzmax < tmax)
        tmax = tzmax;

    if (tmax > t_start && tmax < t_end)
    {
        hit_at = tmax;
        return true;
    }

    return false;
}

std::shared_ptr<Transformation> Transformation::create(double const &rotate_x,
                                                     double const &rotate_y,
                                                     double const &rotate_z,
                                                     double const &t_x,
                                                     double const &t_y,
                                                     double const &t_z,
                                                     std::shared_ptr<Transformation> const &parent)
{
    return std::shared_ptr<Transformation>(new Transformation(rotate_x, rotate_y, rotate_z,
                                               t_x, t_y, t_z,
                                               parent));
}

Transformation::Transformation(double const &rotate_x,
                               double const &rotate_y,
                               double const &rotate_z,
                               double const &t_x,
                               double const &t_y,
                               double const &t_z,
                               std::shared_ptr<Transformation> const &parent)
    : _dev_ptr(nullptr), _need_upload(true)
{
    setParams(rotate_x, rotate_y, rotate_z,
              t_x, t_y, t_z, parent);
}

Transformation::~Transformation()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

Transformation * Transformation::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(Transformation)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 this,
                                 sizeof(Transformation),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void Transformation::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
Point Transformation::point(double const &x,
                            double const &y,
                            double const &z)
{
    return {_r00 * x + _r10 * y + _r20 * z + _t00,
            _r01 * x + _r11 * y + _r21 * z + _t01,
            _r02 * x + _r12 * y + _r22 * z + _t02};
}
PX_CUDA_CALLABLE
Direction Transformation::direction(Direction const &d)
{
    return {_r00 * d.x + _r10 * d.y + _r20 * d.z,
            _r01 * d.x + _r11 * d.y + _r21 * d.z,
            _r02 * d.x + _r12 * d.y + _r22 * d.z};
}
PX_CUDA_CALLABLE
Direction Transformation::normal(Direction const &n)
{
    return {_r00 * n.x + _r01 * n.y + _r02 * n.z,
            _r10 * n.x + _r11 * n.y + _r12 * n.z,
            _r20 * n.x + _r21 * n.y + _r22 * n.z};
}

void Transformation::setParams(double const &rotate_x,
                               double const &rotate_y,
                               double const &rotate_z,
                               double const &t_x,
                               double const &t_y,
                               double const &t_z,
                               std::shared_ptr<Transformation> const &parent)
{
    auto sx = std::sin(rotate_x);
    auto cx = std::cos(rotate_x);
    auto sy = std::sin(rotate_y);
    auto cy = std::cos(rotate_y);
    auto sz = std::sin(rotate_z);
    auto cz = std::cos(rotate_z);

    if (parent == nullptr)
    {
        _r00 = cy * cz;
        _r01 = -cy * sz;
        _r02 = sy;
        _r10 = sx * sy * cz + cx * sz;
        _r11 = cx * cz - sx * sy * sz;
        _r12 = -sx * cy;
        _r20 = sx * sz - cx * sy * cz;
        _r21 = cx * sy * sz + sx * cz;
        _r22 = cx * cy;

        _t0 = t_x;
        _t1 = t_y;
        _t2 = t_z;
    }
    else
    {
        auto r00 = cy * cz;
        auto r01 = -cy * sz;
        auto r02 = sy;
        auto r10 = sx * sy * cz + cx * sz;
        auto r11 = cx * cz - sx * sy * sz;
        auto r12 = -sx * cy;
        auto r20 = sx * sz - cx * sy * cz;
        auto r21 = cx * sy * sz + sx * cz;
        auto r22 = cx * cy;

        auto t0 = t_x;
        auto t1 = t_y;
        auto t2 = t_z;

        _r00 = parent->r00() * r00 + parent->r01() * r10 + parent->r02() * r20;
        _r01 = parent->r00() * r01 + parent->r01() * r11 + parent->r02() * r21;
        _r02 = parent->r00() * r02 + parent->r01() * r12 + parent->r02() * r22;
        _r10 = parent->r10() * r00 + parent->r11() * r10 + parent->r12() * r20;
        _r11 = parent->r10() * r01 + parent->r11() * r11 + parent->r12() * r21;
        _r12 = parent->r10() * r02 + parent->r11() * r12 + parent->r12() * r22;
        _r20 = parent->r20() * r00 + parent->r21() * r10 + parent->r22() * r20;
        _r21 = parent->r20() * r01 + parent->r21() * r11 + parent->r22() * r21;
        _r22 = parent->r20() * r02 + parent->r21() * r12 + parent->r22() * r22;

        _t0 = parent->r00() * t0 + parent->r01() * t1 + parent->r02() * t2 + parent->t0();
        _t1 = parent->r10() * t0 + parent->r11() * t1 + parent->r12() * t2 + parent->t1();
        _t2 = parent->r20() * t0 + parent->r21() * t1 + parent->r22() * t2 + parent->t2();
    }

    _t00 = _r00 * _t0 + _r10 * _t1 + _r20 * _t2;
    _t01 = _r01 * _t0 + _r11 * _t1 + _r21 * _t2;
    _t02 = _r02 * _t0 + _r12 * _t1 + _r22 * _t2;


#ifdef USE_CUDA
    _need_upload = true;
#endif
}