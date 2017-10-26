#include <limits>
#include "base_object.hpp"

using namespace px;

PX_CUDA_CALLABLE
Direction::Direction(Vec3<PREC> const &v)
    : Vec3<PREC>(v)
{
    Vec3::normalize();
}

PX_CUDA_CALLABLE
Direction::Direction(PREC const &x, PREC const &y, PREC const &z)
    : Vec3<PREC>(x, y, z)
{
    Vec3::normalize();
}

PX_CUDA_CALLABLE
void Direction::set(PREC const &x, PREC const &y, PREC const &z)
{
    Vec3<PREC>::x = x;
    Vec3<PREC>::y = y;
    Vec3<PREC>::z = z;
    Vec3::normalize();
}

PX_CUDA_CALLABLE
Ray::Ray(Point const &o, Direction const &d)
    : original(o), direction(d)
{}

std::shared_ptr<Camera> Camera::create(Point const &pos,
                                       Direction const &d,
                                       Direction const &u,
                                       PREC const &ha)
{
    return std::shared_ptr<Camera>(new Camera(pos, d, u, ha));
}

Camera::Camera(Point const &pos,
               Direction const &d,
               Direction const &u,
               PREC const &ha)
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

void Camera::setHalfAngle(PREC const ha)
{
    half_angle = ha;
}

void Camera::setDirection(Direction const &d, Direction const &u)
{
    _d = d;
    _u = u;
    _r = d.cross(u);
}

std::shared_ptr<Transformation> Transformation::create(PREC const &rotate_x,
                                                     PREC const &rotate_y,
                                                     PREC const &rotate_z,
                                                     PREC const &t_x,
                                                     PREC const &t_y,
                                                     PREC const &t_z,
                                                     std::shared_ptr<Transformation> const &parent)
{
    return std::shared_ptr<Transformation>(new Transformation(rotate_x, rotate_y, rotate_z,
                                               t_x, t_y, t_z,
                                               parent));
}

Transformation::Transformation(PREC const &rotate_x,
                               PREC const &rotate_y,
                               PREC const &rotate_z,
                               PREC const &t_x,
                               PREC const &t_y,
                               PREC const &t_z,
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

Transformation * const & Transformation::devPtr() const noexcept
{
    return _dev_ptr;
}

void Transformation::up2Gpu()
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
#endif
}

void Transformation::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr))
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
Point Transformation::point(PREC const &x,
                            PREC const &y,
                            PREC const &z) const noexcept
{
    return {_r00 * x + _r10 * y + _r20 * z + _t00,
            _r01 * x + _r11 * y + _r21 * z + _t01,
            _r02 * x + _r12 * y + _r22 * z + _t02};
}
PX_CUDA_CALLABLE
Direction Transformation::direction(Direction const &d) const noexcept
{
    Direction nd;
    nd.x = _r00 * d.x + _r10 * d.y + _r20 * d.z;
    nd.y = _r01 * d.x + _r11 * d.y + _r21 * d.z;
    nd.z = _r02 * d.x + _r12 * d.y + _r22 * d.z;
    return nd;
}
PX_CUDA_CALLABLE
Direction Transformation::normal(Direction const &n) const noexcept
{
    return {_r00 * n.x + _r01 * n.y + _r02 * n.z,
            _r10 * n.x + _r11 * n.y + _r12 * n.z,
            _r20 * n.x + _r21 * n.y + _r22 * n.z};
}

void Transformation::setParams(PREC const &rotate_x,
                               PREC const &rotate_y,
                               PREC const &rotate_z,
                               PREC const &t_x,
                               PREC const &t_y,
                               PREC const &t_z,
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