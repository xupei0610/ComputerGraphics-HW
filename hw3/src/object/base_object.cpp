#include <limits>
#include "object/base_object.hpp"

using namespace px;

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
    _r = d.cross(_u);
    _u = _r.cross(d);
}

std::shared_ptr<Transformation> Transformation::create(PREC const &s_x,
                                                       PREC const &s_y,
                                                       PREC const &s_z,
                                                       PREC const &rotate_x,
                                                     PREC const &rotate_y,
                                                     PREC const &rotate_z,
                                                     PREC const &t_x,
                                                     PREC const &t_y,
                                                     PREC const &t_z,
                                                     std::shared_ptr<Transformation> const &parent)
{
    return std::shared_ptr<Transformation>(new Transformation(s_x, s_y, s_z,
                                                              rotate_x, rotate_y, rotate_z,
                                               t_x, t_y, t_z,
                                               parent));
}

Transformation::Transformation(PREC const &s_x,
                               PREC const &s_y,
                               PREC const &s_z,
                               PREC const &rotate_x,
                               PREC const &rotate_y,
                               PREC const &rotate_z,
                               PREC const &t_x,
                               PREC const &t_y,
                               PREC const &t_z,
                               std::shared_ptr<Transformation> const &parent)
    : _dev_ptr(nullptr), _need_upload(true)
{
    setParams(s_x, s_y, s_z, rotate_x, rotate_y, rotate_z,
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

//PX_CUDA_CALLABLE
//Point Transformation::point2ObjCoord(PREC const &x,
//                                     PREC const &y,
//                                     PREC const &z) const noexcept
//{
//    return {_inv00 * x + _inv01 * y + _inv02 * z + _inv03,
//            _inv10 * x + _inv11 * y + _inv12 * z + _inv13,
//            _inv20 * x + _inv21 * y + _inv22 * z + _inv23};
//}
//PX_CUDA_CALLABLE
//Point Transformation::pointFromObjCoord(PREC const &x,
//                                        PREC const &y,
//                                        PREC const &z) const noexcept
//{
//    return {_m00 * x + _m01 * y + _m02 * z + _m03,
//            _m10 * x + _m11 * y + _m12 * z + _m13,
//            _m20 * x + _m21 * y + _m22 * z + _m23};
//}
//
//PX_CUDA_CALLABLE
//Direction Transformation::direction(Direction const &d) const noexcept
//{
//    Direction nd;
//    nd.x = _inv00 * d.x + _inv01 * d.y + _inv02 * d.z;
//    nd.y = _inv10 * d.x + _inv11 * d.y + _inv12 * d.z;
//    nd.z = _inv20 * d.x + _inv21 * d.y + _inv22 * d.z;
//    return nd;
//}
//PX_CUDA_CALLABLE
//Direction Transformation::normal(Direction const &n) const noexcept
//{
//    return {_inv00 * n.x + _inv10 * n.y + _inv20 * n.z,
//            _inv01 * n.x + _inv11 * n.y + _inv21 * n.z,
//            _inv02 * n.x + _inv12 * n.y + _inv22 * n.z};
//}

void Transformation::setParams(PREC const &s_x,
                               PREC const &s_y,
                               PREC const &s_z,
                               PREC const &rotate_x,
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
        _m00 = cy * cz - sx*sy*sz;
        _m01 = - cx * sz;
        _m02 = cy*sx*sz+cz*sy;
        _m10 = cz*sx*sy + cy*sz;
        _m11 = cx*cz;
        _m12 = sy*sz - cy*cz*sx;
        _m20 = - cx*sy;
        _m21 = sx;
        _m22 = cx * cy;
        _m03 = t_x;
        _m13 = t_y;
        _m23 = t_z;

        _m00 *= s_x;
        _m01 *= s_x;
        _m02 *= s_x;
        _m03 *= s_x;

        _m10 *= s_y;
        _m11 *= s_y;
        _m12 *= s_y;
        _m13 *= s_y;

        _m20 *= s_z;
        _m21 *= s_z;
        _m22 *= s_z;
        _m23 *= s_z;

    }
    else
    {
        auto m00 = cy * cz - sx*sy*sz;
        auto m01 = - cx * sz;
        auto m02 = cy*sx*sz+cz*sy;
        auto m10 = cz*sx*sy + cy*sz;
        auto m11 = cx*cz;
        auto m12 = sy*sz - cy*cz*sx;
        auto m20 = - cx*sy;
        auto m21 = sx;
        auto m22 = cx * cy;
        auto m03 = t_x;
        auto m13 = t_y;
        auto m23 = t_z;

        m00 *= s_x;
        m01 *= s_x;
        m02 *= s_x;
        m03 *= s_x;

        m10 *= s_y;
        m11 *= s_y;
        m12 *= s_y;
        m13 *= s_y;

        m20 *= s_z;
        m21 *= s_z;
        m22 *= s_z;
        m23 *= s_z;

        _m00 = parent->m00() * m00 + parent->m01() * m10 + parent->m02() * m20;
        _m01 = parent->m00() * m01 + parent->m01() * m11 + parent->m02() * m21;
        _m02 = parent->m00() * m02 + parent->m01() * m12 + parent->m02() * m22;
        _m03 = parent->m00() * m03 + parent->m01() * m13 + parent->m02() * m23 + parent->m03();

        _m10 = parent->m10() * m00 + parent->m11() * m10 + parent->m12() * m20;
        _m11 = parent->m10() * m01 + parent->m11() * m11 + parent->m12() * m21;
        _m12 = parent->m10() * m02 + parent->m11() * m12 + parent->m12() * m22;
        _m13 = parent->m10() * m03 + parent->m11() * m13 + parent->m12() * m23 + parent->m13();

        _m20 = parent->m20() * m00 + parent->m21() * m10 + parent->m22() * m20;
        _m21 = parent->m20() * m01 + parent->m21() * m11 + parent->m22() * m21;
        _m22 = parent->m20() * m02 + parent->m21() * m12 + parent->m22() * m22;
        _m23 = parent->m20() * m03 + parent->m21() * m13 + parent->m22() * m23 + parent->m23();
    }


    auto coef = 1/(_m00*_m11*_m22 + _m01*_m12*_m20
                  + _m02*_m10*_m21 - _m00*_m12*_m21
                  - _m01*_m10*_m22 - _m02*_m11*_m20);

    _inv00 = _m11*_m22 - _m12*_m21;
    _inv01 = _m02*_m21 - _m01*_m22;
    _inv02 = _m01*_m12 - _m02*_m11;
    _inv03 = _m01*_m13*_m22 + _m02*_m11*_m23 + _m03*_m12*_m21 - _m01*_m12*_m23 - _m02*_m13*_m21 - _m03*_m11*_m22;

    _inv10 = _m12*_m20 - _m10*_m22;
    _inv11 = _m00*_m22 - _m02*_m20;
    _inv12 = _m02*_m10 - _m00*_m12;
    _inv13 = _m00*_m12*_m23 + _m02*_m13*_m20 + _m03*_m10*_m22 - _m00*_m13*_m22 - _m02*_m10*_m23 - _m03*_m12*_m20;

    _inv20 = _m10*_m21 - _m11*_m20;
    _inv21 = _m01*_m20 - _m00*_m21;
    _inv22 = _m00*_m11 - _m01*_m10;
    _inv23 = _m00*_m13*_m21 + _m01*_m10*_m23 + _m03*_m11*_m20 - _m00*_m11*_m23 - _m01*_m13*_m20 - _m03*_m10*_m21;

    _inv00 *= coef;
    _inv01 *= coef;
    _inv02 *= coef;
    _inv03 *= coef;
    _inv10 *= coef;
    _inv11 *= coef;
    _inv12 *= coef;
    _inv13 *= coef;
    _inv20 *= coef;
    _inv21 *= coef;
    _inv22 *= coef;
    _inv23 *= coef;

//    printf("%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n\n",
//           _m00, _m01, _m02, _m03,
//           _m10, _m11, _m12, _m13,
//           _m20, _m21, _m22, _m23);
//
//
//    printf("%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n",
//           _inv00, _inv01, _inv02, _inv03,
//           _inv10, _inv11, _inv12, _inv13,
//           _inv20, _inv21, _inv22, _inv23);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}