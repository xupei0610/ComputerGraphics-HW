#ifndef PX_CG_UTIL_VEC3_HPP
#define PX_CG_UTIL_VEC3_HPP

#include "cuda.hpp"

namespace px
{

template<typename T>
class Vec3
{
public:

    T x;
    T y;
    T z;

    PX_CUDA_CALLABLE
    Vec3()
        : x(0), y(0), z(0)
    {}

    PX_CUDA_CALLABLE
    Vec3(T const &x, T const &y, T const &z)
            : x(x), y(y), z(z)
    {}

    PX_CUDA_CALLABLE
    Vec3(Vec3<T> const &v)
            : x(v.x), y(v.y), z(v.z)
    {}

    PX_CUDA_CALLABLE
    ~Vec3() = default;

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> &operator=(Vec3<T_IN> const &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> &operator+=(Vec3<T_IN> const &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> &operator-=(Vec3<T_IN> const &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> &operator*=(Vec3<T_IN> const &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> &operator/=(Vec3<T_IN> const &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    PX_CUDA_CALLABLE
    Vec3<T> &operator*=(double const &v)
    {
        x *= v;
        y *= v;
        z *= v;
        return *this;
    }

    PX_CUDA_CALLABLE
    Vec3<T> &operator/=(double const &v)
    {
        x /= v;
        y /= v;
        z /= v;
        return *this;
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    Vec3<T> cross(Vec3<T_IN> const &v) const noexcept
    {
        return Vec3<T>(y * v.z - z * v.y,
                       z * v.x - x * v.z,
                       x * v.y - y * v.x);
    }

    template<typename T_IN>
    PX_CUDA_CALLABLE
    double dot(Vec3<T_IN> const &v) const noexcept
    {
        return x * v.x + y * v.y + z * v.z;
    }

    PX_CUDA_CALLABLE
    void normalize()
    {
        if (x != 0 || y != 0 || z != 0)
        {
            auto norm = std::sqrt(x*x + y*y + z*z);
            x /= norm;
            y /= norm;
            z /= norm;
        }
    }

    PX_CUDA_CALLABLE
    double norm()
    {
        return std::sqrt(x*x + y*y + z*z);
    }

    PX_CUDA_CALLABLE
    double norm2()
    {
        return x*x + y*y + z*z;
    }
};

}

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator*(px::Vec3<T_IN> const &v, double const &factor)
{
    return px::Vec3<double>(v.x*factor, v.y*factor, v.z*factor);
}

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator/(px::Vec3<T_IN> const &v, double const &factor)
{
    return px::Vec3<double>(v.x/factor, v.y/factor, v.z/factor);
}

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator+(px::Vec3<T_IN> const &v, double const &factor)
{
    return px::Vec3<double>(v.x+factor, v.y+factor, v.z+factor);
};

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator-(px::Vec3<T_IN> const &v, double const &factor)
{
    return px::Vec3<double>(v.x-factor, v.y-factor, v.z-factor);
};

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator*(px::Vec3<T_IN> const &v1, px::Vec3<T_IN> const &v2)
{
    return px::Vec3<double>(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator/(px::Vec3<T_IN> const &v1, px::Vec3<T_IN> const &v2)
{
    return px::Vec3<double>(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z);
}

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator+(px::Vec3<T_IN> const &v1, px::Vec3<T_IN> const &v2)
{
    return px::Vec3<double>(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
};

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator-(px::Vec3<T_IN> const &v1, px::Vec3<T_IN> const &v2)
{
    return px::Vec3<double>(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
};

template<typename T_IN>
PX_CUDA_CALLABLE
px::Vec3<double> operator-(double const &factor, px::Vec3<T_IN> const &v)
{
    return px::Vec3<double>(factor-v.x, factor-v.y, factor-v.z);
};

#endif // PX_CG_UTIL_VEC3_HPP
