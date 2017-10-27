#ifndef PX_CG_UTIL_MATH_CONSTANT_HPP
#define PX_CG_UTIL_MATH_CONSTANT_HPP

#ifdef USE_CUDA
#  include <cfloat>
#else
#  include <cmath>
#  include <limits>
#endif

namespace px
{

#define EPSILON 1e-6
#define DOUBLE_EPSILON 1e-3

#ifdef USE_CUDA

#  define PI 3.14159265358979323846
#  define PI2 6.28318530717958647692
#  define PI_by_4 0.785398163397448309616
#  define DEG2RAD 0.01745329251994329576922

#else

PREC const static PI = std::acos(-1.0);
PREC const static PI2 = 2 * PI;
PREC const static PI_by_4 = PI/4.0;
PREC const static DEG2RAD = PI/180;

#endif

}

#endif // PX_CG_UTIL_MATH_CONSTANT_HPP
