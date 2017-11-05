#ifndef PX_CG_TRACE_HPP
#define PX_CG_TRACE_HPP

#include "scene.hpp"

namespace px { namespace RayTrace
{

template<typename T>
PX_CUDA_CALLABLE
inline
T ambientReflect(T  const &light,
                 T const &material)
{
    return light * material;
}

PX_CUDA_CALLABLE
inline
Light diffuseReflect(Light const &light,
                 Light const &material,
                 Direction const &to_light_vec,
                 Direction const &norm_vec)
{
    auto cosine = to_light_vec.dot(norm_vec);
    if (cosine < 0) cosine *= -1;
    return light * material * cosine;
}

PX_CUDA_CALLABLE
inline
Light specularReflect(Light const &light,
                      Light const &material,
                      Direction const &to_light_vec,
                      Direction const &reflect_vec,
                      int const &specular_exponent)
{
    auto f = to_light_vec.dot(reflect_vec);
    if (f < 0) return {0, 0, 0};
    return light * material * std::pow(f, specular_exponent);
}
PX_CUDA_CALLABLE
inline
PREC specularReflect(PREC const &light,
                     PREC const &material,
                     Direction const &to_light_vec,
                     Direction const &reflect_vec,
                     int const &specular_exponent)
{
    auto f = to_light_vec.dot(reflect_vec);
    if (f < 0) return 0;
    return light * material * std::pow(f, specular_exponent);
}


Light traceCpu(bool const &stop_flag,
               const Scene *const &scene,
               Ray const &ray,
//               Light const &coef = {1, 1, 1},
               int const &depth = 0);

}}
#endif // PX_CG_TRACE_HPP
