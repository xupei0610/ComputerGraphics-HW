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
                 Direction const &to_light_dir,
                 Direction const &norm_vec, bool const &double_face)
{
    auto cosine = to_light_dir.dot(norm_vec);
    if (cosine < 0)
    {
        if (double_face) cosine *= -1;
        else
            return {0, 0, 0};
    }
    return light * material * cosine;
}

PX_CUDA_CALLABLE
inline
Light specularReflect(Light const &light,
                      Light const &material,
                      Direction const &to_camera_dir,
                      Direction const &reflect_dir,
                      PREC const &shininessonent)
{
    auto f = to_camera_dir.dot(reflect_dir);
    if (f < 0) return {0, 0, 0};
    return light * material * std::pow(f, shininessonent);
}

Light traceCpu(bool const &stop_flag,
               const Scene *const &scene,
               Ray const &ray,
//               Light const &coef = {1, 1, 1},
               int const &depth = 0);

}}
#endif // PX_CG_TRACE_HPP
