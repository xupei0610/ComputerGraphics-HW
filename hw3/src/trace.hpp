#ifndef PX_CG_TRACE_HPP
#define PX_CG_TRACE_HPP

#include "scene.hpp"

namespace px
{
class RayTrace
{
public:
    static
    Light traceCpu(const Scene *const &scene,
                   Ray const &ray,
                   double const &refractive_index = 1.0,
                   int const &depth = 0);

    __device__ static
    Light traceGpu(const Scene::Param *const &scene,
                   Ray const &ray,
                   double const &refractive_index = 1.0,
                   int const &depth = 0);

private:

    PX_CUDA_CALLABLE
    inline static
    Light ambientReflect(Light const &light,
                         Light const &material)
    {
        return light * material;
    }
    PX_CUDA_CALLABLE
    inline static
    Light diffuseReflect(Light const &light,
                         Light const &material,
                         Direction const &to_light_vec,
                         Direction const &norm_vec)
    {
        auto cosine = to_light_vec.dot(norm_vec);
        if (cosine < 0) cosine *= - 1;
        return light * material * cosine;
    }
    PX_CUDA_CALLABLE
    inline static
    Light specularReflect(Light const &light,
                          Light const &material,
                          Direction const &to_light_vec,
                          Direction const &reflect_vec,
                          int const &specular_exponent)
    {
        auto f = to_light_vec.dot(reflect_vec);
        if (f < 0) return Light(0,0,0);
        return light*material*std::pow(f, specular_exponent);
    }

};
}
#endif // PX_CG_TRACE_HPP
