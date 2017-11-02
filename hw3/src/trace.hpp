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
               int const &depth = 0);

// the following is for gpu
struct TraceQueue
{
    struct Node
    {
        Ray ray;
        Light coef;
        int depth;

        PX_CUDA_CALLABLE
        Node() = default;
        PX_CUDA_CALLABLE
        Node(Ray const &ray, Light const &coef, int const &depth);
        PX_CUDA_CALLABLE
        ~Node() = default;
    };

    Node *ptr;
    int n;
    const int &size;

    PX_CUDA_CALLABLE
    TraceQueue(Node *const &ptr, int const &size);
    PX_CUDA_CALLABLE
    ~TraceQueue() = default;

    PX_CUDA_CALLABLE
    bool prepend(Point const &ray_o, Direction const &ray_d,
                 Light const &coef, int const &depth);
    PX_CUDA_CALLABLE
    void pop();
};

__device__
GeometryObj *hitCheck(Ray const & ray,
                             const Scene::Param *const &scene,
                             Point &intersection);
__device__
Light reflect(Point const &intersect,
              Point const &texture_coord,
              const GeometryObj *const &obj,
              const Scene::Param *const &scene,
              curandState_t * const &state,
              Direction const &n, Direction const &r);
__device__
void recursive(Point const &intersect,
                TraceQueue::Node const &current,
                Point const &texture_coord,
               GeometryObj const &obj,
                Direction &n,
                Direction const &r,
                TraceQueue &trace,
                Scene::Param const &scene);

}}
#endif // PX_CG_TRACE_HPP
