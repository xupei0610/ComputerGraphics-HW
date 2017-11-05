#ifndef PX_CG_TRACE_CUH
#define PX_CG_TRACE_CUH

#include "trace.hpp"

#include <cfloat>

namespace px { namespace RayTrace
{

struct TraceQueue
{
    struct Node
    {
        Ray ray;
        Light coef;
        int depth;

        __device__
        Node() = default;
        __device__
        Node(Point const &pos, Direction const &dir, Light const &coef, int const &depth);
        __device__
        ~Node() = default;
    };

    Node *ptr;
    int n;
    int size;

    __device__
    TraceQueue(Node *const &ptr, int const &size);
    __device__
    ~TraceQueue() = default;

    __device__
    inline void reset() { n = 0; }
    __device__
    bool prepend(Point const &ray_o, Direction const &ray_d,
                 Light const &coef, int const &depth);
    __device__
    inline void pop() { if (n>0) --n; }
};

__device__
Light reflect(Point const &intersect,
              Point const &texture_coord,
              const GeometryObj *__restrict__ const &obj,
              const Scene::Param *__restrict__ const &scene,
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


#endif
