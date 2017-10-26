#include <cuda_profiler_api.h>
#include "scene.hpp"
#include "trace.hpp"

using namespace px;

#ifndef NDEBUG
#define NO_STDIO_REDIRECT
#endif
#define NO_STDIO_REDIRECT

struct CameraParam
{
    Direction right_vector;
    Direction up_vector;
    Direction dir;
    Point pos;
    PREC dist;

    PX_CUDA_CALLABLE
    CameraParam(Direction const &r, Direction const &u, Direction const &d,
                Point const &p, PREC const &dist)
        : right_vector(r), up_vector(u), dir(d), pos(p), dist(dist)
    {};

    PX_CUDA_CALLABLE
    CameraParam() {};
    PX_CUDA_CALLABLE
    ~CameraParam() {};
};

__constant__ CameraParam cam_param[1];
__constant__ Scene::Param scene_param[1];

PX_CUDA_KERNEL rayCast(bool *stop, int *progress,
                       Light *lights,
                       RayTrace::TraceQueue::Node *node,
                       PREC v_offset, PREC u_offset,
                       int n_nodes)
{
    RayTrace::TraceQueue tr(nullptr, n_nodes);

    auto tid = blockIdx.x * blockDim.x+threadIdx.x;

    PX_CUDA_LOOP(index, scene_param->dimension)
    {
        if (*stop == true)
            break;

        auto v = (scene_param->height - 1) * 0.5 -
                 (index / scene_param->width) + v_offset;
        auto u = (scene_param->width - 1) * 0.5 -
                 (index % scene_param->width) + u_offset;

        auto x = u * cam_param->right_vector.x + v * cam_param->up_vector.x +
                 cam_param->dist * cam_param->dir.x;
        auto y = u * cam_param->right_vector.y + v * cam_param->up_vector.y +
                 cam_param->dist * cam_param->dir.y;
        auto z = u * cam_param->right_vector.z + v * cam_param->up_vector.z +
                 cam_param->dist * cam_param->dir.z;

        tr.ptr = node + tid*n_nodes;
        tr.n = 0;

        RayTrace::TraceQueue::Node current({cam_param->pos, {x, y, z}},
                                           {1, 1, 1}, 0);

        PREC t;
        do
        {
            auto obj = RayTrace::hitCheck(current.ray, scene_param, t);

            if (obj == nullptr)
            {
                lights[index] += scene_param->bg * current.coef;
            }
            else
            {
                auto intersect = current.ray[t];;
                auto texture_coord = obj->textureCoord(intersect);

                Direction r, n;
                lights[index] += RayTrace::reflect(intersect, current.ray.direction,
                                           texture_coord, obj, scene_param,
                                           n, r) * current.coef;
                if (current.depth < scene_param->recursion_depth)
                    RayTrace::recursive(intersect, current,
                                        texture_coord, *obj,
                                        n, r,
                                        tr, *scene_param);
            }
            if (tr.n > 0 || *stop == true)
            {
                current = tr.ptr[tr.n - 1];
                tr.pop();
            }
            else
                break;
        } while (true);

            *progress += 3;
    }
}

PX_CUDA_KERNEL toColor(Light *input,
                       Scene::Color *output,
                       int dim,
                       PREC weight)
{
    PX_CUDA_LOOP(index, dim)
    {
        output[index] = input[index] * weight;
    }
}

void Scene::renderGpu(int const &width, int const &height,
                      PREC const &cam_dist,
                      int const &sampling_r,
                      PREC const &sampling_offset,
                      PREC const &sampling_weight)
{
    _param->n_geometries = geometries.size();
    _param->n_lights = lights.size();

    BaseLight **pl[_param->n_lights];
    BaseGeometry **pg[_param->n_geometries];

    for (auto &l : lights) l->up2Gpu();
    *_rendering_progress = _param->dimension * 0.1;
    for (auto &g : geometries) g->up2Gpu();
    *_rendering_progress = _param->dimension * 0.2;

    cudaDeviceSynchronize();

    auto i = 0;
    for (auto &l : lights) pl[i++] = l->devPtr();
    i = 0;
    for (auto &l : geometries) pg[i++] = l->devPtr();

    PX_CUDA_CHECK(cudaMalloc(&(_param->lights),
                             sizeof(BaseLight **) * _param->n_lights));
    PX_CUDA_CHECK(cudaMemcpy(_param->lights, pl,
                             sizeof(BaseLight **) * _param->n_lights,
                             cudaMemcpyHostToDevice));

    PX_CUDA_CHECK(cudaMalloc(&(_param->geometries),
                             sizeof(BaseGeometry **) * _param->n_geometries));
    PX_CUDA_CHECK(cudaMemcpy(_param->geometries, pg,
                             sizeof(BaseGeometry **) * _param->n_geometries,
                             cudaMemcpyHostToDevice));

    dim3 threads(PX_CUDA_THREADS_PER_BLOCK, 1, 1);
    dim3 blocks(px::cuda::blocks(_param->dimension), 1, 1);

    CameraParam cp[1];
    cp[0].up_vector = cam->up_vector;
    cp[0].right_vector = cam->right_vector;
    cp[0].dir = cam->direction;
    cp[0].dist = cam_dist;
    cp[0].pos = cam->position;


    PX_CUDA_CHECK(cudaMemcpyToSymbol(scene_param, _param,
                                     sizeof(Scene::Param), 0,
                                     cudaMemcpyHostToDevice))
    PX_CUDA_CHECK(cudaMemcpyToSymbol(cam_param, &cp,
                                     sizeof(CameraParam), 0,
                                     cudaMemcpyHostToDevice))

    auto n_nodes = 20*_param->recursion_depth;
    RayTrace::TraceQueue::Node *nodes;
    Light *lights;

    PX_CUDA_CHECK(cudaMalloc(&nodes, blocks.x*threads.x*
                                     sizeof(RayTrace::TraceQueue::Node)*n_nodes))
    PX_CUDA_CHECK(cudaMalloc(&lights, _param->dimension*sizeof(Light)))
    PX_CUDA_CHECK(cudaMemset(lights, 0, _param->dimension*sizeof(Light)))

    bool *stop_flag;
    PX_CUDA_CHECK(cudaHostGetDevicePointer(&stop_flag, _gpu_stop_flag, 0))
    int *progress;
    PX_CUDA_CHECK(cudaHostGetDevicePointer(&progress, _rendering_progress, 0))

#ifndef NDEBUG
    TIC(1)
#endif
//    cudaProfilerStart();

    if (*_gpu_stop_flag == true)
        return;

    *_rendering_progress = _param->dimension * 0.3;

    for (auto k0 = -sampling_r + 1; k0 < sampling_r; k0 += 2)
    {
        for (auto k1 = -sampling_r + 1; k1 < sampling_r; k1 += 2)
        {
            auto v_offset = k0 * sampling_offset;
            auto u_offset = k1 * sampling_offset;

            rayCast<<<blocks, threads>>> (
                    stop_flag, progress,
                    lights, nodes,
                    v_offset, u_offset,
                    n_nodes);
            if (*_gpu_stop_flag)
                break;
        }
        if (*_gpu_stop_flag)
            break;
    }

    PX_CUDA_CHECK(cudaDeviceSynchronize());

    *_rendering_progress = _param->dimension * 0.9;

    if (*_gpu_stop_flag == false)
    {
        toColor<<<blocks, threads>>>(lights, _pixels_gpu, _param->dimension, sampling_weight);
    }



//    cudaProfilerStop();

#ifndef NDEBUG
    TOC(1)
#endif

    PX_CUDA_CHECK(cudaFree(_param->geometries))
    PX_CUDA_CHECK(cudaFree(_param->lights))
    PX_CUDA_CHECK(cudaFree(nodes));
    PX_CUDA_CHECK(cudaFree(lights));
}
