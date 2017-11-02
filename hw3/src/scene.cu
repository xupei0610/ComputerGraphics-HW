#include <cuda_profiler_api.h>
#include "scene.hpp"
#include "trace.hpp"

using namespace px;

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

__global__
void rayCast(bool *  __restrict__ stop,
             Light *  __restrict__ lights,
             RayTrace::TraceQueue::Node *  __restrict__ node,
             PREC v_offset, PREC u_offset,
             int n_nodes)
{
    RayTrace::TraceQueue tr(nullptr, n_nodes);
    curandState_t state;
    curand_init(clock()+blockIdx.x * blockDim.x+threadIdx.x, 0, 0, &state);

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

        tr.ptr = node + (blockIdx.x * blockDim.x+threadIdx.x)*n_nodes;
        tr.n = 0;

        RayTrace::TraceQueue::Node current({cam_param->pos, {x, y, z}},
                                           {1, 1, 1}, 0);
        do
        {
            Point intersect;
            auto obj = RayTrace::hitCheck(current.ray, scene_param, intersect);

            if (obj == nullptr)
            {
                lights[index] += scene_param->bg * current.coef;
            }
            else
            {
                auto texture_coord = obj->textureCoord(intersect);

                Direction n(obj->normal(intersect));
                Direction r(current.ray.direction-n*(2*current.ray.direction.dot(n)));

                lights[index] += RayTrace::reflect(intersect, texture_coord,
                                                   obj, scene_param, &state,
                                                   n, r) * current.coef;
                if (current.depth < scene_param->recursion_depth)
                    RayTrace::recursive(intersect, current,
                                        texture_coord, *obj,
                                        n, r,
                                        tr, *scene_param);
            }

            if (tr.n > 0)
            {
                current = tr.ptr[tr.n - 1];
                tr.pop();
            }
            else
                break;
        } while (*stop == false);
    }
}

__global__ void toColor(Light * __restrict__ input,
                        Scene::Color *  __restrict__ output,
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
    std::cout << "[Info] Upload data to GPU..." << std::flush;

    _param->n_geometries = geometries.size();
    _param->n_lights = lights.size();

    LightObj *pl[_param->n_lights];
    GeometryObj *pg[_param->n_geometries];

    for (auto &l : lights) l->up2Gpu();
    for (auto &g : geometries) g->up2Gpu();

    PX_CUDA_CHECK(cudaDeviceSynchronize());

    auto i = 0;
    for (auto &l : lights) pl[i++] = l->devPtr();
    i = 0;
    for (auto &l : geometries) pg[i++] = l->devPtr();

    PX_CUDA_CHECK(cudaMalloc(&(_param->lights),
                             sizeof(LightObj *) * _param->n_lights));
    PX_CUDA_CHECK(cudaMemcpy(_param->lights, pl,
                             sizeof(LightObj *) * _param->n_lights,
                             cudaMemcpyHostToDevice));

    PX_CUDA_CHECK(cudaMalloc(&(_param->geometries),
                             sizeof(GeometryObj *) * _param->n_geometries));
    PX_CUDA_CHECK(cudaMemcpy(_param->geometries, pg,
                             sizeof(GeometryObj *) * _param->n_geometries,
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

    auto n_nodes = 1+_param->recursion_depth;
    RayTrace::TraceQueue::Node *nodes;
    Light *lights;

    PX_CUDA_CHECK(cudaMalloc(&nodes, blocks.x*threads.x*
                                     sizeof(RayTrace::TraceQueue::Node)*n_nodes))
    PX_CUDA_CHECK(cudaMalloc(&lights, _param->dimension*sizeof(Light)))
    PX_CUDA_CHECK(cudaMemset(lights, 0, _param->dimension*sizeof(Light)))
    bool *stop_flag;
    PX_CUDA_CHECK(cudaHostGetDevicePointer(&stop_flag, _gpu_stop_flag, 0))

    std::cout << "\n[Info] Begin rendering..." << std::flush;

#ifndef NDEBUG
    TIC(1)
#endif
//    cudaProfilerStart();
    if (*_gpu_stop_flag == true)
        return;

    for (auto k0 = -sampling_r + 1; k0 < sampling_r; k0 += 2)
    {
        for (auto k1 = -sampling_r + 1; k1 < sampling_r; k1 += 2)
        {
            auto v_offset = k0 * sampling_offset;
            auto u_offset = k1 * sampling_offset;

            rayCast<<<blocks, threads>>> (
                    stop_flag,
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
