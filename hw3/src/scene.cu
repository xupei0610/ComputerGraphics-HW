//#include <cuda_profiler_api.h>
#include "scene.hpp"
#include "trace.cuh"

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
void rayCast(const bool *stop,
             Light *lights,
             RayTrace::TraceQueue::Node *node,
             const PREC v_offset, const PREC u_offset,
             const int n_nodes, const int start, const int size)
{

    auto tid = blockIdx.x * blockDim.x+threadIdx.x;

    int index;
    Light tmp_l;
    Direction n;
    bool double_face;
    Point intersect;
    Vec3<PREC> texture_coord;
    GeometryObj *obj;
    curandState_t state;
    RayTrace::TraceQueue tr(node + (tid)* n_nodes, n_nodes);
    RayTrace::TraceQueue::Node current;

    curand_init(clock()+tid, 0, 0, &state);

    PX_CUDA_LOOP(i, size)
    {
        if (*stop == true)
            return;

        index = i+start;

        tr.reset();

        {
            auto v = (scene_param->height - 1) * 0.5 -
                     (index / scene_param->width) + v_offset;
            auto u = (scene_param->width - 1) * 0.5 -
                     (index % scene_param->width) + u_offset;

            auto x =
                    u * cam_param->right_vector.x + v * cam_param->up_vector.x +
                    cam_param->dist * cam_param->dir.x;
            auto y =
                    u * cam_param->right_vector.y + v * cam_param->up_vector.y +
                    cam_param->dist * cam_param->dir.y;
            auto z =
                    u * cam_param->right_vector.z + v * cam_param->up_vector.z +
                    cam_param->dist * cam_param->dir.z;


            current.ray.direction.set(x, y, z);
        }

        current.ray.original = cam_param->pos;
        current.coef.x = 1;
        current.coef.y = 1;
        current.coef.z = 1;
        current.depth = 0;

        tmp_l.x = 0;
        tmp_l.y = 0;
        tmp_l.z = 0;

        do
        {
            obj = scene_param->geometries->hit(current.ray,
                                               scene_param->hit_min_tol,
                                               scene_param->hit_max_tol,
                                               intersect);

            if (obj == nullptr)
            {
                tmp_l += scene_param->bg * current.coef;
            }
            else
            {
                texture_coord = obj->textureCoord(intersect);

                n = obj->normal(intersect, double_face);

                tmp_l += RayTrace::reflect(intersect, current.ray.direction,
                                           texture_coord,
                                           obj, scene_param, &state,
                                           n, double_face) * current.coef;
                if (current.depth < scene_param->recursion_depth)
                {
                    RayTrace::recursive(intersect, current,
                                        texture_coord, *obj,
                                        n,
                                        tr, *scene_param);
                }
            }

            if (tr.n > 0)
            {
                current = tr.ptr[tr.n - 1];
                tr.pop();
            }
            else
                break;
        } while (*stop == false);

        lights[i] += tmp_l;
    }

    __syncthreads();
}

__global__ void toColor(Light *input,
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
    std::cout << "\r[Info] Upload data to GPU..." << std::flush;

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    _param->n_lights = _lights.size();

    LightObj *pl[_param->n_lights];

    for (auto &l : _lights) l->up2Gpu();
    _geometries->up2Gpu();

    auto i = 0;
    for (auto &l : _lights) pl[i++] = l->devPtr();
    _param->geometries = _geometries->devPtr();

    PX_CUDA_CHECK(cudaMalloc(&(_param->lights),
                             sizeof(LightObj *) * _param->n_lights));
    PX_CUDA_CHECK(cudaMemcpy(_param->lights, pl,
                             sizeof(LightObj *) * _param->n_lights,
                             cudaMemcpyHostToDevice));


    dim3 threads(PX_CUDA_THREADS_PER_BLOCK, 1, 1);


    auto num_kernels = std::min(PX_CUDA_MAX_STREAMS,
                               (_param->dimension + PX_CUDA_MIN_BLOCKS - 1)/PX_CUDA_MIN_BLOCKS);

    auto streams = new cudaStream_t[num_kernels];
    auto blocks = new dim3[num_kernels];
    auto dim_end = new int[num_kernels+1];
    dim_end[num_kernels] = _param->dimension;
    for (auto i = num_kernels; --i > -1;)
    {
        cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
        dim_end[i] = _param->dimension*i/num_kernels;
        blocks[i].x = cuda::blocks(dim_end[i+1] - dim_end[i]);
    }

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

    PX_CUDA_CHECK(cudaMalloc(&nodes, _param->dimension*
                                     sizeof(RayTrace::TraceQueue::Node)*n_nodes))
    PX_CUDA_CHECK(cudaMalloc(&lights, _param->dimension*sizeof(Light)))
    PX_CUDA_CHECK(cudaMemset(lights, 0, _param->dimension*sizeof(Light)))
    bool *stop_flag;
    PX_CUDA_CHECK(cudaHostGetDevicePointer(&stop_flag, _gpu_stop_flag, 0))

    PX_CUDA_CHECK(cudaFuncSetCacheConfig((void*)rayCast, cudaFuncCachePreferL1))

    std::cout << "\r[Info] Begin rendering..." << std::flush;

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

            for (auto i = 0; i < num_kernels; ++i)
            {
                rayCast<<<blocks[i], threads, 0, streams[i]>>> (stop_flag,
                                lights+dim_end[i], nodes+dim_end[i]*n_nodes,
                                v_offset, u_offset,
                                n_nodes, dim_end[i], dim_end[i+1]-dim_end[i]);
            }
            if (*_gpu_stop_flag)
                break;
        }
        if (*_gpu_stop_flag)
            break;
    }

    PX_CUDA_CHECK(cudaDeviceSynchronize());

    if (*_gpu_stop_flag == false)
        toColor<<<cuda::blocks(_param->dimension), threads>>>(lights, _pixels_gpu, _param->dimension, sampling_weight);

//    cudaProfilerStop();

#ifndef NDEBUG
    TOC(1)
#endif

    PX_CUDA_CHECK(cudaFree(_param->lights))
    PX_CUDA_CHECK(cudaFree(nodes));
    PX_CUDA_CHECK(cudaFree(lights));
    for (auto i = 0; i < num_kernels; ++i)
    {
        PX_CUDA_CHECK(cudaStreamDestroy(streams[i]))
    }
    delete [] dim_end;
    delete [] streams;
    delete [] blocks;
    _param->geometries = nullptr;
    _param->lights = nullptr;
}
