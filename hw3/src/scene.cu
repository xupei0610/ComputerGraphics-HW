#include "scene.hpp"
#include "trace.hpp"

using namespace px;

#ifndef NDEBUG
#define NO_STDIO_REDIRECT
#endif
#define NO_STDIO_REDIRECT

PX_CUDA_KERNEL rayCast(const int width,
                       const int height,
                       const double cam_dist,
                       const int sampling_r,
                       const double sampling_offset,
                       const double sampling_weight,
                       const Point cam_pos,
                       const Direction cam_up_vector,
                       const Direction cam_right_vector,
                       const Direction cam_dir,
                       Scene::Color *pixels,
                       Scene::Param *scene,
                       const int size)
{

//    PX_CUDA_LOOP(index, size)
//    {
//        auto h = index / width;
//        auto w = index % width;
//        auto v0 = (height - 1) * 0.5 - h;
//        auto u0 = (width - 1) * 0.5 - w;
//
//        Light light(0, 0, 0);
//        Ray ray(cam_pos, Direction(0, 0, 0));
//
//        for (auto k0 = -sampling_r + 1; k0 < sampling_r; k0 += 2)
//        {
//            for (auto k1 = -sampling_r + 1; k1 < sampling_r; k1 += 2)
//            {
//
//                auto v = v0 + k0  * sampling_offset;
//                auto u = u0 + k1  * sampling_offset;
//
//                auto x = u * cam_right_vector.x + v * cam_up_vector.x +
//                         cam_dist * cam_dir.x;
//                auto y = u * cam_right_vector.y + v * cam_up_vector.y +
//                         cam_dist * cam_dir.y;
//                auto z = u * cam_right_vector.z + v * cam_up_vector.z +
//                         cam_dist * cam_dir.z;
//
//                ray.direction.set(x, y, z);
//
//                light += RayTrace::traceGpu(scene, ray);
//            }
//        }
//
//        pixels[index] = light * sampling_weight;
//    }
    printf("Hello thread %d, f=%f\n", scene->n_lights, scene->lights[0]->attenuate(1, 2, 3));
    __syncthreads();
}

void Scene::renderGpu(int const &width, int const &height,
                      double const &cam_dist,
                      int const &sampling_r,
                      double const &sampling_offset,
                      double const &sampling_weight)
{
//    if (_need_upload_gpu_param)
//    {
        if (_param_host != nullptr)
            clearGpuData();

        PX_CUDA_CHECK(cudaHostAlloc(&_param_host, sizeof(Param), cudaHostAllocMapped));
        PX_CUDA_CHECK(cudaMemcpy(_param_host, _param, sizeof(Param), cudaMemcpyHostToHost));

        _param_host->n_geometries = geometries.size();
        _param_host->n_lights = lights.size();

//        for (const auto &g : geometries)
//            g->up2Gpu();

        for (const auto &l : lights)
            l->up2Gpu();

        cudaDeviceSynchronize();

        auto i = 0;
//        BaseGeometry* geo[_param_host->n_geometries];
//        for (const auto &g : geometries)
//            g->up2Gpu();
        i = 0;
        BaseLight* lig[_param_host->n_geometries];
        for (const auto &l : lights)
            lig[i++] = l->devPtr();

//        PX_CUDA_CHECK(cudaMalloc(&(_param_host->geometries),
//                                 sizeof(BaseGeometry*)*_param_host->n_geometries));
//        PX_CUDA_CHECK(cudaMemcpy(_param_host->geometries, &geo,
//                                 sizeof(BaseGeometry*)*_param_host->n_geometries,
//                                 cudaMemcpyHostToDevice));

        PX_CUDA_CHECK(cudaMalloc(&(_param_host->lights),
                                 sizeof(BaseLight*)*_param_host->n_lights));
        PX_CUDA_CHECK(cudaMemcpy(_param_host->lights, &lig,
                                 sizeof(BaseLight*)*_param_host->n_lights,
                                 cudaMemcpyHostToDevice));

        PX_CUDA_CHECK(cudaHostGetDevicePointer(&_param_dev, _param_host, 0))

        _need_upload_gpu_param = false;
//    }

    PX_CUDA_LAUNCH_KERNEL(rayCast, 1, //_param_host->dimension,
                          width, height,
                          cam_dist,
                          sampling_r, sampling_offset, sampling_weight,
                          cam->position, cam->up_vector, cam->right_vector,
                          cam->direction,
                          _pixels_gpu,
                          _param_dev,
                          _param_host->dimension);

    cudaDeviceSynchronize();

}
