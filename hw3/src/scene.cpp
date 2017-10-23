#include "scene.hpp"
#include "trace.hpp"

#ifndef NDEBUG
#include <iostream>
#include <chrono>
#define TIC(id) \
    auto _tic_##id = std::chrono::system_clock::now();
#define TOC(id) \
    auto _toc_##id = std::chrono::system_clock::now(); \
    std::cout << "\033[1K\r[Info] Process time: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(_toc_##id - _tic_##id).count() \
              << "ms" << std::endl;
#endif

using namespace px;

Scene::Param::Param(int const &width, int const &height, int const &dimension,
                    Light bg, Light ambient,
                    int const &recursion_depth,
                    int const &area_light_sampling,
                    int const &diffuse_sampling,
                    int const &diffuse_recursion_depth,
                    double const &hit_min_tol, double const &hit_max_tol)
        : width(width), height(height), dimension(dimension),
          bg(bg), ambient(ambient),
          recursion_depth(recursion_depth),
          area_light_sampling(area_light_sampling),
          diffuse_sampling(diffuse_sampling),
          diffuse_recursion_depth(diffuse_recursion_depth),
          hit_min_tol(hit_min_tol), hit_max_tol(hit_max_tol)
{}


Light const Scene::DEFAULT_SCENE_BG = Light(0, 0, 0);

Scene::Scene()
        : _param(new Param(DEFAULT_SCENE_WIDTH, DEFAULT_SCENE_HEIGHT,
             DEFAULT_SCENE_HEIGHT * DEFAULT_SCENE_WIDTH,
             DEFAULT_SCENE_BG, DEFAULT_SCENE_BG,
             DEFAULT_RECURSION_DEPTH,
             DEFAULT_AREA_LIGHT_SAMPLING,
             DEFAULT_DIFFUSE_SAMPLING,
             DEFAULT_DIFFUSE_RECURSION_DEPTH,
             DEFAULT_HIT_MIN_TOL,
             DEFAULT_HIT_MAX_TOL)),
          _param_host(nullptr),
          _param_dev(_param),
          _pixels(nullptr),
          _pixels_gpu(nullptr),
          is_rendering(_is_rendering),
          stop_rendering(false),
          rendering_process(_rendering_processing),
          width(_param->width),
          height(_param->height),
          dimension(_param->dimension),
          bg(_param->bg),
          ambient(_param->ambient),
          sampling_radius(_sampling_radius),
          area_light_sampling(_param->area_light_sampling),
          diffuse_sampling(_param->diffuse_sampling),
          recursion_depth(_param->recursion_depth),
          diffuse_recursion_depth(_param->diffuse_recursion_depth),
          hit_min_tol(_param->hit_min_tol),
          hit_max_tol(_param->hit_max_tol),
          mode(_mode),
          cam(_cam),
          _is_rendering(false),
          _rendering_processing(0),
          _sampling_radius(DEFAULT_SAMPLING_RADIUS),
          _mode(DEFAULT_COMPUTATION_MODE),
          _cam(Camera::create()),
          _allocate_gpu_pixels(false),
          _need_upload_gpu_param(false)
{
    pixels.color = nullptr;
}

Scene::~Scene()
{
    clearPixels();
    clearGpuData();
}

void Scene::clearGpuData()
{
#ifdef USE_CUDA
    if (_param_dev != nullptr)
    {
        for (const auto &g : geometries)
            g->clearGpuData();
        for (const auto &l : lights)
            l->clearGpuData();
        PX_CUDA_CHECK(cudaFree(_param_host->geometries))
        PX_CUDA_CHECK(cudaFree(_param_host->lights))
        PX_CUDA_CHECK(cudaFreeHost(_param_host))
        _param_dev = nullptr;
        _param_host = nullptr;
    }
#endif
}

void Scene::clearPixels()
{
#ifdef USE_CUDA
    if (_allocate_gpu_pixels)
    {
        PX_CUDA_CHECK(cudaFreeHost(_pixels));
        _pixels_gpu = nullptr;
        _allocate_gpu_pixels = false;
    }
    else
#endif
        delete [] _pixels;

    _pixels = nullptr;
}

void Scene::allocatePixels()
{
#ifdef USE_CUDA
    if (mode == ComputationMode::GPU)
    {
        PX_CUDA_CHECK(cudaHostAlloc(&_pixels, sizeof(Color)*_param->dimension, cudaHostAllocMapped));
        PX_CUDA_CHECK(cudaHostGetDevicePointer(&_pixels_gpu, _pixels, 0))
        _allocate_gpu_pixels = true;
    }
    else
#endif
        _pixels = new Color[_param->dimension];

    pixels.color = _pixels;
}

void Scene::setSceneSize(int const &width, int const &height)
{
    if (width < 0)
        throw std::invalid_argument("Failed to set scene width as a negative value.");
    if (height < 0)
        throw std::invalid_argument("Failed to set scene height as a negative value.");
    _param->width = width;
    _param->height = height;
    _param->dimension = _param->width * _param->height;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setBackground(double const &light_r,
                          double const &light_g,
                          double const &light_b)
{
    _param->bg = Light(light_r, light_g, light_b);
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setBackground(Light const &light)
{
    _param->bg = light;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setCamera(std::shared_ptr<Camera> const &cam)
{
    _cam = cam;
}

void Scene::setAmbientLight(Light const &c)
{
    _param->ambient = c;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::addAmbientLight(Light const &c)
{
    _param->ambient += c;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setSamplingRadius(int const &radius)
{
    if (radius < 0)
        throw std::invalid_argument("Failed to set sampling radius as a negative value.");
    _sampling_radius = radius;
}

void Scene::setAreaLightSampling(int const &n)
{
    if (n < 0)
        throw std::invalid_argument("Failed to set sampling number for area light as a negative value.");
    _param->area_light_sampling = n;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setDiffuseSampling(int const &depth, int const &n)
{
    _param->diffuse_recursion_depth = depth;
    _param->diffuse_sampling = n;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setRecursionDepth(int const &depth)
{
    if (depth < 0)
        throw std::invalid_argument("Failed to set recursion depth as a negative value.");
    _param->recursion_depth = depth;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setHitMinTol(double const &tol)
{
    _param->hit_min_tol = tol;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setHitMaxTol(double const &tol)
{
    _param->hit_max_tol = tol;
#ifdef USE_CUDA
    _need_upload_gpu_param = true;
#endif
}

void Scene::setComputationMode(ComputationMode const &mode)
{
#ifdef USE_CUDA
    _mode = mode;
#endif
}

void Scene::render()
{
    _is_rendering = true;
    _rendering_processing = 0;

    clearPixels();
    allocatePixels();

    auto cam_dist = (height*0.5)/std::tan(cam->half_angle);
    auto sampling_weight = sampling_radius == 0 ? 1.0 : 0.25 / (sampling_radius*sampling_radius);
    auto sampling_r      = sampling_radius == 0 ? 1.0 : sampling_radius*2;
    auto sampling_offset = sampling_radius == 0 ? 1.0 : 0.25/sampling_radius;

#ifndef NDEBUG
    TIC(1)
#endif

#ifdef USE_CUDA
    if (mode == ComputationMode::GPU)
        renderGpu(width, height,
                  cam_dist,
                  sampling_r, sampling_offset, sampling_weight);
    else
#endif
        renderCpu(width, height,
                  cam_dist,
                  sampling_r, sampling_offset, sampling_weight);

    if (stop_rendering)
        stop_rendering = false;

    _is_rendering = false;

#ifndef NDEBUG
    TOC(1)
#endif

}

void Scene::renderCpu(int const &width,
                      int const &height,
                      double const &cam_dist,
                      int const &sampling_r,
                      double const &sampling_offset,
                      double const &sampling_weight)
{
    // TODO Adaptive supersampling
    // TODO Motion Blur
    // TODO Depth of Field
    // TODO Zoom Lens

    // TODO Ambient Occlusion

#pragma omp parallel for num_threads(8)
    for (auto i = 0; i < _param->dimension; ++i)
    {
        if (stop_rendering) // OpenMP not support break statement
            continue;

        auto h = i / width;
        auto w = i % width;
        auto v0 = (height - 1) * 0.5 - h;
        auto u0 = (width - 1) * 0.5 - w;

        Light light(0, 0, 0);

        Ray ray(cam->position, Direction(0, 0, 0));

#ifdef ADAPTIVE_SAMPLING
        auto min_r = std::numeric_limits<double>::max();
        auto min_g = min_r;
        auto min_b = min_r;
        auto max_r = -min_r;
        auto max_g = max_r;
        auto max_b = max_r;
#endif

        for (auto k0 = -sampling_r + 1; k0 < sampling_r; k0 += 2)
        {
            for (auto k1 = -sampling_r + 1; k1 < sampling_r; k1 += 2)
            {
#if defined(ADAPTIVE_SAMPLING) || !defined(JITTER_SAMPLING)
                auto v = v0 + k0  * sampling_offset;
                auto u = u0 + k1  * sampling_offset;
#else
                auto v = v0 + (k0 + rnd()) * sampling_offset;
                auto u = u0 + (k1 + rnd()) * sampling_offset;
#endif

                auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                         cam_dist * cam->direction.x;
                auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                         cam_dist * cam->direction.y;
                auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                         cam_dist * cam->direction.z;

                ray.direction.set(x, y, z);

                light += RayTrace::traceCpu(this, ray);

#ifdef ADAPTIVE_SAMPLING
                max_r = std::max(light.x, max_r);
                max_g = std::max(light.y, max_g);
                max_b = std::max(light.z, max_b);
                min_r = std::min(light.x, min_r);
                min_g = std::min(light.y, min_g);
                min_b = std::min(light.z, min_b);
#endif
            }
        }

#ifdef ADAPTIVE_SAMPLING
        if (max_r - min_r < 1e-2 && max_g - min_g < 1e-2 && max_b - max_b < 1e-2)
        {
            pixels.color[i] = light * sampling_w;
            continue;
        }
#endif

#if defined(JITTER_SAMPLING) || defined(APAPTIVE_SAMLING)
#  ifndef JITTER_SAMPLING
#    define JITTER_SAMPLING 5
#  endif
        for (auto n = 1; n < JITTER_SAMPLING; ++n)
        {
            for (auto k0 = -sampling_r + 1; k0 < sampling_r; k0 += 2)
            {
                for (auto k1 = -sampling_r + 1; k1 < sampling_r; k1 += 2)
                {
                    auto v = v0 + (k0 + rnd()) * sampling_offset;
                    auto u = u0 + (k1 + rnd()) * sampling_offset;

                    auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                             cam_dist * cam->direction.x;
                    auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                             cam_dist * cam->direction.y;
                    auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                             cam_dist * cam->direction.z;

                    ray.direction.set(x, y, z);

                    light += trace(ray);

                }
            }
        }
        pixels.color[i] = light * sampling_weight/JITTER_SAMPLING;
#else
        pixels.color[i] = light * sampling_weight;
#endif

        ++_rendering_processing;
    }

}
