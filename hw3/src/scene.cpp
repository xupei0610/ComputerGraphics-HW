#include "scene.hpp"
#include "trace.hpp"


using namespace px;

Scene::Param::Param(int const &width, int const &height, int const &dimension,
                    Light bg, Light ambient,
                    int const &recursion_depth,
                    int const &area_light_sampling,
                    int const &diffuse_sampling,
                    int const &diffuse_recursion_depth,
                    PREC const &hit_min_tol, PREC const &hit_max_tol)
        : width(width), height(height), dimension(dimension),
          bg(bg), ambient(ambient),
          recursion_depth(recursion_depth),
          area_light_sampling(area_light_sampling),
          diffuse_sampling(diffuse_sampling),
          diffuse_recursion_depth(diffuse_recursion_depth),
          hit_min_tol(hit_min_tol), hit_max_tol(hit_max_tol)
{}


Light const Scene::DEFAULT_SCENE_BG = Light(0, 0, 0);
int const Scene::DEFAULT_SCENE_WIDTH = 640;
int const Scene::DEFAULT_SCENE_HEIGHT = 480;
int const Scene::DEFAULT_SAMPLING_RADIUS = 0;
int const Scene::DEFAULT_AREA_LIGHT_SAMPLING = 16;
int const Scene::DEFAULT_DIFFUSE_SAMPLING = 16;
int const Scene::DEFAULT_RECURSION_DEPTH = 5;
int const Scene::DEFAULT_DIFFUSE_RECURSION_DEPTH = 0;
PREC const Scene::DEFAULT_HIT_MIN_TOL = DOUBLE_EPSILON; // minimum tol to check if a ray hits an object or not.
PREC const Scene::DEFAULT_HIT_MAX_TOL = std::numeric_limits<PREC>::max(); // maximum tol to check if a ray hits an object or not

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
          _pixels(nullptr),
          _pixels_gpu(nullptr),
          is_rendering(_is_rendering),
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
          _rendering_progress(nullptr),
          _rendering_time(0),
          _sampling_radius(DEFAULT_SAMPLING_RADIUS),
          _mode(DEFAULT_COMPUTATION_MODE),
          _cam(Camera::create()),
          _allocate_gpu_pixels(false),
          _gpu_stop_flag(nullptr),
          _cpu_stop_flag(false)
{
    pixels.color = nullptr;
#ifdef USE_CUDA
    PX_CUDA_CHECK(cudaHostAlloc(&_gpu_stop_flag, sizeof(bool),
                                cudaHostAllocMapped));
    PX_CUDA_CHECK(cudaHostAlloc(&_rendering_progress, sizeof(int),
                                cudaHostAllocMapped));
#else
    _gpu_stop_flag = nullptr;
    _rendering_progress = new int;
#endif
}

Scene::~Scene()
{
    clearPixels();
#ifdef USE_CUDA
    if (_gpu_stop_flag != nullptr)
        PX_CUDA_CHECK(cudaFreeHost(_gpu_stop_flag));
#endif
    if (_rendering_progress != nullptr)
    {
#ifdef USE_CUDA
        PX_CUDA_CHECK(cudaFreeHost(_rendering_progress));
#else
        delete _rendering_progress;
#endif
    }
}

void Scene::clearPixels()
{
#ifdef USE_CUDA
    if (_pixels_gpu != nullptr)
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
}

void Scene::setBackground(PREC const &light_r,
                          PREC const &light_g,
                          PREC const &light_b)
{
    _param->bg = Light(light_r, light_g, light_b);
}

void Scene::setBackground(Light const &light)
{
    _param->bg = light;

}

void Scene::setCamera(std::shared_ptr<Camera> const &cam)
{
    _cam = cam;
}

void Scene::setAmbientLight(Light const &c)
{
    _param->ambient = c;

}

void Scene::addAmbientLight(Light const &c)
{
    _param->ambient += c;

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

}

void Scene::setDiffuseSampling(int const &depth, int const &n)
{
    _param->diffuse_recursion_depth = depth;
    _param->diffuse_sampling = n;

}

void Scene::setRecursionDepth(int const &depth)
{
    if (depth < 0)
        throw std::invalid_argument("Failed to set recursion depth as a negative value.");
    _param->recursion_depth = depth;

}

void Scene::setHitMinTol(PREC const &tol)
{
    _param->hit_min_tol = tol;

}

void Scene::setHitMaxTol(PREC const &tol)
{
    _param->hit_max_tol = tol;

}

bool Scene::setComputationMode(ComputationMode const &mode)
{
#ifdef USE_CUDA
    _mode = mode;
#else
    if (_mode == ComputationMode::GPU)
        return false;
#endif
    return true;
}

void Scene::stopRendering()
{
    if (_is_rendering)
    {
#ifdef USE_CUDA
        if (_gpu_stop_flag != nullptr)
        {
            *_gpu_stop_flag = true;
        }
#endif
        _cpu_stop_flag = true;
    }
}

int Scene::renderingProgress()
{
    return *_rendering_progress;
}

int Scene::renderingTime()
{
    return _rendering_time;
}

void Scene::render()
{
    _is_rendering = true;
#ifdef USE_CUDA
    *_gpu_stop_flag = false;
#endif
    _cpu_stop_flag = false;
    *_rendering_progress = 0;

    clearPixels();
    allocatePixels();

    auto cam_dist = (height*0.5)/std::tan(cam->half_angle);
    auto sampling_weight = sampling_radius == 0 ? 1.0 : 0.25 / (sampling_radius*sampling_radius);
    auto sampling_r      = sampling_radius == 0 ? 1.0 : sampling_radius*2;
    auto sampling_offset = sampling_radius == 0 ? 1.0 : 0.25/sampling_radius;

#ifdef USE_CUDA
    if (mode == ComputationMode::GPU)
    {
        renderGpu(width, height,
                  cam_dist,
                  sampling_r, sampling_offset, sampling_weight);
    }
    else
#endif
        renderCpu(width, height,
                  cam_dist,
                  sampling_r, sampling_offset, sampling_weight);

    if (_cpu_stop_flag)
        _cpu_stop_flag = false;
#ifdef USE_CUDA
    if (*_gpu_stop_flag)
        *_gpu_stop_flag = false;
#endif
    _is_rendering = false;


}

void Scene::renderCpu(int const &width,
                      int const &height,
                      PREC const &cam_dist,
                      int const &sampling_r,
                      PREC const &sampling_offset,
                      PREC const &sampling_weight)
{
    // TODO Adaptive supersampling
    // TODO Motion Blur
    // TODO Depth of Field
    // TODO Zoom Lens

    // TODO Ambient Occlusion

    std::cout << "\r[Info] Begin rendering..." << std::flush;
#ifndef NDEBUG
    TIC(1)
#endif

#pragma omp parallel for num_threads(8)
    for (auto i = 0; i < _param->dimension; ++i)
    {
        if (_cpu_stop_flag) // OpenMP not support break statement
            continue;

        auto h = i / width;
        auto w = i % width;
        auto v0 = (height - 1) * 0.5 - h;
        auto u0 = (width - 1) * 0.5 - w;

        Light light(0, 0, 0);

        Ray ray(cam->position, Direction(0, 0, 0));

#ifdef ADAPTIVE_SAMPLING
        auto min_r = std::numeric_limits<PREC>::max();
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
                auto v = v0 + (k0 + rnd::rnd_cpu()) * sampling_offset;
                auto u = u0 + (k1 + rnd::rnd_cpu()) * sampling_offset;
#endif

                auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                         cam_dist * cam->direction.x;
                auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                         cam_dist * cam->direction.y;
                auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                         cam_dist * cam->direction.z;

                ray.direction.set(x, y, z);

                light += RayTrace::traceCpu(_cpu_stop_flag,
                                            this, ray);

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
                    auto v = v0 + (k0 + rnd::rnd_cpu()) * sampling_offset;
                    auto u = u0 + (k1 + rnd::rnd_cpu()) * sampling_offset;

                    auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                             cam_dist * cam->direction.x;
                    auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                             cam_dist * cam->direction.y;
                    auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                             cam_dist * cam->direction.z;

                    ray.direction.set(x, y, z);

                    light += RayTrace::traceCpu(this, ray);

                }
            }
        }
        pixels.color[i] = light * sampling_weight/JITTER_SAMPLING;
#else
        pixels.color[i] = light * sampling_weight;
#endif

        ++(*_rendering_progress);
    }

#ifndef NDEBUG
    TOC(1)
#endif
}
