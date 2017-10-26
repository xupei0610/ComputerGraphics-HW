#ifndef PX_CG_SCENE_HPP
#define PX_CG_SCENE_HPP


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


#include "object/base_object.hpp"
#include "object/geometry.hpp"
#include "object/light.hpp"

#include <unordered_set>
#include <string>
#include <memory>
#include <limits>
#include <atomic>

namespace px
{
class Scene;
} // namespace px

class px::Scene
{
public:
    enum class ComputationMode
    {
        CPU,
        GPU
    };
    struct Param
    {
        int width;
        int height;
        int dimension;
        Light bg;
        Light ambient;
        int recursion_depth;
        int area_light_sampling;
        int diffuse_sampling;
        int diffuse_recursion_depth;
        PREC hit_min_tol;
        PREC hit_max_tol;

        int n_geometries;
        BaseGeometry ***geometries;
        int n_lights;
        BaseLight ***lights;

        PX_CUDA_CALLABLE
        Param() = default;
        Param(int const &width, int const &height,
              int const &dimension,
              Light bg, Light ambient,
              int const &recursion_depth,
              int const &area_light_sampling,
              int const &diffuse_sampling,
              int const &diffuse_recursion_depth,
              PREC const &hit_min_tol, PREC const &hit_max_tol);
    };

    struct Color
    {
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;

        template<typename T>
        PX_CUDA_CALLABLE
        std::uint8_t clamp(T const & x)
        {
            return x > 255 ? 255 : x < 0 ? 0 : static_cast<std::uint8_t>(x);
        }

        PX_CUDA_CALLABLE
        Color &operator=(Light const &c)
        {
            r = clamp(c.x*255);
            g = clamp(c.y*255);
            b = clamp(c.z*255);
            return *this;
        }
    };

    union Pixels
    {
        Color* color;
        std::uint8_t* data;
    };

private:
    Param *_param;
    Color *_pixels;
    Color *_pixels_gpu;

public:

    int static constexpr DEFAULT_SCENE_WIDTH = 640;
    int static constexpr DEFAULT_SCENE_HEIGHT = 480;
    Light static const DEFAULT_SCENE_BG; // {0, 0, 0}
    int static constexpr DEFAULT_SAMPLING_RADIUS = 1;
    int static constexpr DEFAULT_AREA_LIGHT_SAMPLING = 16;
    int static constexpr DEFAULT_DIFFUSE_SAMPLING = 16;
    int static constexpr DEFAULT_RECURSION_DEPTH = 5;
    int static constexpr DEFAULT_DIFFUSE_RECURSION_DEPTH = 0;
    PREC static constexpr DEFAULT_HIT_MIN_TOL = 1e-3; // minimum tol to check if a ray hits an object or not.
    PREC static constexpr DEFAULT_HIT_MAX_TOL = std::numeric_limits<PREC>::max(); // maximum tol to check if a ray hits an object or not
#ifdef USE_CUDA
    ComputationMode static constexpr DEFAULT_COMPUTATION_MODE = ComputationMode::GPU;
#else
    ComputationMode static constexpr DEFAULT_COMPUTATION_MODE = ComputationMode::CPU;
#endif

public:
    bool const &is_rendering;

    std::unordered_set<std::shared_ptr<Geometry> > geometries;
    std::unordered_set<std::shared_ptr<BaseLight> > lights;

    Pixels pixels;

    int const &width;
    int const &height;
    int const &dimension;

    Light const &bg;
    Light const &ambient;
    int const &sampling_radius;
    int const &area_light_sampling;
    int const &diffuse_sampling;
    int const &recursion_depth;
    int const &diffuse_recursion_depth;
    PREC const &hit_min_tol;
    PREC const &hit_max_tol;
    ComputationMode const &mode;

    std::shared_ptr<Camera> const & cam;

    Scene();
    ~Scene();

    void clearPixels();

    void setSceneSize(int const &width, int const &height);

    void setBackground(PREC const &light_r,
                       PREC const &light_g,
                       PREC const &light_b);
    void setBackground(Light const &light);

    void setCamera(std::shared_ptr<Camera> const &cam);

    void setAmbientLight(Light const &light);
    void addAmbientLight(Light const &light);

    void setSamplingRadius(int const &radius);
    void setAreaLightSampling(int const &n);
    void setDiffuseSampling(int const &depth, int const &n);
    void setRecursionDepth(int const &depth);
    void setHitMinTol(PREC const &tol);
    void setHitMaxTol(PREC const &tol);

    void render();
    void stopRendering();
    int renderProgress();

    void setComputationMode(ComputationMode const &mode);

protected:
    void renderCpu(int const &width, int const &height,
                   PREC const &cam_dist,
                   int const &sampling_r,
                   PREC const &sampling_offset,
                   PREC const &sampling_weight);
    void renderGpu(int const &width, int const &height,
                   PREC const &cam_dist,
                   int const &sampling_r,
                   PREC const &sampling_offset,
                   PREC const &sampling_weight);
    void allocatePixels();

private:
    bool _is_rendering;
    int *_rendering_progress; // not atomic

    int _sampling_radius;

    ComputationMode _mode;
    std::shared_ptr<Camera> _cam;

    bool _allocate_gpu_pixels;
    bool *_gpu_stop_flag;

    std::atomic<bool>_cpu_stop_flag;
};
#endif // PX_CG_SCENE_HPP