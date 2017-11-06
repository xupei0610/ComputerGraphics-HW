#include "scene.hpp"

#include <random>

#ifndef NDEBUG
#include <iostream>
#include <chrono>
#define TIC(id) \
    auto _tic_##id = std::chrono::system_clock::now();
#define TOC(id) \
    auto _toc_##id = std::chrono::system_clock::now(); \
    std::cout << "[Info] Process time: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(_toc_##id - _tic_##id).count() \
              << "ms" << std::endl;
#endif

using namespace px;
Light const Scene::DEFAULT_SCENE_BG = Light(0, 0, 0);

Scene::Scene()
    : width(_width),
      height(_height),
      bg(_bg),
      ambient(_ambient),
      sampling_radius(_sampling_radius),
      recursion_depth(_recursion_depth),
      hit_min_tol(_hit_min_tol),
      hit_max_tol(_hit_max_tol),
      cam(_cam),
      _width(DEFAULT_SCENE_WIDTH),
      _height(DEFAULT_SCENE_HEIGHT),
      _bg(DEFAULT_SCENE_BG),
      _ambient(DEFAULT_SCENE_BG),
      _sampling_radius(DEFAULT_SAMPLING_RADIUS),
      _recursion_depth(5),
      _hit_min_tol(DEFAULT_HIT_MIN_TOL),
      _hit_max_tol(DEFAULT_HIT_MAX_TOL),
      _cam(Camera::create())
{
  pixels.data = nullptr;
}

Scene::~Scene()
{
    clearPixels();
}

void Scene::clearPixels()
{
    delete [] pixels.color;
}

void Scene::setSceneSize(int const &width, int const &height)
{
    if (width < 0)
        throw std::invalid_argument("Failed to set scene width as a negative value.");
    if (height < 0)
        throw std::invalid_argument("Failed to set scene height as a negative value.");
    _width = width;
    _height = height;
}

void Scene::setBackground(double const &light_r,
                          double const &light_g,
                          double const &light_b)
{
    _bg = Light(light_r, light_g, light_b);
}

void Scene::setBackground(Light const &light)
{
    _bg = light;
}

void Scene::setCamera(std::shared_ptr<Camera> const &cam)
{
    _cam = cam;
}

void Scene::setAmbientLight(Light const &c)
{
    _ambient = c;
}

void Scene::setSamplingRadius(int const &radius)
{
    if (radius < 0)
        throw std::invalid_argument("Failed to set sampling radius as a negative value.");
    _sampling_radius = radius;
}

void Scene::setRecursionDepth(int const &depth)
{
    if (depth < 0)
        throw std::invalid_argument("Failed to set recursion depth as a negative value.");
    _recursion_depth = depth;
}

void Scene::setHitMinTol(double const &tol)
{
    _hit_min_tol = tol;
}

void Scene::setHitMaxTol(double const &tol)
{
    _hit_max_tol = tol;
}

Light Scene::ambientReflect(Light const &light,
                            Light const &material)
{
    return light * material;
}

Light Scene::diffuseReflect(Light const &light,
                            Light const &material,
                            Direction const &to_light_vec,
                            Direction const &norm_vec)
{
    auto cosine = to_light_vec.dot(norm_vec);
    if (cosine < 0) cosine *= - 1;
    return light * material * cosine;
}

Light Scene::specularReflect(Light const &light,
                             Light const &material,
                             Direction const &to_light_vec,
                             Direction const &reflect_vec,
                             int const &specular_exponent)
{
    auto f = to_light_vec.dot(reflect_vec);
    if (f < 0)
        return Light(0,0,0);
    return light*material*std::pow(f, specular_exponent);
}

void Scene::render()
{
    if (pixels.data != nullptr)
        clearPixels();

    pixels.color = new Color[height * width];

    auto d = (height*0.5)/std::tan(cam->half_angle);
    auto sampling_w = sampling_radius == 0 ? 1.0 : 0.25 / (sampling_radius*sampling_radius);
    auto sampling_d = sampling_radius == 0 ? 1.0 : sampling_radius*2;
    auto sampling_offset = sampling_radius == 0 ? 1.0 : 0.25/sampling_radius;
    auto size = height * width;

    std::random_device rd;
    std::mt19937 sd(rd());
    std::uniform_real_distribution<double> rand(-1, 1);

#ifndef NDEBUG
    TIC(1)
#endif

#pragma omp parallel for num_threads(8)
    for (auto i = 0; i < size; ++i)
    {
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

        for (auto k0 = -sampling_d + 1; k0 < sampling_d; k0 += 2)
        {
            for (auto k1 = -sampling_d + 1; k1 < sampling_d; k1 += 2)
            {
#if defined(ADAPTIVE_SAMPLING) || !defined(JITTER_SAMPLING)
                auto v = v0 + k0  * sampling_offset;
                auto u = u0 + k1  * sampling_offset;
#else
                auto v = v0 + (k0 + rand(sd)) * sampling_offset;
                auto u = u0 + (k1 + rand(sd)) * sampling_offset;
#endif

                auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                         d * cam->direction.x;
                auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                         d * cam->direction.y;
                auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                         d * cam->direction.z;

                ray.direction.set(x, y, z);

                light += trace(ray);

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
#ifndef JITTER_SAMPLING
#define JITTER_SAMPLING 5
#endif
        for (auto n = 1; n < JITTER_SAMPLING; ++n)
        {
            for (auto k0 = -sampling_d + 1; k0 < sampling_d; k0 += 2)
            {
                for (auto k1 = -sampling_d + 1; k1 < sampling_d; k1 += 2)
                {
                    auto v = v0 + (k0 + rand(sd)) * sampling_offset;
                    auto u = u0 + (k1 + rand(sd)) * sampling_offset;

                    auto x = u * cam->right_vector.x + v * cam->up_vector.x +
                             d * cam->direction.x;
                    auto y = u * cam->right_vector.y + v * cam->up_vector.y +
                             d * cam->direction.y;
                    auto z = u * cam->right_vector.z + v * cam->up_vector.z +
                             d * cam->direction.z;

                    ray.direction.set(x, y, z);

                    light += trace(ray);

                }
            }
        }
        pixels.color[i] = light*sampling_w/JITTER_SAMPLING;
        std::cout << JITTER_SAMPLING << std::endl;
#else
        pixels.color[i] = light * sampling_w;
#endif
    }

#ifndef NDEBUG
    TOC(1)
#endif
}

Light Scene::trace(Ray const & ray,
                   double const &refractive_index,
                   int const &depth)
{
    auto end_range = hit_max_tol;
    double t;
    BaseObject * obj = nullptr;
    for (const auto & o : objects)
    {
        if (o->hit(ray, 0, end_range, t))
        {
            end_range = t;
            obj = o.get();
        }
    }

    if (obj == nullptr)
        return bg;

    auto intersect = ray[t];
    auto n = obj->normVec(intersect); // norm vector at the hit point
    Ray I(intersect, {0, 0, 0});      // from hit point to light source
//    Direction h(0, 0, 0);             // half vector
    Direction r;     // reflect vector

    double attenuate;
    auto L = ambientReflect(ambient, obj->ambient(intersect));
    for (const auto & light : lights)
    {
        I.direction = light->dirFrom(intersect);
        attenuate = light->distTo(intersect); // distance from intersect point to the light

//        h = I.direction - ray.direction;

        bool shadow = false;

        for (auto const &o: objects)
        {
            if (o->hit(I, hit_min_tol, attenuate, t))
            {
                shadow = true;
                break;
            }
        }

        if (shadow == false)
        {
            attenuate = light->attenuate(intersect);
            if (attenuate == 0)
                continue;

            L += diffuseReflect(light->light,
                                obj->diffuse(intersect),
                                I.direction, n) * attenuate;

            r = I.direction-n*2*I.direction.dot(n);
            L += specularReflect(light->light,
                                 obj->specular(intersect),
//                                 h, n, // Blinn Phong model
                                 ray.direction, r, // Phong model
                                 obj->material->specularExponent()) * attenuate;
        }

    }

    if (depth < recursion_depth)
    {
        // reflect
        auto ref = obj->specular(intersect);
        if (ref.x != 0 || ref.y != 0 || ref.z != 0)
//        if (ref.norm2() > 1e-5)
        {
            r = ray.direction-n*2*ray.direction.dot(n);
            ref *= trace({intersect+r*hit_min_tol, r}, refractive_index, depth+1);
            L += ref;
        }

        // refract
        ref = obj->transmissive(intersect);
        if (ref.x != 0 || ref.y != 0 || ref.z != 0)
//        if (ref.norm2() > 1e-5)
        {
            auto cos_theta = ray.direction.dot(n);
            auto nt = cos_theta > 0 ? (n *= -1, 1.0)
                                    : (cos_theta *= -1, obj->material->refractiveIndex());

            auto n_ratio = refractive_index / nt;
            auto cos_phi_2 =
                    1 - n_ratio * n_ratio * (1 - cos_theta * cos_theta);
            if (cos_phi_2 >= 0)
            {
                auto t = n * cos_theta;
                t += ray.direction;
                t *= n_ratio;
                if (cos_phi_2 != 0)
                    t -= n * std::sqrt(cos_phi_2);
                ref *= trace({intersect+t*hit_min_tol, t}, nt, depth + 1);
                L += ref;
            }
        }
    }
    return L;
}
