#include "trace.hpp"

using namespace px;
#include <cfloat>

PX_CUDA_CALLABLE
RayTrace::TraceQueue::Node::Node(Ray const &ray,
                                 Light const &coef,
                                 int const &depth)
    : ray(ray), coef(coef), depth(depth)
{}

PX_CUDA_CALLABLE
RayTrace::TraceQueue::TraceQueue(Node *const &ptr, int const &size)
    : ptr(ptr), n(0), size(size)
{}

PX_CUDA_CALLABLE
bool RayTrace::TraceQueue::prepend(Point const &ray_o, Direction const &ray_d,
                                   Light const &coef, int const &depth)
{
    if (n < size)
    {
        ptr[n].ray.original = ray_o;
        ptr[n].ray.direction = ray_d;
        ptr[n].coef = coef;
        ptr[n].depth = depth;
        ++n;
        return true;
    }
    return false;
}

PX_CUDA_CALLABLE
void RayTrace::TraceQueue::pop()
{
    if (n>0)
        --n;
}

#define VAR_LIG(j) (*(scene->lights[j]))
#define VAR_GEO(i) (*(scene->geometries[i]))

__device__
const BaseGeometry *RayTrace::hitCheck(Ray const & ray,
                             const Scene::Param *const &scene,
                             PREC &t)
{
    auto end_range = scene->hit_max_tol;
    const BaseGeometry * obj = nullptr;
    for (auto i = 0; i < scene->n_geometries; ++i)
    {
        auto tmp_obj = VAR_GEO(i)->hit(ray, scene->hit_min_tol, end_range, t);
        if (tmp_obj == nullptr)
            continue;

        end_range = t;
        obj = tmp_obj;
    }
    return obj;
}

__device__
Light RayTrace::reflect(Point const &intersect,
                        Direction const &direction,
                        Point const &texture_coord,
                        const BaseGeometry *const &obj,
                        const Scene::Param *const &scene,
                        Direction &n, Direction &r)
{
    n = obj->normVec(intersect); // norm vector at the hit point
    r = direction-n*(2*direction.dot(n));     // reflect vector
    Ray I(intersect, {0, 0, 0});      // from hit point to light source
    Direction h(0, 0, 0);             // half vector

    auto diffuse = obj->material()->diffuse(texture_coord.x, texture_coord.y, texture_coord.z);
    auto specular = obj->material()->specular(texture_coord.x, texture_coord.y, texture_coord.z);
    auto specular_exp = obj->material()->specularExp(texture_coord.x, texture_coord.y, texture_coord.z);

    auto L = ambientReflect(scene->ambient, obj->material()->ambient(texture_coord));

    PREC dist, t;
    for (auto j = 0; j < scene->n_lights; ++j)
    {
        // soft shadow for area light
        int sampling = VAR_LIG(j)->type() == BaseLight::Type::AreaLight ? scene->area_light_sampling : 1;
        int shadow_hit = sampling;

        for (auto k = 0; k < sampling; ++k)
        {
            I.direction = VAR_LIG(j)->dirFromDevice(intersect, dist);
            // attenuate represents distance from intersect point to the light here

//        h = I.direction - ray.direction;
            for (auto i = 0; i < scene->n_geometries; ++i)
            {
                if (VAR_GEO(i)->hit(I, scene->hit_min_tol, dist, t))
                {
                    --shadow_hit;
                    break;
                }
            }
        }

        if (shadow_hit != 0) // shadow_hit == 0 means that the pixel is completely in shadow.
        {
            dist = VAR_LIG(j)->attenuate(intersect) * shadow_hit / sampling;
            if (dist == 0)
                continue;
            L += diffuseReflect(VAR_LIG(j)->light(), diffuse,
                                I.direction, n) * dist;

            L += specularReflect(VAR_LIG(j)->light(), specular,
//                                 h, n, // Blinn Phong model
                                 I.direction, r, // Phong model
                                 specular_exp) * dist;
        }
    }
    return L;
}

__device__
void RayTrace::recursive(Point const &intersect,
                         TraceQueue::Node const &current,
                         Point const &texture_coord,
                         BaseGeometry const &obj,
                         Direction &n,
                         Direction const &r,
                         TraceQueue &trace,
                         Scene::Param const &scene)
{
    auto ref = obj.material()->transmissive(texture_coord.x, texture_coord.y, texture_coord.z);
    ref *= current.coef;
    if (ref.x > -FLT_MIN && ref.x < FLT_MIN)
        ref.x = 0;
    if (ref.y > -FLT_MIN && ref.y < FLT_MIN)
        ref.y = 0;
    if (ref.z > -FLT_MIN && ref.z < FLT_MIN)
        ref.z = 0;
    if (ref.x != 0 || ref.y != 0 || ref.z != 0)
    {
        auto cos_theta = current.ray.direction.dot(n); // cos_theta

        // ior
        auto ior = cos_theta > 0 ? (n *= PREC(-1), obj.material()->refractiveIndex(texture_coord))
                          : (cos_theta *= -1, PREC(1.0) / obj.material()->refractiveIndex(texture_coord));
        // cos_phi_2
        auto cos_phi_2 = 1 - ior * ior * (1 - cos_theta * cos_theta);
        if (cos_phi_2  >= 0)
        {
            // refractive vector
            if (cos_phi_2 >= 0)
            {
                auto t = n * cos_theta;
                t += current.ray.direction;
                t *= ior;
                if (cos_phi_2 != 0)
                    t -= n * std::sqrt(cos_phi_2);
                trace.prepend(intersect + current.ray.direction*scene.hit_min_tol,
                              t,
                              ref, current.depth+1);
            }

        }
    }

    ref = obj.material()->specular(texture_coord.x, texture_coord.y, texture_coord.z);
    ref *= current.coef;
    if (ref.x > -FLT_MIN && ref.x < FLT_MIN)
        ref.x = 0;
    if (ref.y > -FLT_MIN && ref.y < FLT_MIN)
        ref.y = 0;
    if (ref.z > -FLT_MIN && ref.z < FLT_MIN)
        ref.z = 0;
    if (ref.x != 0 || ref.y != 0 || ref.z != 0)
    {
        trace.prepend(intersect + r*scene.hit_min_tol, r,
                      ref, current.depth + 1);
    }
}

