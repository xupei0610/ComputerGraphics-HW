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

#define LIGHT(j) (*(scene->lights[j]))
#define GEOMETRY(i) (*(scene->geometries[i]))

__device__
const BaseGeometry *RayTrace::hitCheck(Ray const & ray,
                             const Scene::Param *const &scene,
                             Point &intersection)
{
    intersection.y = scene->hit_max_tol;
    const BaseGeometry * obj = nullptr;
    for (auto i = 0; i < scene->n_geometries; ++i)
    {
        auto tmp_obj = GEOMETRY(i)->hit(ray, scene->hit_min_tol, intersection.y, intersection.x);
        if (tmp_obj == nullptr)
            continue;

        intersection.y = intersection.x;
        obj = tmp_obj;
    }
    intersection = ray[intersection.x];
    return obj;
}

__device__
Light RayTrace::reflect(Point const &intersect,
                        Point const &texture_coord,
                        const BaseGeometry *const &obj,
                        const Scene::Param *const &scene,
                        curandState_t * const &state,
                        Direction const &n, Direction const &r)
{
    Ray I(intersect, {0, 0, 0});      // from hit point2ObjCoord to light source
//    Direction h(0, 0, 0);             // half vector

    auto diffuse = obj->material()->diffuse(texture_coord.x, texture_coord.y, texture_coord.z);
    auto specular = obj->material()->specular(texture_coord.x, texture_coord.y, texture_coord.z);
    auto specular_exp = obj->material()->specularExp(texture_coord.x, texture_coord.y, texture_coord.z);

    auto L = ambientReflect(scene->ambient, obj->material()->ambient(texture_coord));

    PREC dist, t;
    for (auto j = 0; j < scene->n_lights; ++j)
    {
        // soft shadow for area light
        int sampling = LIGHT(j)->type() == BaseLight::Type::AreaLight ? scene->area_light_sampling : 1;
        int shadow_hit = sampling;

        for (auto k = 0; k < sampling; ++k)
        {
            I.direction = LIGHT(j)->dirFromDevice(I.original, dist, state);
            // attenuate represents distance from intersect point2ObjCoord to the light here

//        h = I.direction - ray.direction;
            for (auto i = 0; i < scene->n_geometries; ++i)
            {
                if (GEOMETRY(i)->hit(I, scene->hit_min_tol, dist, t))
                {
                    --shadow_hit;
                    break;
                }
            }
        }

        if (shadow_hit == 0) // shadow_hit == 0 means that the pixel is completely in shadow.
            continue;

        dist = LIGHT(j)->attenuate(intersect) * shadow_hit / sampling;
        if (dist == 0)
            continue;
        L += diffuseReflect(LIGHT(j)->light(), diffuse,
                            I.direction, n) * dist;

        L += specularReflect(LIGHT(j)->light(), specular,
//                                 h, n, // Blinn Phong model
                             I.direction, r, // Phong model
                             specular_exp) * dist;
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
    if (ref.x > -EPSILON && ref.x < EPSILON)
        ref.x = 0;
    if (ref.y > -EPSILON && ref.y < EPSILON)
        ref.y = 0;
    if (ref.z > -EPSILON && ref.z < EPSILON)
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
                auto t = n;
                t *= cos_theta;
                t += current.ray.direction;
                t *= ior;
                if (cos_phi_2 != 0)
                    t -= n * std::sqrt(cos_phi_2);
                trace.prepend(intersect, t,
                              ref, current.depth+1);
            }

        }
    }

    ref = obj.material()->specular(texture_coord.x, texture_coord.y, texture_coord.z);
    ref *= current.coef;
    if (ref.x > -EPSILON && ref.x < EPSILON)
        ref.x = 0;
    if (ref.y > -EPSILON && ref.y < EPSILON)
        ref.y = 0;
    if (ref.z > -EPSILON && ref.z < EPSILON)
        ref.z = 0;
    if (ref.x != 0 || ref.y != 0 || ref.z != 0)
    {
        trace.prepend(intersect, r,
                      ref, current.depth + 1);
    }
}

#undef GEOMETRY
#undef LIGHT
