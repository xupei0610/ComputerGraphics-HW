#include "trace.cuh"

using namespace px;

__device__
RayTrace::TraceQueue::Node::Node(Point const &pos,
                                     Direction const &dir,
                                     Light const &coef,
                                     int const &depth)
        : ray(pos, dir), coef(coef), depth(depth)
{}

__device__
RayTrace::TraceQueue::TraceQueue(Node *const &ptr, int const &size)
        : ptr(ptr), n(0), size(size)
{}

__device__
bool
RayTrace::TraceQueue::prepend(Point const &ray_o, Direction const &ray_d,
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

__device__
Light RayTrace::reflect(Point const &intersect,
                                Point const &texture_coord,
                                const GeometryObj *__restrict__ const &obj,
                                const Scene::Param *__restrict__ const &scene,
                                curandState_t *const &state,
                                Direction const &n, Direction const &r)
{
    Ray I(intersect, {0, 0, 0});      // from hit point2ObjCoord to light source
//    Direction h(0, 0, 0);             // half vector

    auto diffuse = obj->material()->diffuse(texture_coord);
    auto specular = obj->material()->specular(texture_coord);
    auto specular_exp = obj->material()->specularExp(texture_coord);

    auto L = ambientReflect(scene->ambient,
                            obj->material()->ambient(texture_coord));

    PREC dist, t;
    for (auto j = 0; j < scene->n_lights; ++j)
    {
// soft shadow for area light
        int sampling = scene->lights[j]->type == LightType::AreaLight
                       ? scene->area_light_sampling : 1;
        int shadow_hit = sampling;

        for (auto k = 0; k < sampling; ++k)
        {
            I.direction = scene->lights[j]->dirFrom(I.original, dist, state);

//        h = I.direction - direction;
            if (dist > scene->hit_min_tol && scene->geometries->hit(I, scene->hit_min_tol, dist))
                --shadow_hit;
        }

        if (shadow_hit ==
            0) // shadow_hit == 0 means that the pixel is completely in shadow.
            continue;

        dist = scene->lights[j]->attenuate(intersect) * shadow_hit / sampling;
        if (dist == 0)
            continue;

        if (dist < FLT_MAX)
        {
            L += diffuseReflect(scene->lights[j]->light, diffuse,
                                I.direction, n) * dist;

            L += specularReflect(scene->lights[j]->light, specular,
//                                 h, n, // Blinn Phong model
                                 I.direction, r, // Phong model
                                 specular_exp) * dist;
        }
        else
            L = Light(1, 1, 1);
    }
    return L;
}

__device__
void RayTrace::recursive(Point const &intersect,
                             TraceQueue::Node const &current,
                             Point const &texture_coord,
                             GeometryObj const &obj,
                             Direction &n,
                             Direction const &r,
                             TraceQueue &trace,
                             Scene::Param const &scene)
{
    auto ref = obj.material()->transmissive(texture_coord);
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
        auto ior = cos_theta > 0 ? (n *= -1, obj.material()->refractiveIndex(texture_coord))
                                 : (cos_theta *= -1, PREC(1.0) / obj.material()->refractiveIndex(texture_coord));

// cos_phi_2
        auto cos_phi_2 = 1 - ior * ior * (1 - cos_theta * cos_theta);
        if (cos_phi_2 >= 0)
        {
            auto t = n;
            t *= cos_theta;
            t += current.ray.direction;
            t *= ior;
            if (cos_phi_2 != 0)
                t -= n * std::sqrt(cos_phi_2);
            trace.prepend(intersect, t,
                          ref, current.depth + 1);
        }
    }

    ref = obj.material()->specular(texture_coord);
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
