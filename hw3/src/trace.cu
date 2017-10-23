#include "trace.hpp"

using namespace px;

__device__
Light RayTrace::traceGpu(const Scene::Param *const &scene,
                         Ray const & ray,
                         double const &refractive_index,
                         int const &depth)
{
////    if (stop_rendering)
////        return bg;
//
//    auto end_range = scene->hit_max_tol;
//    double t;
//    BaseGeometry *obj = nullptr, *tmp_obj;
//
//    for (auto i = 0; i < scene->n_geometries; ++i)
//    {
//        tmp_obj = scene->geometries[i]->hit(ray, 0, end_range, t);
//        if (tmp_obj == nullptr)
//            continue;
//
//        end_range = t;
//        obj = tmp_obj;
//    }
//
//    if (obj == nullptr)
//        return scene->bg;
//
//    auto intersect = ray[t];
//    auto n = obj->normVec(intersect); // norm vector at the hit point
//    Ray I(intersect, {0, 0, 0});      // from hit point to light source
////    Direction h(0, 0, 0);             // half vector
//    Direction r = ray.direction-n*2*ray.direction.dot(n);     // reflect vector
//
//    auto texture_coord = obj->textureCoord(intersect);
//    auto diffuse = obj->material->diffuse(texture_coord);
//    auto specular = obj->material->specular(texture_coord);
//    auto specular_exp = obj->material->specularExp(texture_coord);
//
//    double attenuate;
//    auto L = ambientReflect(scene->ambient, obj->material->ambient(texture_coord));
//    for (auto j = 0; j < scene->n_lights; ++j)
//    {
//        // soft shadow for area light
//        int sampling = scene->lights[j]->type() == BaseLight::Type::AreaLight ? scene->area_light_sampling : 1;
//        int shadow_hit = sampling;
//
//        for (auto i = 0; i < sampling; ++i)
//        {
//            I.direction = scene->lights[j]->dirFrom(intersect, attenuate);
//            // attenuate represents distance from intersect point to the light here
//
////        h = I.direction - ray.direction;
//            for (auto i = 0; i < scene->n_geometries; ++i)
//            {
//                if (scene->geometries[i]->hit(I, scene->hit_min_tol, attenuate, t))
//                {
//                    --shadow_hit;
//                    break;
//                }
//            }
//        }
//
//        if (shadow_hit != 0) // shadow_hit == 0 means that the pixel is completely in shadow.
//        {
//            attenuate = scene->lights[j]->attenuate(intersect) * shadow_hit / sampling;
//
//            if (attenuate == 0)
//                continue;
//
//            L += diffuseReflect(scene->lights[j]->light, diffuse,
//                                I.direction, n) * attenuate;
//
//            L += specularReflect(scene->lights[j]->light, specular,
////                                 h, n, // Blinn Phong model
//                                 I.direction, r, // Phong model
//                                 specular_exp) * attenuate;
//        }
//    }
//
//    if (depth < scene->recursion_depth)
//    {
//        auto ref = obj->material->transmissive(texture_coord);
//        if (ref.x != 0 || ref.y != 0 || ref.z != 0)
////        if (ref.norm2() > 1e-5)
//        {
//            // refract
//            auto cos_theta = ray.direction.dot(n);
//            auto nt = cos_theta > 0 ? (n *= -1, 1.0)
//                                    : (cos_theta *= -1, obj->refractiveIndex(texture_coord));
//
//            auto n_ratio = refractive_index / nt;
//            auto cos_phi_2 =
//                    1 - n_ratio * n_ratio * (1 - cos_theta * cos_theta);
//            if (cos_phi_2 >= 0)
//            {
//                auto t = n * cos_theta;
//                t += ray.direction;
//                t *= n_ratio;
//                if (cos_phi_2 != 0)
//                    t -= n * std::sqrt(cos_phi_2);
//                ref *= trace(scene,
//                             {intersect+t*scene->hit_min_tol, t}, nt, depth + 1);
//                L += ref;
//            }
//        }
//
//        // reflect
//        ref = obj->material->specular(texture_coord);
//        if (ref.x != 0 || ref.y != 0 || ref.z != 0)
////        if (ref.norm2() > 1e-5)
//        {
//            ref *= trace(scene,
//                         {intersect+r*scene->hit_min_tol, r}, refractive_index, depth+1);
//            L += ref;
//        }
//    }
//
//    // indirect diffuse using Monte-Carlo integration
//    if (depth < scene->diffuse_recursion_depth)
//    {
//        auto N = scene->diffuse_sampling / (depth + 1);
//        if (N > 0)
//        {
//            Light indirect_diffuse(0, 0, 0);
//            for (auto i = 0; i < N; ++i)
//            {
//                auto r1 = std::abs(rnd());
//                auto r2 = std::abs(rnd());
//
//                auto s = std::sqrt(1 - r1 * r1);
//                auto phi = 2 * PI * r2;
//                auto x = s * std::cos(phi);
//                auto z = s * std::sin(phi);
//
//                Direction Nt;
//                if (std::abs(n.x) > std::abs(n.y))
//                {
//                    Nt.x = n.z;
//                    Nt.z = -n.x;
//                }
//                else
//                {
//                    Nt.y = -n.z;
//                    Nt.z = n.y;
//                }
//                Direction Nb = n.cross(Nt);
//                Direction sample(x * Nb.x + r1 * n.x + z * Nt.x,
//                                 x * Nb.y + r1 * n.y + z * Nt.y,
//                                 x * Nb.z + r1 * n.z + z * Nt.z);
//
//                indirect_diffuse +=
//                        trace(scene,
//                              {intersect + sample * scene->hit_min_tol, sample},
//                              refractive_index,
//                              depth + 1) * r1;
//            }
//            indirect_diffuse *= diffuse * 2 / (PI * N);
//            L += indirect_diffuse;
//        }
//    }
//    return L;
    return {0, 0, 0};
}
