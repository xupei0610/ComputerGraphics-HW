#include "trace.hpp"

using namespace px;

Light RayTrace::traceCpu(bool const &stop_flag,
                         const Scene *const &scene,
                         Ray const &ray,
                         int const &depth)
{
    if (stop_flag)
        return {0, 0, 0};

    auto end_range = scene->hit_max_tol;
    Point intersect;
    const BaseGeometry *obj = scene->geometries->hit(ray,
                                                     scene->hit_min_tol,
                                                     end_range, intersect);
    if (obj == nullptr)
        return scene->bg;

    bool double_face;
    auto n = obj->normal(intersect, double_face); // norm vector at the hit point2ObjCoord
    Ray I(intersect, {0, 0, 0});      // from hit point2ObjCoord to light source
    Direction h;             // half vector
//    Direction r;
    auto texture_coord = obj->textureCoord(intersect);
    auto diffuse = obj->material()->diffuse(texture_coord);
    auto specular = obj->material()->specular(texture_coord);
    auto shininess = obj->material()->Shininess(texture_coord);

    PREC attenuate;
    auto L = ambientReflect(scene->ambient, obj->material()->ambient(texture_coord));
    for (const auto &light : scene->lights)
    {
        // soft shadow for area light
        int sampling = light->type == LightType::AreaLight ? scene->area_light_sampling : 1;
        int shadow_hit = sampling;

        for (auto k = 0; k < sampling; ++k)
        {
            I.direction = light->dirFrom(intersect, attenuate);
            // attenuate represents distance from intersect point2ObjCoord to the light here

            if (attenuate > scene->hit_min_tol && scene->geometries->hit(I, scene->hit_min_tol, attenuate))
                --shadow_hit;
        }

        if (shadow_hit == 0) // shadow_hit == 0 means that the pixel is completely in shadow.
            continue;

        attenuate = light->attenuate(intersect) * shadow_hit / sampling ;

        if (attenuate == 0)
            continue;

        L += diffuseReflect(light->light(), diffuse,
                            I.direction, n, double_face) * attenuate;

//        r = I.direction-n*(2*I.direction.dot(n));
        h = I.direction - ray.direction;
        L += specularReflect(light->light(), specular,
                                 h, n, // Blinn Phong model
//                             ray.direction, r, // Phong model
                             shininess) * attenuate;

    }

    if (depth < scene->recursion_depth)
    {
        auto ref = obj->material()->transmissive(texture_coord);
//        ref *= coef;
        if (ref.x > -EPSILON && ref.x < EPSILON)
            ref.x = 0;
        if (ref.y > -EPSILON && ref.y < EPSILON)
            ref.y = 0;
        if (ref.z > -EPSILON && ref.z < EPSILON)
            ref.z = 0;
        if (ref.x != 0 || ref.y != 0 || ref.z != 0)
        {
            // refract
            auto cos_theta = ray.direction.dot(n);
            auto ior = cos_theta > 0 ? (n *= -1, obj->material()->refractiveIndex(texture_coord))
                                    : (cos_theta *= -1, 1.0 / obj->material()->refractiveIndex(texture_coord));

            auto cos_phi_2 =
                    1 - ior*ior * (1 - cos_theta * cos_theta);
            if (cos_phi_2 >= 0)
            {
                auto t = n;
                t *= cos_theta;
                t += ray.direction;
                t *= ior;
                if (cos_phi_2 != 0)
                    t -= n * std::sqrt(cos_phi_2);
                ref *= traceCpu(stop_flag,
                                scene,
                                {intersect, t},
                                //ref,
                                depth + 1);
                L += ref;
            }
        }

        // reflect
//        specular *= coef;
        if (specular.x > -EPSILON && specular.x < EPSILON)
            specular.x = 0;
        if (specular.y > -EPSILON && specular.y < EPSILON)
            specular.y = 0;
        if (specular.z > -EPSILON && specular.z < EPSILON)
            specular.z = 0;
        if (specular.x != 0 || specular.y != 0 || specular.z != 0)
        {
            specular *= traceCpu(stop_flag,
                                 scene,
                                 {intersect, ray.direction-n*(2*ray.direction.dot(n))},
                                 //specular,
                                 depth+1);
            L += specular;
        }

        // indirect diffuse using Monte-Carlo integration
        if (depth < scene->diffuse_recursion_depth)
        {
            auto N = scene->diffuse_sampling / (depth + 1);
            if (N > 0)
            {
                Light indirect_diffuse(0, 0, 0);
                for (auto i = 0; i < N; ++i)
                {
                    auto r1 = std::abs(rnd::rnd_cpu());
                    auto r2 = std::abs(rnd::rnd_cpu());

                    auto s = std::sqrt(1 - r1 * r1);
                    auto phi = 2 * PI * r2;
                    auto x = s * std::cos(phi);
                    auto z = s * std::sin(phi);

                    Direction Nt;
                    if (std::abs(n.x) > std::abs(n.y))
                    {
                        Nt.x = n.z;
                        Nt.y = 0;
                        Nt.z = -n.x;
                    }
                    else
                    {
                        Nt.x = 0;
                        Nt.y = -n.z;
                        Nt.z = n.y;
                    }
                    Direction Nb = n.cross(Nt);
                    Direction sample(x * Nb.x + r1 * n.x + z * Nt.x,
                                     x * Nb.y + r1 * n.y + z * Nt.y,
                                     x * Nb.z + r1 * n.z + z * Nt.z);

                    indirect_diffuse += traceCpu(stop_flag,
                                                 scene, {intersect, sample},
//                                                 coef,
                                                 depth + 1) * r1;
                }
                indirect_diffuse *= diffuse * 2 / (PI * N);
                L += indirect_diffuse;
            }
        }
    }

    return L;
}
