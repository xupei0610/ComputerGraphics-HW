#ifndef PX_CG_GPU_CREATOR_HPP
#define PX_CG_GPU_CREATOR_HPP

#include "util/cuda.hpp"
#include "object/light.hpp"
#include "object/material.hpp"
#include "object/geometry.hpp"

namespace px { namespace GpuCreator {

__global__
void destroyLight(BaseLight *dev_ptr)
{
    delete dev_ptr;
}
//__global__
//void destroyGeometry(BaseGeometry *dev_ptr)
//{
//    delete dev_ptr;
//}
//__global__
//void destroyMaterial(BaseMaterial *dev_ptr)
//{
//    delete dev_ptr;
//}

void destroy(BaseLight * dev_ptr)
{
    destroyLight<<<1, 1>>>(dev_ptr);
}
//void destroy(BaseGeometry * dev_ptr)
//{
//    destroyGeometry<<<1, 1>>>(dev_ptr);
//}
//void destroy(BaseMaterial *dev_ptr)
//{
//    destroyMaterial<<<1, 1>>>(dev_ptr);
//}

__global__
void createDirectionalLight(BaseLight* dev_ptr, Light light, Direction dir)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        delete dev_ptr;
        dev_ptr = new DirectionalLight(light, dir);
    }
}
__global__
void createPointLight(BaseLight* dev_ptr, Light light, Point pos)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        delete dev_ptr;
        dev_ptr = new PointLight(light, pos);
    }
}
__global__
void createSpotLight(BaseLight* dev_ptr,
                     Light light,
                     Point pos,
                     Direction direction,
                     double half_angle1,
                     double half_angle2,
                     double falloff)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        delete dev_ptr;
        dev_ptr = new SpotLight(light, pos, direction,
                                half_angle1, half_angle2,
                                falloff);
    }
}
__global__
void createAreaLight(BaseLight* dev_ptr,
                     Light light,
                     Point center,
                     double radius)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        delete dev_ptr;
        dev_ptr = new AreaLight(light, center, radius);
    }
}

void DirectionalLight(BaseLight* const dev_ptr,
                      Light const &light, Direction const &dir)
{
    createDirectionalLight<<<1, 1>>>(dev_ptr, light, dir);
}
void PointLight(BaseLight* const dev_ptr,
                      Light const &light, Point const &pos)
{
    createPointLight<<<1, 1>>>(dev_ptr, light, pos);
}
void SpotLight(BaseLight* const dev_ptr,
               Light const &light,
               Point const &pos,
               Direction const &direction,
               double const &half_angle1,
               double const &half_angle2,
               double const &falloff)
{
    createSpotLight<<<1, 1>>>(dev_ptr,
            light, pos, direction,
            half_angle1, half_angle2,
            falloff);
}
void AreaLight(BaseLight* const dev_ptr,
               Light const &light,
               Point const &center,
               double const &radius)
{
    createAreaLight<<<1, 1>>>(dev_ptr, light, center, radius);
}

}}

#endif // PX_CG_GPU_CREATOR_HPP
