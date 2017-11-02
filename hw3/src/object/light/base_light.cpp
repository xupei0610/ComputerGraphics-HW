#include "object/light/base_light.hpp"

using namespace px;

LightObj::LightObj(void * obj,
                   LightType const &type, Light const &light,
                   fnAttenuate_t const &fn_attenuate, fnDirFrom_t const &fn_dir_from)
        : obj(obj), fn_attenuate(fn_attenuate), fn_dir_from(fn_dir_from),
          type(type), light(light)
{}

BaseLight::BaseLight(LightType const &type, Light const &light)
        : _light(light),
          dev_ptr(nullptr), need_upload(true),
          type(type)
{}

void BaseLight::setLight(Light const &light)
{
    _light = light;
#ifdef USE_CUDA
    need_upload = true;
#endif
}

void BaseLight::setLight(PREC const &r, PREC const &g, PREC const &b)
{
    _light.x = r;
    _light.y = g;
    _light.z = b;
#ifdef USE_CUDA
    need_upload = true;
#endif
}

void BaseLight::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;
    PX_CUDA_CHECK(cudaFree(dev_ptr));
    dev_ptr = nullptr;
    need_upload = true;
#endif
}
